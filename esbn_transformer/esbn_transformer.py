import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# 辅助函数：判断值是否存在
def exists(val):
    return val is not None

# 辅助函数：返回val或默认值d（当val不存在时）
def default(val, d):
    return val if exists(val) else d

# 辅助函数：获取张量数据类型所能表示的最大负数值
def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# 使用einops对多个张量进行相同的变换
def rearrange_all(tensors, *args, **kwargs):
    return map(lambda t: rearrange(t, *args, **kwargs), tensors)

# 组归一化层：对输入数据按照group进行归一化处理
class GroupLayerNorm(nn.Module):
    def __init__(self, dim, groups = 1, eps = 1e-5):
        super().__init__()
        # 初始化时定义归一化偏移和缩放参数
        self.eps = eps
        self.groups = groups
        self.g = nn.Parameter(torch.ones(1, groups, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, groups, dim, 1))

    def forward(self, x):
        # 将数据按照 groups 分组后归一化：格式转换为 [b, groups, dim, n]
        x = rearrange(x, 'b (g d) n -> b g d n', g = self.groups)
        std = torch.var(x, dim = 2, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 2, keepdim = True)
        # 标准化并应用缩放和偏移
        out = (x - mean) / (std + self.eps) * self.g + self.b
        # 恢复原来数据格式
        return rearrange(out, 'b g d n -> b (g d) n')

# 预归一化模块：先归一化再调用后续函数
class PreNorm(nn.Module):
    def __init__(self, dim, fn, groups = 1):
        super().__init__()
        self.norm = GroupLayerNorm(dim, groups = groups)
        self.fn = fn

    def forward(self, x, **kwargs):
        # 先归一化后调用目标函数fn
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 前馈网络层：使用两层卷积和 GELU 激活函数实现
class FeedForward(nn.Module):
    def __init__(self, *, dim, mult = 4, groups = 1):
        super().__init__()
        input_dim = dim * groups
        hidden_dim = dim * mult * groups
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1, groups = groups),
            nn.GELU(),
            nn.Conv1d(hidden_dim, input_dim, 1, groups = groups)
        )

    def forward(self, x):
        return self.net(x)

# 注意力层：实现多头自注意力机制，同时支持可选的因果遮罩和group绑定
class Attention(nn.Module):
    def __init__(self, *, dim, dim_head = 64, heads = 8, causal = False, groups = 1):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.groups = groups
        self.heads = heads
        self.causal = causal
        input_dim = dim * groups
        inner_dim = dim_head * heads * groups

        # 将输入映射到查询、键和值
        self.to_q = nn.Conv1d(input_dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv1d(input_dim, inner_dim * 2, 1, bias = False)
        self.to_out = nn.Conv1d(inner_dim, input_dim, 1)

    def forward(self, x, mask = None, context = None):
        # x: [b, channels, n]，获取batch大小、设备信息以及头、group数量
        n, device, h, g, causal = x.shape[2], x.device, self.heads, self.groups, self.causal
        context = default(context, x)

        # 计算查询、键和值
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = 1))
        q, k, v = rearrange_all((q, k, v), 'b (g h d) n -> (b g h) n d', g = g, h = h)
        q = q * self.scale

        # 计算注意力得分
        sim = einsum('b i d, b j d -> b i j', q, k)

        if g > 1:
            # 当group数大于1时，对各组累加，允许网络通过注意力矩阵绑定符号信息
            sim = rearrange(sim, '(b g h) i j -> b g h i j', g = g, h = h)
            sim = sim.cumsum(dim = 1)
            sim = rearrange(sim, 'b g h i j -> (b g h) i j')

        if exists(mask):
            # 如果提供了mask，则重复mask并应用到注意力矩阵中
            mask = repeat(mask, 'b n -> (b g h) n', h = h, g = g)
            mask = rearrange(mask, 'b n -> b n ()') * rearrange(mask, 'b n -> b () n')
            mask_value = max_neg_value(sim)
            sim = sim.masked_fill(~mask, mask_value)

        if causal:
            # 如果是因果模型，则应用上三角遮罩
            causal_mask = torch.ones((n, n), device = device).triu(1).bool()
            mask_value = max_neg_value(sim)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b g h) n d -> b (g h d) n', h = h, g = g)
        return self.to_out(out)

# Transformer块：封装注意力层和前馈网络，并采用残差连接
class TransformerBlock(nn.Module):
    def __init__(self, *, dim, causal = False, dim_head = 64, heads = 8, ff_mult = 4, groups = 1):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, groups = groups), groups = groups)
        self.ff = PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, groups = groups), groups = groups)

    def forward(self, x, mask = None):
        # 使用残差连接：先经过注意力层，再经过前馈网络后叠加输入
        x = self.attn(x, mask = mask) + x
        x = self.ff(x) + x
        return x

# 主模型：EsbnTransformer
class EsbnTransformer(nn.Module):
    def __init__(self, *, dim, depth, num_tokens, max_seq_len, causal = False, dim_head = 64, heads = 8, ff_mult = 4):
        super().__init__()
        # 模型主要参数设置
        self.dim = dim
        self.max_seq_len = max_seq_len
        # token嵌入层，将token id映射到特征空间
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 位置嵌入层，用于表示序列中各位置的位置信息
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([])
        # 预Transformer块：用于处理输入token初步交互
        self.pre_transformer_block = TransformerBlock(dim = dim, causal = causal, dim_head = dim_head, heads = heads)

        # 参数symbols，在后面用于特定的符号绑定
        self.symbols = nn.Parameter(torch.randn(max_seq_len, dim))

        # 中间Transformer层，group数设置为2支持符号绑定
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim = dim, causal = causal, dim_head = dim_head, heads = heads, groups = 2))

        # 后Transformer块，用于输出转换
        self.post_transformer_block = TransformerBlock(dim = dim, causal = causal, dim_head = dim_head, heads = heads)

        # 输出层：变换为logits（用于分类或下游任务）
        self.to_logits = nn.Sequential(
            Rearrange('b d n -> b n d'),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, mask = None):
        # 输入 x 的形状：[batch, seq_len]
        b, n, d, device = *x.shape, self.dim, x.device
        # 将token id映射到嵌入向量
        x = self.token_emb(x)

        # 生成位置嵌入并加到token嵌入上
        pos_emb = self.pos_emb(torch.arange(n, device = device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb

        # 将数据转为适合卷积操作的形状 [b, d, n]
        x = rearrange(x, 'b n d -> b d n')

        # 预Transformer块处理
        x = self.pre_transformer_block(x, mask = mask)

        # 扩展维度为 [b, 1, d, n]，为后续拼接symbol预留空间
        x = rearrange(x, 'b d n -> b () d n')
        symbols = self.symbols[:, :n]
        symbols = repeat(symbols, 'n d -> b () d n', b = b)
        # 拼接symbols，构成新的特征维度
        x = torch.cat((x, symbols), dim = 1)
        x = rearrange(x, 'b ... n -> b (...) n')

        # 依次通过中间的Transformer层
        for block in self.layers:
            x = block(x, mask = mask)

        # 分离出symbol通道（取第二个通道）
        x = rearrange(x, 'b (s d) n -> b s d n', s = 2)
        x = x[:, 1]

        # 后Transformer块进一步处理
        x = self.post_transformer_block(x, mask = mask)
        # 输出到线性层生成logits
        return self.to_logits(x)
