import torch  # 引入 PyTorch 核心库，用于处理 Tensor (张量) 和自动微分
import torch.nn as nn  # 引入神经网络模块，包含所有的层定义，如 Embedding, Linear, Transformer
import math  # 引入数学库，用于进行平方根等计算 (math.sqrt)


# 定义一个名为 RoPE 的类，继承自 nn.Module，用于实现旋转位置编码
class RoPE(nn.Module):
    def __init__(self, dim: int, base=10000.0):  # 初始化函数，dim 是向量维度，base 是频率基数
        super().__init__()  # 调用父类初始化方法
        self.dim = dim  # 将维度参数存入实例
        self.base = base  # 将基数参数存入实例
        half = self.dim // 2  # RoPE 通过旋转复平面实现，所以需要将维度减半
        # 计算旋转频率：使用 base^-(i/d/2) 的公式生成频率数列
        freqs = torch.pow(base, -torch.arange(0, half) / half)
        # 将频率数据注册为 buffer，这意味着它会被保存到模型参数里，但不会被优化器更新
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播函数，x 是输入张量
        seq_len = x.shape[0]  # 获取当前输入序列的长度
        # 生成一个 0 到 seq_len-1 的位置序列，并在对应设备上转化为浮点数
        pos = torch.arange(seq_len, device=x.device).float()
        # 计算位置与频率的外积，得到每一个位置对应的旋转角度
        freqs = torch.outer(pos, self.freqs)  # 得到 [seq_len, half] 的旋转角度矩阵
        cos = torch.cos(freqs)  # 计算所有频率的 cos 值
        sin = torch.sin(freqs)  # 计算所有频率的 sin 值

        # 将 x 在最后一个维度按 2 分割，变成 x1 和 x2 (即向量的前半部分和后半部分)
        x1, x2 = x.chunk(2, dim=-1)
        # 应用旋转矩阵：rx1 = x1*cos - x2*sin，这是将位置信息注入特征向量的数学手段
        rx1 = x1 * cos.unsqueeze(1) - x2 * sin.unsqueeze(1)
        # 应用旋转矩阵：rx2 = x1*sin + x2*cos
        rx2 = x1 * sin.unsqueeze(1) + x2 * cos.unsqueeze(1)
        # 将旋转后的两个部分拼接回原始维度 [seq_len, batch_size, d_model]
        return torch.cat([rx1, rx2], dim=-1)

    # 定义翻译模型类


class Translation(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, ...):  # 初始化参数设置
        super().__init__()  # 初始化父类
        self.d_model = d_model  # 保存向量维度
        self.pad_token_id = pad_token_id  # 保存填充符号ID

        # Embedding 层：将单词 ID 映射为 d_model 维的稠密向量空间
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = RoPE(d_model)  # 实例化位置编码

        # 核心 Transformer 层：包含 Encoder 和 Decoder 的所有组件
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False  # 指明数据格式为 [序列长度, 批次, 维度]
        )

        # 输出线性层：将模型输出的 128 维特征映射回词表大小，准备进行分类预测
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    # 屏蔽函数：如果输入的 ID 是 0 (PAD)，则返回 True，供 Transformer 忽略
    def generate_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        return seq == self.pad_token_id

    # 掩码函数：生成上三角矩阵，实现 Decoder “看不见未来词”的约束
    def generate_tgt_mask(self, tgt_len: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1)  # 创建上三角
        return mask.masked_fill(mask == 1, float('-inf'))  # 将上三角变为负无穷

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor):
        src_seq_len, batch_size = src_ids.shape  # 获取输入长度和批次大小
        tgt_seq_len, _ = tgt_ids.shape  # 获取目标长度

        # 将 ID 转为向量，并乘上根号 d_model (缩放比例)，这是 Transformer 的标准处理
        src_emb = self.src_embedding(src_ids) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt_ids) * math.sqrt(self.d_model)

        # 注入位置信息
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        # 构造掩码，转置是因为 Transformer 要求 [batch, seq_len] 格式
        src_pad_mask = self.generate_padding_mask(src_ids).T
        tgt_pad_mask = self.generate_padding_mask(tgt_ids).T
        tgt_mask = self.generate_tgt_mask(tgt_seq_len).to(src_ids.device)

        # 调用 Transformer 模型核心：传入源向量、目标向量以及对应的掩码
        transformer_out = self.transformer(
            src=src_emb, tgt=tgt_emb, src_mask=None, tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )

        # 将输出向量映射为词表大小的预测分数
        return self.fc_out(transformer_out)
