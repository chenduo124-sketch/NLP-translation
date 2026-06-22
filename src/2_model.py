import torch
import torch.nn as nn
import math

# 位置编码 RoPE
class RoPE(nn.Module):
    def __init__(self, dim: int, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        # 两两一组
        half = self.dim // 2
        freqs = torch.pow(base, -torch.arange(0, half) / half)
        # 存入张量，不参与梯度更新，不保存进模型参数
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: 输入形状[seq_len, batch_size, d_model]
        seq_len = x.shape[0]
        pos = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(pos, self.freqs)  # [seq_len, half]
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # 拆分前后两半
        x1, x2 = x.chunk(2, dim=-1)
        # 旋转公式
        rx1 = x1 * cos.unsqueeze(1) - x2 * sin.unsqueeze(1)
        rx2 = x1 * sin.unsqueeze(1) + x2 * cos.unsqueeze(1)
        return torch.cat([rx1, rx2], dim=-1)


# 完整Transformer翻译模型
class Translation(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,    # 英文词表总大小 en_vocab
        tgt_vocab_size: int,    # 中文词表总大小 zh_vocab
        d_model: int = 128,    # 词向量/模型隐藏维度，越小训练越快
        nhead: int = 4,        # 多头注意力头数，d_model必须能被nhead整除
        num_encoder_layers: int = 3,  # 编码器堆叠层数
        num_decoder_layers: int = 3,  # 解码器堆叠层数
        dim_feedforward: int = 512,   # FFN中间层维度
        dropout: float = 0.1,
        pad_token_id: int = 0   # <pad> 的id，你词表里固定是0
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # ========== 嵌入层 Embedding ==========
        # 【作用】把数字ID映射为稠密向量，让模型能理解词语语义
        # 【乘sqrt(d_model)原因】平衡词向量与位置编码数值尺度，训练更稳定
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = RoPE(d_model)

        # ========== 官方封装Transformer核心块 ==========
        # 内部包含：多头自注意力 + 前馈网络FFN + 层归一化 + 残差连接
        # 残差连接：防止深层网络梯度消失；层归一化加速收敛
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # 输入格式 [seq_len, batch, feature] 标准格式
        )

        # ========== 输出线性层 ==========
        # 【作用】把Decoder输出的d_model维向量映射回目标词表维度
        # 后续CrossEntropyLoss内置log_softmax，这里不用加softmax
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    # ========== 生成Padding掩码：屏蔽<pad>，不让注意力看到填充符 ==========
    def generate_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [seq_len, batch]
        # True = 遮蔽位置，False = 可见
        return seq == self.pad_token_id

    # ========== 生成解码器上三角掩码：防止看到未来token（翻译生成时只能看前面文字） ==========
    def generate_tgt_mask(self, tgt_len: int) -> torch.Tensor:
        # 上三角全True，遮蔽未来位置，保证生成时序
        mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor):
        """
        训练阶段前向传播
        :param src_ids: 英文输入ID [src_seq_len, batch_size]
        :param tgt_ids: 中文输入ID [tgt_seq_len, batch_size]
        :return: 每个位置所有中文词概率分数 [tgt_seq_len, batch, tgt_vocab_size]
        """
        src_seq_len, batch_size = src_ids.shape
        tgt_seq_len, _ = tgt_ids.shape

        # 1. 词嵌入 + 缩放
        src_emb = self.src_embedding(src_ids) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt_ids) * math.sqrt(self.d_model)

        # 2. 叠加位置编码
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        # 3. 构造各类掩码 + 关键：转置padding mask
        src_pad_mask = self.generate_padding_mask(src_ids).T
        tgt_pad_mask = self.generate_padding_mask(tgt_ids).T
        tgt_mask = self.generate_tgt_mask(tgt_seq_len).to(src_ids.device)

        # 4. Transformer Encoder+Decoder核心计算
        transformer_out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=None,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )

        # 5. 映射到词表维度输出分数
        output = self.fc_out(transformer_out)
        return output


if __name__ == "__main__":
    model = Translation(src_vocab_size=1000, tgt_vocab_size=1000, d_model=128, nhead=4)
    batch = 4
    src = torch.randint(0, 1000, (30, batch))  # [seq=30, batch=4]
    tgt = torch.randint(0, 1000, (35, batch))
    out = model(src, tgt)
    print("输出shape:", out.shape)  # [35, 4, 1000]