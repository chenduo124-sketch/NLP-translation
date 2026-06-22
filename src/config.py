"""
初始化：路径、文件名、超参数
"""
from pathlib import Path


# 0. 特殊符号定义
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
ALL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
ALL_IDXS = [PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX]


# 1. 根目录（自动获取当前 config.py 所在目录的上一级，即项目根目录）
BASE_DIR = Path(__file__).resolve().parent.parent


# 2. 路径定义（直接用 / 拼接）
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"


# 3. 具体文件名路径
TRAIN_DATA = RAW_DATA_DIR / "cmn.txt"
ZH_PROCESSED_DATA = PROCESSED_DATA_DIR / "zh_token.txt"
EN_PROCESSED_DATA = PROCESSED_DATA_DIR / "en_token.txt"
ZH_VOCAB_PATH = MODELS_DIR / "zh_vocab.json"
EN_VOCAB_PATH = MODELS_DIR / "en_vocab.json"
MODEL_WEIGHTS = MODELS_DIR / "transformer_model.pth"


# 4. --- CPU 训练优化版超参数 ---
MAX_LEN = 50           # 句子最长度，缺少的补PAD，多的剪切
D_MODEL = 128          # 向量维度
N_HEAD = 4             # 多头注意力头数，D_MODEL 需要能被 N_HEAD 整除
N_LAYERS = 3           # Transformer 层数：模型的“深度”，3 层足以处理 3 万条语料
DIM_FEEDFORWARD = 512  # 前馈网络中间层维度：通常设为 D_MODEL 的 4 倍，增加非线性表达能力（编码器/解码器）


# 5. 训练控制
BATCH_SIZE = 16         # 适合 CPU 处理的小批次，避免单次运算耗时过长
EPOCHS = 40             # 较小的模型需要多跑几轮来收敛
LEARNING_RATE = 0.0005  # 学习率，适用于Adam进行反向传播。新参数 = 旧参数 - （学习率 * 梯度）