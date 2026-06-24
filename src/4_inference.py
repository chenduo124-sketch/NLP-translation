import torch
import jieba  # 确保已安装 jieba
import importlib
from config import *

# 动态加载你的组件
mod = importlib.import_module("2_model")
Translation = mod.Translation
mod2 = importlib.import_module("1_dataset")
Vocab = mod2.Vocab


def translate(model, raw_text, zh_vocab, en_vocab, device):
    model.eval()

    # 1. 使用 jieba 分词，保证逻辑与训练时的一致性
    # 确保此处的分词方式与你训练时的 preprocess 逻辑一致
    tokens = list(jieba.cut(raw_text))

    # 2. 转换为 ID (这里建议在 Vocab 类里加一个处理 UNK 的逻辑)
    src_ids = torch.tensor(zh_vocab.sent_to_ids(tokens, add_sos_eos=False)).unsqueeze(1).to(device)

    # 3. 贪婪解码
    tgt_input = torch.tensor([[en_vocab.sos_id]]).to(device)

    with torch.no_grad():
        for _ in range(50):
            output = model(src_ids, tgt_input)
            next_token = output.argmax(2)[-1, :].item()
            if next_token == en_vocab.eos_id:
                break
            tgt_input = torch.cat([tgt_input, torch.tensor([[next_token]]).to(device)], dim=0)

    return en_vocab.ids_to_sent(tgt_input.squeeze().tolist())


# --- 主逻辑 ---
device = torch.device("cpu")
zh_vocab = Vocab(ZH_VOCAB_PATH)
en_vocab = Vocab(EN_VOCAB_PATH)

model = Translation(src_vocab_size=12447, tgt_vocab_size=7972, d_model=D_MODEL).to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS))

# 用户无需输入空格
user_input = "曾经的我很善良，现在的我很善变"
print(f"输入: {user_input}")
print(f"翻译: {translate(model, user_input, zh_vocab, en_vocab, device)}")