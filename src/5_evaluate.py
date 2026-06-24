import json
import torch
from torch.utils.data import DataLoader, Subset
from config import *
from inference import translate  # 复用推理脚本中的 translate 函数

# 1. 加载组件 (复用训练时的 Vocab)
mod = importlib.import_module("2_model")
Translation = mod.Translation
mod2 = importlib.import_module("1_dataset")
Vocab = mod2.Vocab

zh_vocab = Vocab(ZH_VOCAB_PATH)
en_vocab = Vocab(EN_VOCAB_PATH)
model = Translation(src_vocab_size=12447, tgt_vocab_size=7972, d_model=D_MODEL)
model.load_state_dict(torch.load(MODEL_WEIGHTS))
model.to(device)

# 2. 加载验证集索引并构建 Subset
full_dataset = TranslationDataset(ZH_PROCESSED_DATA, EN_PROCESSED_DATA, zh_vocab, en_vocab)
with open(VAL_PATH, "r") as f:
    val_indices = json.load(f)
val_dataset = Subset(full_dataset, val_indices)

# 3. 生成翻译并保存
references = []
hypotheses = []

print("正在进行批量评测...")
for i in range(len(val_dataset)):
    src_ids, tgt_ids = val_dataset[i]

    # 还原原文和参考译文
    src_sent = zh_vocab.ids_to_sent(src_ids.tolist())
    ref_sent = en_vocab.ids_to_sent(tgt_ids.tolist(), ignore_sos_eos=True)

    # 模型预测 (src_sent 是空格分隔的词，直接传入 translate)
    pred_sent = translate(model, src_sent, zh_vocab, en_vocab, device)

    references.append(ref_sent)
    hypotheses.append(pred_sent)

    if i % 100 == 0:
        print(f"进度: {i}/{len(val_dataset)}")

# 4. 计算 BLEU 分数
from sacrebleu import corpus_bleu

bleu = corpus_bleu(hypotheses, [references])
print(f"最终 BLEU Score: {bleu.score:.2f}")