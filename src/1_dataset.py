"""
为干净的token词源，配上id形成词表，存储到模型内部
加载本地 json 词表
单个词转 id（未知词返回 <unk>）
id 转回词
分词句子 → id 序列（自动加 sos/eos）
id 序列还原成句子（自动过滤特殊标记）
"""
import json
from config import *


class Vocab:
    def __init__(self, vocab_json_path):
        # 1. 读取词表json文件
        with open(vocab_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # {单词: id}
        self.word2id = data["word2id"]
        # {id: 单词}  json读取后key是字符串，转成数字id
        self.id2word = {int(k): v for k, v in data["id2word"].items()}

        # 预存特殊标记id，方便后续使用
        self.pad_id = PAD_IDX
        self.unk_id = UNK_IDX
        self.sos_id = SOS_IDX
        self.eos_id = EOS_IDX

    # 函数1：单个单词 → id
    def word_to_id(self, word):
        # 词不在词表里返回unk的id
        return self.word2id.get(word, self.unk_id)

    # 函数2：单个id → 单词
    def id_to_word(self, idx):
        return self.id2word.get(idx, "<unk>")

    # 函数3：分词后的词列表 → id列表（核心！给模型输入用）
    def sent_to_ids(self, word_list, add_sos_eos=True):
        ids = [self.word_to_id(w) for w in word_list]
        # 翻译任务标准：开头<sos>，结尾<eos>
        if add_sos_eos:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    # 函数4：id数字列表 → 还原成可读句子
    def ids_to_sent(self, id_list):
        words = [self.id_to_word(i) for i in id_list]
        # 过滤掉特殊标记，只保留正常文字
        real_words = [w for w in words if w not in ALL_TOKENS]
        return " ".join(real_words)