import json
import nltk
import jieba
from config import *

# 加载英文模型
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import word_tokenize


# 读取数据，清洗成列表
def txt2list(path):
    result = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if not line:
                continue
            parts = line.strip().split("\t")

            if len(parts) >= 2:
                en = parts[0]
                zh = parts[1]
                result.append([en, zh])
    return result


# 将列表进行分词，再放入集合去重，存入分词清单
def list2token(txt_list):
    # 先分词
    en_tokens_all = []
    zh_tokens_all = []
    for en, zh in txt_list:
        # 中文jieba精确分词
        zh_words = jieba.lcut(zh)
        zh_tokens_all.append(zh_words)
        # 英文简单按空格分词
        en_words = word_tokenize(en)
        en_tokens_all.append(en_words)

    with open(ZH_PROCESSED_DATA, "w", encoding="utf-8") as f:
        for word in zh_tokens_all:
            line = " ".join(word)
            f.write(line + "\n")

    with open(EN_PROCESSED_DATA, "w", encoding="utf-8") as f:
        for word in en_tokens_all:
            line = " ".join(word)
            f.write(line + "\n")

    # 然后去重
    zh_all_words, en_all_words = [], []

    for sub_list in zh_tokens_all:
        zh_all_words.extend(sub_list)

    for sub_list in en_tokens_all:
        en_all_words.extend(sub_list)

    return ALL_TOKENS + list(set(zh_all_words)), ALL_TOKENS + list(set(en_all_words))


def words2json(words, vocab_path):
    word2id, id2word = {}, {}
    for idx, word in enumerate(words):
        word2id[word] = idx

    for idx, word in enumerate(words):
        id2word[idx] = word

    vocab_data = {
        "word2id": word2id,
        "id2word": id2word
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)


# text = "我是AI合作者"
# tokens = list(jieba.cut(text)) # 切成词：['我', '是', 'AI', '合作者']


if __name__ == '__main__':
    words = txt2list(TRAIN_DATA)
    zh_all_words, en_all_words = list2token(words)
    words2json(en_all_words, EN_VOCAB_PATH)
    words2json(zh_all_words, ZH_VOCAB_PATH)
