import importlib
import time
import json
import torch

mod = importlib.import_module("2_model")
Translation = mod.Translation  # 获取类
mod2 = importlib.import_module("1_dataset")
Vocab = mod2.Vocab

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from config import *

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter # 1. 导入 TensorBoard


# --- 1. 数据集加载器 ---
class TranslationDataset(Dataset):
    def __init__(self, zh_path, en_path, zh_vocab, en_vocab):
        # 读取数据行
        with open(zh_path, 'r', encoding='utf-8') as f:
            self.zh_lines = [line.strip().split() for line in f.readlines()]
        with open(en_path, 'r', encoding='utf-8') as f:
            self.en_lines = [line.strip().split() for line in f.readlines()]

        self.zh_vocab = zh_vocab  # 这是传入的 Vocab 类实例
        self.en_vocab = en_vocab

    def __len__(self):
        return len(self.zh_lines)

    def __getitem__(self, idx):
        # 1. 直接调用Vocab 类里的 sent_to_ids 方法
        # 英文加 SOS/EOS，中文这里设为 add_sos_eos=False
        zh_ids = self.zh_vocab.sent_to_ids(self.zh_lines[idx], add_sos_eos=False)
        en_ids = self.en_vocab.sent_to_ids(self.en_lines[idx], add_sos_eos=True)

        # 2. 转换为 Tensor
        return torch.tensor(zh_ids), torch.tensor(en_ids)


def collate_fn(batch):
    # 对这个批次的数据进行补零，然后输出。
    zh_batch, en_batch = zip(*batch)
    zh_padded = pad_sequence(zh_batch, padding_value=PAD_IDX)
    en_padded = pad_sequence(en_batch, padding_value=PAD_IDX)
    return zh_padded, en_padded


# --- 2. 训练循环 ---
def train(model, train_loader, val_loader, optimizer, criterion, device):
    writer = SummaryWriter(Train_LOG)  # 2. 创建日志记录器
    best_val_loss = float('inf')  # 用于保存最佳模型
    patience = 5  # 耐心值
    no_improve_epochs = 0  # 记录验证集未下降的次数
    global_step = 0  # 用于记录全局步数，曲线才平滑

    for epoch in range(EPOCHS):
        start_time = time.time()  # 2. 记录每个 Epoch 开始的时间
        model.train()  # 开启训练
        total_loss = 0  # 记录本轮总 Loss，用来算平均值

        # 3. 使用 tqdm 包裹 DataLoader
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        # 从 train_loader 里一批一批拿数据，然后放入cpu，
        for i, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            # 给到解码器使用，前者训练用，后者计算损失用
            tgt_input, tgt_out = tgt[:-1, :], tgt[1:, :]

            optimizer.zero_grad()  # 清空梯度
            output = model(src, tgt_input)  # 前向传播

            # 计算 Loss: [序列长度*Batch, 词表大小]
            loss = criterion(output.view(-1, output.shape[-1]), tgt_out.reshape(-1))

            loss.backward()
            # CPU 训练梯度裁剪非常重要，防止 Loss 跳变
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # 更新权重

            total_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch + 1} | Batch {i} | Loss: {loss.item():.4f}")
            # 更新进度条描述
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 记录到 TensorBoard
            writer.add_scalar("Loss/Train", loss.item(), global_step)
        # --- 验证阶段 ---
        model.eval()    # 冻结所有参数，进入推理阶段
        val_loss = 0
        # 在此块中不计算梯度，减少内存占用，加快计算速度
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input, tgt_out = tgt[:-1, :], tgt[1:, :]
                output = model(src, tgt_input)
                val_loss += criterion(output.view(-1, output.shape[-1]), tgt_out.reshape(-1)).item()

        avg_val_loss = val_loss / len(val_loader)   # 计算本轮验证集的平均 Loss。
        avg_train_loss = total_loss / len(train_loader) # # 计算本轮训练集的平均 Loss。
        # 记录验证 Loss 到 TensorBoard
        writer.add_scalar(Train_LOG, avg_val_loss, epoch)
        # 3. 计算并格式化耗时
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(int(end_time - start_time), 60)

        print(f"--- Epoch {epoch + 1} 完成 | 耗时: {epoch_mins}m {epoch_secs}s ---")
        print(f"--- Epoch {epoch + 1} 完成 | 训练平均 Loss: {avg_train_loss:.4f} | 验证 Loss: {avg_val_loss:.4f} ---")

        # --- 最佳模型保存与提前停止逻辑 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), MODEL_WEIGHTS)
            print(">>> 验证集 Loss 下降，已保存最佳模型。")
        else:
            no_improve_epochs += 1
            print(f">>> 验证集 Loss 未下降，当前耐心值: {no_improve_epochs}/{patience}")
            if no_improve_epochs >= patience:
                print(">>> 达到耐心值上限，停止训练。")
                break


# 冒烟测试函数：跑之前验证下数据是否可靠
def run_smoke_test(model, loader, device, criterion):
    print("正在进行冒烟测试...")
    model.train()
    try:
        src, tgt = next(iter(loader))
        src, tgt = src.to(device), tgt.to(device)
        tgt_input, tgt_out = tgt[:-1, :], tgt[1:, :]

        output = model(src, tgt_input)
        loss = criterion(output.view(-1, output.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        print(f"冒烟测试通过！单批次 Loss: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"冒烟测试失败，错误信息: {e}")
        return False


# --- 3. 主程序 ---
def main():
    print("开始初始化环境...")
    device = torch.device("cpu")

    # 初始化组件（加载数据集）
    zh_vocab = Vocab(ZH_VOCAB_PATH)
    en_vocab = Vocab(EN_VOCAB_PATH)
    full_dataset = TranslationDataset(ZH_PROCESSED_DATA, EN_PROCESSED_DATA, zh_vocab, en_vocab)

    # 划分数据集
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 实例化 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 保存推理集索引，用于下一步推理
    val_indices = val_dataset.indices
    with open(VAL_PATH, "w") as f:
        json.dump(val_indices, f)
    print("验证集索引已保存至 val_indices.json")

    # 初始化模型
    model = Translation(src_vocab_size=12447, tgt_vocab_size=7972, d_model=D_MODEL).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 执行流程
    if run_smoke_test(model, train_loader, device, criterion):
        print("冒烟测试通过，准备开始正式训练...")
        # 传入 train_loader 和 val_loader
        train(model, train_loader, val_loader, optimizer, criterion, device)
    else:
        print("程序终止。")


if __name__ == "__main__":
    main()