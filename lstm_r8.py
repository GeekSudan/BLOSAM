import os
import torch
import random
import numpy as np
import sys; sys.path.append("..")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utility.bypass_bn import enable_running_stats, disable_running_stats
import utils.DataProcessing as DP
import utils.LSTMClassifier as LSTMC
from utils.logger import Logger
from samori import SAM 
from blosam_2step import BLOSAM

# ========================= 固定随机种子 ========================= #
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ========================= Smooth CrossEntropy ========================= #
def smooth_crossentropy(preds, targets, smoothing=0.1):
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(preds, dim=-1)
    true_dist = torch.zeros_like(log_probs)
    true_dist.scatter_(1, targets.unsqueeze(1), confidence)
    true_dist += smoothing / (preds.size(1) - 1)
    return (-true_dist * log_probs).sum(dim=1).mean()

# ========================= Top-k Accuracy ========================= #
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# ========================= 配置参数 ========================= #
DATA_DIR = '/workspace/dansu/PSOSGD-main/data'
TRAIN_DIR = 'train_txt'
TEST_DIR = 'test_txt'
TRAIN_FILE = 'train_txt.txt'
TEST_FILE = 'test_txt.txt'
TRAIN_LABEL = 'train_label.txt'
TEST_LABEL = 'test_label.txt'

embedding_dim = 100
hidden_dim = 50
sentence_len = 32
nlabel = 8
batch_size = 5
epochs = 50
learning_rate = 0.1
rho = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('outputs', exist_ok=True)
os.makedirs('saved_model', exist_ok=True)

logger = Logger('outputs/R8_blosam.txt', title='')
logger.set_names(['Valid Loss', 'Valid Acc.'])

# ========================= 数据加载 ========================= #
with open(os.path.join(DATA_DIR, TRAIN_FILE), 'r') as f:
    train_filenames = [os.path.join(TRAIN_DIR, line.strip()) for line in f]
with open(os.path.join(DATA_DIR, TEST_FILE), 'r') as f:
    test_filenames = [os.path.join(TEST_DIR, line.strip()) for line in f]
filenames = train_filenames + test_filenames
corpus = DP.Corpus(DATA_DIR, filenames)

train_set = DP.TxtDatasetProcessing(DATA_DIR, TRAIN_DIR, TRAIN_FILE, TRAIN_LABEL, sentence_len, corpus)
test_set = DP.TxtDatasetProcessing(DATA_DIR, TEST_DIR, TEST_FILE, TEST_LABEL, sentence_len, corpus)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# ========================= 模型与优化器 ========================= #
model = LSTMC.LSTMClassifier(
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    vocab_size=len(corpus.dictionary),
    label_size=nlabel,
    num_layers=1,
    dropout=0.0,
    device=device
).to(device)
optimizer = BLOSAM(model.parameters(),lr=learning_rate,rho=rho,adaptive=True, p=2, xi_lr_ratio=2,momentum_theta=0.9, weight_decay=0.0005, dampening=0) # L2 regularization

# base_optimizer = optim.SGD
# optimizer = SAM(model.parameters(), base_optimizer, rho=rho, adaptive=True, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
criterion = lambda outputs, targets: smooth_crossentropy(outputs, targets, smoothing=0.1)

# ========================= 调整学习率函数 ========================= #
def adjust_learning_rate(optimizer, epoch, base_lr):
    lr = base_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"[Epoch {epoch}] Learning rate adjusted to: {lr:.6f}")
    return optimizer

# ========================= 训练主循环 ========================= #
best_acc = 0.0
train_loss_, test_loss_, train_acc_, test_acc_ = [], [], [], []
save_path = 'saved_model/best_model_sam.pth'

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
    @property
    def avg(self):
        return self.sum / self.count if self.count != 0 else 0

for epoch in range(epochs):
    optimizer = adjust_learning_rate(optimizer, epoch, learning_rate)
    model.train()
    # model.lstm.flatten_parameters()
    total_loss, total_acc, total = 0.0, 0.0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # optimizer.zero_grad()
        enable_running_stats(model)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        
        disable_running_stats(model)
        outputs2 = model(inputs)
        loss = criterion(outputs2, labels)
        loss.backward()
        optimizer.second_step(zero_grad=True)

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_acc += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss_.append(total_loss / total)
    train_acc_.append(total_acc / total)

    # 验证
    model.eval()
    val_loss, val_acc, val_total = 0.0, 0.0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            val_acc += (predicted == targets).sum().item()
            val_total += targets.size(0)
            val_loss += loss.item()

    test_loss_.append(val_loss / val_total)
    test_acc_.append(val_acc / val_total)
    logger.append([f"{test_loss_[-1]:.6f}", f"{test_acc_[-1]:.6f}"])

    print('[Epoch: %3d/%3d] Train Loss: %.4f | Test Loss: %.4f | Train Acc: %.4f | Test Acc: %.4f'
          % (epoch + 1, epochs, train_loss_[-1], test_loss_[-1], train_acc_[-1], test_acc_[-1]))

    if test_acc_[-1] > best_acc:
        best_acc = test_acc_[-1]
        torch.save(model.state_dict(), save_path)
        print(f">> Best model saved with acc: {best_acc:.4f}")

# ========================= 保存结果 ========================= #
param = {
    'lr': learning_rate,
    'batch size': batch_size,
    'embedding dim': embedding_dim,
    'hidden dim': hidden_dim,
    'sentence len': sentence_len,
    'rho': rho
}
result = {
    'train loss': train_loss_,
    'test loss': test_loss_,
    'train acc': train_acc_,
    'test acc': test_acc_,
    'param': param
}
