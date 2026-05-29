# ============================================================
# train_beam_dropout.py
# 方案 1: Beam Dropout 训练 + 稳健性评估
#
# 目标:
#   实验 D 显示当前 BT 模型对单波束移除极度敏感(单波束移除 → Top-1 跌 50%+)
#   方案 1 用训练时随机丢波束,强迫网络学习稳健表示
#
# 实验设置:
#   - 模型: Beam_Estimator_BT(同 train_ablation.py 配置)
#   - 损失: KL 散度
#   - 增强: 训练时 50% 概率丢 1~2 个波束(基于在线噪声后的功率)
#   - 测试: 固定加噪 + 完整输入(与之前 BT+KL 公平对比)
#
# 评估:
#   1. 完整输入 Top-1(对比 BT+KL=61.98%)
#   2. 单波束移除消融(对比之前每个 4~10%)
#   3. Top-K 强波束消融
# ============================================================

import os
import sys
import io
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

from models import Beam_Estimator_BT, count_parameters
from utils import (
    add_noise_to_power_torch,
    normalize_power_torch,
    random_beam_dropout_torch,
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                              errors='replace')

# ===================== 配置 =====================
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.8
TEMPERATURE = 0.5
DROPOUT_P = 0.5      # 50% 样本应用 beam dropout
DROPOUT_MAX = 2      # 最多丢 2 个波束

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'beam_dataset')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'results', 'ablation')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def kl_divergence_loss(pred, target, temperature=1.0):
    pred_log_soft = F.log_softmax(pred / temperature, dim=1)
    target_soft = F.softmax(target / temperature, dim=1)
    return F.kl_div(pred_log_soft, target_soft, reduction='batchmean')


def load_data_with_signal():
    """加载含无噪声信号功率的数据"""
    data_path = os.path.join(DATA_DIR, 'static_beam_data.npz')
    data = np.load(data_path, allow_pickle=True)

    X_norm = data['X_wide']
    Y_norm = data['Y_narrow']
    X_signal = data['X_wide_signal']
    Y_signal = data['Y_narrow_signal']
    Y_raw = data['Y_narrow_raw']
    X_raw = data['X_wide_raw']
    snr_linear = float(data['snr_linear'])

    num_samples = X_signal.shape[0]
    num_train = int(num_samples * TRAIN_RATIO)

    np.random.seed(0)
    indices = np.random.permutation(num_samples)
    train_idx = indices[:num_train]
    test_idx = indices[num_train:]

    return {
        'X_train_signal': X_signal[train_idx],
        'Y_train_signal': Y_signal[train_idx],
        'X_test_norm': X_norm[test_idx],
        'Y_test_norm': Y_norm[test_idx],
        'X_test_raw': X_raw[test_idx],
        'Y_raw_test': Y_raw[test_idx],
        'snr_linear': snr_linear,
    }


def train_with_beam_dropout(model, data, model_name='bt_kl_beamdrop'):
    """训练:在线噪声 + Beam Dropout + KL"""
    print(f"\n{'='*60}")
    print(f"  训练: {model_name}")
    print(f"  增强: 在线噪声 + Beam Dropout(p={DROPOUT_P}, max_drop={DROPOUT_MAX})")
    print(f"{'='*60}")
    print(f"  模型参数量: {count_parameters(model):,}")
    print(f"  设备: {DEVICE}")

    snr_linear = data['snr_linear']
    X_train_t = torch.FloatTensor(data['X_train_signal']).to(DEVICE)
    Y_train_t = torch.FloatTensor(data['Y_train_signal']).to(DEVICE)
    X_test_t = torch.FloatTensor(data['X_test_norm']).to(DEVICE)
    Y_test_t = torch.FloatTensor(data['Y_test_norm']).to(DEVICE)

    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t),
                             batch_size=BATCH_SIZE, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = float('inf')
    best_epoch = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        # 训练
        model.train()
        epoch_loss = 0.0

        for batch_x_signal, batch_y_signal in train_loader:
            optimizer.zero_grad()

            # Step 1: 在线加噪
            batch_x_noisy = add_noise_to_power_torch(batch_x_signal, snr_linear)
            batch_y_noisy = add_noise_to_power_torch(batch_y_signal, snr_linear)

            # Step 2: Beam Dropout(只对输入,不动标签)
            batch_x_drop = random_beam_dropout_torch(
                batch_x_noisy, p_apply=DROPOUT_P, max_drop=DROPOUT_MAX)

            # Step 3: 归一化
            batch_x = normalize_power_torch(batch_x_drop)
            batch_y = normalize_power_torch(batch_y_noisy)

            pred = model(batch_x)
            loss = kl_divergence_loss(pred, batch_y, TEMPERATURE)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        avg_train = epoch_loss / len(train_loader.dataset)

        # 测试(完整输入,无 dropout)
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for bx, by in test_loader:
                pred = model(bx)
                test_loss += kl_divergence_loss(pred, by, TEMPERATURE).item() * bx.size(0)
        avg_test = test_loss / len(test_loader.dataset)

        if avg_test < best_loss:
            best_loss = avg_test
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(RESULT_DIR, f'{model_name}.pth'))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:3d}/{EPOCHS}] "
                  f"Train: {avg_train:.6f} | Test: {avg_test:.6f}")

    print(f"\n训练完成! 耗时 {time.time() - start_time:.1f}s")
    print(f"  最优 Epoch: {best_epoch+1}, Loss: {best_loss:.6f}")
    return model


def evaluate_top1(model, X_norm_or_raw, true_best, is_raw=False):
    """评估 Top-1; is_raw=True 时输入是原始线性功率,需要归一化"""
    model.eval()
    if is_raw:
        X_dB = 10 * np.log10(X_norm_or_raw + 1e-30)
        p_min = X_dB.min(axis=-1, keepdims=True)
        p_max = X_dB.max(axis=-1, keepdims=True)
        denom = np.where(p_max - p_min < 1e-10, 1.0, p_max - p_min)
        X_norm = (X_dB - p_min) / denom
    else:
        X_norm = X_norm_or_raw
    X_t = torch.FloatTensor(X_norm).to(DEVICE)
    with torch.no_grad():
        Y_pred = model(X_t).cpu().numpy()
    pred_best = np.argmax(Y_pred, axis=1)
    return float(np.mean(pred_best == true_best))


def mask_keep_topk(power_linear, k, mode='strong'):
    """复制自 experiment_D.py"""
    N, n_beams = power_linear.shape
    masked = np.zeros_like(power_linear)
    if mode == 'strong':
        topk_idx = np.argsort(power_linear, axis=1)[:, -k:]
    else:
        topk_idx = np.argsort(power_linear, axis=1)[:, :k]
    rows = np.repeat(np.arange(N), k)
    cols = topk_idx.flatten()
    masked[rows, cols] = power_linear[rows, cols]
    return masked


def run_robustness_test(model, X_test_raw, true_best_test, label):
    """对模型做单波束移除 + Top-K 强波束消融"""
    print(f"\n{'='*60}")
    print(f"  稳健性测试: {label}")
    print(f"{'='*60}")

    n_wide = X_test_raw.shape[1]

    # 完整基准
    top1_full = evaluate_top1(model, X_test_raw, true_best_test, is_raw=True)
    print(f"  完整 8 波束:  Top-1 = {top1_full*100:.2f}%")

    # 单波束移除
    print(f"\n  --- 逐个移除单波束 ---")
    print(f"  {'移除波束':<12} {'Top-1':<10} {'退化':<10}")
    deltas = []
    for j in range(n_wide):
        masked = X_test_raw.copy()
        masked[:, j] = 0
        top1 = evaluate_top1(model, masked, true_best_test, is_raw=True)
        delta = (top1 - top1_full) * 100
        deltas.append(delta)
        print(f"  波束 {j}        {top1*100:>6.2f}%   {delta:>+6.2f}%")
    avg_delta = np.mean(deltas)
    print(f"\n  平均单波束移除退化: {avg_delta:+.2f}%")

    # Top-K 强波束
    print(f"\n  --- 只保留 Top-K 强波束 ---")
    print(f"  {'K':<5} {'Top-1':<10}")
    for k in [1, 2, 4, 6, 7, 8]:
        masked = mask_keep_topk(X_test_raw, k, 'strong')
        top1 = evaluate_top1(model, masked, true_best_test, is_raw=True)
        print(f"  {k:<5} {top1*100:>6.2f}%")

    return top1_full, avg_delta


def main():
    print("\n" + "#" * 60)
    print("#  方案 1: Beam Dropout 训练 + 稳健性评估")
    print("#" * 60)

    os.makedirs(RESULT_DIR, exist_ok=True)
    data = load_data_with_signal()
    print(f"训练 {data['X_train_signal'].shape[0]} / "
          f"测试 {data['X_test_norm'].shape[0]}")

    # 训练 Beam Dropout 版本
    model = Beam_Estimator_BT(
        n_wide=8, n_narrow=32,
        dim=64, m_cat=4, num_heads=2,
        mlp_dim=64,
        num_encoder_layers=1,
        num_classifier_layers=4,
        dropout=0.1,
    ).to(DEVICE)
    model = train_with_beam_dropout(model, data, 'bt_kl_beamdrop')

    # 测试集稳健性评估
    new_full, new_avg_delta = run_robustness_test(
        model, data['X_test_raw'], np.argmax(data['Y_raw_test'], axis=1),
        label='BT + KL + Beam Dropout(本实验)')

    # 加载之前的 BT+KL 模型做对比
    print("\n" + "="*60)
    print("  对照: 之前的 BT + KL(无 Beam Dropout)")
    print("="*60)

    old_model = Beam_Estimator_BT(
        n_wide=8, n_narrow=32,
        dim=64, m_cat=4, num_heads=2, mlp_dim=64,
        num_encoder_layers=1, num_classifier_layers=4,
        dropout=0.1,
    ).to(DEVICE)
    old_path = os.path.join(RESULT_DIR, 'bt_kl.pth')
    if os.path.exists(old_path):
        old_model.load_state_dict(torch.load(old_path, map_location=DEVICE,
                                             weights_only=True))
        old_full, old_avg_delta = run_robustness_test(
            old_model, data['X_test_raw'],
            np.argmax(data['Y_raw_test'], axis=1),
            label='BT + KL(无 Beam Dropout)')
    else:
        print(f"  未找到 {old_path}, 跳过对照")
        old_full, old_avg_delta = None, None

    # 总结
    print("\n" + "=" * 60)
    print(" 方案 1 总结")
    print("=" * 60)
    print(f"  完整 Top-1:")
    if old_full is not None:
        print(f"    BT + KL                : {old_full*100:.2f}%")
    print(f"    BT + KL + Beam Dropout : {new_full*100:.2f}%")
    if old_full is not None:
        print(f"    完整 Top-1 变化         : {(new_full - old_full)*100:+.2f}%")

    print(f"\n  单波束移除平均退化(越接近 0 越稳健):")
    if old_avg_delta is not None:
        print(f"    BT + KL                : {old_avg_delta:+.2f}%")
    print(f"    BT + KL + Beam Dropout : {new_avg_delta:+.2f}%")
    if old_avg_delta is not None:
        print(f"    稳健性提升              : {old_avg_delta - new_avg_delta:+.2f}%")


if __name__ == '__main__':
    main()
