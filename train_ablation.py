# ============================================================
# train_ablation.py
# 消融实验:对比架构(1D-CNN vs 全连接)和损失函数(MSE vs KL)
#
# 实验配置:
#   1. Baseline: Beam_Estimator_1D + MSE
#   2. FC + MSE: Beam_Estimator_FC + MSE
#   3. FC + KL: Beam_Estimator_FC + KL Divergence
# ============================================================

import os
import sys
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

from models import (
    Beam_Estimator_1D,
    Beam_Estimator_FC,
    Beam_Estimator_BT,
    count_parameters,
)
from utils import add_noise_to_power_torch, normalize_power_torch

# 强制 UTF-8 输出
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                                  errors='replace')

# ===================== 全局配置 =====================
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.8
TEMPERATURE = 0.5  # KL 散度的温度参数

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'beam_dataset')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'results', 'ablation')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data():
    """加载静态数据"""
    data_path = os.path.join(DATA_DIR, 'static_beam_data.npz')
    data = np.load(data_path, allow_pickle=True)

    X = data['X_wide']           # 归一化宽波束功率 (N, 8)
    Y = data['Y_narrow']         # 归一化窄波束功率 (N, 32)
    Y_raw = data['Y_narrow_raw'] # 原始窄波束功率

    # 划分训练集和测试集
    num_samples = X.shape[0]
    num_train = int(num_samples * TRAIN_RATIO)

    indices = np.random.permutation(num_samples)
    train_idx = indices[:num_train]
    test_idx = indices[num_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    Y_raw_test = Y_raw[test_idx]

    print(f"数据加载完成: 训练 {X_train.shape[0]} / 测试 {X_test.shape[0]}")

    return X_train, X_test, Y_train, Y_test, Y_raw_test


def load_data_with_signal():
    """
    加载静态数据 + 无噪声信号功率(用于在线噪声增强)

    返回的训练集 X/Y 是无噪声信号功率(线性域),
    需要在训练循环中在线加噪再归一化。
    测试集仍返回固定加噪版本,保证与其他实验对比公平。
    """
    data_path = os.path.join(DATA_DIR, 'static_beam_data.npz')
    data = np.load(data_path, allow_pickle=True)

    if 'X_wide_signal' not in data.files:
        raise RuntimeError(
            "static_beam_data.npz 不含 X_wide_signal,"
            "请先重跑 data_generation.py"
        )

    X_norm = data['X_wide']
    Y_norm = data['Y_narrow']
    X_signal = data['X_wide_signal']      # 无噪声宽波束功率
    Y_signal = data['Y_narrow_signal']    # 无噪声窄波束功率
    Y_raw = data['Y_narrow_raw']
    snr_linear = float(data['snr_linear'])

    num_samples = X_signal.shape[0]
    num_train = int(num_samples * TRAIN_RATIO)

    indices = np.random.permutation(num_samples)
    train_idx = indices[:num_train]
    test_idx = indices[num_train:]

    # 训练集用无噪声信号(训练时在线加噪)
    X_train_signal = X_signal[train_idx]
    Y_train_signal = Y_signal[train_idx]

    # 测试集用固定加噪(归一化版本,与其他实验一致)
    X_test_norm = X_norm[test_idx]
    Y_test_norm = Y_norm[test_idx]
    Y_raw_test = Y_raw[test_idx]

    print(f"数据加载(含信号功率): "
          f"训练 {X_train_signal.shape[0]} / 测试 {X_test_norm.shape[0]}")

    return (X_train_signal, Y_train_signal,
            X_test_norm, Y_test_norm, Y_raw_test, snr_linear)


def kl_divergence_loss(pred, target, temperature=1.0):
    """
    KL 散度损失(把功率向量当概率分布)

    参数:
        pred: 模型输出 [B, 32],范围 [0,1]
        target: 真实标签 [B, 32],范围 [0,1]
        temperature: 温度参数,越小越接近硬分类

    返回:
        KL(target || pred) 的平均值
    """
    # 转换为对数概率分布
    # 加小量避免 log(0)
    pred_log_soft = F.log_softmax(pred / temperature, dim=1)
    target_soft = F.softmax(target / temperature, dim=1)

    # KL 散度: sum(target * log(target / pred))
    loss = F.kl_div(pred_log_soft, target_soft, reduction='batchmean')

    return loss


def train_model(model, X_train, Y_train, X_test, Y_test,
                loss_type='mse', model_name='Model'):
    """
    训练单个模型

    参数:
        model: 模型实例
        loss_type: 'mse' 或 'kl'
        model_name: 模型名称(用于保存)
    """
    print(f"\n{'='*60}")
    print(f"  训练: {model_name} + {loss_type.upper()} Loss")
    print(f"{'='*60}")
    print(f"  模型参数量: {count_parameters(model):,}")
    print(f"  设备: {DEVICE}")

    # 转换为 PyTorch 张量
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    Y_train_t = torch.FloatTensor(Y_train).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    Y_test_t = torch.FloatTensor(Y_test).to(DEVICE)

    # 如果是 1D-CNN 模型,需要增加通道维度
    if isinstance(model, Beam_Estimator_1D):
        X_train_t = X_train_t.unsqueeze(1)
        X_test_t = X_test_t.unsqueeze(1)

    # DataLoader
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(X_test_t, Y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    best_epoch = 0

    start_time = time.time()

    for epoch in range(EPOCHS):
        # ---- 训练阶段 ----
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)

            # 选择损失函数
            if loss_type == 'mse':
                loss = F.mse_loss(pred, batch_y)
            elif loss_type == 'kl':
                loss = kl_divergence_loss(pred, batch_y, temperature=TEMPERATURE)
            else:
                raise ValueError(f"未知损失类型: {loss_type}")

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        avg_train_loss = epoch_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        # ---- 验证阶段 ----
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                pred = model(batch_x)

                if loss_type == 'mse':
                    loss = F.mse_loss(pred, batch_y)
                elif loss_type == 'kl':
                    loss = kl_divergence_loss(pred, batch_y, temperature=TEMPERATURE)

                test_loss += loss.item() * batch_x.size(0)

        avg_test_loss = test_loss / len(test_dataset)
        test_losses.append(avg_test_loss)

        # 保存最优模型
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_epoch = epoch
            save_path = os.path.join(RESULT_DIR, f'{model_name}.pth')
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:3d}/{EPOCHS}] "
                  f"Train: {avg_train_loss:.6f} | Test: {avg_test_loss:.6f}")

    elapsed = time.time() - start_time
    print(f"\n训练完成! 耗时 {elapsed:.1f}s")
    print(f"  最优 Epoch: {best_epoch+1}, 最优 Loss: {best_loss:.6f}")

    return model, {'train_loss': train_losses, 'test_loss': test_losses}


def train_model_online_noise(model, X_train_signal, Y_train_signal,
                              X_test, Y_test, snr_linear,
                              loss_type='kl', model_name='Model'):
    """
    带在线噪声重采样的训练

    每个 batch:
        1. 取无噪声信号功率 (B, n_beams)
        2. 在线生成新噪声并加上
        3. 逐样本 dB 归一化
        4. 喂入模型计算 loss

    测试集仍用固定加噪版本,保证与其他实验对比公平。
    """
    print(f"\n{'='*60}")
    print(f"  训练: {model_name} + {loss_type.upper()} Loss + 在线噪声增强")
    print(f"{'='*60}")
    print(f"  模型参数量: {count_parameters(model):,}")
    print(f"  设备: {DEVICE}")
    print(f"  SNR (linear): {snr_linear:.2f}")

    # 训练集: 无噪声信号功率 (线性域)
    X_train_t = torch.FloatTensor(X_train_signal).to(DEVICE)
    Y_train_t = torch.FloatTensor(Y_train_signal).to(DEVICE)

    # 测试集: 固定加噪+归一化版本
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    Y_test_t = torch.FloatTensor(Y_test).to(DEVICE)

    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(X_test_t, Y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    test_losses = []
    best_loss = float('inf')
    best_epoch = 0

    start_time = time.time()

    for epoch in range(EPOCHS):
        # ---- 训练阶段 (在线加噪) ----
        model.train()
        epoch_loss = 0.0

        for batch_x_signal, batch_y_signal in train_loader:
            optimizer.zero_grad()

            # 在线加噪 + 归一化(每个 batch 噪声不同)
            batch_x_noisy = add_noise_to_power_torch(batch_x_signal, snr_linear)
            batch_y_noisy = add_noise_to_power_torch(batch_y_signal, snr_linear)

            batch_x = normalize_power_torch(batch_x_noisy)
            batch_y = normalize_power_torch(batch_y_noisy)

            pred = model(batch_x)

            if loss_type == 'mse':
                loss = F.mse_loss(pred, batch_y)
            elif loss_type == 'kl':
                loss = kl_divergence_loss(pred, batch_y, temperature=TEMPERATURE)
            else:
                raise ValueError(f"未知损失类型: {loss_type}")

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        avg_train_loss = epoch_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        # ---- 验证阶段 (固定加噪) ----
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                pred = model(batch_x)

                if loss_type == 'mse':
                    loss = F.mse_loss(pred, batch_y)
                elif loss_type == 'kl':
                    loss = kl_divergence_loss(pred, batch_y, temperature=TEMPERATURE)

                test_loss += loss.item() * batch_x.size(0)

        avg_test_loss = test_loss / len(test_dataset)
        test_losses.append(avg_test_loss)

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_epoch = epoch
            save_path = os.path.join(RESULT_DIR, f'{model_name}.pth')
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:3d}/{EPOCHS}] "
                  f"Train: {avg_train_loss:.6f} | Test: {avg_test_loss:.6f}")

    elapsed = time.time() - start_time
    print(f"\n训练完成! 耗时 {elapsed:.1f}s")
    print(f"  最优 Epoch: {best_epoch+1}, 最优 Loss: {best_loss:.6f}")

    return model, {'train_loss': train_losses, 'test_loss': test_losses}


def evaluate_model(model, X_test, Y_test, Y_raw_test, model_name='Model'):
    """
    评估模型性能

    返回:
        top1_acc: Top-1 准确率
        top3_acc: Top-3 准确率
        snr_ratio_dB: 平均 SNR 损失(dB)
    """
    print(f"\n{'='*50}")
    print(f"  评估: {model_name}")
    print(f"{'='*50}")

    model.eval()

    # 准备输入
    X_tensor = torch.FloatTensor(X_test).to(DEVICE)
    if isinstance(model, Beam_Estimator_1D):
        X_tensor = X_tensor.unsqueeze(1)

    # 预测
    with torch.no_grad():
        Y_pred = model(X_tensor).cpu().numpy()

    # Top-1 Accuracy
    pred_best = np.argmax(Y_pred, axis=1)
    true_best = np.argmax(Y_raw_test, axis=1)
    top1_acc = np.mean(pred_best == true_best)

    # Top-3 Accuracy
    pred_top3 = np.argsort(Y_pred, axis=1)[:, -3:]
    top3_hits = np.array([true_best[i] in pred_top3[i]
                          for i in range(len(true_best))])
    top3_acc = np.mean(top3_hits)

    # Effective SNR Ratio
    optimal_power = np.array([Y_raw_test[i, true_best[i]]
                              for i in range(len(true_best))])
    achieved_power = np.array([Y_raw_test[i, pred_best[i]]
                               for i in range(len(pred_best))])

    snr_ratio = achieved_power / (optimal_power + 1e-10)
    snr_ratio = np.clip(snr_ratio, 0, 1)
    snr_ratio_dB = 10 * np.log10(snr_ratio + 1e-10)
    avg_snr_dB = np.mean(snr_ratio_dB)

    print(f"  Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"  Top-3 Accuracy: {top3_acc*100:.2f}%")
    print(f"  Avg SNR Ratio: {avg_snr_dB:.2f} dB")

    return {
        'top1_acc': top1_acc,
        'top3_acc': top3_acc,
        'snr_ratio_dB': avg_snr_dB,
        'pred_best': pred_best,
        'true_best': true_best
    }


def main():
    """消融实验主流程"""
    print("\n" + "#" * 60)
    print("#  消融实验:架构 + 损失函数对比")
    print("#" * 60)

    os.makedirs(RESULT_DIR, exist_ok=True)

    # 加载数据
    X_train, X_test, Y_train, Y_test, Y_raw_test = load_data()

    results = {}

    # ============ 实验 1: Baseline (1D-CNN + MSE) ============
    print("\n" + "="*60)
    print("  实验 1: Baseline (1D-CNN + MSE)")
    print("="*60)

    model_1 = Beam_Estimator_1D().to(DEVICE)
    model_1, hist_1 = train_model(
        model_1, X_train, Y_train, X_test, Y_test,
        loss_type='mse', model_name='baseline_cnn_mse')

    results['Baseline (1D-CNN + MSE)'] = evaluate_model(
        model_1, X_test, Y_test, Y_raw_test,
        model_name='Baseline (1D-CNN + MSE)')

    # ============ 实验 2: 全连接 + MSE ============
    print("\n" + "="*60)
    print("  实验 2: 全连接 + MSE")
    print("="*60)

    model_2 = Beam_Estimator_FC().to(DEVICE)
    model_2, hist_2 = train_model(
        model_2, X_train, Y_train, X_test, Y_test,
        loss_type='mse', model_name='fc_mse')

    results['FC + MSE'] = evaluate_model(
        model_2, X_test, Y_test, Y_raw_test,
        model_name='FC + MSE')

    # ============ 实验 3: 全连接 + KL ============
    print("\n" + "="*60)
    print("  实验 3: 全连接 + KL 散度")
    print("="*60)

    model_3 = Beam_Estimator_FC().to(DEVICE)
    model_3, hist_3 = train_model(
        model_3, X_train, Y_train, X_test, Y_test,
        loss_type='kl', model_name='fc_kl')

    results['FC + KL'] = evaluate_model(
        model_3, X_test, Y_test, Y_raw_test,
        model_name='FC + KL')

    # ============ 实验 4: Beam Transformer + KL ============
    print("\n" + "="*60)
    print("  实验 4: Beam Transformer + KL 散度")
    print("="*60)

    model_4 = Beam_Estimator_BT(
        n_wide=8, n_narrow=32,
        dim=64, m_cat=4, num_heads=2,
        mlp_dim=64,                # 与原 FC 隐层规模相当,避免过拟合
        num_encoder_layers=1,
        num_classifier_layers=4,
        dropout=0.1,
    ).to(DEVICE)
    model_4, hist_4 = train_model(
        model_4, X_train, Y_train, X_test, Y_test,
        loss_type='kl', model_name='bt_kl')

    results['BT + KL'] = evaluate_model(
        model_4, X_test, Y_test, Y_raw_test,
        model_name='BT + KL')

    # ============ 实验 5: BT + KL + 在线噪声增强 ============
    print("\n" + "="*60)
    print("  实验 5: BT + KL + 在线噪声重采样")
    print("="*60)

    # 加载含无噪声信号功率的数据
    (X_train_signal, Y_train_signal,
     X_test_aug, Y_test_aug, Y_raw_test_aug,
     snr_linear) = load_data_with_signal()

    model_5 = Beam_Estimator_BT(
        n_wide=8, n_narrow=32,
        dim=64, m_cat=4, num_heads=2,
        mlp_dim=64,
        num_encoder_layers=1,
        num_classifier_layers=4,
        dropout=0.1,
    ).to(DEVICE)
    model_5, hist_5 = train_model_online_noise(
        model_5, X_train_signal, Y_train_signal,
        X_test_aug, Y_test_aug, snr_linear,
        loss_type='kl', model_name='bt_kl_online_noise')

    results['BT + KL + 在线噪声'] = evaluate_model(
        model_5, X_test_aug, Y_test_aug, Y_raw_test_aug,
        model_name='BT + KL + 在线噪声')

    # ============ 总结对比 ============
    print("\n" + "="*60)
    print("  消融实验总结")
    print("="*60)
    print(f"{'配置':<32} {'Top-1':<10} {'Top-3':<10} {'SNR Loss':<10}")
    print("-" * 62)

    for name, res in results.items():
        print(f"{name:<32} {res['top1_acc']*100:>6.2f}%   "
              f"{res['top3_acc']*100:>6.2f}%   {res['snr_ratio_dB']:>6.2f} dB")

    # 计算提升
    baseline_top1 = results['Baseline (1D-CNN + MSE)']['top1_acc']
    fc_mse_top1 = results['FC + MSE']['top1_acc']
    fc_kl_top1 = results['FC + KL']['top1_acc']
    bt_kl_top1 = results['BT + KL']['top1_acc']
    bt_kl_aug_top1 = results['BT + KL + 在线噪声']['top1_acc']

    print("\n提升分析:")
    print(f"  全连接 vs 1D-CNN (MSE):       {(fc_mse_top1 - baseline_top1)*100:+.2f}%")
    print(f"  KL vs MSE (全连接):            {(fc_kl_top1 - fc_mse_top1)*100:+.2f}%")
    print(f"  BT vs FC (KL):                 {(bt_kl_top1 - fc_kl_top1)*100:+.2f}%")
    print(f"  在线噪声 vs 固定噪声 (BT+KL):  {(bt_kl_aug_top1 - bt_kl_top1)*100:+.2f}%")
    print(f"  最佳配置 vs Baseline:          {(bt_kl_aug_top1 - baseline_top1)*100:+.2f}%")

    print(f"\n所有结果已保存至: {RESULT_DIR}")


if __name__ == '__main__':
    main()
