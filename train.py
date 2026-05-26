# ============================================================
# train.py
# 训练与评估流程：波束估计模型 + 波束预测模型
# 参考论文：A Deep Learning-Based Low Overhead Beam Selection
#           in mmWave Communications
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import time

from models import Beam_Estimator_1D, Beam_Predictor_1D_LSTM

# ===================== 全局配置 =====================
# 训练超参数
BATCH_SIZE = 64
EPOCHS_ESTIMATOR = 50      # 估计模型训练轮数
EPOCHS_PREDICTOR = 50      # 预测模型训练轮数
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.8          # 训练集比例

# 数据路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'beam_dataset')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'results')

# 设备选择
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_static_data():
    """
    加载静态波束数据（用于估计模型）

    返回:
        X_train, X_test: 宽波束功率 (归一化)
        Y_train, Y_test: 窄波束功率 (归一化)
        raw_data: 原始未归一化数据（用于评估）
    """
    data_path = os.path.join(DATA_DIR, 'static_beam_data.npz')

    if not os.path.exists(data_path):
        print("未找到数据文件，正在生成...")
        from data_generation import main as generate_data
        generate_data()

    data = np.load(data_path, allow_pickle=True)
    X = data['X_wide']           # 归一化宽波束功率 (N, 8)
    Y = data['Y_narrow']         # 归一化窄波束功率 (N, 32)
    X_raw = data['X_wide_raw']   # 原始宽波束功率
    Y_raw = data['Y_narrow_raw'] # 原始窄波束功率

    # 划分训练集和测试集
    num_samples = X.shape[0]
    num_train = int(num_samples * TRAIN_RATIO)

    # 随机打乱
    indices = np.random.permutation(num_samples)
    train_idx = indices[:num_train]
    test_idx = indices[num_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    X_raw_test = X_raw[test_idx]
    Y_raw_test = Y_raw[test_idx]

    print(f"静态数据加载完成: 训练 {X_train.shape[0]} / 测试 {X_test.shape[0]}")

    raw_data = {
        'X_raw_test': X_raw_test,
        'Y_raw_test': Y_raw_test,
        'norm_wide': data['norm_params_wide'],
        'norm_narrow': data['norm_params_narrow']
    }

    return X_train, X_test, Y_train, Y_test, raw_data


def load_temporal_data():
    """
    加载时序波束数据（用于预测模型）

    返回:
        X_train, X_test: 时序宽波束功率 (N, L, 8)
        Y_train, Y_test: 当前帧窄波束功率 (N, 32)
        raw_data: 原始数据
    """
    data_path = os.path.join(DATA_DIR, 'temporal_beam_data.npz')

    if not os.path.exists(data_path):
        print("未找到时序数据文件，正在生成...")
        from data_generation import main as generate_data
        generate_data()

    data = np.load(data_path, allow_pickle=True)
    X = data['X_seq']       # 归一化时序输入 (N, L, 8)
    Y = data['Y_seq']       # 归一化标签 (N, 32)
    X_raw = data['X_seq_raw']
    Y_raw = data['Y_seq_raw']

    # 划分训练集和测试集
    num_samples = X.shape[0]
    num_train = int(num_samples * TRAIN_RATIO)

    indices = np.random.permutation(num_samples)
    train_idx = indices[:num_train]
    test_idx = indices[num_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    Y_raw_test = Y_raw[test_idx]

    print(f"时序数据加载完成: 训练 {X_train.shape[0]} / 测试 {X_test.shape[0]}")

    raw_data = {
        'Y_raw_test': Y_raw_test,
        'norm_Y': data['norm_params_Y']
    }

    return X_train, X_test, Y_train, Y_test, raw_data


# ===================== 训练函数 =====================

def train_estimator():
    """
    训练波束质量估计模型 (Beam_Estimator_1D)

    输入: 宽波束功率 [B, 1, 8]
    输出: 窄波束功率 [B, 32]
    损失: MSE Loss
    """
    print("\n" + "=" * 60)
    print("  训练波束质量估计模型 (Beam_Estimator_1D)")
    print("=" * 60)

    # 加载数据
    X_train, X_test, Y_train, Y_test, raw_data = load_static_data()

    # 转换为 PyTorch 张量
    # 估计模型输入需要增加通道维度: (N, 8) → (N, 1, 8)
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1).to(DEVICE)
    Y_train_t = torch.FloatTensor(Y_train).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1).to(DEVICE)
    Y_test_t = torch.FloatTensor(Y_test).to(DEVICE)

    # 构建 DataLoader
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(X_test_t, Y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = Beam_Estimator_1D().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"设备: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS_ESTIMATOR}, LR: {LEARNING_RATE}")

    # 训练循环
    train_losses = []
    test_losses = []
    best_loss = float('inf')

    for epoch in range(EPOCHS_ESTIMATOR):
        # ---- 训练阶段 ----
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
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
                loss = criterion(pred, batch_y)
                test_loss += loss.item() * batch_x.size(0)

        avg_test_loss = test_loss / len(test_dataset)
        test_losses.append(avg_test_loss)

        # 保存最优模型
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(),
                       os.path.join(RESULT_DIR, 'best_estimator.pth'))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:3d}/{EPOCHS_ESTIMATOR}] "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Test Loss: {avg_test_loss:.6f}")

    print(f"\n训练完成! 最优测试 Loss: {best_loss:.6f}")

    # 保存训练曲线
    history = {'train_loss': train_losses, 'test_loss': test_losses}

    return model, history, raw_data


def train_predictor():
    """
    训练波束质量预测模型 (Beam_Predictor_1D_LSTM)

    输入: 时序宽波束功率 [B, L, 1, 8]
    输出: 当前帧窄波束功率 [B, 32]
    损失: MSE Loss
    """
    print("\n" + "=" * 60)
    print("  训练波束质量预测模型 (Beam_Predictor_1D_LSTM)")
    print("=" * 60)

    # 加载时序数据
    X_train, X_test, Y_train, Y_test, raw_data = load_temporal_data()

    # 转换为 PyTorch 张量
    # 预测模型输入: (N, L, 8) → (N, L, 1, 8)
    X_train_t = torch.FloatTensor(X_train).unsqueeze(2).to(DEVICE)
    Y_train_t = torch.FloatTensor(Y_train).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(2).to(DEVICE)
    Y_test_t = torch.FloatTensor(Y_test).to(DEVICE)

    # 构建 DataLoader
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(X_test_t, Y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = Beam_Predictor_1D_LSTM().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"设备: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS_PREDICTOR}, LR: {LEARNING_RATE}")

    # 训练循环
    train_losses = []
    test_losses = []
    best_loss = float('inf')

    for epoch in range(EPOCHS_PREDICTOR):
        # ---- 训练阶段 ----
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
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
                loss = criterion(pred, batch_y)
                test_loss += loss.item() * batch_x.size(0)

        avg_test_loss = test_loss / len(test_dataset)
        test_losses.append(avg_test_loss)

        # 保存最优模型
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(),
                       os.path.join(RESULT_DIR, 'best_predictor.pth'))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:3d}/{EPOCHS_PREDICTOR}] "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Test Loss: {avg_test_loss:.6f}")

    print(f"\n训练完成! 最优测试 Loss: {best_loss:.6f}")

    history = {'train_loss': train_losses, 'test_loss': test_losses}

    return model, history, raw_data


# ===================== 评估函数 =====================

def evaluate_model(model, X_test, Y_test, Y_raw_test, norm_params,
                   model_name="Model", is_temporal=False):
    """
    综合评估模型性能

    评估指标:
        1. MSE: 预测误差（归一化域）
        2. Top-1 Accuracy: 最优波束命中率
        3. Effective SNR Ratio: 实际 SNR / 最优 SNR

    参数:
        model: 训练好的模型
        X_test: 测试输入 (归一化)
        Y_test: 测试标签 (归一化)
        Y_raw_test: 原始未归一化的窄波束功率
        norm_params: 归一化参数（逐样本的 min/max 数组）
        model_name: 模型名称
        is_temporal: 是否为时序模型
    """
    print(f"\n{'=' * 50}")
    print(f"  评估: {model_name}")
    print(f"{'=' * 50}")

    model.eval()

    # 准备输入数据
    if is_temporal:
        X_tensor = torch.FloatTensor(X_test).unsqueeze(2).to(DEVICE)
    else:
        X_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(DEVICE)

    # 模型预测（归一化域）
    with torch.no_grad():
        Y_pred_norm = model(X_tensor).cpu().numpy()

    # ---- 指标 1: MSE ----
    mse = np.mean((Y_pred_norm - Y_test) ** 2)
    print(f"  MSE (归一化): {mse:.6f}")

    # ---- 指标 2: Top-1 Accuracy ----
    # 归一化后保持了相对顺序，直接用预测值选波束
    pred_best_beam = np.argmax(Y_pred_norm, axis=1)
    true_best_beam = np.argmax(Y_raw_test, axis=1)
    top1_acc = np.mean(pred_best_beam == true_best_beam)
    print(f"  Top-1 Accuracy: {top1_acc * 100:.2f}%")

    # Top-3 Accuracy
    pred_top3 = np.argsort(Y_pred_norm, axis=1)[:, -3:]
    top3_hits = np.array([true_best_beam[i] in pred_top3[i]
                          for i in range(len(true_best_beam))])
    top3_acc = np.mean(top3_hits)
    print(f"  Top-3 Accuracy: {top3_acc * 100:.2f}%")

    # ---- 指标 3: Effective SNR Ratio ----
    optimal_power = np.array([Y_raw_test[i, true_best_beam[i]]
                              for i in range(len(true_best_beam))])
    achieved_power = np.array([Y_raw_test[i, pred_best_beam[i]]
                               for i in range(len(pred_best_beam))])

    snr_ratio = achieved_power / (optimal_power + 1e-10)
    snr_ratio = np.clip(snr_ratio, 0, 1)
    snr_ratio_dB = 10 * np.log10(snr_ratio + 1e-10)

    avg_snr_ratio_dB = np.mean(snr_ratio_dB)
    print(f"  Effective SNR Ratio: {np.mean(snr_ratio):.4f} "
          f"({avg_snr_ratio_dB:.2f} dB)")

    results = {
        'mse': mse,
        'top1_acc': top1_acc,
        'top3_acc': top3_acc,
        'snr_ratio': snr_ratio,
        'snr_ratio_dB': snr_ratio_dB,
        'pred_best_beam': pred_best_beam,
        'true_best_beam': true_best_beam,
        'Y_pred_norm': Y_pred_norm
    }

    return results


# ===================== 可视化函数 =====================

def plot_training_curves(history, model_name, save_path):
    """绘制训练/验证损失曲线"""
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['test_loss'], label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'{model_name} - Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  训练曲线已保存: {save_path}")


def plot_snr_cdf(results_dict, save_path):
    """
    绘制 Effective SNR 的 CDF 图

    对比不同模型预测波束的实际 SNR 与全局最优 SNR 的比值分布
    """
    plt.figure(figsize=(8, 6))

    for name, results in results_dict.items():
        snr_ratio_dB = results['snr_ratio_dB']
        sorted_snr = np.sort(snr_ratio_dB)
        cdf = np.arange(1, len(sorted_snr) + 1) / len(sorted_snr)
        plt.plot(sorted_snr, cdf, label=name, linewidth=2)

    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5,
                label='Ideal (Exhaustive)')
    plt.xlabel('Effective SNR Loss (dB)')
    plt.ylabel('CDF')
    plt.title('Effective SNR CDF Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([-15, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  SNR CDF 图已保存: {save_path}")


def plot_beam_pattern_comparison(Y_true, Y_pred, sample_idx, save_path):
    """绘制单个样本的波束功率对比图"""
    plt.figure(figsize=(10, 4))
    beam_indices = np.arange(32)

    plt.bar(beam_indices - 0.2, Y_true[sample_idx], width=0.4,
            label='Ground Truth', alpha=0.7)
    plt.bar(beam_indices + 0.2, Y_pred[sample_idx], width=0.4,
            label='Predicted', alpha=0.7)

    plt.xlabel('Beam Index')
    plt.ylabel('Beam Power')
    plt.title(f'Beam Power Comparison (Sample #{sample_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ===================== 主函数 =====================

def main():
    """
    完整的训练与评估流程
    """
    print("\n" + "#" * 60)
    print("#  深度学习波束选择 Baseline - 训练与评估")
    print("#  参考: A Deep Learning-Based Low Overhead Beam Selection")
    print("#" * 60)

    # 创建结果目录
    os.makedirs(RESULT_DIR, exist_ok=True)

    start_time = time.time()

    # ============ 阶段 1: 训练估计模型 ============
    estimator, est_history, est_raw_data = train_estimator()

    # 评估估计模型
    X_train_e, X_test_e, Y_train_e, Y_test_e, raw_data_e = load_static_data()
    est_results = evaluate_model(
        estimator, X_test_e, Y_test_e,
        raw_data_e['Y_raw_test'],
        raw_data_e['norm_narrow'],
        model_name="Beam_Estimator_1D",
        is_temporal=False
    )

    # ============ 阶段 2: 训练预测模型 ============
    predictor, pred_history, pred_raw_data = train_predictor()

    # 评估预测模型
    X_train_p, X_test_p, Y_train_p, Y_test_p, raw_data_p = load_temporal_data()
    pred_results = evaluate_model(
        predictor, X_test_p, Y_test_p,
        raw_data_p['Y_raw_test'],
        raw_data_p['norm_Y'],
        model_name="Beam_Predictor_1D_LSTM",
        is_temporal=True
    )

    # ============ 阶段 3: 可视化 ============
    print("\n\n生成可视化图表...")

    plot_training_curves(
        est_history, "Beam_Estimator_1D",
        os.path.join(RESULT_DIR, 'estimator_training_curve.png'))
    plot_training_curves(
        pred_history, "Beam_Predictor_1D_LSTM",
        os.path.join(RESULT_DIR, 'predictor_training_curve.png'))

    plot_snr_cdf(
        {'Estimator (1D-SR)': est_results,
         'Predictor (CNN+LSTM)': pred_results},
        os.path.join(RESULT_DIR, 'snr_cdf_comparison.png'))

    for i in range(min(3, len(raw_data_e['Y_raw_test']))):
        plot_beam_pattern_comparison(
            raw_data_e['Y_raw_test'], est_results['Y_pred_norm'], i,
            os.path.join(RESULT_DIR, f'beam_pattern_sample_{i}.png'))

    # ============ 总结 ============
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  训练与评估总结")
    print("=" * 60)
    print(f"  总耗时: {elapsed:.1f} 秒")
    print(f"\n  [估计模型 Beam_Estimator_1D]")
    print(f"    MSE: {est_results['mse']:.6f}")
    print(f"    Top-1 Acc: {est_results['top1_acc']*100:.2f}%")
    print(f"    Top-3 Acc: {est_results['top3_acc']*100:.2f}%")
    print(f"    Avg SNR Ratio: {np.mean(est_results['snr_ratio_dB']):.2f} dB")
    print(f"\n  [预测模型 Beam_Predictor_1D_LSTM]")
    print(f"    MSE: {pred_results['mse']:.6f}")
    print(f"    Top-1 Acc: {pred_results['top1_acc']*100:.2f}%")
    print(f"    Top-3 Acc: {pred_results['top3_acc']*100:.2f}%")
    print(f"    Avg SNR Ratio: {np.mean(pred_results['snr_ratio_dB']):.2f} dB")
    print("=" * 60)
    print(f"\n所有结果已保存至: {RESULT_DIR}")


if __name__ == '__main__':
    main()
