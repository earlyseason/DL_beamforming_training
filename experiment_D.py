# ============================================================
# experiment_D.py
# 输入消融实验:验证网络是否真的利用了所有 8 个宽波束的信息
#
# 实验设置:
#   不重训模型(用 bt_kl.pth),在测试集上做 4 种消融:
#     - 完整: 用所有 8 个波束
#     - Top-K 强: 只保留最强 K 个,其他在线性功率域置零
#     - Top-K 弱: 只保留最弱 K 个
#     - 单波束(K=1): 只保留最强 / 只保留最弱
#
# 关键:
#   消融在"线性功率"上做,然后重新走 dB 归一化流程,
#   保证模型看到的输入分布与训练时一致(归一化后的 [0,1])。
# ============================================================

import os
import sys
import io
import numpy as np
import torch

from models import Beam_Estimator_BT

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                              errors='replace')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'beam_dataset')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'results', 'ablation')

TRAIN_RATIO = 0.8
SEED = 42  # 与 train_ablation.py 不一致没关系,我们只关心相对退化


def normalize_per_sample_dB(power_linear, eps=1e-30):
    """逐样本 dB 域归一化,与 utils.normalize_power 一致"""
    power_dB = 10 * np.log10(power_linear + eps)
    p_min = power_dB.min(axis=-1, keepdims=True)
    p_max = power_dB.max(axis=-1, keepdims=True)
    denom = p_max - p_min
    denom = np.where(denom < 1e-10, 1.0, denom)
    return (power_dB - p_min) / denom


def mask_keep_topk(power_linear, k, mode='strong'):
    """
    保留每个样本中 top-k 个波束(按线性功率值),其他置零

    参数:
        power_linear: (N, n_beams) 线性功率(必须 >=0)
        k: 保留的波束数
        mode: 'strong' 保留最强 K 个 / 'weak' 保留最弱 K 个

    返回:
        masked: 同 shape,被丢弃的波束位置为 0
    """
    N, n_beams = power_linear.shape
    masked = np.zeros_like(power_linear)

    if mode == 'strong':
        # 取每行最大的 k 个索引
        topk_idx = np.argsort(power_linear, axis=1)[:, -k:]
    elif mode == 'weak':
        # 取每行最小的 k 个索引
        topk_idx = np.argsort(power_linear, axis=1)[:, :k]
    else:
        raise ValueError(mode)

    rows = np.repeat(np.arange(N), k)
    cols = topk_idx.flatten()
    masked[rows, cols] = power_linear[rows, cols]

    return masked


def evaluate_top1(model, X_norm, true_best):
    """给定归一化输入和真实最优波束,返回 Top-1 准确率和模型预测"""
    model.eval()
    X_t = torch.FloatTensor(X_norm).to(DEVICE)
    with torch.no_grad():
        Y_pred = model(X_t).cpu().numpy()
    pred_best = np.argmax(Y_pred, axis=1)
    top1 = np.mean(pred_best == true_best)
    return top1, pred_best


def main():
    # ============ 加载数据 ============
    data_path = os.path.join(DATA_DIR, 'static_beam_data.npz')
    data = np.load(data_path, allow_pickle=True)

    X_norm_full = data['X_wide']         # (N, 8) 已归一化(参考)
    X_raw = data['X_wide_raw']           # (N, 8) 含噪声线性功率
    Y_raw = data['Y_narrow_raw']

    N, n_wide = X_raw.shape
    true_best = np.argmax(Y_raw, axis=1)

    # 划分测试集(用同样的 80/20 比例,但这里我们用全部数据避免依赖训练划分)
    # 因为模型训练时见过的样本和测试样本的相对差异在消融下应该一致
    # 为严格起见,用最后 20% 作为"近似测试集"
    n_test = N - int(N * TRAIN_RATIO)
    np.random.seed(SEED)
    perm = np.random.permutation(N)
    test_idx = perm[-n_test:]

    X_raw_test = X_raw[test_idx]
    true_best_test = true_best[test_idx]

    print("=" * 70)
    print("  实验 D:输入消融——验证网络使用了哪些宽波束的信息")
    print("=" * 70)
    print(f"  测试样本数: {len(test_idx)}")
    print(f"  宽波束数: {n_wide}")
    print(f"  设备: {DEVICE}")

    # ============ 加载已训练 BT 模型 ============
    model = Beam_Estimator_BT(
        n_wide=8, n_narrow=32,
        dim=64, m_cat=4, num_heads=2,
        mlp_dim=64,
        num_encoder_layers=1,
        num_classifier_layers=4,
        dropout=0.1,
    ).to(DEVICE)
    weight_path = os.path.join(RESULT_DIR, 'bt_kl.pth')
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE,
                                     weights_only=True))

    # ============ 基准:完整输入 ============
    print("\n--- 基准 ---")
    X_full_norm = normalize_per_sample_dB(X_raw_test)
    top1_full, _ = evaluate_top1(model, X_full_norm, true_best_test)
    print(f"  完整 8 波束输入:  Top-1 = {top1_full*100:.2f}%")

    # ============ 强波束消融 ============
    print("\n--- 只保留 Top-K 强波束 ---")
    print(f"  {'K':<5} {'Top-1':<10} {'相对完整':<12} {'相对随机猜':<12}")
    random_baseline = 1 / 32  # 完全瞎猜
    for k in [1, 2, 3, 4, 5, 6, 7, 8]:
        masked = mask_keep_topk(X_raw_test, k, mode='strong')
        X_norm = normalize_per_sample_dB(masked)
        top1, _ = evaluate_top1(model, X_norm, true_best_test)
        delta = (top1 - top1_full) * 100
        rel = top1 / top1_full * 100
        print(f"  {k:<5} {top1*100:>6.2f}%   "
              f"{delta:>+6.2f}%      ({rel:>5.1f}%)")

    # ============ 弱波束消融 ============
    print("\n--- 只保留 Top-K 弱波束(强波束置零)---")
    print(f"  {'K':<5} {'Top-1':<10} {'相对完整':<12}")
    for k in [1, 2, 3, 4, 5, 6, 7, 8]:
        masked = mask_keep_topk(X_raw_test, k, mode='weak')
        X_norm = normalize_per_sample_dB(masked)
        top1, _ = evaluate_top1(model, X_norm, true_best_test)
        delta = (top1 - top1_full) * 100
        print(f"  {k:<5} {top1*100:>6.2f}%   {delta:>+6.2f}%")

    # ============ 单波束消融:逐个移除每个位置 ============
    print("\n--- 逐个移除单个波束(其他 7 个保留)---")
    print(f"  {'移除波束':<10} {'Top-1':<10} {'退化':<10}")
    for j in range(n_wide):
        # 把第 j 个波束置零
        masked = X_raw_test.copy()
        masked[:, j] = 0
        X_norm = normalize_per_sample_dB(masked)
        top1, _ = evaluate_top1(model, X_norm, true_best_test)
        delta = (top1 - top1_full) * 100
        print(f"  波束 {j}      {top1*100:>6.2f}%   {delta:>+6.2f}%")

    # ============ 解读 ============
    print("\n" + "=" * 70)
    print(" 解读说明")
    print("=" * 70)
    print("  - 如果 Top-K 强波束的 K=1~3 已经接近完整 Top-1")
    print("    → 网络几乎只用最强的几个波束,弱波束信息没用上")
    print("  - 如果 K 必须 ≥6 才接近完整 Top-1")
    print("    → 网络确实用了所有波束,信息利用充分")
    print("  - 如果 Top-K 弱波束完全失败(接近随机)")
    print("    → 弱波束单独不够,但不代表它在完整输入下没用")
    print("  - 移除单个波束的退化越大,该波束越重要")


if __name__ == '__main__':
    main()
