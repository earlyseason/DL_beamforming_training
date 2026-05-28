# ============================================================
# experiment_B_fc.py
# 用全连接模型重新验证实验 B:预测落点统计
# ============================================================

import os
import sys
import io
import numpy as np
import torch

from models import Beam_Estimator_FC

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                              errors='replace')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'beam_dataset')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'results', 'ablation')


def beam_response_matrix(codebook_wide, codebook_narrow):
    return np.abs(codebook_wide.conj().T @ codebook_narrow) ** 2


def build_coverage_main_lobe(resp, threshold_db=-3.0):
    coverage = []
    for j in range(resp.shape[0]):
        peak = resp[j].max()
        thresh = peak * 10 ** (threshold_db / 10)
        cov = np.where(resp[j] >= thresh)[0]
        coverage.append(cov)
    return coverage


def main():
    # 加载数据
    static = np.load(os.path.join(DATA_DIR, 'static_beam_data.npz'),
                     allow_pickle=True)
    X_wide_norm = static['X_wide']
    Y_narrow_raw = static['Y_narrow_raw']
    X_wide_raw = static['X_wide_raw']
    codebook_wide = static['codebook_wide']
    codebook_narrow = static['codebook_narrow']

    N_wide = codebook_wide.shape[1]
    N_narrow = codebook_narrow.shape[1]
    N = X_wide_norm.shape[0]

    # 几何关系
    resp = beam_response_matrix(codebook_wide, codebook_narrow)
    coverage_main = build_coverage_main_lobe(resp, threshold_db=-3.0)

    # 真实信息
    true_best = np.argmax(Y_narrow_raw, axis=1)
    j_argmax = np.argmax(X_wide_raw, axis=1)

    # 加载 FC + KL 模型
    model = Beam_Estimator_FC().to(DEVICE)
    weight_path = os.path.join(RESULT_DIR, 'fc_kl.pth')
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE,
                                     weights_only=True))
    model.eval()

    with torch.no_grad():
        x = torch.FloatTensor(X_wide_norm).to(DEVICE)
        y_pred_norm = model(x).cpu().numpy()

    model_pred = np.argmax(y_pred_norm, axis=1)
    model_acc = (model_pred == true_best).mean()

    print("=" * 60)
    print(" 实验 B 重验:FC + KL 模型的预测落点分析")
    print("=" * 60)
    print(f"  模型 Top-1: {model_acc*100:.2f}%")

    # 模型预测是否落在 argmax 宽波束的主瓣覆盖内
    in_cov = np.array([model_pred[i] in coverage_main[j_argmax[i]]
                       for i in range(N)])

    # 判断模型预测落入哪个宽波束的主瓣
    narrow_to_wide = np.argmax(resp, axis=0)
    pred_belongs = narrow_to_wide[model_pred]
    cross_wide = pred_belongs != j_argmax

    print(f"\n  模型预测落在 argmax 宽波束的 -3dB 主瓣覆盖内: "
          f"{in_cov.mean()*100:6.2f}%")
    print(f"  模型预测跨到了非 argmax 的宽波束(响应归属):     "
          f"{cross_wide.mean()*100:6.2f}%")

    if cross_wide.sum() > 0:
        within_acc = (model_pred[~cross_wide] == true_best[~cross_wide]).mean()
        cross_acc = (model_pred[cross_wide] == true_best[cross_wide]).mean()
        print(f"\n  分组 Top-1:")
        print(f"    不跨宽波束(局限本宽波束)({(~cross_wide).sum():5d} 个) "
              f"Top-1 = {within_acc*100:6.2f}%")
        print(f"    跨宽波束修正           ({cross_wide.sum():5d} 个) "
              f"Top-1 = {cross_acc*100:6.2f}%")

    # 真实最优是否落在 argmax 覆盖内
    true_in_cov = np.array([true_best[i] in coverage_main[j_argmax[i]]
                            for i in range(N)])
    inside_acc = (model_pred[true_in_cov] == true_best[true_in_cov]).mean()
    if (~true_in_cov).sum() > 0:
        outside_acc = (model_pred[~true_in_cov] ==
                       true_best[~true_in_cov]).mean()
    else:
        outside_acc = float('nan')

    print(f"\n  真实最优窄波束落在 argmax 宽波束覆盖内的比例: "
          f"{true_in_cov.mean()*100:6.2f}%")
    print(f"  分组 Top-1:")
    print(f"    真实最优在覆盖内 ({true_in_cov.sum():5d} 个) "
          f"模型 Top-1 = {inside_acc*100:6.2f}%")
    print(f"    真实最优在覆盖外 ({(~true_in_cov).sum():5d} 个) "
          f"模型 Top-1 = {outside_acc*100:6.2f}%")

    print("\n" + "=" * 60)
    print(" 对比 Baseline (1D-CNN)")
    print("=" * 60)
    print("  Baseline 模型:")
    print("    跨宽波束修正比例: 22.63%")
    print("    真实最优在覆盖外时 Top-1: 44.68%")
    print(f"\n  FC + KL 模型:")
    print(f"    跨宽波束修正比例: {cross_wide.mean()*100:.2f}%")
    print(f"    真实最优在覆盖外时 Top-1: {outside_acc*100:.2f}%")
    print(f"\n  改进:")
    print(f"    跨宽波束修正比例提升: {(cross_wide.mean() - 0.2263)*100:+.2f}%")
    print(f"    覆盖外命中率提升: {(outside_acc - 0.4468)*100:+.2f}%")


if __name__ == '__main__':
    main()
