# ============================================================
# experiment_AB.py
# 验证神经网络是否真的利用了跨宽波束信息进行超分辨率
#
# 实验 A: Lazy baseline Top-1 对比
#   A0 完全随机 / A1 几何中心 / A2 经验众数 / A3 主瓣覆盖内随机
#   M  Beam_Estimator_1D 实测 Top-1
#
# 实验 B: 预测落点统计
#   - 模型预测窄波束是否落在最强宽波束的主瓣覆盖范围内
#   - 模型是否做"跨宽波束修正"
#   - 真实最优落在覆盖外时,模型还能命中吗?
# ============================================================

import os
import sys
import io
import numpy as np
import torch

from models import Beam_Estimator_1D

# 强制 stdout/stderr 使用 utf-8,避免 Windows GBK 控制台编码报错
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                                  errors='replace')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'beam_dataset')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'results')


def beam_response_matrix(codebook_wide, codebook_narrow):
    """
    计算每个宽波束在每个窄波束方向上的能量响应:
        R[j, k] = | w_wide_j^H * w_narrow_k |^2

    输出 shape: (N_wide, N_narrow)
    """
    return np.abs(codebook_wide.conj().T @ codebook_narrow) ** 2


def build_coverage_main_lobe(resp, threshold_db=-3.0):
    """
    每个宽波束的"主瓣覆盖窄波束集合":
        在 -threshold_db 阈值内的窄波束。
    """
    coverage = []
    for j in range(resp.shape[0]):
        peak = resp[j].max()
        thresh = peak * 10 ** (threshold_db / 10)
        cov = np.where(resp[j] >= thresh)[0]
        coverage.append(cov)
    return coverage


def build_geometric_center(resp):
    """每个宽波束响应最强的窄波束 → 该宽波束的几何中心窄波束。"""
    return np.argmax(resp, axis=1)


def build_empirical_mode(wide_powers_raw, narrow_powers_raw, n_wide, n_narrow):
    """
    经验众数映射:在数据集中,当宽波束 j 是最强宽波束时,
    真实最优窄波束的众数是哪个。
    """
    j_max = np.argmax(wide_powers_raw, axis=1)
    k_max = np.argmax(narrow_powers_raw, axis=1)
    table = np.zeros(n_wide, dtype=int)
    for j in range(n_wide):
        mask = j_max == j
        if mask.sum() == 0:
            table[j] = -1
            continue
        counts = np.bincount(k_max[mask], minlength=n_narrow)
        table[j] = int(np.argmax(counts))
    return table


def main():
    # ============ 加载数据 ============
    static = np.load(os.path.join(DATA_DIR, 'static_beam_data.npz'),
                     allow_pickle=True)
    X_wide_norm = static['X_wide']           # (N, 8)
    Y_narrow_raw = static['Y_narrow_raw']    # (N, 32)
    X_wide_raw = static['X_wide_raw']        # (N, 8)
    codebook_wide = static['codebook_wide']      # (M, N_wide) complex
    codebook_narrow = static['codebook_narrow']  # (M, N_narrow) complex

    N_wide = codebook_wide.shape[1]
    N_narrow = codebook_narrow.shape[1]
    N = X_wide_norm.shape[0]

    print("=" * 60)
    print("  实验 AB:验证模型是否真在利用跨宽波束信息")
    print("=" * 60)
    print(f"  数据样本数: {N}")
    print(f"  宽波束: {N_wide}, 窄波束: {N_narrow}")
    print(f"  设备: {DEVICE}")

    # ============ 几何关系 ============
    resp = beam_response_matrix(codebook_wide, codebook_narrow)
    geom_center = build_geometric_center(resp)
    coverage_main = build_coverage_main_lobe(resp, threshold_db=-3.0)

    print("\n--- 码本几何关系(基于 |w_j^H w_k|^2) ---")
    print(f"  几何中心映射 wide→narrow: {geom_center.tolist()}")
    for j, cov in enumerate(coverage_main):
        peak_dB = 10 * np.log10(resp[j].max() + 1e-30)
        print(f"  宽波束 {j} 峰值 {peak_dB:6.2f} dB,主瓣 -3dB 覆盖窄波束: "
              f"{cov.tolist()} (共 {len(cov)})")

    # ============ 经验众数(在全集构造,作为最强 lazy baseline) ============
    emp_mode = build_empirical_mode(X_wide_raw, Y_narrow_raw,
                                    N_wide, N_narrow)
    print(f"\n  经验众数 wide→narrow: {emp_mode.tolist()}")

    # ============ 真实信息 ============
    true_best = np.argmax(Y_narrow_raw, axis=1)
    j_argmax = np.argmax(X_wide_raw, axis=1)

    # ============ 实验 A: 各 lazy baseline 与模型 Top-1 对比 ============
    print("\n" + "=" * 60)
    print(" 实验 A: Lazy baseline vs 模型 Top-1")
    print("=" * 60)

    # A0 完全均匀随机
    rng = np.random.default_rng(0)
    a0_pred = rng.integers(0, N_narrow, size=N)
    a0 = (a0_pred == true_best).mean()

    # A1 最强宽波束 → 该宽波束几何中心窄波束
    a1_pred = geom_center[j_argmax]
    a1 = (a1_pred == true_best).mean()

    # A2 最强宽波束 → 经验众数(训练数据中该宽波束最常对应的窄波束)
    a2_pred = emp_mode[j_argmax]
    a2 = (a2_pred == true_best).mean()

    # A3 最强宽波束 → 在主瓣覆盖范围内随机选一个
    rng = np.random.default_rng(1)
    a3_pred = np.array([rng.choice(coverage_main[j]) for j in j_argmax])
    a3 = (a3_pred == true_best).mean()

    # 模型预测(全集评估;模型已训练过,仅作为"上界"参考)
    estimator = Beam_Estimator_1D().to(DEVICE)
    weight_path = os.path.join(RESULT_DIR, 'best_estimator.pth')
    estimator.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    estimator.eval()

    with torch.no_grad():
        x = torch.FloatTensor(X_wide_norm).unsqueeze(1).to(DEVICE)
        y_pred_norm = estimator(x).cpu().numpy()
    model_pred = np.argmax(y_pred_norm, axis=1)
    model_acc = (model_pred == true_best).mean()

    print(f"  A0 完全随机                                Top-1 = {a0*100:6.2f}%")
    print(f"  A1 几何中心  (最强宽波束→中心窄波束)         Top-1 = {a1*100:6.2f}%")
    print(f"  A2 经验众数  (最强宽波束→数据集众数)         Top-1 = {a2*100:6.2f}%")
    print(f"  A3 主瓣覆盖内随机 (最强宽波束→覆盖范围随机)  Top-1 = {a3*100:6.2f}%")
    print(f"  -----------------------------------------------------")
    print(f"  M  Beam_Estimator_1D 模型预测              Top-1 = {model_acc*100:6.2f}%")

    print(f"\n  解读:")
    print(f"    模型超过最强 lazy baseline 的提升 = "
          f"{(model_acc - max(a1, a2, a3))*100:+.2f}%")
    if model_acc - max(a1, a2, a3) < 0.02:
        print(f"    [!] 模型几乎没有超越 lazy baseline,"
              f"可能没有真正学到跨宽波束信息")
    elif model_acc - max(a1, a2, a3) < 0.05:
        print(f"    [~] 模型仅勉强超越 lazy baseline,跨宽波束利用有限")
    else:
        print(f"    [OK] 模型显著超过 lazy baseline,确实学到了额外信息")

    # ============ 实验 B: 落点统计 ============
    print("\n" + "=" * 60)
    print(" 实验 B: 模型预测落点 vs 最强宽波束的覆盖范围")
    print("=" * 60)

    # 模型预测是否落在 argmax 宽波束的主瓣覆盖内
    in_cov = np.array([model_pred[i] in coverage_main[j_argmax[i]]
                       for i in range(N)])

    # 模型预测是否就是 argmax 宽波束的几何中心
    is_center = (model_pred == geom_center[j_argmax])

    # 判断模型预测落入哪个宽波束的主瓣 → 是否跨宽波束
    # 用响应最强的宽波束作为该窄波束的"归属"
    narrow_to_wide = np.argmax(resp, axis=0)  # (N_narrow,)
    pred_belongs = narrow_to_wide[model_pred]
    cross_wide = pred_belongs != j_argmax

    print(f"  模型预测落在 argmax 宽波束的 -3dB 主瓣覆盖内: "
          f"{in_cov.mean()*100:6.2f}%")
    print(f"  模型预测 = argmax 宽波束的几何中心窄波束:       "
          f"{is_center.mean()*100:6.2f}%")
    print(f"  模型预测跨到了非 argmax 的宽波束(响应归属):     "
          f"{cross_wide.mean()*100:6.2f}%")

    if cross_wide.sum() > 0:
        within_acc = (model_pred[~cross_wide] == true_best[~cross_wide]).mean()
        cross_acc = (model_pred[cross_wide] == true_best[cross_wide]).mean()
        print(f"\n  对模型预测的两种行为分别计算 Top-1:")
        print(f"    不跨宽波束(局限本宽波束)({(~cross_wide).sum():5d} 个) "
              f"Top-1 = {within_acc*100:6.2f}%")
        print(f"    跨宽波束修正           ({cross_wide.sum():5d} 个) "
              f"Top-1 = {cross_acc*100:6.2f}%")

    # ===== 真实最优是否落在 argmax 覆盖内 → "局限本宽波束"的理论上限 =====
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
    print(f"   → 这是任何'只局限本宽波束'方法的 Top-1 上限")
    print(f"  分组 Top-1:")
    print(f"    真实最优在覆盖内 ({true_in_cov.sum():5d} 个) "
          f"模型 Top-1 = {inside_acc*100:6.2f}%")
    print(f"    真实最优在覆盖外 ({(~true_in_cov).sum():5d} 个) "
          f"模型 Top-1 = {outside_acc*100:6.2f}%")

    # ============ 结论摘要 ============
    print("\n" + "=" * 60)
    print(" 结论摘要")
    print("=" * 60)
    delta_a1 = (model_acc - a1) * 100
    delta_a2 = (model_acc - a2) * 100
    delta_a3 = (model_acc - a3) * 100
    print(f"  模型 Top-1 = {model_acc*100:.2f}%")
    print(f"    比 A1 几何中心  高 {delta_a1:+.2f}%")
    print(f"    比 A2 经验众数  高 {delta_a2:+.2f}%")
    print(f"    比 A3 覆盖内随机 高 {delta_a3:+.2f}%")
    print(f"  模型预测局限本宽波束的比例: {(~cross_wide).mean()*100:.2f}%")
    print(f"  当真实最优在本宽波束覆盖外时,模型命中率: "
          f"{outside_acc*100:.2f}%")


if __name__ == '__main__':
    main()
