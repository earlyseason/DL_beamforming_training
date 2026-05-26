# ============================================================
# codebook_analysis.py
# 码本设计分析与可视化
#
# 分析内容:
#   1. 宽/窄波束的方向图（极坐标 + 笛卡尔坐标）
#   2. 宽波束之间的交叠情况
#   3. 宽波束与窄波束的角度对应关系
#   4. 码本相关性矩阵（评估冗余度）
#   5. 不同宽波束设计对比（子阵列 vs 相位组合）
#
# 用法: python codebook_analysis.py
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from utils import generate_dft_codebook, generate_wide_codebook_subarray

# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
M = 32           # 天线数
N_narrow = 32    # 窄波束数
N_wide = 8       # 宽波束数

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, 'results', 'codebook_analysis')


def compute_beam_pattern(codebook, num_angles=361):
    """
    计算码本的方向图

    参数:
        codebook: shape = (M, N) 复数码本
        num_angles: 采样角度数

    返回:
        angles_deg: 角度数组（度）
        pattern_dB: shape = (N, num_angles) 各波束在各角度的增益（dB）
    """
    M_local = codebook.shape[0]
    angles_deg = np.linspace(-90, 90, num_angles)
    angles_rad = np.deg2rad(angles_deg)

    # ULA 导向向量: a(θ) = [1, e^{jπsin(θ)}, ..., e^{jπ(M-1)sin(θ)}] / sqrt(M)
    pattern = np.zeros((codebook.shape[1], num_angles), dtype=complex)
    for i, theta in enumerate(angles_rad):
        steering = np.exp(1j * np.pi * np.sin(theta) * np.arange(M_local))
        steering = steering / np.sqrt(M_local)
        # 增益 = |w^H * a(θ)|^2
        pattern[:, i] = codebook.conj().T @ steering

    pattern_power = np.abs(pattern) ** 2
    pattern_dB = 10 * np.log10(pattern_power + 1e-30)

    return angles_deg, pattern_dB


def plot_beam_patterns_cartesian(codebook_n, codebook_w, save_path):
    """
    分析 1: 笛卡尔坐标下的方向图对比
    展示窄波束（精细）和宽波束（粗糙）的覆盖差异
    """
    angles, pattern_n_dB = compute_beam_pattern(codebook_n)
    _, pattern_w_dB = compute_beam_pattern(codebook_w)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 上图：32 个窄波束
    cmap_n = plt.cm.viridis(np.linspace(0, 1, N_narrow))
    for i in range(N_narrow):
        axes[0].plot(angles, pattern_n_dB[i], color=cmap_n[i], linewidth=1, alpha=0.8)
    axes[0].set_xlabel('角度 (度)')
    axes[0].set_ylabel('波束增益 (dB)')
    axes[0].set_title(f'窄波束码本 (N={N_narrow}) - 高分辨率覆盖')
    axes[0].set_ylim([-30, 5])
    axes[0].set_xlim([-90, 90])
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(-3, color='red', linestyle='--', alpha=0.5, label='-3dB线')
    axes[0].legend(loc='upper right')

    # 下图：8 个宽波束
    cmap_w = plt.cm.plasma(np.linspace(0, 1, N_wide))
    for i in range(N_wide):
        axes[1].plot(angles, pattern_w_dB[i], color=cmap_w[i],
                     linewidth=2, label=f'宽波束 {i}', alpha=0.8)
    axes[1].set_xlabel('角度 (度)')
    axes[1].set_ylabel('波束增益 (dB)')
    axes[1].set_title(f'宽波束码本 (N={N_wide}) - 子阵列方法')
    axes[1].set_ylim([-30, 5])
    axes[1].set_xlim([-90, 90])
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(-3, color='red', linestyle='--', alpha=0.5, label='-3dB线')
    axes[1].legend(loc='upper right', ncol=4, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")


def plot_beam_patterns_polar(codebook_n, codebook_w, save_path):
    """
    分析 2: 极坐标方向图
    更直观地展示波束的空间覆盖
    """
    angles, pattern_n_dB = compute_beam_pattern(codebook_n)
    _, pattern_w_dB = compute_beam_pattern(codebook_w)

    fig = plt.figure(figsize=(14, 6))

    # 窄波束极坐标
    ax1 = fig.add_subplot(121, projection='polar')
    cmap_n = plt.cm.viridis(np.linspace(0, 1, N_narrow))
    angles_rad = np.deg2rad(angles)
    for i in range(N_narrow):
        # 截断太弱的部分以便可视化
        gain = np.maximum(pattern_n_dB[i], -25) + 25  # 平移到正值
        ax1.plot(angles_rad, gain, color=cmap_n[i], linewidth=1, alpha=0.7)
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_thetamin(-90)
    ax1.set_thetamax(90)
    ax1.set_title(f'窄波束方向图 (N={N_narrow})', pad=20)

    # 宽波束极坐标
    ax2 = fig.add_subplot(122, projection='polar')
    cmap_w = plt.cm.plasma(np.linspace(0, 1, N_wide))
    for i in range(N_wide):
        gain = np.maximum(pattern_w_dB[i], -25) + 25
        ax2.plot(angles_rad, gain, color=cmap_w[i], linewidth=2,
                 label=f'宽 {i}', alpha=0.8)
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_thetamin(-90)
    ax2.set_thetamax(90)
    ax2.set_title(f'宽波束方向图 (N={N_wide})', pad=20)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")


def plot_overlap_analysis(codebook_w, save_path):
    """
    分析 3: 宽波束之间的交叠情况
    展示相邻宽波束在 -3dB / -10dB 处的重叠程度
    """
    angles, pattern_w_dB = compute_beam_pattern(codebook_w)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 上图：所有 8 个宽波束 + 包络线
    cmap_w = plt.cm.plasma(np.linspace(0, 1, N_wide))
    for i in range(N_wide):
        axes[0].plot(angles, pattern_w_dB[i], color=cmap_w[i],
                     linewidth=2, label=f'波束 {i}')
    # 计算所有波束的最大值包络
    envelope = pattern_w_dB.max(axis=0)
    axes[0].plot(angles, envelope, 'k--', linewidth=2, alpha=0.7,
                 label='最大值包络')
    axes[0].axhline(-3, color='red', linestyle=':', alpha=0.7, label='-3dB')
    axes[0].axhline(-10, color='orange', linestyle=':', alpha=0.7, label='-10dB')
    axes[0].set_xlabel('角度 (度)')
    axes[0].set_ylabel('增益 (dB)')
    axes[0].set_title('宽波束交叠分析（最大值包络越平坦 = 覆盖越均匀）')
    axes[0].set_ylim([-25, 5])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=5, fontsize=9, loc='lower center')

    # 下图：相邻波束的交叠（计算两波束的最小值，越高表示交叠越好）
    overlap_levels = []
    for i in range(N_wide - 1):
        overlap = np.minimum(pattern_w_dB[i], pattern_w_dB[i+1])
        overlap_levels.append(overlap.max())  # 两波束都强的最高点
        axes[1].plot(angles, overlap, linewidth=2,
                     label=f'波束 {i}∩{i+1}: 交点={overlap.max():.1f}dB')

    axes[1].set_xlabel('角度 (度)')
    axes[1].set_ylabel('交叠增益 (dB)')
    axes[1].set_title('相邻宽波束的交叠（越高代表覆盖越连续）')
    axes[1].set_ylim([-25, 5])
    axes[1].axhline(-3, color='red', linestyle=':', alpha=0.7, label='-3dB线')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=4, fontsize=8, loc='lower center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")

    # 打印交叠统计
    print(f"\n  [宽波束交叠统计]")
    avg_overlap = np.mean(overlap_levels)
    print(f"    相邻波束平均交叠点: {avg_overlap:.1f} dB")
    if avg_overlap > -3:
        print(f"    [OK] 交叠良好（> -3dB），覆盖连续")
    elif avg_overlap > -10:
        print(f"    [警告] 交叠一般（-10dB ~ -3dB），可能有覆盖漏洞")
    else:
        print(f"    [问题] 交叠很差（< -10dB），存在明显覆盖空洞")


def plot_wide_to_narrow_mapping(codebook_n, codebook_w, save_path):
    """
    分析 4: 宽波束与窄波束的对应关系
    每个宽波束实际"代表"了哪些窄波束？
    """
    angles, pattern_n_dB = compute_beam_pattern(codebook_n)
    _, pattern_w_dB = compute_beam_pattern(codebook_w)

    # 对每个宽波束，找出在其覆盖范围内的窄波束
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for w_idx in range(N_wide):
        ax = axes[w_idx]
        # 画当前宽波束
        ax.fill_between(angles, -30, pattern_w_dB[w_idx],
                         where=pattern_w_dB[w_idx] > -10,
                         alpha=0.3, color='red', label='宽波束覆盖区')
        ax.plot(angles, pattern_w_dB[w_idx], 'r-', linewidth=2.5,
                label=f'宽波束 {w_idx}')

        # 找出在该宽波束 -10dB 范围内的窄波束（峰值位置）
        narrow_peaks = []
        for n_idx in range(N_narrow):
            peak_angle = angles[np.argmax(pattern_n_dB[n_idx])]
            wide_gain_at_peak = pattern_w_dB[w_idx][np.argmax(pattern_n_dB[n_idx])]
            if wide_gain_at_peak > -10:
                narrow_peaks.append((n_idx, peak_angle, wide_gain_at_peak))

        # 画对应的窄波束
        cmap = plt.cm.viridis(np.linspace(0.2, 1, len(narrow_peaks) + 1))
        for k, (n_idx, peak_angle, _) in enumerate(narrow_peaks):
            ax.plot(angles, pattern_n_dB[n_idx], color=cmap[k], alpha=0.6,
                    linewidth=1)
            ax.axvline(peak_angle, color=cmap[k], linestyle=':', alpha=0.4)

        ax.set_title(f'宽波束 {w_idx} 覆盖了 {len(narrow_peaks)} 个窄波束')
        ax.set_xlabel('角度 (度)')
        ax.set_ylabel('增益 (dB)')
        ax.set_ylim([-25, 5])
        ax.set_xlim([-90, 90])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')

    plt.suptitle('宽-窄波束角度对应关系（理想：每个宽波束对应 4 个窄波束）',
                  fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")


def plot_correlation_matrix(codebook_n, codebook_w, save_path):
    """
    分析 5: 码本相关性矩阵
    评估码本的"信息冗余度"
    """
    # 宽-窄相关性
    corr_w_n = np.abs(codebook_w.conj().T @ codebook_n) ** 2  # (N_wide, N_narrow)

    # 宽-宽相关性
    corr_w_w = np.abs(codebook_w.conj().T @ codebook_w) ** 2  # (N_wide, N_wide)

    # 窄-窄相关性
    corr_n_n = np.abs(codebook_n.conj().T @ codebook_n) ** 2  # (N_narrow, N_narrow)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 宽-窄相关性
    im0 = axes[0].imshow(corr_w_n, cmap='hot', aspect='auto')
    axes[0].set_xlabel('窄波束 Index')
    axes[0].set_ylabel('宽波束 Index')
    axes[0].set_title('宽-窄码本相关性\n（每个宽波束"覆盖"哪些窄波束）')
    plt.colorbar(im0, ax=axes[0])

    # 宽-宽相关性（对角线应该高，非对角线越低越好）
    im1 = axes[1].imshow(corr_w_w, cmap='hot')
    axes[1].set_xlabel('宽波束 Index')
    axes[1].set_ylabel('宽波束 Index')
    axes[1].set_title('宽波束之间相关性\n（非对角线低=波束独立）')
    plt.colorbar(im1, ax=axes[1])

    # 窄-窄相关性
    im2 = axes[2].imshow(corr_n_n, cmap='hot')
    axes[2].set_xlabel('窄波束 Index')
    axes[2].set_ylabel('窄波束 Index')
    axes[2].set_title('窄波束之间相关性\n（DFT 应该接近正交）')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")

    # 评估指标
    off_diag_w = corr_w_w - np.diag(np.diag(corr_w_w))
    print(f"\n  [码本相关性统计]")
    print(f"    宽-宽非对角线最大值: {off_diag_w.max():.3f} (低=好)")
    print(f"    宽-宽非对角线均值: {off_diag_w.mean():.3f}")
    off_diag_n = corr_n_n - np.diag(np.diag(corr_n_n))
    print(f"    窄-窄非对角线均值: {off_diag_n.mean():.3f} (DFT理论上应接近0)")


def compare_wide_codebook_designs(save_path):
    """
    分析 6: 不同宽波束设计的对比
    对比子阵列方法 vs 相位组合方法 vs 直接 DFT-8
    """
    # 方案 A: 子阵列（你当前的方法）
    cb_subarray = generate_wide_codebook_subarray(M, N_wide)

    # 方案 B: 直接 32 天线 DFT-8（更窄的波束）
    cb_direct_dft8 = generate_dft_codebook(M, N_wide)

    # 方案 C: 相邻 DFT 波束相位组合（4 个相邻 DFT-32 波束加和）
    cb_dft32 = generate_dft_codebook(M, N_narrow)
    cb_combined = np.zeros((M, N_wide), dtype=complex)
    for i in range(N_wide):
        # 把相邻 4 个 DFT-32 波束加权组合
        cb_combined[:, i] = np.sum(cb_dft32[:, i*4:(i+1)*4], axis=1)
        cb_combined[:, i] /= np.linalg.norm(cb_combined[:, i])

    angles, pat_a = compute_beam_pattern(cb_subarray)
    _, pat_b = compute_beam_pattern(cb_direct_dft8)
    _, pat_c = compute_beam_pattern(cb_combined)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    for i in range(N_wide):
        axes[0].plot(angles, pat_a[i], linewidth=1.5, alpha=0.7)
        axes[1].plot(angles, pat_b[i], linewidth=1.5, alpha=0.7)
        axes[2].plot(angles, pat_c[i], linewidth=1.5, alpha=0.7)

    titles = [
        '方案 A: 子阵列方法（当前实现）\n中间 8 根天线 + DFT-8，其余 24 根置零',
        '方案 B: 直接 32 天线 DFT-8\n所有 32 根天线参与，但码本只有 8 个方向',
        '方案 C: 相邻 DFT 组合\n4 个相邻 DFT-32 波束加和，等效宽波束'
    ]

    for ax, title in zip(axes, titles):
        ax.set_xlabel('角度 (度)')
        ax.set_ylabel('增益 (dB)')
        ax.set_title(title)
        ax.set_ylim([-30, 10])
        ax.set_xlim([-90, 90])
        ax.grid(True, alpha=0.3)
        ax.axhline(-3, color='red', linestyle=':', alpha=0.5)

    plt.suptitle('三种宽波束设计方案对比', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")

    # 计算各方案的 -3dB 主瓣宽度（平均）
    print(f"\n  [三种方案的 -3dB 主瓣宽度对比]")
    for name, pat in [('方案A 子阵列', pat_a),
                       ('方案B 直接DFT-8', pat_b),
                       ('方案C 相邻组合', pat_c)]:
        peak_idx = pat.argmax(axis=1)
        widths = []
        for i in range(N_wide):
            peak_val = pat[i].max()
            mask = pat[i] > peak_val - 3
            indices = np.where(mask)[0]
            if len(indices) > 1:
                width = angles[indices[-1]] - angles[indices[0]]
                widths.append(width)
        print(f"    {name}: 平均主瓣宽度 = {np.mean(widths):.1f}°")


def main():
    print("=" * 60)
    print("  码本设计分析与可视化")
    print("=" * 60)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 生成两种码本
    codebook_n = generate_dft_codebook(M, N_narrow)
    codebook_w = generate_wide_codebook_subarray(M, N_wide)

    print(f"\n窄波束码本 shape: {codebook_n.shape}")
    print(f"宽波束码本 shape: {codebook_w.shape}")

    print("\n[1/6] 笛卡尔坐标方向图...")
    plot_beam_patterns_cartesian(
        codebook_n, codebook_w,
        os.path.join(SAVE_DIR, '01_patterns_cartesian.png'))

    print("\n[2/6] 极坐标方向图...")
    plot_beam_patterns_polar(
        codebook_n, codebook_w,
        os.path.join(SAVE_DIR, '02_patterns_polar.png'))

    print("\n[3/6] 宽波束交叠分析...")
    plot_overlap_analysis(
        codebook_w,
        os.path.join(SAVE_DIR, '03_overlap_analysis.png'))

    print("\n[4/6] 宽-窄波束对应关系...")
    plot_wide_to_narrow_mapping(
        codebook_n, codebook_w,
        os.path.join(SAVE_DIR, '04_wide_to_narrow_mapping.png'))

    print("\n[5/6] 码本相关性矩阵...")
    plot_correlation_matrix(
        codebook_n, codebook_w,
        os.path.join(SAVE_DIR, '05_correlation_matrix.png'))

    print("\n[6/6] 不同宽波束设计对比...")
    compare_wide_codebook_designs(
        os.path.join(SAVE_DIR, '06_wide_codebook_comparison.png'))

    print("\n" + "=" * 60)
    print(f"所有码本分析图表已保存至: {SAVE_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
