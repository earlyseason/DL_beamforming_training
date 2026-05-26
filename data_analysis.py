# ============================================================
# data_analysis.py
# 数据集统计分析与可视化
#
# 分析内容:
#   1. 原始功率分布（强/弱信号用户）
#   2. 功率集中度（信道是否平坦）
#   3. 最优波束分布
#   4. 归一化前后的功率对比
#   5. 边缘样本占比
#
# 用法: python data_analysis.py
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'data', 'beam_dataset',
                         'static_beam_data.npz')
SAVE_DIR = os.path.join(SCRIPT_DIR, 'results', 'data_analysis')


def load_data():
    """加载数据集"""
    data = np.load(DATA_PATH)
    Y_raw = data['Y_narrow_raw']    # 原始窄波束功率 (N, 32)
    Y_norm = data['Y_narrow']        # 归一化后的窄波束功率 (N, 32)
    X_raw = data['X_wide_raw']       # 原始宽波束功率 (N, 8)
    print(f"加载数据: {Y_raw.shape[0]} 个样本")
    print(f"  窄波束功率: {Y_raw.shape}")
    print(f"  宽波束功率: {X_raw.shape}")
    return Y_raw, Y_norm, X_raw


def analyze_power_distribution(Y_raw, save_path):
    """
    分析 1: 原始最大波束功率分布
    展示强/弱信号用户的分布
    """
    max_power = Y_raw.max(axis=1)        # 每个样本的最强波束功率
    max_power_dB = 10 * np.log10(max_power + 1e-30)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图 1: 线性域分布（log 横轴）
    axes[0].hist(max_power, bins=80, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('最大波束功率 (W, log scale)')
    axes[0].set_ylabel('用户数量')
    axes[0].set_title('原始功率分布（线性域）')
    axes[0].grid(True, alpha=0.3)

    # 子图 2: dB 域分布（更直观）
    axes[1].hist(max_power_dB, bins=80, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('最大波束功率 (dBW)')
    axes[1].set_ylabel('用户数量')
    axes[1].set_title('原始功率分布 (dB 域)')
    axes[1].grid(True, alpha=0.3)

    # 添加分位线
    p25, p50, p75 = np.percentile(max_power_dB, [25, 50, 75])
    for p, label, color in [(p25, '25%', 'green'), (p50, '中位数', 'red'),
                             (p75, '75%', 'blue')]:
        axes[1].axvline(p, color=color, linestyle='--', alpha=0.7,
                        label=f'{label}: {p:.1f} dBW')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")

    # 打印统计信息
    print(f"\n  [原始功率统计]")
    print(f"    最小值: {max_power.min():.2e} W ({max_power_dB.min():.1f} dBW)")
    print(f"    中位数: {np.median(max_power):.2e} W ({p50:.1f} dBW)")
    print(f"    最大值: {max_power.max():.2e} W ({max_power_dB.max():.1f} dBW)")
    print(f"    跨度: {max_power.max() / max_power.min():.1e} 倍 "
          f"({max_power_dB.max() - max_power_dB.min():.1f} dB)")


def analyze_concentration(Y_raw, save_path):
    """
    分析 2: 功率集中度
    集中度 = max_power / mean_power
    高集中度 = LOS 用户（信道集中在某个方向）
    低集中度 = NLOS / 边缘用户（功率分散在多个方向）
    """
    max_power = Y_raw.max(axis=1)
    mean_power = Y_raw.mean(axis=1)
    concentration = max_power / (mean_power + 1e-30)
    concentration_dB = 10 * np.log10(concentration)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图 1: 集中度直方图
    axes[0].hist(concentration_dB, bins=60, color='purple',
                 edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('功率集中度 (dB)\n[10*log10(max/mean)]')
    axes[0].set_ylabel('用户数量')
    axes[0].set_title('功率集中度分布')
    axes[0].grid(True, alpha=0.3)

    # 添加阈值线
    flat_threshold = 10 * np.log10(2)    # 集中度 < 2 倍 = 平坦
    sharp_threshold = 10 * np.log10(5)   # 集中度 > 5 倍 = 集中
    axes[0].axvline(flat_threshold, color='red', linestyle='--',
                    label=f'平坦阈值 (max/mean<2): {flat_threshold:.1f} dB')
    axes[0].axvline(sharp_threshold, color='green', linestyle='--',
                    label=f'集中阈值 (max/mean>5): {sharp_threshold:.1f} dB')
    axes[0].legend()

    # 子图 2: 集中度 vs 信号强度（散点图）
    max_power_dB = 10 * np.log10(max_power + 1e-30)
    axes[1].scatter(max_power_dB, concentration_dB, s=2, alpha=0.3, c='teal')
    axes[1].set_xlabel('最大波束功率 (dBW)')
    axes[1].set_ylabel('功率集中度 (dB)')
    axes[1].set_title('信号强度 vs 集中度')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")

    # 统计信息
    flat_ratio = (concentration < 2).mean()
    sharp_ratio = (concentration > 5).mean()
    print(f"\n  [功率集中度统计]")
    print(f"    平坦用户 (max/mean < 2): {flat_ratio*100:.1f}%  ← 难以选准最优波束")
    print(f"    集中用户 (max/mean > 5): {sharp_ratio*100:.1f}%  ← 容易选准")
    print(f"    中位数集中度: {np.median(concentration):.2f}x ({np.median(concentration_dB):.1f} dB)")


def analyze_best_beam_distribution(Y_raw, save_path):
    """
    分析 3: 最优波束的分布
    检查是否存在波束偏置（某些波束被频繁选中）
    """
    best_beams = np.argmax(Y_raw, axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    counts, _, _ = ax.hist(best_beams, bins=np.arange(33) - 0.5,
                            color='goldenrod', edgecolor='black', alpha=0.7)

    ax.set_xlabel('窄波束 Index')
    ax.set_ylabel('被选为最优的次数')
    ax.set_title('最优窄波束分布（共 32 个波束）')
    ax.set_xticks(range(0, 32, 2))
    ax.grid(True, alpha=0.3, axis='y')

    # 添加均匀分布参考线
    uniform_count = len(best_beams) / 32
    ax.axhline(uniform_count, color='red', linestyle='--',
               label=f'均匀分布参考: {uniform_count:.0f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")

    # 统计信息
    print(f"\n  [最优波束分布统计]")
    print(f"    最常被选: 波束 {counts.argmax()} ({int(counts.max())} 次)")
    print(f"    最少被选: 波束 {counts.argmin()} ({int(counts.min())} 次)")
    print(f"    选中比例不均度: {counts.max() / counts.min():.1f}x")


def analyze_normalization_effect(Y_raw, Y_norm, save_path):
    """
    分析 4: 归一化前后的对比
    展示边缘样本和强信号样本归一化后是否被"等价化"
    """
    max_power = Y_raw.max(axis=1)
    max_power_dB = 10 * np.log10(max_power + 1e-30)

    # 选 3 个不同强度的代表样本
    p10 = np.percentile(max_power, 10)
    p50 = np.percentile(max_power, 50)
    p90 = np.percentile(max_power, 90)

    weak_idx = np.argmin(np.abs(max_power - p10))     # 弱信号代表
    medium_idx = np.argmin(np.abs(max_power - p50))   # 中等信号代表
    strong_idx = np.argmin(np.abs(max_power - p90))   # 强信号代表

    samples = [
        (weak_idx, '弱信号 (10%分位)', max_power_dB[weak_idx]),
        (medium_idx, '中等信号 (50%分位)', max_power_dB[medium_idx]),
        (strong_idx, '强信号 (90%分位)', max_power_dB[strong_idx])
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    beam_idx = np.arange(32)

    for col, (idx, label, power_dB) in enumerate(samples):
        # 上排：原始功率（dB）
        raw_dB = 10 * np.log10(Y_raw[idx] + 1e-30)
        axes[0, col].bar(beam_idx, raw_dB, color='steelblue', alpha=0.7)
        axes[0, col].set_title(f'{label}\n(峰值={power_dB:.1f} dBW)')
        axes[0, col].set_xlabel('窄波束 Index')
        axes[0, col].set_ylabel('原始功率 (dB)')
        axes[0, col].grid(True, alpha=0.3)

        # 下排：归一化后的功率
        axes[1, col].bar(beam_idx, Y_norm[idx], color='coral', alpha=0.7)
        axes[1, col].set_title(f'{label}（归一化后）')
        axes[1, col].set_xlabel('窄波束 Index')
        axes[1, col].set_ylabel('归一化功率 [0,1]')
        axes[1, col].set_ylim([0, 1.1])
        axes[1, col].grid(True, alpha=0.3)

    plt.suptitle('归一化前后对比：边缘样本是否被"等价化"', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")


def analyze_edge_users(Y_raw, save_path):
    """
    分析 5: 边缘用户的占比和影响
    边缘用户定义: 最大功率低于中位数 1/100（-20 dB）
    """
    max_power = Y_raw.max(axis=1)
    max_power_dB = 10 * np.log10(max_power + 1e-30)
    median_dB = np.median(max_power_dB)

    edge_threshold = median_dB - 20  # 比中位数低 20 dB
    edge_mask = max_power_dB < edge_threshold

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图 1: 边缘用户 vs 普通用户的累积分布
    sorted_dB = np.sort(max_power_dB)
    cdf = np.arange(1, len(sorted_dB) + 1) / len(sorted_dB)
    axes[0].plot(sorted_dB, cdf, linewidth=2, color='navy')
    axes[0].axvline(edge_threshold, color='red', linestyle='--',
                    label=f'边缘阈值: {edge_threshold:.1f} dBW')
    axes[0].axvline(median_dB, color='green', linestyle='--',
                    label=f'中位数: {median_dB:.1f} dBW')
    axes[0].fill_between(sorted_dB, 0, cdf, where=(sorted_dB < edge_threshold),
                          alpha=0.3, color='red', label=f'边缘用户 ({edge_mask.mean()*100:.1f}%)')
    axes[0].set_xlabel('最大波束功率 (dBW)')
    axes[0].set_ylabel('累积分布 (CDF)')
    axes[0].set_title('信号强度 CDF')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图 2: 边缘 vs 普通用户的功率集中度对比
    mean_power = Y_raw.mean(axis=1)
    concentration = max_power / (mean_power + 1e-30)
    concentration_dB = 10 * np.log10(concentration)

    axes[1].hist(concentration_dB[~edge_mask], bins=40, alpha=0.6,
                  label=f'普通用户 ({(~edge_mask).mean()*100:.1f}%)',
                  color='blue', edgecolor='black')
    axes[1].hist(concentration_dB[edge_mask], bins=40, alpha=0.6,
                  label=f'边缘用户 ({edge_mask.mean()*100:.1f}%)',
                  color='red', edgecolor='black')
    axes[1].set_xlabel('功率集中度 (dB)')
    axes[1].set_ylabel('用户数量')
    axes[1].set_title('集中度对比：边缘 vs 普通用户')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")

    print(f"\n  [边缘用户统计]")
    print(f"    边缘用户阈值: {edge_threshold:.1f} dBW")
    print(f"    边缘用户数量: {edge_mask.sum()} / {len(max_power)} "
          f"({edge_mask.mean()*100:.1f}%)")
    print(f"    边缘用户中位集中度: "
          f"{np.median(concentration[edge_mask]):.2f}x")
    print(f"    普通用户中位集中度: "
          f"{np.median(concentration[~edge_mask]):.2f}x")


def main():
    print("=" * 60)
    print("  波束选择数据集分析")
    print("=" * 60)

    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 加载数据
    Y_raw, Y_norm, X_raw = load_data()

    # 5 项分析
    print("\n[1/5] 原始功率分布分析...")
    analyze_power_distribution(
        Y_raw, os.path.join(SAVE_DIR, '01_power_distribution.png'))

    print("\n[2/5] 功率集中度分析...")
    analyze_concentration(
        Y_raw, os.path.join(SAVE_DIR, '02_concentration.png'))

    print("\n[3/5] 最优波束分布分析...")
    analyze_best_beam_distribution(
        Y_raw, os.path.join(SAVE_DIR, '03_best_beam_distribution.png'))

    print("\n[4/5] 归一化效果分析...")
    analyze_normalization_effect(
        Y_raw, Y_norm, os.path.join(SAVE_DIR, '04_normalization_effect.png'))

    print("\n[5/5] 边缘用户分析...")
    analyze_edge_users(
        Y_raw, os.path.join(SAVE_DIR, '05_edge_users.png'))

    print("\n" + "=" * 60)
    print(f"所有分析图表已保存至: {SAVE_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
