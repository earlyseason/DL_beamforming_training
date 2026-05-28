# ============================================================
# data_generation.py
# 数据集生成主流程：调用 channel_generator 和 utils 完成数据生成
#
# 使用环境: conda activate mmwave_py12
# 运行: python data_generation.py
# ============================================================

import numpy as np
import os

from channel_generator import (
    load_deepmimo_channels,
    load_deepmimo_temporal_channels,
    generate_synthetic_channels,
    generate_temporal_channels,
)
from utils import (
    generate_dft_codebook,
    generate_wide_codebook_subarray,
    compute_beam_power,
    process_beam_data,
    construct_sliding_window,
    normalize_power,
    filter_valid_channels,
)

# ===================== 全局参数配置 =====================
# 天线配置
M = 32              # 基站天线数（ULA: 32x1）
Nr = 1              # 用户天线数（单天线）

# 码本配置
N_narrow = 32       # 窄波束（高分辨率 DFT 码本）波束数
N_wide = 8          # 宽波束（低分辨率）波束数

# 信噪比与功率配置
P_tx_dBm = 30       # 发送功率 (dBm)
SNR_dB = 10         # 接收端信噪比 (dB)

# 时序滑动窗口长度
L = 3                # 用于 LSTM 预测的历史帧数

# DeepMIMO 场景配置
SCENARIO_NAME = 'city_0_newyork_28'  # New York 28GHz 毫米波场景

# 数据保存路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, 'data', 'beam_dataset')


def main():
    """
    主函数：生成完整的训练数据集

    优先使用 DeepMIMO 真实射线追踪数据，
    若 DeepMIMO 不可用则回退到合成信道。
    """
    print("=" * 60)
    print("  波束选择数据集生成")
    print("  配置: BS天线={}, 窄波束={}, 宽波束={}".format(
        M, N_narrow, N_wide))
    print("=" * 60)
    # ! 注意生成宽波束时，本程序应用了功率归一化，这导致子阵总功率并没有像每天线功率那样衰减
    # ============ 1. 生成码本 ============
    print("\n[1/5] 生成码本...")
    narrow_codebook = generate_dft_codebook(M, N_narrow)
    # 当前最佳: 子阵列方法（即使有栅瓣，但宽主瓣保证了角度覆盖）
    wide_codebook = generate_wide_codebook_subarray(M, N_wide)
    print(f"  窄波束码本 (DFT-32): shape = {narrow_codebook.shape}")
    print(f"  宽波束码本 (子阵列-8): shape = {wide_codebook.shape}")

    # ============ 2. 功率参数 ============
    P_tx_linear = 10 ** (P_tx_dBm / 10) / 1000  # dBm → Watt
    snr_linear = 10 ** (SNR_dB / 10)
    print(f"\n[2/5] 功率配置: P_tx = {P_tx_dBm} dBm = {P_tx_linear:.4f} W, "
          f"SNR = {SNR_dB} dB")

    # ============ 3. 加载/生成信道 ============
    print("\n[3/5] 加载信道数据...")
    use_deepmimo = False
    try:
        channels = load_deepmimo_channels(
            scenario_name=SCENARIO_NAME, M=M, Nr=Nr)
        use_deepmimo = True
        print(f"  DeepMIMO 信道加载成功!")
    except Exception as e:
        print(f"  DeepMIMO 不可用 ({type(e).__name__}: {e})")
        print(f"  回退到合成信道...")
        channels = generate_synthetic_channels(num_users=2000, M=M)

    # 过滤无效（全零）信道：被建筑物完全遮挡的用户没有有效路径，
    # 对应信道为零，这些样本会稀释训练数据，需要剔除
    channels, _ = filter_valid_channels(channels)
    num_users = channels.shape[0]
    print(f"  有效用户数: {num_users}")

    # ============ 4. 计算波束功率（静态数据，用于估计模型） ============
    print("\n[4/5] 计算波束接收功率...")
    wide_powers, narrow_powers = process_beam_data(
        channels, narrow_codebook, wide_codebook,
        snr_linear, P_tx_linear, N_wide, N_narrow)

    print(f"  宽波束功率 X: shape = {wide_powers.shape}")
    print(f"  窄波束功率 Y: shape = {narrow_powers.shape}")

    # 归一化
    wide_norm, w_min, w_max = normalize_power(wide_powers)
    narrow_norm, n_min, n_max = normalize_power(narrow_powers)

    # ============ 5. 生成时序数据（用于预测模型） ============
    print("\n[5/5] 生成时序数据并构造滑动窗口...")

    try:
        if use_deepmimo:
            temporal_channels = load_deepmimo_temporal_channels(
                num_timestamps=10, scenario_name=SCENARIO_NAME, M=M, Nr=Nr)
        else:
            raise RuntimeError("使用合成时序信道")
    except Exception as e:
        print(f"  时序信道回退到合成模式 ({type(e).__name__})")
        temporal_channels = generate_temporal_channels(
            num_users=500, num_frames=10, num_paths=3, M=M)

    # 过滤无效时序信道（同样剔除全零用户）
    temporal_channels, _ = filter_valid_channels(temporal_channels)

    # 计算时序波束功率
    num_t_users, num_frames, _ = temporal_channels.shape
    wide_powers_seq = np.zeros((num_t_users, num_frames, N_wide))
    narrow_powers_seq = np.zeros((num_t_users, num_frames, N_narrow))

    for u in range(num_t_users):
        for t in range(num_frames):
            h = temporal_channels[u, t]
            wide_powers_seq[u, t] = compute_beam_power(
                h, wide_codebook, snr_linear, P_tx_linear)
            narrow_powers_seq[u, t] = compute_beam_power(
                h, narrow_codebook, snr_linear, P_tx_linear)

    # 构造滑动窗口
    X_seq, Y_seq = construct_sliding_window(
        wide_powers_seq, narrow_powers_seq, window_size=L)
    print(f"  时序输入 X_seq: shape = {X_seq.shape}  (samples, L={L}, {N_wide})")
    print(f"  时序标签 Y_seq: shape = {Y_seq.shape}  (samples, {N_narrow})")

    # 归一化时序数据
    X_seq_norm, xs_min, xs_max = normalize_power(X_seq)
    Y_seq_norm, ys_min, ys_max = normalize_power(Y_seq)

    # ============ 保存数据 ============
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 静态数据（估计模型用）
    np.savez(os.path.join(SAVE_DIR, 'static_beam_data.npz'),
             X_wide=wide_norm,
             Y_narrow=narrow_norm,
             X_wide_raw=wide_powers,
             Y_narrow_raw=narrow_powers,
             norm_params_wide=np.array([w_min, w_max]),
             norm_params_narrow=np.array([n_min, n_max]),
             codebook_narrow=narrow_codebook,
             codebook_wide=wide_codebook)

    # 时序数据（预测模型用）
    np.savez(os.path.join(SAVE_DIR, 'temporal_beam_data.npz'),
             X_seq=X_seq_norm,
             Y_seq=Y_seq_norm,
             X_seq_raw=X_seq,
             Y_seq_raw=Y_seq,
             norm_params_X=np.array([xs_min, xs_max]),
             norm_params_Y=np.array([ys_min, ys_max]))

    print(f"\n数据已保存至: {SAVE_DIR}")
    print("  - static_beam_data.npz  (估计模型训练数据)")
    print("  - temporal_beam_data.npz (预测模型训练数据)")

    # 打印数据统计
    print("\n" + "=" * 40)
    print("数据统计:")
    print(f"  数据来源: {'DeepMIMO (' + SCENARIO_NAME + ')' if use_deepmimo else '合成信道'}")
    print(f"  静态样本数: {wide_powers.shape[0]}")
    print(f"  时序样本数: {X_seq.shape[0]}")
    best_beams = np.argmax(narrow_powers, axis=1)
    print(f"  窄波束最优波束分布: "
          f"均值={best_beams.mean():.1f}, "
          f"标准差={best_beams.std():.1f}")
    print("=" * 40)


if __name__ == '__main__':
    main()
