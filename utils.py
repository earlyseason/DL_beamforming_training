# ============================================================
# utils.py
# 工具函数模块：码本生成、波束功率计算、数据预处理
# ============================================================

import numpy as np


def generate_dft_codebook(num_antennas, num_beams):
    """
    生成标准 DFT 码本（窄波束码本）

    参数:
        num_antennas: 天线数量 M
        num_beams: 波束数量 N

    返回:
        codebook: 复数矩阵，shape = (num_antennas, num_beams)
                  每一列是一个波束的导向向量
    """
    m_indices = np.arange(num_antennas).reshape(-1, 1)  # (M, 1)
    k_indices = np.arange(num_beams).reshape(1, -1)     # (1, N)

    # w_k[m] = (1/sqrt(M)) * exp(j * 2*pi * m * k / N)
    codebook = (1.0 / np.sqrt(num_antennas)) * \
        np.exp(1j * 2 * np.pi * m_indices * k_indices / num_beams)

    return codebook  # shape: (M, N)


def generate_wide_codebook_subarray(num_antennas, num_wide_beams):
    """
    生成宽波束码本 —— 子阵列方法（旧方案）

    原理：从 M 根天线中取中间 N_wide 根天线构成子阵列，
    对子阵列生成 DFT 码本，其余天线位置填零。
    子阵列孔径小 → 波束宽度大 → 实现低分辨率覆盖。

    缺陷:
        - 旁瓣高（-9 dB），存在严重栅瓣
        - 峰值低（-6 dB，因为 24 根天线置零）
        - 主瓣交叠仅 -10 dB（重叠不足）

    参数:
        num_antennas: 总天线数 M
        num_wide_beams: 宽波束数量 N_wide

    返回:
        wide_codebook: shape = (num_antennas, num_wide_beams)
    """
    sub_array_size = num_wide_beams
    start_idx = (num_antennas - sub_array_size) // 2
    end_idx = start_idx + sub_array_size

    sub_codebook = generate_dft_codebook(sub_array_size, num_wide_beams)

    wide_codebook = np.zeros((num_antennas, num_wide_beams), dtype=complex)
    wide_codebook[start_idx:end_idx, :] = sub_codebook

    for i in range(num_wide_beams):
        wide_codebook[:, i] /= np.linalg.norm(wide_codebook[:, i])

    return wide_codebook


def compute_beam_power(channel, codebook, snr_linear, P_tx_linear):
    """
    计算信道经过波束赋形后的接收功率（含噪声）

    参数:
        channel: 信道向量 h，shape = (M,) 复数
        codebook: 码本矩阵，shape = (M, N_beams)
        snr_linear: 线性信噪比
        P_tx_linear: 线性发送功率

    返回:
        rx_power: 各波束的接收功率，shape = (N_beams,)
    """
    num_beams = codebook.shape[1]

    # y_k = h^H * w_k，计算各波束的接收增益
    beam_gains = channel.conj() @ codebook  # shape: (N_beams,)

    # 无噪声接收功率
    signal_power = P_tx_linear * np.abs(beam_gains) ** 2

    # 噪声功率：基于平均信号功率和 SNR
    avg_signal = np.mean(signal_power)
    if avg_signal > 0:
        noise_power = avg_signal / snr_linear
    else:
        noise_power = 1e-10

    # 添加高斯白噪声（功率域）
    noise = noise_power * np.abs(
        np.random.randn(num_beams) + 1j * np.random.randn(num_beams)
    ) ** 2 / 2

    rx_power = signal_power + noise
    return rx_power


def compute_beam_signal_power(channel, codebook, P_tx_linear):
    """
    计算无噪声的接收信号功率(用于在线噪声重采样)

    与 compute_beam_power 相比,本函数不加噪声,
    便于训练时每个 batch 在线生成不同的噪声实现,
    起到数据增强的作用。

    参数:
        channel: 信道向量 h, shape = (M,) 复数
        codebook: 码本矩阵, shape = (M, N_beams)
        P_tx_linear: 线性发送功率

    返回:
        signal_power: 无噪声功率, shape = (N_beams,)
    """
    beam_gains = channel.conj() @ codebook
    return P_tx_linear * np.abs(beam_gains) ** 2


def add_noise_to_power_torch(signal_power, snr_linear, generator=None):
    """
    Torch 版本的加噪函数,用于训练循环中批量在线加噪 (GPU 友好)

    噪声模型与 compute_beam_power 保持一致:
        noise_power = mean(signal_power_per_sample) / SNR
        noise ~ noise_power * |N(0,1) + j*N(0,1)|^2 / 2

    参数:
        signal_power: 无噪声功率 tensor, shape = (B, n_beams)
                     每个样本独立加噪
        snr_linear: 线性 SNR
        generator: torch.Generator (可选, 控制随机性)

    返回:
        rx_power: 含噪声功率 tensor, 同 shape
    """
    import torch

    avg_signal = signal_power.mean(dim=-1, keepdim=True)  # (B, 1)
    noise_power = avg_signal / snr_linear

    shape = signal_power.shape
    if generator is not None:
        re = torch.randn(shape, generator=generator,
                         device=signal_power.device,
                         dtype=signal_power.dtype)
        im = torch.randn(shape, generator=generator,
                         device=signal_power.device,
                         dtype=signal_power.dtype)
    else:
        re = torch.randn(shape, device=signal_power.device,
                         dtype=signal_power.dtype)
        im = torch.randn(shape, device=signal_power.device,
                         dtype=signal_power.dtype)

    noise = noise_power * (re ** 2 + im ** 2) / 2
    return signal_power + noise


def normalize_power_torch(power_data, eps=1e-30):
    """
    Torch 版本的逐样本 dB 归一化(用于在线增强后的归一化)

    参数:
        power_data: shape = (B, n_beams), 线性功率(>=0)

    返回:
        normalized: shape = (B, n_beams), 范围 [0, 1]
    """
    import torch

    power_dB = 10 * torch.log10(power_data + eps)
    p_min = power_dB.min(dim=-1, keepdim=True).values
    p_max = power_dB.max(dim=-1, keepdim=True).values
    denom = p_max - p_min
    denom = torch.where(denom < 1e-10, torch.ones_like(denom), denom)
    return (power_dB - p_min) / denom


def random_beam_dropout_torch(power_linear, p_apply=0.5, max_drop=2):
    """
    Beam Dropout: 每个样本以 p_apply 概率丢弃 1~max_drop 个波束

    在线性功率域置零,后续 dB 归一化时被置零位置自然成为最弱(归一化为 0)。
    用于训练时强迫网络学习"缺失某波束读数也能推出整体分布"的稳健表示。

    参数:
        power_linear: (B, n_beams) 线性功率(>=0)
        p_apply: 每个样本应用 dropout 的概率
        max_drop: 单样本最多丢弃的波束数

    返回:
        masked: 同 shape, 被丢弃位置为 0
    """
    import torch

    B, n_beams = power_linear.shape
    device = power_linear.device

    # 每个样本是否应用 dropout
    apply_mask = torch.rand(B, device=device) < p_apply

    # 每个样本丢弃的数量(1 ~ max_drop)
    n_drops = torch.randint(1, max_drop + 1, (B,), device=device)

    # 用随机数排序后取前 k 个,等价于无放回随机选 k 个位置
    rand_vals = torch.rand(B, n_beams, device=device)
    sorted_indices = rand_vals.argsort(dim=1)

    # 构造每个样本的 keep mask
    pos_idx = torch.arange(n_beams, device=device).unsqueeze(0).expand(B, -1)
    drop_count = n_drops.unsqueeze(1)            # (B, 1)
    drop_in_sorted = (pos_idx < drop_count)       # (B, n_beams) bool

    keep_mask = torch.ones_like(power_linear)
    keep_mask.scatter_(1, sorted_indices, (~drop_in_sorted).float())

    # 不应用 dropout 的样本保持全 1
    keep_mask[~apply_mask] = 1.0

    return power_linear * keep_mask


def process_beam_data(channels, narrow_codebook, wide_codebook,
                      snr_linear, P_tx_linear, N_wide, N_narrow):
    """
    对所有用户信道计算宽波束和窄波束的接收功率

    参数:
        channels: shape = (num_users, M)
        narrow_codebook: shape = (M, N_narrow)
        wide_codebook: shape = (M, N_wide)
        snr_linear: 线性 SNR
        P_tx_linear: 线性发送功率
        N_wide: 宽波束数量
        N_narrow: 窄波束数量

    返回:
        wide_powers: shape = (num_users, N_wide) — 模型输入
        narrow_powers: shape = (num_users, N_narrow) — 模型标签
    """
    num_users = channels.shape[0]
    wide_powers = np.zeros((num_users, N_wide))
    narrow_powers = np.zeros((num_users, N_narrow))

    for i in range(num_users):
        h = channels[i]
        wide_powers[i] = compute_beam_power(
            h, wide_codebook, snr_linear, P_tx_linear)
        narrow_powers[i] = compute_beam_power(
            h, narrow_codebook, snr_linear, P_tx_linear)

    return wide_powers, narrow_powers


def construct_sliding_window(wide_powers_seq, narrow_powers_seq, window_size):
    """
    构造滑动窗口时序数据，用于 LSTM 预测模型

    参数:
        wide_powers_seq: shape = (num_users, num_frames, N_wide)
        narrow_powers_seq: shape = (num_users, num_frames, N_narrow)
        window_size: 滑动窗口长度 L

    返回:
        X_seq: shape = (num_samples, L, N_wide) — 过去 L 帧的宽波束功率
        Y_seq: shape = (num_samples, N_narrow) — 当前帧的窄波束功率
    """
    num_users, num_frames, _ = wide_powers_seq.shape
    X_list = []
    Y_list = []

    for u in range(num_users):
        for t in range(window_size, num_frames):
            x_window = wide_powers_seq[u, t - window_size:t, :]
            y_current = narrow_powers_seq[u, t, :]
            X_list.append(x_window)
            Y_list.append(y_current)

    X_seq = np.array(X_list)
    Y_seq = np.array(Y_list)
    return X_seq, Y_seq


def filter_valid_channels(channels, threshold=1e-10):
    """
    过滤掉全零（无效）的信道行

    DeepMIMO 射线追踪场景中，部分用户位置（建筑物内、完全遮挡区域）
    没有任何有效路径，对应的信道为全零。这些样本对训练无意义且会稀释数据。

    参数:
        channels: shape = (num_users, M) 静态信道
                  或 (num_users, num_frames, M) 时序信道
        threshold: 信道幅值的判定阈值，低于此值视为零信道

    返回:
        valid_channels: 过滤后的信道
        valid_mask: shape = (num_users,) 的布尔数组，True 表示有效用户
    """
    if channels.ndim == 2:
        # 静态信道: 每个用户的信道范数
        channel_norm = np.linalg.norm(channels, axis=1)
    elif channels.ndim == 3:
        # 时序信道: 所有帧的总能量都为零才视为无效用户
        channel_norm = np.linalg.norm(
            channels.reshape(channels.shape[0], -1), axis=1)
    else:
        raise ValueError(f"不支持的信道维度: {channels.shape}")

    valid_mask = channel_norm > threshold
    valid_channels = channels[valid_mask]

    print(f"  过滤无效信道: {valid_mask.sum()} / {len(valid_mask)} "
          f"个有效用户 ({100 * valid_mask.mean():.1f}%)")

    return valid_channels, valid_mask


def normalize_power(power_data):
    """
    对功率数据进行 dB 转换 + 逐样本归一化

    步骤:
    1. 转换到 dB 域: 10*log10(power)，使功率分布更均匀
    2. 逐样本 Min-Max 归一化到 [0, 1]

    参数:
        power_data: shape = (N, num_beams) 或 (N, L, num_beams)

    返回:
        normalized: 归一化后的数据
        p_min: 每样本最小值
        p_max: 每样本最大值
    """
    # 转换到 dB 域（加小量避免 log(0)）
    power_dB = 10 * np.log10(power_data + 1e-30)

    if power_dB.ndim == 2:
        p_min = power_dB.min(axis=1, keepdims=True)
        p_max = power_dB.max(axis=1, keepdims=True)
    elif power_dB.ndim == 3:
        p_min = power_dB.min(axis=(1, 2), keepdims=True)
        p_max = power_dB.max(axis=(1, 2), keepdims=True)
    else:
        p_min = power_dB.min()
        p_max = power_dB.max()

    denom = p_max - p_min
    denom = np.where(denom < 1e-10, 1.0, denom)
    normalized = (power_dB - p_min) / denom

    return normalized, p_min.squeeze(), p_max.squeeze()
