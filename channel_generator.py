# ============================================================
# channel_generator.py
# 信道生成模块：基于 DeepMIMO v4 生成毫米波 MIMO 信道
# ============================================================

import numpy as np


# 默认参数（可通过函数参数覆盖）
DEFAULT_M = 32
DEFAULT_Nr = 1
DEFAULT_SCENARIO = 'city_0_newyork_28'


def   load_deepmimo_channels(scenario_name=DEFAULT_SCENARIO,
                           M=DEFAULT_M, Nr=DEFAULT_Nr):
    """
    使用 DeepMIMO v4 API 加载静态信道数据

    流程: dm.download(场景) → dm.load(场景, tx_sets={1:[0]}) → dataset.compute_channels()
    输出 shape: [n_ue, n_rx_ant, n_tx_ant, n_subcarriers]

    参数:
        scenario_name: DeepMIMO 场景名称
        M: 基站天线数
        Nr: 用户天线数

    返回:
        channels: shape = (num_users, M) 的复数信道矩阵
    """
    import deepmimo as dm

    print(f"正在加载 DeepMIMO 场景: {scenario_name}")
    print("  (首次运行会自动下载场景数据，可能需要几分钟)")
    dm.download(scenario_name)

    # 只取第 1 个 TX set，避免返回 MacroDataset
    dataset = dm.load(scenario_name, tx_sets={1: [0]})
    print(f"  场景加载成功")

    # 配置信道参数
    ch_params = dm.ChannelParameters(
        bs_antenna={'shape': [M, 1], 'spacing': 0.5},  # ULA: 32x1
        ue_antenna={'shape': [Nr, 1], 'spacing': 0.5},  # 单天线
        freq_domain=True,
        ofdm={
            'subcarriers': 64,
            'selected_subcarriers': [0],  # 单子载波
            'bandwidth': 100e6
        },
        num_paths=10
    )

    # 生成信道矩阵: [n_ue, Nr=1, M=32, n_sub=1]
    channels_raw = dataset.compute_channels(ch_params)
    print(f"  原始信道 shape: {channels_raw.shape}")

    # 提取为 (num_users, M)
    channels = channels_raw[:, 0, :, 0]  # [n_ue, M]
    print(f"  处理后信道 shape: {channels.shape} "
          f"(用户数={channels.shape[0]}, 天线数={M})")

    return channels


def load_deepmimo_temporal_channels(num_timestamps=10,
                                    scenario_name=DEFAULT_SCENARIO,
                                    M=DEFAULT_M, Nr=DEFAULT_Nr):
    """
    使用 DeepMIMO v4 生成时序信道（多时间戳）

    利用 compute_channels 的 num_timestamps 参数生成多普勒时变信道。

    参数:
        num_timestamps: 时间戳数量
        scenario_name: DeepMIMO 场景名称
        M: 基站天线数
        Nr: 用户天线数

    返回:
        temporal_channels: shape = (num_users, num_timestamps, M)
    """
    import deepmimo as dm

    dataset = dm.load(scenario_name, tx_sets={1: [0]})

    ch_params = dm.ChannelParameters(
        bs_antenna={'shape': [M, 1], 'spacing': 0.5},
        ue_antenna={'shape': [Nr, 1], 'spacing': 0.5},
        freq_domain=True,
        doppler=True,
        ofdm={
            'subcarriers': 64,
            'selected_subcarriers': [0],
            'bandwidth': 100e6
        },
        num_paths=10
    )

    # 生成多时间戳信道: [n_ue, Nr=1, M=32, n_sub=1, N_t]
    channels_raw = dataset.compute_channels(
        ch_params, num_timestamps=num_timestamps)
    print(f"  时序信道原始 shape: {channels_raw.shape}")

    # [n_ue, 1, M, 1, N_t] → [n_ue, N_t, M]
    channels = channels_raw[:, 0, :, 0, :]  # [n_ue, M, N_t]
    channels = channels.transpose(0, 2, 1)   # [n_ue, N_t, M]

    print(f"  时序信道 shape: {channels.shape}")
    return channels


def generate_synthetic_channels(num_users=2000, num_paths=5, M=DEFAULT_M):
    """
    生成合成信道数据（DeepMIMO 不可用时的备选方案）

    使用多径信道模型：h = sum_l alpha_l * a(theta_l)

    参数:
        num_users: 用户数量
        num_paths: 多径数
        M: 天线数

    返回:
        channels: shape = (num_users, M)
    """
    print(f"  生成合成多径信道：{num_users} 个用户，{num_paths} 条路径...")

    channels = np.zeros((num_users, M), dtype=complex)

    for i in range(num_users):
        h = np.zeros(M, dtype=complex)
        for p in range(num_paths):
            theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            alpha = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
            alpha *= (0.5 ** p)
            steering = np.exp(
                1j * np.pi * np.sin(theta) * np.arange(M)
            ) / np.sqrt(M)
            h += alpha * steering
        channels[i] = h

    return channels


def generate_temporal_channels(num_users=500, num_frames=10, num_paths=3,
                               M=DEFAULT_M):
    """
    生成时序合成信道（模拟用户移动场景，备选方案）

    参数:
        num_users: 用户数量
        num_frames: 每个用户的时间帧数
        num_paths: 多径数
        M: 天线数

    返回:
        temporal_channels: shape = (num_users, num_frames, M)
    """
    print(f"  生成时序合成信道：{num_users} 用户 x {num_frames} 帧...")

    fc = 28e9
    c = 3e8
    wavelength = c / fc
    d = wavelength / 2

    temporal_channels = np.zeros((num_users, num_frames, M), dtype=complex)

    for u in range(num_users):
        speed = np.random.uniform(1, 5)
        direction = np.random.uniform(0, 2 * np.pi)

        base_aods = np.random.uniform(-np.pi / 2, np.pi / 2, num_paths)
        base_alphas = (np.random.randn(num_paths) +
                       1j * np.random.randn(num_paths)) / np.sqrt(2)
        path_weights = np.array([0.5 ** p for p in range(num_paths)])
        base_alphas *= path_weights

        for t in range(num_frames):
            h = np.zeros(M, dtype=complex)
            dt = t * 0.01

            for p in range(num_paths):
                displacement = speed * dt
                delta_theta = 0.01 * displacement * np.cos(
                    direction - base_aods[p])
                theta_t = base_aods[p] + delta_theta

                doppler = (2 * np.pi * fc / c) * speed * \
                    np.cos(theta_t - direction) * dt
                alpha_t = base_alphas[p] * np.exp(1j * doppler)

                steering = np.exp(
                    1j * 2 * np.pi * d / wavelength *
                    np.sin(theta_t) * np.arange(M)
                ) / np.sqrt(M)

                h += alpha_t * steering

            temporal_channels[u, t] = h

    return temporal_channels
