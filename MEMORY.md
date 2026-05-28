# MEMORY.md — 项目工作记忆

> 用途：跨会话恢复上下文。新会话时让 AI 读取本文件继续工作。

## 1. 项目概览

**项目名称**: 基于深度学习的低开销毫米波波束选择 (DL Beam Selection)

**参考论文**: *A Deep Learning-Based Low Overhead Beam Selection in mmWave Communications*

**核心思想**: 用 8 个低分辨率宽波束的接收功率，通过 1D-CNN 超分辨率重建出 32 个高分辨率窄波束的功率分布，并通过 LSTM 做时序预测。

**项目根目录**: `d:\AllPrograme\studycommon\middledaima\DLbeamforming`

## 2. 运行环境

- **Conda 环境**: `mmwave_py12`（Python 3.12）
- **Python 路径**: `D:/mydownload-softwares/anacond32013/envs/mmwave_py12/python.exe`
- **关键依赖**: `deepmimo` 4.0.1（已 pip install -e .）, `torch`（CUDA）, `numpy`, `scipy`, `matplotlib`
- **GPU**: 已确认 CUDA 可用
- **运行方式（避免 conda run 的 GBK 编码问题）**:
  ```bash
  D:/mydownload-softwares/anacond32013/envs/mmwave_py12/python.exe data_generation.py
  ```

## 3. 代码结构（已清理至最简稳定版）

```
DLbeamforming/
├── data_generation.py       # 主流程：调用以下两个模块生成数据
├── channel_generator.py     # DeepMIMO 信道生成（含合成信道备选）
├── utils.py                 # 工具函数（码本、功率计算、归一化、过滤）
├── models.py                # PyTorch 模型（Beam_Estimator_1D + Beam_Predictor_1D_LSTM）
├── train.py                 # 训练 + 评估 + 可视化
├── data_analysis.py         # 数据集统计分析（5 张图）
├── codebook_analysis.py     # 码本方向图分析（6 张图）
├── README.md                # 使用说明
├── MEMORY.md                # 本文件
├── deepmimo_scenarios/      # DeepMIMO 原始 .mat 数据（gitignore）
├── data/beam_dataset/       # 生成的训练数据 .npz（gitignore）
├── results/                 # 训练结果、模型权重、图表（gitignore）
│   ├── data_analysis/       # 数据集分析图
│   └── codebook_analysis/   # 码本方向图分析
└── DeepMIMO/                # 本地 DeepMIMO 源码（已 pip install -e .）
```

**utils.py 当前导出的函数**（已清理）：
- `generate_dft_codebook` — DFT-N 码本
- `generate_wide_codebook_subarray` — 子阵列宽波束（**当前最佳方案**）
- `compute_beam_power` — 信道→波束功率
- `process_beam_data` — 批量功率计算
- `construct_sliding_window` — 时序滑动窗口
- `normalize_power` — 逐样本 dB 域归一化
- `filter_valid_channels` — 过滤全零信道

## 4. 关键参数（默认配置）

```python
M = 32                  # 基站天线数（ULA）
Nr = 1                  # 用户天线数
N_narrow = 32           # 窄波束（DFT 码本）
N_wide = 8              # 宽波束（子阵列码本）
SNR_dB = 10             # 信噪比
P_tx_dBm = 30           # 发送功率
L = 3                   # LSTM 时序窗口长度
SCENARIO_NAME = 'city_0_newyork_28'  # DeepMIMO 28GHz 场景

# 训练超参
BATCH_SIZE = 64
EPOCHS_ESTIMATOR = 50
EPOCHS_PREDICTOR = 50
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.8
```

## 5. 当前最佳性能（基线，子阵列码本）

```
[估计模型 Beam_Estimator_1D]  1745 参数
  MSE: 0.0319 | Top-1: 39.24% | Top-3: 76.57% | SNR Ratio: -36.46 dB

[预测模型 Beam_Predictor_1D_LSTM]  51808 参数
  MSE: 0.0208 | Top-1: 66.40% | Top-3: 90.76% | SNR Ratio: -32.30 dB
```

## 6. 关键决策与重要踩坑

### 6.1 DeepMIMO v4 API 必须用 `tx_sets={1:[0]}`
- 否则 `dm.load()` 返回 `MacroDataset`（包含 3 个 TX set），`compute_channels()` 返回 list 而非 ndarray
- 信道 shape: `[n_ue=31719, n_rx=1, n_tx=32, n_sub=1]`，提取方式: `channels_raw[:, 0, :, 0]`

### 6.2 必须过滤全零信道
- DeepMIMO 中 49.7% 用户被建筑遮挡，信道全零
- 不过滤 → Top-1 仅 19%；过滤后 → Top-1 提升到 39%
- 实现于 `utils.py::filter_valid_channels()`

### 6.3 归一化必须用"逐样本 + dB 域"
- ❌ 全局 Min-Max：99% 样本被压到接近 0，模型学不到东西
- ✅ 逐样本归一化 + 先转 dB 域：保留波束间相对功率分布
- 实现于 `utils.py::normalize_power()`

### 6.4 输出层必须用 Sigmoid，不是 ReLU
- ReLU 输出无界，与 [0,1] 标签不匹配，训练 Loss 不下降
- Sigmoid 让输出范围 [0,1] 与归一化标签匹配
- 实现于 `models.py`

### 6.5 ⭐⭐⭐ 子阵列码本是当前最优，已删除其他探索代码
**经过完整实验验证**，新码本方案（密集指向 + Kaiser 加窗）**反而比旧方案差**：

| 方案 | 估计 Top-1 | 预测 Top-1 |
|---|---|---|
| **子阵列（保留）** | **39.24%** | **66.40%** |
| 密集指向 cov=80°, β=8 | 5.52% | 59.00% |
| 密集指向 cov=160°, β=6 | 4.96% | 53.18% |
| 密集指向 cov=140°, β=4 | 8.53% | 51.14% |

**失败原因（重要教训）**:
1. 加 Kaiser 窗**只能压制旁瓣，无法消除栅瓣**（栅瓣由阵列结构决定）
2. 全 32 天线 + 加窗后**主瓣变窄到 5°**（变成窄波束，失去"宽"的特性）
3. 子阵列虽然有栅瓣，但 **17° 的主瓣**真正能"宽"覆盖角度
4. **栅瓣的负面影响被高估了**——实验证明模型能从栅瓣中提取真信号

**已清理**: 删除了 `new_codebook_design.py`、`new_codebook_v2.py`、`new_codebook_v3.py`、`kaiser_codebook_test.py` 以及 `utils.py` 中的 `generate_wide_codebook_subarray_kaiser`、`generate_wide_codebook_dense_kaiser` 函数和 `results/new_codebook/` 目录。

## 7. 已完成的工作

- [x] DeepMIMO v4 API 集成（download/load/compute_channels）
- [x] DFT-32 窄波束 + 子阵列-8 宽波束码本生成
- [x] 信道→波束功率→归一化→.npz 数据流水线
- [x] Beam_Estimator_1D（1745 参数）训练流程
- [x] Beam_Predictor_1D_LSTM（51808 参数）训练流程
- [x] 评估指标：MSE / Top-1 / Top-3 / Effective SNR Ratio + CDF 图
- [x] 数据集统计分析（5 张图，`data_analysis.py`）
- [x] 码本方向图分析（6 张图，`codebook_analysis.py`）
- [x] 新码本方案探索（虽失败但获经验）— 已清理回滚
- [x] 重构代码：data_generation.py / channel_generator.py / utils.py 三模块解耦
- [x] 过滤全零信道（数据质量提升）
- [x] README.md 文档
- [x] MEMORY.md 工作记忆

## 8. 待办事项（按性价比排序）

### 优先级 1：改损失函数（预期 +3~5%）
- 当前 MSE Loss 不直接优化 Top-1
- 试 KL 散度（把功率向量当概率分布）或 Cross-Entropy
- 修改位置：`train.py::criterion = nn.MSELoss()` → 改成混合 loss

### 优先级 2：数据采样平衡（预期 +3~5%）
- 当前最优波束分布不均（波束 26 vs 14，差 23.6 倍）
- 加权采样：让每个最优波束类别等概率出现
- 修改位置：`train.py::DataLoader` 加 `WeightedRandomSampler`

### 优先级 3：动态场景（影响泛化）
- DeepMIMO 静态场景下 LSTM 实际是"多次测量平均"，不是真预测
- 改用动态场景，或人工注入用户运动
- 警告日志已确认：`No doppler because all velocities are zero`

### 优先级 4：加大模型容量（+5%，但失去低开销优势）
- Layer 2 通道 32 → 64
- 修改位置：`models.py::Beam_Estimator_1D::feature_extract`

### 优先级 5：git 仓库管理
- 用户已 `git init`，需要：
  - 添加 `.gitignore` 忽略 `data/`, `results/`, `deepmimo_scenarios/`, `__pycache__/`, `*.pth`, `*.npz`, `*.zip`, `DeepMIMO/`
  - 首次 commit 当前代码

## 9. 数据集关键统计

```
DeepMIMO city_0_newyork_28:
- 总用户: 31,719
- 有效用户（过滤零信道后）: 15,940 (50.3%)
- 静态训练样本: 15,940
- 时序训练样本: 111,580（滑动窗口 L=3）

用户角度分布:
- ±40° 内: 57.5%
- ±60° 内: 83.6%
- ±90° 内: 100%
- 平均绝对角度: 37.1°

功率统计:
- 最大波束功率范围: -184 ~ -76 dBW（跨度 108 dB）
- 边缘用户（功率 < 中位数 -20dB）: 8.8%
- 功率集中度（max/mean）中位数: 15.46x（11.9 dB）
- 最优波束分布不均度: 23.6x（波束 26 最常被选 1487 次, 波束 14 最少 63 次）
```

## 10. 重要文件路径

| 文件 | 内容 |
|---|---|
| `data/beam_dataset/static_beam_data.npz` | 静态数据：X_wide, Y_narrow, X_wide_raw, Y_narrow_raw, codebooks |
| `data/beam_dataset/temporal_beam_data.npz` | 时序数据：X_seq, Y_seq, raw versions |
| `results/best_estimator.pth` | 估计模型最优权重 |
| `results/best_predictor.pth` | 预测模型最优权重 |
| `results/snr_cdf_comparison.png` | SNR CDF 对比图 |
| `deepmimo_scenarios/city_0_newyork_28/` | DeepMIMO 原始 .mat 数据 |

## 11. 工作流命令

```bash
# 激活环境并直接用 python 可执行文件运行（避免 conda run 编码问题）
D:/mydownload-softwares/anacond32013/envs/mmwave_py12/python.exe data_generation.py
D:/mydownload-softwares/anacond32013/envs/mmwave_py12/python.exe train.py
D:/mydownload-softwares/anacond32013/envs/mmwave_py12/python.exe data_analysis.py
D:/mydownload-softwares/anacond32013/envs/mmwave_py12/python.exe codebook_analysis.py
```

## 12. 用户偏好（沟通风格）

- 用户是物理层信号处理 + 深度学习方向的研究者，对通信原理熟悉
- 喜欢深度技术讨论，提问尖锐（多次抓住关键点：栅瓣 vs 重叠、归一化副作用、码本几何形状）
- 不喜欢额外创建工具脚本（除非要求），偏好命令行直接查看
- 要求中文回复，重视分析图表（统计 + 可视化）
- 关键反馈：理论分析必须配合实验验证；不要在结论上过度乐观
- 注重代码整洁：失败的探索代码及时清理，保持代码库简洁
