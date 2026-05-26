# 基于深度学习的低开销毫米波波束选择 (Beam Selection)

参考论文：*A Deep Learning-Based Low Overhead Beam Selection in mmWave Communications*

利用低分辨率宽波束测量数据，通过 1D 超分辨率重建估计高分辨率窄波束质量，并通过 1D-CNN + LSTM 预测下一时刻的波束质量。

## 项目结构

```
DLbeamforming/
├── data_generation.py      # 主流程：生成训练数据集
├── channel_generator.py    # DeepMIMO 信道生成 + 合成信道备选
├── utils.py                # 工具函数（码本、功率计算、归一化）
├── models.py               # PyTorch 模型定义
├── train.py                # 训练与评估流程
├── DeepMIMO/               # DeepMIMO 源码（已 pip install -e .）
├── deepmimo_scenarios/     # 下载的场景数据（自动生成）
├── data/beam_dataset/      # 生成的训练数据（自动生成）
└── results/                # 训练结果、模型权重、图表（自动生成）
```

## 环境要求

- Python >= 3.11
- Conda 环境：`mmwave_py12`
- 依赖包：`deepmimo`, `numpy`, `torch`, `matplotlib`

```bash
conda activate mmwave_py12
pip install numpy torch matplotlib
cd DeepMIMO && pip install -e .
```

## 使用步骤

### 第 1 步：生成数据

```bash
conda activate mmwave_py12
python data_generation.py
```

首次运行会自动下载 DeepMIMO 场景 `city_0_newyork_28`（约几百 MB）。完成后在 `data/beam_dataset/` 下生成：
- `static_beam_data.npz` — 静态波束功率数据（估计模型用）
- `temporal_beam_data.npz` — 时序波束功率数据（预测模型用）

### 第 2 步：训练模型

```bash
python train.py
```

自动训练两个模型（各 50 epoch），完成后在 `results/` 下生成：
- `best_estimator.pth` — 估计模型权重
- `best_predictor.pth` — 预测模型权重
- `estimator_training_curve.png` — 估计模型训练曲线
- `predictor_training_curve.png` — 预测模型训练曲线
- `snr_cdf_comparison.png` — Effective SNR CDF 对比图
- `beam_pattern_sample_*.png` — 波束功率对比图

### 第 3 步：查看结果

训练结束后终端会打印评估指标：
- **MSE**：归一化域预测误差
- **Top-1 Accuracy**：预测最优波束命中率
- **Top-3 Accuracy**：前 3 波束包含最优波束的概率
- **Effective SNR Ratio**：预测波束 vs 全局最优波束的 SNR 损失 (dB)

## 模型说明

| 模型 | 输入 | 输出 | 用途 |
|------|------|------|------|
| Beam_Estimator_1D | 宽波束功率 `[B, 1, 8]` | 窄波束功率 `[B, 32]` | 从当前宽波束测量估计窄波束质量 |
| Beam_Predictor_1D_LSTM | 历史 L=3 帧宽波束 `[B, 3, 1, 8]` | 窄波束功率 `[B, 32]` | 利用时序信息预测当前窄波束质量 |

## 参数配置

在 `data_generation.py` 顶部修改：

```python
M = 32                  # 基站天线数
N_narrow = 32           # 窄波束数
N_wide = 8              # 宽波束数
SNR_dB = 10             # 信噪比
L = 3                   # LSTM 时序窗口长度
SCENARIO_NAME = 'city_0_newyork_28'  # DeepMIMO 场景
```

在 `train.py` 顶部修改：

```python
BATCH_SIZE = 64
EPOCHS_ESTIMATOR = 50
EPOCHS_PREDICTOR = 50
LEARNING_RATE = 1e-3
```

## 切换场景

将 `data_generation.py` 中的 `SCENARIO_NAME` 改为其他 DeepMIMO 场景名即可，首次运行会自动下载。可用场景参考 [DeepMIMO 官网](https://deepmimo.net/)。
