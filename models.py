# ============================================================
# models.py
# 深度学习模型模块：波束质量估计网络 + 波束质量预测网络
# 参考论文：A Deep Learning-Based Low Overhead Beam Selection
#           in mmWave Communications
# ============================================================

import torch
import torch.nn as nn


class Beam_Estimator_1D(nn.Module):
    """
    波束质量估计网络 (1D Super-Resolution)

    功能：将长度为 8 的低分辨率宽波束功率向量，
         通过转置卷积上采样 + 卷积特征提取，
         映射为长度为 32 的高分辨率窄波束功率向量。

    网络结构:
        Input:  [B, 1, 8]
        Layer1: ConvTranspose1d → [B, 16, 32]  (上采样 4 倍)
        Layer2: Conv1d + ReLU  → [B, 32, 32]  (特征提取)
        Layer3: Conv1d + ReLU  → [B, 1, 32]   (输出层)
        Output: [B, 32]  (squeeze 后)
    """

    def __init__(self, in_channels=1, n_wide=8, n_narrow=32):
        """
        参数:
            in_channels: 输入通道数（默认 1，单通道功率信号）
            n_wide: 宽波束数量（输入长度）
            n_narrow: 窄波束数量（输出长度）
        """
        super(Beam_Estimator_1D, self).__init__()

        # Layer 1: 转置卷积上采样层
        # 将长度 8 上采样到长度 32 (stride=4, kernel=4)
        # 输出长度 = (input_len - 1) * stride + kernel_size = (8-1)*4 + 4 = 32
        self.upsample = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=16,
            kernel_size=4,
            stride=4,
            padding=0
        )

        # Layer 2: 1D 卷积特征提取层
        # padding=1 保持长度不变: 32 → 32
        self.feature_extract = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Layer 3: 1D 卷积输出层
        # 将 32 通道压缩为 1 通道，Sigmoid 将输出映射到 [0,1] 匹配归一化标签
        self.output_layer = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=1,
                      kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，shape = [B, 1, 8]
               B = batch_size, 1 = 通道, 8 = 宽波束数

        返回:
            out: 预测的窄波束功率，shape = [B, 32]
        """
        # Layer 1: 上采样 [B, 1, 8] → [B, 16, 32]
        x = self.upsample(x)

        # Layer 2: 特征提取 [B, 16, 32] → [B, 32, 32]
        x = self.feature_extract(x)

        # Layer 3: 输出 [B, 32, 32] → [B, 1, 32]
        x = self.output_layer(x)

        # Squeeze 通道维度: [B, 1, 32] → [B, 32]
        out = x.squeeze(1)

        return out


class Beam_Estimator_FC(nn.Module):
    """
    波束质量估计网络 - 全连接版本

    改进动机:
        原 1D-CNN 架构的转置卷积(kernel=4, stride=4)导致每个输出位置
        只能看到 1 个输入宽波束,无法提取跨宽波束的功率比信息。
        全连接层让网络能同时看到所有 8 个宽波束的全局关系。

    网络结构:
        Input:  [B, 8]
        FC1:    Linear(8 → 64) + ReLU
        FC2:    Linear(64 → 128) + ReLU + Dropout(0.2)
        FC3:    Linear(128 → 32) + Sigmoid
        Output: [B, 32]

    参数量: 8*64 + 64*128 + 128*32 = 512 + 8192 + 4096 = 12800
    """

    def __init__(self, n_wide=8, n_narrow=32, hidden_dims=[64, 128]):
        """
        参数:
            n_wide: 宽波束数量(输入维度)
            n_narrow: 窄波束数量(输出维度)
            hidden_dims: 隐藏层维度列表
        """
        super(Beam_Estimator_FC, self).__init__()

        self.n_wide = n_wide
        self.n_narrow = n_narrow

        # 全连接网络
        self.fc_net = nn.Sequential(
            nn.Linear(n_wide, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], n_narrow),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，shape = [B, 8] 或 [B, 1, 8]

        返回:
            out: 预测的窄波束功率，shape = [B, 32]
        """
        # 如果输入是 [B, 1, 8],squeeze 掉通道维度
        if x.dim() == 3:
            x = x.squeeze(1)

        # [B, 8] → [B, 32]
        out = self.fc_net(x)

        return out


class TransformerEncoderBlock(nn.Module):
    """
    单层 Transformer 编码器(Pre-LN 结构)

    结构:
        x → LN → MSA → +residual → LN → MLP(SiLU) → +residual
    """

    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # MSA 子模块
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        # MLP 子模块
        x = x + self.mlp(self.norm2(x))
        return x


class Beam_Estimator_BT(nn.Module):
    """
    Beam Transformer 波束估计网络(方案 B:BT 架构 + KL 软标签)

    参考论文: Beam Transformer (BT) 算法
    改进点(相对原 BT):
        - 输出层:Sigmoid 替代 Softmax,保留功率分布的软排序信息
        - 损失:KL 散度(在 train 端实现)替代 CE,保留超分辨率语义

    架构:
        Input  [B, N_wide=8]
          ↓ 复制 M_cat 次          [B, M_cat, N_wide]
          ↓ Linear(N_wide → dim)   [B, M_cat, dim]
          ↓ 拼接 CLS token         [B, M_cat+1, dim]
          ↓ + 位置编码              [B, M_cat+1, dim]
          ↓ Transformer × L_BT     [B, M_cat+1, dim]
          ↓ 取 CLS token           [B, dim]
          ↓ MLP × L_FC (SiLU)
          ↓ Sigmoid → [B, N_narrow=32]
    """

    def __init__(self,
                 n_wide=8,
                 n_narrow=32,
                 dim=64,
                 m_cat=4,
                 num_heads=2,
                 mlp_dim=None,        # 默认 2*n_narrow
                 num_encoder_layers=1,
                 num_classifier_layers=4,
                 dropout=0.1):
        super().__init__()

        if mlp_dim is None:
            mlp_dim = 2 * n_narrow  # 论文默认: mlp_dim = 2N

        self.n_wide = n_wide
        self.n_narrow = n_narrow
        self.dim = dim
        self.m_cat = m_cat

        # 嵌入层: N_wide → dim
        self.embed = nn.Linear(n_wide, dim)

        # 可学习的 CLS token 和位置编码
        # 序列长度 = m_cat + 1 (CLS + m_cat 个复制 token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, m_cat + 1, dim))

        # Transformer 编码器堆叠
        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(dim, num_heads, mlp_dim, dropout)
            for _ in range(num_encoder_layers)
        ])

        # 编码器输出后的 LayerNorm(常见做法,稳定 CLS 表示)
        self.norm = nn.LayerNorm(dim)

        # 分类头(L_FC 层 MLP,前 L_FC-1 层用 SiLU)
        classifier_layers = []
        for _ in range(num_classifier_layers - 1):
            classifier_layers.append(nn.Linear(dim, dim))
            classifier_layers.append(nn.SiLU())
            classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.append(nn.Linear(dim, n_narrow))
        classifier_layers.append(nn.Sigmoid())  # 方案 B:Sigmoid 输出
        self.classifier = nn.Sequential(*classifier_layers)

        # 参数初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        参数:
            x: [B, n_wide] 或 [B, 1, n_wide]

        返回:
            out: [B, n_narrow],范围 [0, 1]
        """
        if x.dim() == 3:
            x = x.squeeze(1)

        B = x.size(0)

        # 复制扩张: [B, n_wide] → [B, m_cat, n_wide]
        x = x.unsqueeze(1).expand(-1, self.m_cat, -1)

        # 嵌入: [B, m_cat, n_wide] → [B, m_cat, dim]
        x = self.embed(x)

        # 拼接 CLS token: [B, m_cat+1, dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 加位置编码
        x = x + self.pos_embed

        # Transformer 编码
        x = self.encoder(x)
        x = self.norm(x)

        # 取 CLS token 作为全局表示
        cls_out = x[:, 0]  # [B, dim]

        # 分类头 → [B, n_narrow]
        out = self.classifier(cls_out)

        return out


class Beam_Predictor_1D_LSTM(nn.Module):
    """
    波束质量预测网络 (1D-CNN + LSTM)

    功能：利用过去 L=3 个时刻的宽波束测量数据，
         通过 TimeDistributed 1D-CNN 提取空间特征，
         再通过 LSTM 建模时序依赖关系，
         预测当前时刻的窄波束功率向量。

    网络结构:
        Input:  [B, L, 1, 8]
        CNN:    TimeDistributed Conv1d → [B, L, 128]
        LSTM:   LSTM(128→64) → [B, L, 64]
        Output: Linear(64→32) + ReLU → [B, 32]
    """

    def __init__(self, n_wide=8, n_narrow=32, seq_len=3,
                 cnn_out_channels=16, lstm_hidden=64):
        """
        参数:
            n_wide: 宽波束数量（每帧输入长度）
            n_narrow: 窄波束数量（输出长度）
            seq_len: 时序长度 L
            cnn_out_channels: CNN 输出通道数
            lstm_hidden: LSTM 隐藏层维度
        """
        super(Beam_Predictor_1D_LSTM, self).__init__()

        self.n_wide = n_wide
        self.n_narrow = n_narrow
        self.seq_len = seq_len
        self.cnn_out_channels = cnn_out_channels
        self.lstm_hidden = lstm_hidden

        # CNN 特征提取后的展平长度
        # Conv1d(1, 16, k=3, p=1) 保持长度 8 → 输出 16*8 = 128
        self.cnn_flat_size = cnn_out_channels * n_wide  # 16 * 8 = 128

        # 空间特征提取器 (TimeDistributed 1D-CNN)
        # 对每个时间步独立提取空间特征
        self.spatial_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()  # (cnn_out_channels * n_wide) = 128
        )

        # 时序建模器 (LSTM)
        self.lstm = nn.LSTM(
            input_size=self.cnn_flat_size,   # 128
            hidden_size=lstm_hidden,          # 64
            num_layers=1,
            batch_first=True
        )

        # 解码输出层
        # 取 LSTM 最后时间步的隐藏状态，映射到窄波束维度
        self.decoder = nn.Sequential(
            nn.Linear(in_features=lstm_hidden, out_features=n_narrow),
            nn.Sigmoid()  # 输出映射到 [0,1] 匹配归一化标签
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，shape = [B, L, 1, 8]
               B = batch_size
               L = 时序长度 (3)
               1 = 通道数
               8 = 宽波束数

        返回:
            out: 预测的窄波束功率，shape = [B, 32]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # ---- TimeDistributed CNN ----
        # 将 Batch 和时序维度合并: [B, L, 1, 8] → [B*L, 1, 8]
        x_reshaped = x.view(batch_size * seq_len, 1, self.n_wide)

        # 通过 CNN 提取空间特征: [B*L, 1, 8] → [B*L, 128]
        cnn_out = self.spatial_cnn(x_reshaped)

        # 还原时序维度: [B*L, 128] → [B, L, 128]
        cnn_out = cnn_out.view(batch_size, seq_len, self.cnn_flat_size)

        # ---- LSTM 时序建模 ----
        # 输入: [B, L, 128]，输出: [B, L, 64]
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)

        # 取最后一个时间步的输出: [B, 64]
        last_hidden = lstm_out[:, -1, :]

        # ---- 解码输出 ----
        # [B, 64] → [B, 32]
        out = self.decoder(last_hidden)

        return out


def count_parameters(model):
    """统计模型可训练参数总数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 模型测试：验证输入输出维度
    print("=" * 50)
    print("模型结构验证")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # ---- 测试 Beam_Estimator_1D ----
    print("\n--- Beam_Estimator_1D ---")
    estimator = Beam_Estimator_1D().to(device)
    print(estimator)
    print(f"参数量: {count_parameters(estimator):,}")

    # 模拟输入: batch=4, channel=1, length=8
    x_est = torch.randn(4, 1, 8).to(device)
    y_est = estimator(x_est)
    print(f"输入 shape: {x_est.shape}")
    print(f"输出 shape: {y_est.shape}")
    assert y_est.shape == (4, 32), f"输出维度错误: {y_est.shape}"
    print("维度验证通过!")

    # ---- 测试 Beam_Predictor_1D_LSTM ----
    print("\n--- Beam_Predictor_1D_LSTM ---")
    predictor = Beam_Predictor_1D_LSTM().to(device)
    print(predictor)
    print(f"参数量: {count_parameters(predictor):,}")

    # 模拟输入: batch=4, seq_len=3, channel=1, length=8
    x_pred = torch.randn(4, 3, 1, 8).to(device)
    y_pred = predictor(x_pred)
    print(f"输入 shape: {x_pred.shape}")
    print(f"输出 shape: {y_pred.shape}")
    assert y_pred.shape == (4, 32), f"输出维度错误: {y_pred.shape}"
    print("维度验证通过!")

    print("\n所有模型测试通过!")
