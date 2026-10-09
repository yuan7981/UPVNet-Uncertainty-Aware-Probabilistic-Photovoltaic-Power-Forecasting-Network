import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

class ReflectionalSwitchFunction(nn.Module):
    def __init__(self, grid_min: float = -2., grid_max: float = 2., num_grids: int = 8, exponent: int = 2, denominator: float = 0.33):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator
        self.inv_denominator = 1 / self.denominator

    def forward(self, x):
        diff = (x[..., None] - self.grid)
        diff_mul = diff.mul(self.inv_denominator)
        diff_tanh = torch.tanh(diff_mul)
        diff_pow = -diff_tanh.mul(diff_tanh)
        diff_pow += 1
        return diff_pow

class RGBFLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, grid_min: float = -2., grid_max: float = 2., num_grids: int = 8, exponent: int = 2, denominator: float = 0.33, use_base_update: bool = True, base_activation=F.silu, spline_weight_init_scale: float = 0.1) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = ReflectionalSwitchFunction(grid_min, grid_max, num_grids, exponent, denominator)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)

    def forward(self, x):
        # x shape: (Batch, Input_Dim)
        spline_basis = self.rbf(self.layernorm(x)).view(x.shape[0], -1)
        ret = self.spline_linear(spline_basis)
        return ret


class PositionEmbedding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.pe = nn.Embedding(max_len, dim)
    def forward(self, x):
        seq_len = x.shape[1]
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(x.shape[0], 1)
        return self.pe(pos_ids)

class CovPositive(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()
    def forward(self, x):
        return self.elu(x) + 1.0 + 1e-6

def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):
    mean_term = torch.sum(mean1**2, dim=-1, keepdim=True) + \
                torch.sum(mean2**2, dim=-1, keepdim=True).transpose(-1, -2) - \
                2.0 * torch.matmul(mean1, mean2.transpose(-1, -2))
    cov1_sqrt = torch.sqrt(torch.clamp(cov1, min=1e-12))
    cov2_sqrt = torch.sqrt(torch.clamp(cov2, min=1e-12))
    cov_term = torch.sum(cov1, dim=-1, keepdim=True) + \
               torch.sum(cov2, dim=-1, keepdim=True).transpose(-1, -2) - \
               2.0 * torch.matmul(cov1_sqrt, cov2_sqrt.transpose(-1, -2))
    return mean_term + cov_term

class ProbAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
    def forward(self, q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, mask=None):
        scores = -wasserstein_distance_matmul(q_mean, q_cov, k_mean, k_cov) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e32)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out_mean = torch.matmul(weights, v_mean)
        out_cov = torch.matmul(weights**2, v_cov)
        return out_mean, out_cov, weights

class NWPEngine(nn.Module):
    def __init__(self, nwp_dim, hidden_dim):
        super().__init__()
        self.mean_net = nn.Sequential(nn.Linear(nwp_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cov_net = nn.Sequential(nn.Linear(nwp_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cov_pos = CovPositive()
    def forward(self, nwp_seq):
        mean = self.mean_net(nwp_seq)
        cov = self.cov_pos(self.cov_net(nwp_seq))
        return mean, cov

class CNNRsBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(out_channels)
        self.residual_proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        residual = self.residual_proj(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        out = out.permute(0, 2, 1)
        return out





class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embed dim must be divisible by num_heads"
        
        # Q 来自一个源，K, V 来自另一个源
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query: (B, L, D) - 来自源 A
        # key, value: (B, L, D) - 来自源 B
        
        B, L, _ = query.shape
        
        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention Score: Q * K^T
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e32)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Output: Weights * V
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        
        return self.out_proj(out), attn_weights

# -------------------------
# 3. UPVNet 主模型 
# -------------------------
class UPVNet(nn.Module):
    def __init__(self, nwp_dim, seq_len=96, cnnrs_channels=22, rgbf_hidden=64, attn_dim=22, dropout=0.1):
        super().__init__()
        
        # === 1. NWP 分支 ===
        self.nwp_encoder = NWPEngine(nwp_dim, hidden_dim=attn_dim)
        self.nwp_self_attn = ProbAttention(d_k=attn_dim, dropout=dropout)
        self.pos_emb = PositionEmbedding(dim=attn_dim, max_len=seq_len)

        # === 2. Power 分支 ===
        self.cnnrs = CNNRsBlock(in_channels=1, out_channels=cnnrs_channels)
        # 维度对齐：道 -> Attention维度
        self.power_proj = nn.Linear(cnnrs_channels, attn_dim)

        # === 3. 双向交叉注意力 (Bidirectional Cross Attention) ===

        self.cross_attn_p2n = CrossAttention(embed_dim=attn_dim, num_heads=2, dropout=dropout)
        

        self.cross_attn_n2p = CrossAttention(embed_dim=attn_dim, num_heads=2, dropout=dropout)
        

        self.fusion_norm = nn.LayerNorm(attn_dim * 2)
        self.fusion_linear = nn.Linear(attn_dim * 2, attn_dim)
        self.fusion_dropout = nn.Dropout(dropout)

        # === 4. 解码器 (RGBF) ===
        self.rgbf_layer = RGBFLayer(
            input_dim=attn_dim, 
            output_dim=rgbf_hidden,
            num_grids=8,
            spline_weight_init_scale=0.1
        )

        # === 5. 输出头 ===
        self.power_head = nn.Linear(rgbf_hidden, 1)
        self.var_head = nn.Sequential(
            nn.Linear(rgbf_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        self._init_var_bias()

    def _init_var_bias(self):
        last_linear = None
        for layer in reversed(self.var_head):
            if isinstance(layer, nn.Linear):
                last_linear = layer
                break
        if last_linear and hasattr(last_linear, 'bias') and last_linear.bias is not None:
            nn.init.constant_(last_linear.bias, 0.54)

    def forward(self, power_seq, nwp_seq, mask=None):
        # ---------------------------
        # A. 特征提取
        # ---------------------------
        
        # 1. NWP 特征
        nwp_mean, nwp_cov = self.nwp_encoder(nwp_seq)
        nwp_mean += self.pos_emb(nwp_mean)
        nwp_cov += self.pos_emb(nwp_cov)
        # ProbAttention 自注意力增强
        nwp_feat, _, _ = self.nwp_self_attn(
            q_mean=nwp_mean, q_cov=nwp_cov,
            k_mean=nwp_mean, k_cov=nwp_cov,
            v_mean=nwp_mean, v_cov=nwp_cov,
            mask=mask
        ) # (B, L, D)

        # 2. Power 特征
        if power_seq.dim() == 3 and power_seq.shape[1] == 1:
            cnnrs_in = power_seq
        else:
            cnnrs_in = power_seq.permute(0, 2, 1)
            
        cnnrs_out = self.cnnrs(cnnrs_in)       # (B, C, L)
        power_feat_raw = cnnrs_out.permute(0, 2, 1) # (B, L, C)
        power_feat = self.power_proj(power_feat_raw) # (B, L, D) - 维度对齐

        # ---------------------------
        # B. 双向交叉注意力融合
        # ---------------------------
        
        # 路数 1: Power 关注 NWP (Context = NWP)
        # Q=Power, K=NWP, V=NWP
        feat_p2n, _ = self.cross_attn_p2n(query=power_feat, key=nwp_feat, value=nwp_feat, mask=mask)
        # 残差连接 (可选，推荐加上)
        feat_p2n = feat_p2n + power_feat 

        # 路数 2: NWP 关注 Power (Context = Power)
        # Q=NWP, K=Power, V=Power
        feat_n2p, _ = self.cross_attn_n2p(query=nwp_feat, key=power_feat, value=power_feat, mask=mask)
        # 残差连接
        feat_n2p = feat_n2p + nwp_feat

        # 融合: 拼接 -> Norm -> Linear
        combined = torch.cat([feat_p2n, feat_n2p], dim=-1) # (B, L, 2*D)
        combined = self.fusion_norm(combined)
        fused_feats = self.fusion_linear(combined)         # (B, L, D)
        fused_feats = self.fusion_dropout(fused_feats)

        # ---------------------------
        # C. 解码 
        # ---------------------------
        
        # 取最后一个时间步进行预测
        last_step_feat = fused_feats[:, -1, :] # (B, D)
        
        rgbf_out = self.rgbf_layer(last_step_feat) 

        # ---------------------------
        # D. 输出
        # ---------------------------
        pred_power = self.power_head(rgbf_out).squeeze(-1)
        pred_var = self.var_head(rgbf_out).squeeze(-1)

        return pred_power, pred_var