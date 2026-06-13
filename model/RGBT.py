"""
YOLOv11-RGBT Multimodal Fusion Core Modules
============================================
Extracted from YOLOv11-RGBT project for open-source use.
Contains the core RGB-T (RGB + Thermal) multimodal fusion modules.

Author: Extracted from YOLOv11-RGBT
License: AGPL-3.0
GitHub: https://github.com/ultralytics/ultralytics

Key Fusion Modules:
1. DynamicAlignFusion - Dynamic alignment fusion with learnable weights
2. SDFM - Superficial Detail Fusion Module
3. PSFM - Profound Semantic Fusion Module  
4. GEFM - Guided Enhancement Fusion Module
5. ContextGuideFusionModule - Context-guided fusion with SE attention
6. CrossAttentionShared - Cross-attention with weight sharing
7. CrossMLCAv2 - Cross Multi-Level Channel Attention
8. CrossTransformerFusion - Transformer-based cross-modal fusion
9. Fusion - Multi-feature fusion (BiFPN, SDI, adaptive, etc.)
10. SDI - Semantics and Detail Infusion
11. CBFuse - Cross-Backbone Feature Fuse
12. PyramidContextExtraction - Multi-scale context extraction
13. DynamicAlignFusion - Dynamic alignment fusion
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# ============================================================================
# Basic Convolution Modules
# ============================================================================

def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with batch normalization and activation."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DSConv(nn.Module):
    """Basic Depthwise Separable Convolution."""
    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, bias=False):
        super().__init__()
        if p is None:
            p = (d * (k - 1)) // 2
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=k, stride=s,
                            padding=p, dilation=d, groups=c_in, bias=bias)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.bn(x))


class GSConv(nn.Module):
    """Grouped Shuffle Convolution."""
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, autopad(k), g=g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, autopad(5), g=c_, act=act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        b, c, h, w = x2.shape
        x2 = x2.reshape(b, 2, c // 2, h, w).transpose(1, 2).reshape(b, c, h, w)
        return torch.cat((x1, x2), 1)


# ============================================================================
# Attention Modules
# ============================================================================

class SEAttention(nn.Module):
    """Squeeze-and-Excitation Attention."""
    def __init__(self, c1, r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // r, c1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class h_sigmoid(nn.Module):
    """Hard Sigmoid activation."""
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


# ============================================================================
# Core Multimodal Fusion Modules
# ============================================================================

class DynamicAlignFusion(nn.Module):
    """
    Dynamic Alignment Fusion Module
    
    Aligns features from two modalities (RGB and Thermal) using:
    1. Channel alignment via 1x1 convolutions
    2. Sigmoid-based dynamic weight generation
    3. Learnable parameters for adaptive fusion
    
    Args:
        c1: Input channels (list or single value)
        inc: Input channel list [rgb_ch, thermal_ch]
        ouc: Output channels
    
    Input: [x_rgb, x_thermal]
    Output: Fused feature map
    """
    def __init__(self, c1, inc=None, ouc=None):
        super().__init__()
        
        if inc is None:
            inc = c1 if isinstance(c1, (list, tuple)) else [c1, c1]
        if isinstance(inc[0], (list, tuple)):
            inc = inc[0]
        inc = [int(x) if isinstance(x, (list, tuple)) else int(x) for x in inc]
        if ouc is None:
            ouc = inc[0]
        if isinstance(ouc, (list, tuple)):
            ouc = ouc[0]
        ouc = int(ouc)
        
        self.conv_align1 = Conv(inc[0], ouc, 1)
        self.conv_align2 = Conv(inc[1], ouc, 1)
        
        self.conv_concat = Conv(ouc * 2, ouc * 2, 3)
        self.sigmoid = nn.Sigmoid()
        
        self.x1_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        self.x2_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5, requires_grad=True)
        
        self.conv_final = Conv(ouc, ouc, 1)
        
    def forward(self, x):
        self._clamp_abs(self.x1_param.data, 1.0)
        self._clamp_abs(self.x2_param.data, 1.0)
        
        x1, x2 = x
        x1, x2 = self.conv_align1(x1), self.conv_align2(x2)
        x_concat = self.sigmoid(self.conv_concat(torch.cat([x1, x2], dim=1)))
        x1_weight, x2_weight = torch.chunk(x_concat, 2, dim=1)
        x1, x2 = x1 * x1_weight, x2 * x2_weight
        
        return self.conv_final(x1 * self.x1_param + x2 * self.x2_param)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign


class SDFM(nn.Module):
    """
    Superficial Detail Fusion Module (SDFM)
    
    Fuses low-level detail features from RGB and Thermal modalities:
    1. Feature recalibration using channel attention
    2. Local (spatial) attention
    3. Global (channel) attention
    4. Weighted fusion based on combined attention
    
    Args:
        c1: Input channels
        channels: Channel number (defaults to c1[0])
        r: Reduction ratio for attention
    
    Input: [x_rgb, x_thermal]
    Output: Fused detail features
    """
    def __init__(self, c1, channels=None, r=4):
        super().__init__()
        if channels is None:
            channels = c1[0] if isinstance(c1, (list, tuple)) else c1
        if isinstance(channels, (list, tuple)):
            channels = channels[0]
        channels = int(channels)
        inter_channels = int(channels // r)

        self.Recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(2 * channels, 2 * inter_channels),
            Conv(2 * inter_channels, 2 * channels, act=nn.Sigmoid()),
        )

        self.channel_agg = Conv(2 * channels, channels)

        self.local_att = nn.Sequential(
            Conv(channels, inter_channels, 1),
            Conv(inter_channels, channels, 1, act=False),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(channels, inter_channels, 1),
            Conv(inter_channels, channels, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x1, x2 = data
        _, c, _, _ = x1.shape
        input_feat = torch.cat([x1, x2], dim=1)
        recal_w = self.Recalibrate(input_feat)
        recal_input = recal_w * input_feat
        recal_input = recal_input + input_feat
        x1, x2 = torch.split(recal_input, c, dim=1)
        agg_input = self.channel_agg(recal_input)
        local_w = self.local_att(agg_input)
        global_w = self.global_att(agg_input)
        w = self.sigmoid(local_w * global_w)
        xo = w * x1 + (1 - w) * x2
        return xo


class DenseLayer(nn.Module):
    """Dense connection layer with DSConv blocks."""
    def __init__(self, in_C, out_C, down_factor=4, k=2):
        super().__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(DSConv(mid_C * i, mid_C, 3))

        self.fuse = DSConv(in_C + mid_C, out_C, 3)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for i in self.denseblock:
            feats = i(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)
        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class GEFM(nn.Module):
    """
    Guided Enhancement Fusion Module (GEFM)
    
    Cross-modal guided enhancement using attention mechanism:
    1. RGB guides Thermal enhancement
    2. Thermal guides RGB enhancement
    3. Concatenation and reduction
    
    Args:
        in_C: Input channels
        out_C: Output channels
    
    Input: (x_rgb, x_thermal)
    Output: Enhanced fused features
    """
    def __init__(self, in_C, out_C):
        super().__init__()
        self.RGB_K = DSConv(out_C, out_C, 3)
        self.RGB_V = DSConv(out_C, out_C, 3)
        self.Q = DSConv(in_C, out_C, 3)
        self.INF_K = DSConv(out_C, out_C, 3)
        self.INF_V = DSConv(out_C, out_C, 3)
        self.Second_reduce = DSConv(in_C, out_C, 3)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, y):
        Q = self.Q(torch.cat([x, y], dim=1))
        RGB_K = self.RGB_K(x)
        RGB_V = self.RGB_V(x)
        m_batchsize, C, height, width = RGB_V.size()
        RGB_V = RGB_V.view(m_batchsize, -1, width * height)
        RGB_K = RGB_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        RGB_Q = Q.view(m_batchsize, -1, width * height)
        RGB_mask = torch.bmm(RGB_K, RGB_Q)
        RGB_mask = self.softmax(RGB_mask)
        RGB_refine = torch.bmm(RGB_V, RGB_mask.permute(0, 2, 1))
        RGB_refine = RGB_refine.view(m_batchsize, -1, height, width)
        RGB_refine = self.gamma1 * RGB_refine + y
        
        INF_K = self.INF_K(y)
        INF_V = self.INF_V(y)
        INF_V = INF_V.view(m_batchsize, -1, width * height)
        INF_K = INF_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        INF_Q = Q.view(m_batchsize, -1, width * height)
        INF_mask = torch.bmm(INF_K, INF_Q)
        INF_mask = self.softmax(INF_mask)
        INF_refine = torch.bmm(INF_V, INF_mask.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, height, width)
        INF_refine = self.gamma2 * INF_refine + x
        
        out = self.Second_reduce(torch.cat([RGB_refine, INF_refine], dim=1))
        return out


class PSFM(nn.Module):
    """
    Profound Semantic Fusion Module (PSFM)
    
    Deep semantic fusion combining DenseLayer and GEFM:
    1. Dense feature extraction for each modality
    2. Cross-modal guided enhancement
    
    Args:
        c1: Input channels
        Channel: Channel number (defaults to c1[0])
    
    Input: [x_rgb, x_thermal]
    Output: Deep fused semantic features
    """
    def __init__(self, c1, Channel=None):
        super().__init__()
        if Channel is None:
            Channel = c1[0] if isinstance(c1, (list, tuple)) else c1
        if isinstance(Channel, (list, tuple)):
            Channel = Channel[0]
        Channel = int(Channel)
        self.RGBobj = DenseLayer(Channel, Channel)
        self.Infobj = DenseLayer(Channel, Channel)
        self.obj_fuse = GEFM(Channel * 2, Channel)
        
    def forward(self, data):
        rgb, depth = data
        rgb_sum = self.RGBobj(rgb)
        Inf_sum = self.Infobj(depth)
        out = self.obj_fuse(rgb_sum, Inf_sum)
        return out


class ContextGuideFusionModule(nn.Module):
    """
    Context-Guided Fusion Module
    
    Uses SE attention to guide cross-modal feature fusion:
    1. Channel alignment if needed
    2. Concatenate and apply SE attention
    3. Split attention weights and apply cross-guidance
    
    Args:
        c1: Input channels
        inc: Input channel list [rgb_ch, thermal_ch]
    
    Input: [x_rgb, x_thermal]
    Output: [x_rgb + guided_thermal, x_thermal + guided_rgb]
    """
    def __init__(self, c1, inc=None):
        super().__init__()
        
        if inc is None:
            inc = c1 if isinstance(c1, (list, tuple)) else [c1, c1]
        if isinstance(inc[0], (list, tuple)):
            inc = inc[0]
        inc = [int(x) if isinstance(x, (list, tuple)) else int(x) for x in inc]
        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)
        
        self.se = SEAttention(inc[1] * 2)
    
    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)
        
        x_concat = torch.cat([x0, x1], dim=1)
        x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)


class CrossAttentionShared(nn.Module):
    """
    Cross-Attention Module with Weight Sharing
    
    Shared attention mechanism for RGB-T feature interaction:
    1. Shared QKV generation
    2. Cross-attention between modalities
    3. Positional encoding
    4. Combined projection
    
    Args:
        dim: Input channel dimension
        num_heads: Number of attention heads
        attn_ratio: Attention key dimension ratio
    
    Input: [x_rgb, x_thermal]
    Output: [x_rgb_out, x_thermal_out, x_combined]
    """
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = self.head_dim * num_heads

        self.qkv = nn.Conv2d(dim, nh_kd * 2 + h, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.proj_all = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.pe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

    def forward(self, x):
        x1, x2 = x
        B_, C, H, W = x1.shape
        
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.num_heads * self.key_dim, 
                             self.num_heads * self.key_dim, 
                             self.num_heads * self.head_dim], dim=1)
        q = q.reshape(B_, self.num_heads, self.key_dim, H * W).transpose(-2, -1)
        k = k.reshape(B_, self.num_heads, self.key_dim, H * W)
        v = v.reshape(B_, self.num_heads, self.head_dim, H * W).transpose(-2, -1)
        
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1_out = (attn @ v).transpose(-2, -1).reshape(B_, C, H, W)
        x1_out = self.proj(x1_out)
        
        qkv2 = self.qkv(x2)
        q2, k2, v2 = qkv2.split([self.num_heads * self.key_dim,
                                  self.num_heads * self.key_dim,
                                  self.num_heads * self.head_dim], dim=1)
        q2 = q2.reshape(B_, self.num_heads, self.key_dim, H * W).transpose(-2, -1)
        k2 = k2.reshape(B_, self.num_heads, self.key_dim, H * W)
        v2 = v2.reshape(B_, self.num_heads, self.head_dim, H * W).transpose(-2, -1)
        
        attn2 = (q2 @ k2) * self.scale
        attn2 = attn2.softmax(dim=-1)
        x2_out = (attn2 @ v2).transpose(-2, -1).reshape(B_, C, H, W)
        x2_out = self.proj(x2_out)
        
        x_out_all = self.proj_all(torch.cat([x1_out, x2_out], dim=1))
        
        return [x1_out + self.pe(x1_out), x2_out + self.pe(x2_out), x_out_all]


class CrossMLCAv2(nn.Module):
    """
    Cross Multi-Level Channel Attention v2
    
    Multi-scale channel attention for RGB-T fusion:
    1. Local and global pooling
    2. 1D convolution for channel attention
    3. Cross-modal attention application
    4. Merge via convolution
    
    Args:
        in_size: Input channel size
        local_size: Local pooling size
        gamma, b: ECA attention parameters
        local_weight: Weight for local vs global attention
    
    Input: [x_rgb, x_thermal]
    Output: [x_rgb_enhanced, x_thermal_enhanced, merged]
    """
    def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.5):
        super().__init__()
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.local_weight = local_weight
        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)
        self.merge_conv = nn.Conv2d(in_channels=in_size * 2, out_channels=in_size, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = x
        local_arv1 = self.local_arv_pool(x1)
        global_arv1 = self.global_arv_pool(local_arv1)
        local_arv2 = self.local_arv_pool(x2)
        global_arv2 = self.global_arv_pool(local_arv2)

        b, c, m, n = x1.shape
        b_local, c_local, m_local, n_local = local_arv1.shape

        temp_local1 = local_arv1.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_local2 = local_arv2.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global1 = global_arv1.view(b, c, -1).transpose(-1, -2)
        temp_global2 = global_arv2.view(b, c, -1).transpose(-1, -2)

        y_local1 = self.conv_local(temp_local1)
        y_global1 = self.conv(temp_global1)
        y_local2 = self.conv_local(temp_local2)
        y_global2 = self.conv(temp_global2)

        y_local_transpose1 = y_local1.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(
            b, c, self.local_size, self.local_size)
        y_global_transpose1 = y_global1.view(b, -1).transpose(-1, -2).unsqueeze(-1)
        y_local_transpose2 = y_local2.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(
            b, c, self.local_size, self.local_size)
        y_global_transpose2 = y_global2.view(b, -1).transpose(-1, -2).unsqueeze(-1)

        att_local1 = y_local_transpose1.sigmoid()
        att_global1 = F.adaptive_avg_pool2d(y_global_transpose1.sigmoid(), [self.local_size, self.local_size])
        att_all1 = F.adaptive_avg_pool2d(att_global1 * (1 - self.local_weight) + (att_local1 * self.local_weight), [m, n])

        att_local2 = y_local_transpose2.sigmoid()
        att_global2 = F.adaptive_avg_pool2d(y_global_transpose2.sigmoid(), [self.local_size, self.local_size])
        att_all2 = F.adaptive_avg_pool2d(att_global2 * (1 - self.local_weight) + (att_local2 * self.local_weight), [m, n])

        x1 = x1 * att_all1 + x1
        x2 = x2 * att_all2 + x2

        merged = torch.cat([x1, x2], dim=1)
        output = self.merge_conv(merged)

        return [x1, x2, output]


# ============================================================================
# Transformer-based Fusion
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention for Transformer."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        q_s = self.W_Q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_O(context), attn_weights


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder for cross-modal fusion."""
    def __init__(self, input_dim, model_dim, num_heads, num_layers, hidden_dim, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src1, src2, mask=None):
        src1 = self.dropout(self.input_proj(src1))
        src2 = self.dropout(self.input_proj(src2))
        
        for layer in self.layers:
            src1 = layer(src1, mask)
            src2 = layer(src2, mask)
        
        return src1, src2


class CrossTransformerFusion(nn.Module):
    """
    Cross-Transformer Fusion Module
    
    Transformer-based cross-modal fusion:
    1. Flatten spatial dimensions
    2. Parallel Transformer encoding
    3. Reshape and concatenate
    
    Args:
        input_dim: Input channel dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
    
    Input: [x_rgb, x_thermal]
    Output: Concatenated [x_rgb_out, x_thermal_out]
    """
    def __init__(self, input_dim, num_heads=2, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = input_dim * 2
        self.model_dim = input_dim
        self.encoder = TransformerEncoder(input_dim, self.model_dim, num_heads, num_layers, self.hidden_dim, dropout)

    def forward(self, x):
        vis, inf = x[0], x[1]
        B, C, H, W = vis.shape
        
        vis = vis.permute(0, 2, 3, 1).reshape(B, -1, C)
        inf = inf.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        vis_out, inf_out = self.encoder(vis, inf)
        
        vis_out = vis_out.view(B, H, W, -1).permute(0, 3, 1, 2)
        inf_out = inf_out.view(B, H, W, -1).permute(0, 3, 1, 2)
        
        out = torch.cat((vis_out, inf_out), dim=1)
        return out


# ============================================================================
# Multi-Feature Fusion Modules
# ============================================================================

class SDI(nn.Module):
    """
    Semantics and Detail Infusion (SDI)
    
    Multi-scale feature fusion via element-wise multiplication:
    1. Align all features to the same spatial size
    2. Apply GSConv to each feature
    3. Element-wise multiplication
    
    Args:
        channels: List of input channel sizes
    
    Input: List of feature maps at different scales
    Output: Fused feature map
    """
    def __init__(self, channels):
        super().__init__()
        self.convs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])

    def forward(self, xs):
        ans = torch.ones_like(xs[0])
        target_size = xs[0].shape[2:]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                  mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)
        return ans


class CBFuse(nn.Module):
    """
    Cross-Backbone Feature Fuse (CBFuse)
    
    Selective feature fusion from different backbone layers:
    1. Select features based on index
    2. Interpolate to target size
    3. Sum all features
    
    Args:
        idx: List of indices for feature selection
    
    Input: List of feature maps
    Output: Fused feature map
    """
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class Fusion(nn.Module):
    """
    Unified Multi-Feature Fusion Module
    
    Supports multiple fusion strategies:
    - 'weight': Simple weighted sum
    - 'adaptive': Adaptive weight generation
    - 'concat': Channel concatenation
    - 'bifpn': BiFPN-style weighted fusion
    - 'SDI': Semantics and Detail Infusion
    
    Args:
        inc_list: List of input channel sizes
        fusion: Fusion strategy name
    
    Input: List of feature maps
    Output: Fused feature map
    """
    def __init__(self, inc_list, fusion='bifpn'):
        super().__init__()
        assert fusion in ['weight', 'adaptive', 'concat', 'bifpn', 'SDI']
        self.fusion = fusion

        if self.fusion == 'bifpn':
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        elif self.fusion == 'SDI':
            self.SDI = SDI(inc_list)
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])
            if self.fusion == 'adaptive':
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)

    def forward(self, x):
        if self.fusion in ['weight', 'adaptive']:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'concat':
            return torch.cat(x, dim=1)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0) + self.epsilon)
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'SDI':
            return self.SDI(x)


# ============================================================================
# Context Extraction Modules
# ============================================================================

class PyramidPoolAgg(nn.Module):
    """Pyramid Pooling Aggregator for multi-scale features."""
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)


class RCA(nn.Module):
    """
    Rectangular Calibration Attention
    
    Combines spatial and channel attention:
    1. Depthwise convolution for spatial context
    2. Horizontal and vertical pooling
    3. Band convolution for attention generation
    
    Args:
        inp: Input channels
        kernel_size: Kernel size
        ratio: Reduction ratio
        band_kernel_size: Band kernel size for attention
        square_kernel_size: Square kernel size for spatial conv
    """
    def __init__(self, inp, kernel_size=1, ratio=2, band_kernel_size=11, dw_size=(1, 1), 
                 padding=(0, 0), stride=1, square_kernel_size=3, relu=True):
        super().__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc = inp // ratio
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()
        )

    def sge(self, x):
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w
        ge = self.excite(x_gather)
        return ge

    def forward(self, x):
        loc = self.dwconv_hw(x)
        att = self.sge(x)
        out = att * loc
        return out


class PyramidContextExtraction(nn.Module):
    """
    Pyramid Context Extraction Module
    
    Multi-scale context extraction using pyramid pooling and RCM:
    1. Pyramid pooling aggregation
    2. Rectangular calibration attention
    3. Split back to original scales
    
    Args:
        c1: Input channels
        dim: Dimension list for each scale
        n: Number of RCM blocks
    
    Input: List of multi-scale features
    Output: List of enhanced multi-scale features
    """
    def __init__(self, c1, dim=None, n=3):
        super().__init__()
        
        if dim is None:
            dim = c1 if isinstance(c1, (list, tuple)) else [c1]
        if isinstance(dim[0], (list, tuple)):
            dim = dim[0]
        dim = [int(x) if isinstance(x, (list, tuple)) else int(x) for x in dim]
        self.dim = dim
        self.ppa = PyramidPoolAgg()
        self.rcm = nn.Sequential(*[RCA(sum(dim), 3, 2, square_kernel_size=1) for _ in range(n)])
        
    def forward(self, x):
        x = self.ppa(x)
        x = self.rcm(x)
        return torch.split(x, self.dim, dim=1)


# ============================================================================
# Utility Modules
# ============================================================================

class Silence(nn.Module):
    """Pass-through module for YOLO graph construction."""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class Concat(nn.Module):
    """Concatenate tensors along specified dimension."""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


# ============================================================================
# Usage Examples
# ============================================================================

"""
Example 1: Early Fusion (Input level)
--------------------------------------
Simply concatenate RGB and Thermal channels:
    input = torch.cat([rgb, thermal], dim=1)  # [B, 6, H, W]
    model = YOLO(ch=6)

Example 2: Mid Fusion (Feature level)
--------------------------------------
Use fusion modules at specific network layers:
    # At backbone stage
    fused = DynamicAlignFusion([64, 64])([rgb_feat, thermal_feat])
    
    # At neck stage
    fused = PSFM([128, 128])([rgb_feat, thermal_feat])
    
    # Multi-scale fusion
    fused = Fusion([64, 128, 256], fusion='bifpn')(features)

Example 3: Late Fusion (Score level)
-------------------------------------
Run separate RGB and Thermal models, then fuse predictions:
    rgb_pred = rgb_model(rgb)
    thermal_pred = thermal_model(thermal)
    fused_pred = (rgb_pred + thermal_pred) / 2

Example 4: Cross-Modal Attention
---------------------------------
    # Cross-attention fusion
    out = CrossAttentionShared(128)([rgb_feat, thermal_feat])
    
    # Transformer fusion
    out = CrossTransformerFusion(128)([rgb_feat, thermal_feat])

Example 5: Building a RGB-T YOLO Model
---------------------------------------
    # In your YAML config:
    backbone:
      - [-1, 1, Conv, [64, 3, 2]]        # RGB branch
      - [-1, 1, Conv, [64, 3, 2]]        # Thermal branch  
      - [[-1, -2], 1, DynamicAlignFusion, [64]]  # Fusion
      - [-1, 1, C2f, [128, 1]]
      ...
"""
