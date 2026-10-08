
import math
import torch
import torch.nn as nn


import torch.nn.functional as F
from timm.layers import trunc_normal_, Mlp, DropPath


class GraphConvGuidance(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        """
        Graph Convolution with Learnable Adjacency Matrix.

        Args:
            in_channels (int): Number of input channels (channels of diff_feature, C//2).
            out_channels (int): Number of output channels ((C//4)*2).
            num_nodes (int): Number of nodes (V).
        """
        super(GraphConvGuidance, self).__init__()

        # Learnable adjacency matrix, initialized as identity
        self.adj = nn.Parameter(torch.eye(num_nodes))
        # Linear transformation
        self.fc = nn.Linear(in_channels, out_channels)
        # Activation function
        self.tanh = nn.Tanh()
        # BatchNorm2d expects input of shape [B, out_channels, T, V]
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Permute to [B, T, V, C] for graph convolution along node dimension
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, T, V, C]
        # Graph convolution using learnable adjacency matrix
        x = torch.einsum('vw,btwc->btwc', self.adj, x)  # [B, T, V, C]
        # Linear transformation + activation
        x = self.fc(x)  # [B, T, V, out_channels]
        x = self.tanh(x)
        # Rearrange to [B, out_channels, T, V] for BatchNorm2d
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.bn(x)

        return x


class TemporalModulationMS(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, dilations=(1, 2, 3, 4)):
        super().__init__()
        self.dilations = tuple(dilations)
        self.branches = nn.ModuleList([
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=(3, 1),
                padding=(d, 0),
                dilation=(d, 1),
                groups=in_channels,  # depthwise
                bias=False
            ) for d in self.dilations
        ])
        # pointwise fuse
        self.pw = nn.Conv2d(in_channels * len(self.dilations), out_channels, kernel_size=1, bias=True)
        self.act = nn.Tanh()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, V]
        h = torch.cat([b(x) for b in self.branches], dim=1)  # [B, C*len(d), T, V]
        z = self.pw(h)  # [B, out_channels, T, V]
        z = self.act(z)
        z = self.bn(z)
        return z


class MGSTFormer(nn.Module):
    def __init__(self, in_channels=2, num_points=44, kernel_size=3, num_heads=4,
                 drop=0., drop_path=0., mlp_ratio=2.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        SAE-SC block.

        Args:
            in_channels (int): Number of input channels.
            num_points (int): Number of skeleton nodes (default: 44).
            kernel_size (int): Retained for interface compatibility; unused in this block.
            num_heads (int): Number of spatial experts and temporal convolution groups (default: 4).
            drop (float): Dropout rate for projection.
            drop_path (float): Stochastic depth rate (default: 0.).
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            act_layer (nn.Module): Activation function class (default: GELU).
            norm_layer (nn.Module): Normalization layer class (default: LayerNorm).
        """
        super(MGSTFormer, self).__init__()

        # -------------------- Linear & Normalization --------------------
        self.mapping = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.norm_1 = norm_layer(in_channels)

        # -------------------- Skeleton Branch (Learnable adjacency matrix) --------------------
        self.gconv = nn.Parameter(torch.zeros(num_heads, num_points, num_points))
        trunc_normal_(self.gconv, std=.02)  # 用截断正态分布初始化

        # -------------------- Spatial Expert Gating --------------------
        hop_in_dim = in_channels // 4
        hidden_dim = max(hop_in_dim // 2, 16)

        self.hop_gate = nn.Sequential(
            nn.Linear(hop_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_heads)  # Predict one gating logit per spatial expert
        )
        self.hop_softmax = nn.Softmax(dim=-1)

        # -------------------- Expert Diversity (Cosine) --------------------
        # Encourage spatial experts to learn complementary representations
        # (mitigate expert collapse / same-output issue).
        self.enable_div_cosine = True
        self.div_cosine_global_weight = 0.2  # Reduce the penalty weight for pairs involving the last spatial expert
        self.aux_loss = None

        # -------------------- Temporal Experts (3/5/7/9) + Gating --------------------
        t_in_dim = in_channels // 4
        t_hidden = max(t_in_dim // 2, 16)

        ks_list = [3, 5, 7, 9]
        self.tconv_experts = nn.ModuleList([
            nn.Conv2d(
                t_in_dim, t_in_dim,
                kernel_size=(k, 1),
                padding=((k - 1) // 2, 0),
                groups=num_heads,
                bias=True
            ) for k in ks_list
        ])

        self.t_gate = nn.Sequential(
            nn.Linear(t_in_dim, t_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(t_hidden, len(ks_list))  # 4
        )
        self.t_softmax = nn.Softmax(dim=-1)

        # -------------------- Motion Branch --------------------
        self.Motion = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2,
                      kernel_size=(3, 1), stride=1,
                      padding=(1, 0), groups=in_channels // 2, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU()
        )

        # -------------------- Projection & Residual --------------------
        self.proj = nn.Linear(in_channels, in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(in_features=in_channels,
                       hidden_features=int(mlp_ratio * in_channels),
                       act_layer=act_layer, drop=drop)

        # -------------------- Guidance Modules --------------------
        # Use motion (diff) branch to generate gamma & beta for both TConv and GConv.
        # Multi-scale MTM provides a larger / mixed temporal receptive field.
        self.MTM = TemporalModulationMS(in_channels // 2, (in_channels // 4) * 2, dilations=(1, 2, 3, 4))

        # MSM uses motion-guided graph convolution to produce modulation parameters.
        self.MSM = GraphConvGuidance(in_channels // 2, (in_channels // 4) * 2, num_points)

        # Learnable scaling factors for multiplicative and additive modulation
        self.mtm_gamma_scale = nn.Parameter(torch.tensor(0.10))
        self.mtm_beta_scale = nn.Parameter(torch.tensor(0.05))
        self.msm_gamma_scale = nn.Parameter(torch.tensor(0.10))
        self.msm_beta_scale = nn.Parameter(torch.tensor(0.05))

        # -------------------- Adaptive Fusion Module --------------------
        # Concatenate outputs: [TConv_out, GConv_out, Diff_feature]
        # Channels: C//4 + C//4 + C//2 = C
        self.aggregation_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 3, kernel_size=1, bias=False)
        )

        # Final fusion projection: ensure output channels = in_channels
        self.aggregation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

    def _cosine_diversity_loss(self, outs: 'torch.Tensor') -> 'torch.Tensor':
        """
        Asymmetric squared-cosine diversity loss on spatial expert outputs.

        Args:
            outs: [B, K, C, T, V] spatial expert outputs before gating fusion.

        Returns:
            Scalar tensor.
        """
        # Pool over (T, V) -> [B, K, C]
        z = outs.mean(dim=(3, 4))
        z = F.normalize(z, p=2, dim=-1, eps=1e-6)

        # Cosine similarity matrix per sample: [B, K, K]
        sim = torch.einsum('bkc,blc->bkl', z, z)

        # We encourage off-diagonal similarities to be small (close to 0).
        K = sim.shape[-1]
        if K < 2:
            return sim.new_tensor(0.0)

        # Apply stronger diversity constraints among the first K-1 spatial experts.
        # Apply weaker constraints between the last expert and the remaining experts.
        part_K = max(K - 1, 1)
        part_idx = list(range(part_K))
        global_idx = K - 1

        # Pairs among the first K-1 spatial experts
        if len(part_idx) >= 2:
            pp = sim[:, part_idx, :][:, :, part_idx]  # [B, part_K, part_K]
            # remove diagonal
            eye = torch.eye(part_K, device=pp.device, dtype=pp.dtype).unsqueeze(0)
            pp_off = (pp * (1.0 - eye))
            # Average squared cosine similarity over off-diagonal expert pairs,
            # then sum over batch samples.
            denom = part_K * (part_K - 1)
            l_pp = (pp_off.pow(2).sum() / max(denom, 1))
        else:
            l_pp = sim.new_tensor(0.0)

        # Pairs between the last spatial expert and the first K-1 experts
        if K >= 2:
            gp = sim[:, global_idx, :part_K]  # [B, part_K]
            l_gp = gp.pow(2).mean()
        else:
            l_gp = sim.new_tensor(0.0)

        return l_pp + self.div_cosine_global_weight * l_gp

    def interpolate_diff(self, x, order=0, dim=2):
        """
        Compute temporal differences with padding.

        Args:
            x (Tensor): Input tensor.
            order (int): Number of differential orders.
            dim (int): Dimension along which to compute differences (default: 2 for T).

        Returns:
            Tensor: Differential-augmented tensor.
        """
        out = x
        for _ in range(order):
            out = torch.diff(out, dim=dim)
            # For 4D tensor, F.pad args: (pad_left, pad_right, pad_top, pad_bottom)
            out = F.pad(out, (0, 0, 0, 1), mode='replicate')
        return out

    def forward(self, X_feat):
        """
        Forward pass of the SAE-SC block.

        Args:
            X_feat (Tensor): Input tensor of shape [B, C, T, V].

        Returns:
            Tensor: Output tensor of shape [B, C, T, V].
        """
        B, C, T, V = X_feat.shape

        # -------------------- Linear & LN & Res  --------------------
        X_feat = X_feat.permute(0, 2, 3, 1).contiguous()  # [B, T, V, C]
        X_feat_res = X_feat

        X_in = self.mapping(self.norm_1(X_feat)).permute(0, 3, 1, 2).contiguous()  # [B, C, T, V]
        f_st, X_in_motion = torch.split(X_in, [C // 2, C // 2], dim=1)
        X_in_tc = torch.chunk(f_st, 2, dim=1)  # [B, C//4, T, V]

        # -------------------- DST: Spatial Experts with Gated Fusion --------------------
        x_s = X_in_tc[0]  # [B, C//4, T, V]

        A_hops = self.gconv.to(device=x_s.device, dtype=x_s.dtype)  # 使用可学习的邻接矩阵

        outs = torch.einsum('n c t u, k v u -> n k c t v', x_s, A_hops)

        # Auxiliary diversity loss (spatial experts)
        if self.enable_div_cosine:
            self.aux_loss = self._cosine_diversity_loss(outs)
        else:
            self.aux_loss = None

        g = x_s.mean(dim=(2, 3))  # [B, C//4]，对 T,V 做全局均值池化
        w = self.hop_softmax(self.hop_gate(g))  # [B, 4]，softmax 后和为 1

        w = w.view(B, -1, 1, 1, 1)  # [B, 4, 1, 1, 1]
        X_gc = (outs * w).sum(dim=1)

        # -------------------- Temporal Experts (3/5/7/9) + gated fusion --------------------
        x_t = X_in_tc[1]  # [B, C//4, T, V]

        t_outs = [conv(x_t) for conv in self.tconv_experts]  # list of 4 tensors
        t_outs = torch.stack(t_outs, dim=1)  # [B, 4, C//4, T, V]

        t_g = x_t.mean(dim=(2, 3))  # [B, C//4]
        t_w = self.t_softmax(self.t_gate(t_g))  # [B, 4]
        t_w = t_w.view(B, 4, 1, 1, 1)  # [B, 4, 1, 1, 1]

        X_tc = (t_outs * t_w).sum(dim=1)  # [B, C//4, T, V]

        # -------------------- Motion Branch --------------------
        X_delta = self.interpolate_diff(X_in_motion, order=1)
        motion = self.Motion(X_delta)  # [B, C//2, T, V]

        # ------------- Motion-guided Skeletal Modulation (MSM) ------------------
        Z_s = self.MSM(motion)
        gamma_s, beta_s = torch.chunk(Z_s, 2, dim=1)

        # Apply tanh and learnable scaling to the spatial modulation parameters
        gamma_s = torch.tanh(gamma_s) * self.msm_gamma_scale
        beta_s = torch.tanh(beta_s) * self.msm_beta_scale
        mean_g = X_gc.mean(dim=(2, 3), keepdim=True)
        std_g = X_gc.std(dim=(2, 3), keepdim=True)
        X_gcm = (X_gc - mean_g) / (std_g + 1e-6) * (1 + gamma_s) + beta_s

        # ------------- Motion-guided Temporal Modulation (MTM) ------------------
        Z_t = self.MTM(motion)
        gamma_t, beta_t = torch.chunk(Z_t, 2, dim=1)

        # Apply tanh and learnable scaling to the temporal modulation parameters
        gamma_t = torch.tanh(gamma_t) * self.mtm_gamma_scale
        beta_t = torch.tanh(beta_t) * self.mtm_beta_scale
        mean_t = X_tc.mean(dim=(2, 3), keepdim=True)
        std_t = X_tc.std(dim=(2, 3), keepdim=True)
        X_tcm = (X_tc - mean_t) / (std_t + 1e-6) * (1 + gamma_t) + beta_t

        # -------------------- Aggregation --------------------
        X_agg = torch.cat([X_tcm, X_gcm, motion], dim=1)  # [B, C, T, V]
        agg_logits = self.aggregation_gate(X_agg)   # [B, 3, T, V]
        agg_weights = F.softmax(agg_logits, dim=1)  # normalize along branch dimension
        weighted_t = X_tcm * agg_weights[:, 0:1, :, :]
        weighted_g = X_gcm * agg_weights[:, 1:2, :, :]
        weighted_d = motion * agg_weights[:, 2:3, :, :]
        agg_feature = torch.cat([weighted_t, weighted_g, weighted_d], dim=1)
        agg_feature = self.aggregation(agg_feature)

        # -------------------- FFN --------------------
        X_f = self.proj_drop(self.proj(agg_feature.permute(0, 2, 3, 1).contiguous()))
        X_f = X_feat_res + self.drop_path(X_f)

        X_f = X_f + self.drop_path(self.mlp(self.norm_2(X_f)))
        X_f = X_f.permute(0, 3, 1, 2).contiguous()

        return X_f



class MFM_Layer(nn.Module):
    def __init__(
            self, depth, in_channels, out_channels, num_points=44, kernel_size=3, num_heads=4,
            drop=0., drop_path=0., mlp_ratio=2.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(MFM_Layer, self).__init__()
        blocks = []
        for index in range(depth):
            blocks.append(
                MGSTFormer(
                    in_channels=in_channels if index == 0 else out_channels,
                    num_points=num_points,
                    kernel_size=kernel_size,
                    num_heads=num_heads,
                    drop=drop,
                    drop_path=drop_path if isinstance(drop_path, float) else drop_path[index],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        output = input
        aux = None
        for block in self.blocks:
            output = block(output)
            # accumulate auxiliary loss if provided by the block
            if hasattr(block, 'aux_loss') and (block.aux_loss is not None):
                aux = block.aux_loss if aux is None else (aux + block.aux_loss)
        # expose summed aux loss to upper modules
        self.aux_loss = aux
        return output


"""Multi-scale feature construction and aggregation."""


class MultiScaleSpatioTemporalFeatureConstructor(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=2, dilation=1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.reduction = nn.Conv2d(dim_in, dim_out, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1),
                                   dilation=(dilation, 1), padding_mode='replicate')
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        x = self.bn(self.reduction(x))
        return x


class MotionConsistencyLearning(nn.Module):
    def __init__(self, in_channels, dropout=0.1):
        """
        Args:
            in_channels (int): Number of input feature channels.
            dropout (float): Dropout probability, used to reduce overfitting.
        """
        super(MotionConsistencyLearning, self).__init__()

        # Downsample features at multiple scales with BN and ReLU
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(8, 1), stride=(8, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(4, 1), stride=(4, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # f4 is already at the target resolution, but still processed with BN and ReLU
        self.f4_proc = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Scale-specific enhancement branches with independent trainable parameters
        self.expert1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.expert2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.expert3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.expert4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Predict scale weights at each temporal-joint position from concatenated multi-scale features
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4, kernel_size=1)
        )

        self.dropout = nn.Dropout(dropout)

        # Use fused C-channel feature as residual and add it to X_fusion (4C), then compress back to C
        self.res_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * 4)
        )
        # Learnable residual scaling initialized conservatively for stable training
        self.res_scale = nn.Parameter(torch.zeros(1, in_channels * 4, 1, 1))
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x (tuple):
                f1: [B, C, T, V]
                f2: [B, C, T//2, V]
                f3: [B, C, T//4, V]
                f4: [B, C, T//8, V]

        Returns:
            torch.Tensor: [B, C, T//8, V] — fused feature representation
        """
        f1, f2, f3, f4 = x

        # Step 1: align different-scale features to the same temporal resolution
        f1_ds = self.down1(f1)      # [B, C, T//8, V]
        f2_ds = self.down2(f2)      # [B, C, T//8, V]
        f3_ds = self.down3(f3)      # [B, C, T//8, V]
        f4_proc = self.f4_proc(f4)  # [B, C, T//8, V]

        # Step 2: concatenate temporally aligned features from all four scales
        X_fusion = torch.cat([f1_ds, f2_ds, f3_ds, f4_proc], dim=1)  # [B, 4C, T//8, V]

        # Step 3: scale-specific fixed-kernel expert enhancement
        f1_enh = self.expert1(f1_ds)   # [B, C, T//8, V]
        f2_enh = self.expert2(f2_ds)   # [B, C, T//8, V]
        f3_enh = self.expert3(f3_ds)   # [B, C, T//8, V]
        f4_enh = self.expert4(f4_proc) # [B, C, T//8, V]

        # Step 4: generate adaptive scale weights from X_fusion
        gate_logits = self.gate_conv(X_fusion)       # [B, 4, T//8, V]
        gate_weights = F.softmax(gate_logits, dim=1) # normalize along scale dimension

        # Step 5: weighted fusion of enhanced scale-specific features
        fused_output = (
            gate_weights[:, 0:1, :, :] * f1_enh +
            gate_weights[:, 1:2, :, :] * f2_enh +
            gate_weights[:, 2:3, :, :] * f3_enh +
            gate_weights[:, 3:4, :, :] * f4_enh
        )  # [B, C, T//8, V]

        fused_output = self.dropout(fused_output)

        # Step 6: use the fused result as a residual term and add it to X_fusion
        fused_residual = self.res_proj(fused_output)                 # [B, 4C, T//8, V]
        output = X_fusion + self.res_scale * fused_residual         # [B, 4C, T//8, V]

        # Step 7: compress back to C channels for downstream classifier compatibility
        output = self.out_proj(output)               # [B, C, T//8, V]

        return output


"""SAE-SC model."""


class SAE_SC(nn.Module):
    def __init__(self, in_channels=2, depths=(3, 3, 3, 3), channels=(96, 96, 96, 96), num_classes=52,
                 embed_dim=96, num_people=1, num_frames=64, num_points=44, kernel_size=3, num_heads=4,
                 head_drop=0., drop=0., drop_path=0., mlp_ratio=2.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, index_t=True, global_pool='avg', ):

        super(SAE_SC, self).__init__()

        assert len(depths) == len(channels), "For each stage a channel dimension must be given."
        assert global_pool in ["avg", "max"], f"Only avg and max is supported but {global_pool} is given"
        self.num_classes: int = num_classes
        self.head_drop = head_drop
        self.index_t = index_t
        self.embed_dim = embed_dim

        if self.head_drop != 0:
            self.dropout = nn.Dropout(p=self.head_drop)
        else:
            self.dropout = None

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=1, stride=1, padding=0),
            act_layer(),
            nn.Conv2d(in_channels=2 * in_channels, out_channels=3 * in_channels, kernel_size=1, stride=1, padding=0),
            act_layer(),
            nn.Conv2d(in_channels=3 * in_channels, out_channels=embed_dim, kernel_size=1, stride=1, padding=0)
        )

        if self.index_t:
            self.STPE = nn.Parameter(torch.zeros(embed_dim, num_points * num_people))
            trunc_normal_(self.STPE, std=.02)
        else:
            self.STPE_ = nn.Parameter(
                torch.zeros(1, embed_dim, num_frames, num_points * num_people))
            trunc_normal_(self.STPE_, std=.02)

        # Init blocks
        drop_path = torch.linspace(0.0, drop_path, sum(depths)).tolist()
        MFM = []
        DS = []
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            MFM.append(
                MFM_Layer(
                    depth=depth,
                    in_channels=embed_dim if index == 0 else channels[index - 1],
                    out_channels=channel,
                    num_points=num_points * num_people,
                    kernel_size=kernel_size,
                    num_heads=num_heads,
                    drop=drop,
                    drop_path=drop_path[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer
                )
            )
            if index != len(depths) - 1:
                DS.append(
                    MultiScaleSpatioTemporalFeatureConstructor(channels[index], channels[index + 1],
                                                               kernel_size=kernel_size)
                )
        self.MFM = nn.ModuleList(MFM)
        self.DSs = nn.ModuleList(DS)

        self.global_pool: str = global_pool

        self.head = nn.Linear(channels[-1], num_classes)

        self.fusion = MotionConsistencyLearning(in_channels=channels[-1])

    def forward_features(self, X_feat):
        X_f_L = []
        DSs = [X_feat]

        ds_output = X_feat
        for ds in self.DSs:
            ds_output = ds(ds_output)
            DSs.append(ds_output)

        aux_total = None
        for X_f, mfm_layer in zip(DSs, self.MFM):
            out = mfm_layer(X_f)
            X_f_L.append(out)
            if hasattr(mfm_layer, 'aux_loss') and (mfm_layer.aux_loss is not None):
                aux_total = mfm_layer.aux_loss if aux_total is None else (aux_total + mfm_layer.aux_loss)

        # expose aux loss on the whole model (unscaled)
        self.aux_loss_total = aux_total

        X_z = self.fusion(X_f_L)
        return X_z

    def classifier(self, input, pre_logits=False):
        if self.global_pool == "avg":
            input = input.mean(dim=(2, 3))
        elif self.global_pool == "max":
            input = torch.amax(input, dim=(2, 3))
        if self.dropout is not None:
            input = self.dropout(input)
        return input if pre_logits else self.head(input)

    def feature_embedding(self, X_raw, index_t):
        B, C, T, V, M = X_raw.shape

        X_raw = X_raw.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)  # [B, C, T, M * V]

        output = self.projection(X_raw)

        if self.index_t:
            te = torch.zeros(B, T, self.embed_dim).to(output.device)  # B, T, C
            div_term = torch.exp(
                (torch.arange(0, self.embed_dim, 2, dtype=torch.float) * -(math.log(10000.0) / self.embed_dim))).to(
                output.device)
            te[:, :, 0::2] = torch.sin(index_t.unsqueeze(-1).float() * div_term)
            te[:, :, 1::2] = torch.cos(index_t.unsqueeze(-1).float() * div_term)
            X_feat = output + torch.einsum('b t c, c v -> b c t v', te, self.STPE)
        else:
            X_feat = output + self.STPE_

        return X_feat

    def forward(self, input, index_t):
        X_feat = self.feature_embedding(input, index_t)
        X_z = self.forward_features(X_feat)
        logits = self.classifier(X_z)
        return logits, getattr(self, "aux_loss_total", None)


def SAE_SC_(**kwargs):
    return SAE_SC(
        depths=(3, 3, 3, 3),
        channels=(96, 96, 96, 96),
        embed_dim=96,
        **kwargs
    )