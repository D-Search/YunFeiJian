#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MaxViT Model with Prototype Learning

基于 SimpleMaxViT (baseline) 整合原型学习模块:
1. PrototypeGuidedAttention (PGA) - 空间特征增强
2. MultiPrototypeBank - 多原型记忆库 + K-means初始化 + EMA更新
3. 双头输出: BCE分类头 + 原型特征头

修改内容:
- forward() 现在返回 spatial feature map，用于原型相似度热图和弱监督病灶定位。
- feature_map 来自 backbone 最后一个空间特征图（经过投影和可选的 PGA 增强）。
- 调用方需同步调整为接收5个返回值。
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PrototypeGuidedAttention(nn.Module):
    """原型引导注意力: 在 Backbone 特征图上进行空间增强 (Soft Attention版本)"""
    def __init__(self, feat_dim, n_cls, K=3, temperature=0.1):
        super().__init__()
        self.n_cls, self.K = n_cls, K
        self.tau = nn.Parameter(torch.tensor(temperature))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.class_mod = nn.Parameter(torch.ones(n_cls))

        #增强特征
        self.enhance = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.GELU()
        )

    def forward(self, x, prototypes, enabled=True, return_attention_map=False):
        if not enabled:
            return x, {"attn_mean": 1.0, "attention_map": None,
                        "spatial_attention": None, "similarity_map": None}

        B, C, H, W = x.shape
        feat_flat = F.normalize(x.view(B, C, -1).permute(0, 2, 1), dim=-1, eps=1e-6)
        proto_flat = F.normalize(prototypes.view(-1, C), dim=-1, eps=1e-6)

        # [B, H*W, n_cls, K] similarity matrix
        sim = torch.matmul(feat_flat, proto_flat.t()).view(B, H*W, self.n_cls, self.K)

        # Soft Attention
        attn_weights = F.softmax(sim / self.tau, dim=-1)
        weighted_sim = (sim * attn_weights).sum(dim=-1)

        # Spatial attention (max response over classes)
        spatial_attn = (weighted_sim * torch.sigmoid(self.class_mod)).max(dim=-1)[0]
        spatial_attn_reshaped = spatial_attn.view(B, H, W)
        mask = torch.sigmoid(spatial_attn_reshaped).unsqueeze(1)  # [B, 1, H, W]

        enhanced = self.enhance(x)
        enhanced_masked = enhanced * mask  # mask [B,1,H,W] broadcasts over feat_dim channels
        return_dict = {
            "attn_mean": mask.mean().item(),
            "attention_map": mask if return_attention_map else None,
            "spatial_attention": spatial_attn_reshaped if return_attention_map else None,
            "similarity_map": F.relu( weighted_sim).view(B,H,W,self.n_cls) if return_attention_map else None
            }

        return x + self.gamma * enhanced_masked, return_dict


class MultiPrototypeBank(nn.Module):
    """多原型记忆库: 每个类别维护 K 个原型"""
    def __init__(self, n_cls, K=3, feat_dim=256, momentum=0.99):
        super().__init__()
        self.n_cls, self.K, self.feat_dim = n_cls, K, feat_dim
        self.momentum = momentum
        self.register_buffer("prototypes", torch.randn(n_cls, K, feat_dim))
        self.initialized = False

    @torch.no_grad()
    def initialize_from_features(self, all_features, all_labels, max_iter=20):
        """使用 K-means 初始化原型库"""
        print(f"==> Initializing Prototype Bank using K-means (K={self.K})...")
        new_prototypes = self.prototypes.clone()
        
        for c in range(self.n_cls):
            mask = all_labels[:, c] > 0.5
            cls_feats = all_features[mask]
            
            if len(cls_feats) < self.K:
                print(f"   Class {c}: insufficient samples ({len(cls_feats)}), random init.")
                if len(cls_feats) > 0:
                    idx = torch.randint(0, len(cls_feats), (self.K,))
                    new_prototypes[c] = cls_feats[idx]
                continue

            indices = torch.randperm(len(cls_feats))[:self.K]
            centroids = cls_feats[indices]
            
            for i in range(max_iter):
                centroids = F.normalize(centroids, dim=-1)
                cls_feats_norm = F.normalize(cls_feats, dim=-1)
                dist = torch.matmul(cls_feats_norm, centroids.t())
                assignments = dist.argmax(dim=-1)
                
                new_centroids = torch.zeros_like(centroids)
                for k in range(self.K):
                    k_mask = (assignments == k)
                    if k_mask.any():
                        new_centroids[k] = cls_feats[k_mask].mean(0)
                    else:
                        new_centroids[k] = centroids[k]
                centroids = new_centroids
            
            new_prototypes[c] = centroids
            print(f"   Class {c}: K-means converged.")

        self.prototypes.copy_(F.normalize(new_prototypes, dim=-1))
        self.initialized = True
        print("==> Prototype Bank Initialization Complete.")

    @torch.no_grad()
    def update_ema(self, features, labels):
        """基于当前 Batch 的在线 EMA 更新"""
        if not self.initialized:
            return
        features = F.normalize(features, dim=-1, eps=1e-6)
        for c in range(self.n_cls):
            mask = labels[:, c] > 0.5
            if not mask.any():
                continue
            cls_feats = features[mask]
            dist = torch.matmul(cls_feats, self.prototypes[c].t())
            target_ids = dist.argmax(dim=-1)
            for k in range(self.K):
                if (target_ids == k).any():
                    new_mean = cls_feats[target_ids == k].mean(0)
                    self.prototypes[c, k] = self.momentum * self.prototypes[c, k] + (1 - self.momentum) * new_mean
        self.prototypes.data = F.normalize(self.prototypes, dim=-1, eps=1e-6)

    def compute_losses(self, features, labels, temp=0.1):
        """计算原型损失和多样性损失 (Soft Attention版本)"""
        # [B, n_cls, K] 相似度矩阵
        sim = torch.matmul(features, self.prototypes.view(-1, self.feat_dim).t()).view(
            features.size(0), self.n_cls, self.K
        )
        
        # Soft Attention: 使用 softmax 替代 max
        # 对 K 个原型进行软加权，避免对噪声原型的过度响应
        attn_weights = F.softmax(sim / temp, dim=-1)  # [B, n_cls, K]
        weighted_sim = (sim * attn_weights).sum(dim=-1)   # [B, n_cls]
        logits = weighted_sim
        
        loss_proto = F.binary_cross_entropy_with_logits(logits, labels)
        
        p = self.prototypes
        intra_sim = torch.matmul(p, p.transpose(1, 2))
        mask = (1.0 - torch.eye(self.K, device=p.device)).unsqueeze(0)
        loss_div = (intra_sim * mask).clamp(min=0).mean()
        
        return loss_proto, loss_div


class MaxViTProtoNet(nn.Module):
    """
    MaxViT 原型网络
    
    整合原型学习的改进版:
    - Backbone 提取空间特征图
    - 可选 PGA 空间增强
    - 双头输出: BCE分类 + 原型特征
    - **新增**: 返回空间 feature_map，用于原型相似度热图
    """
    
    def __init__(
        self,
        backbone_name: str = "maxvit_tiny_tf_224.in1k",
        pretrained: bool = False,
        n_cls: int = 14,
        dropout: float = 0.4,
        feat_dim: int = 256,
        use_pga: bool = True,
        pga_K: int = 3,
        pga_temperature: float = 0.1,
    ):
        super().__init__()
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )
        
        backbone_dim = self.backbone.num_features
        print(f"[Model] Backbone: {backbone_name}, feature dim: {backbone_dim}")
        
        self.feat_dim = feat_dim
        # 投影层：将 backbone 通道数映射到 feat_dim
        self.proj = nn.Conv2d(backbone_dim, feat_dim, 1) if backbone_dim != feat_dim else nn.Identity()
        
        self.use_pga = use_pga
        if use_pga:
            self.pga = PrototypeGuidedAttention(feat_dim, n_cls, K=pga_K, temperature=pga_temperature)
            print(f"[Model] PGA enabled (K={pga_K}, temp={pga_temperature})")
        else:
            self.pga = None
        
        self.head_dropout = nn.Dropout(dropout)
        self.bce_head = nn.Linear(feat_dim, n_cls)
        self.proto_head = nn.Linear(feat_dim, feat_dim)
        
        self.n_cls = n_cls
        self.pga_enabled = True

    def forward(self, x, proto_bank=None, return_viz_params=False):
        """
        Args:
            proto_bank: prototype bank for PGA
            return_viz_params: if True, returns extra visualization data
                (spatial_features, spatial_attention, similarity_map, proto_features, bce_logits, input_image)
        Returns:
            pooled, proto_features, bce_logits, pga_stats, feature_map
            OR (when return_viz_params=True):
            pooled, proto_features, bce_logits, pga_stats, viz_params
        """
        # 1. Backbone feature extraction
        features = self.backbone.forward_features(x)

        # 2. Reshape to 2D spatial feature map if needed
        if features.dim() == 3:
            B, N, C_backbone = features.shape
            H = W = int(N ** 0.5)
            features = features.transpose(1, 2).reshape(B, C_backbone, H, W)
        elif features.dim() == 2:
            features = features.unsqueeze(-1).unsqueeze(-1)

        # 3. Project to feat_dim
        spatial_feats = self.proj(features)

        # 4. Optional PGA spatial enhancement
        pga_stats = {}
        if self.pga and self.pga_enabled and proto_bank is not None and self.use_pga:
            spatial_feats, pga_stats = self.pga(
                spatial_feats, proto_bank.prototypes,
                return_attention_map=return_viz_params
            )

        feature_map = spatial_feats

        # 5. Global pooling -> classification
        pooled = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        pooled = self.head_dropout(pooled)
        bce_logits = self.bce_head(pooled)
        proto_features = F.normalize(self.proto_head(pooled), dim=-1, eps=1e-6)

        if return_viz_params:
            viz_params = {
                "spatial_features": spatial_feats,
                "spatial_attention": pga_stats.get("spatial_attention"),
                "similarity_map": pga_stats.get("similarity_map"),
                "attention_map": pga_stats.get("attention_map"),
                "proto_features": proto_features,
                "bce_logits": bce_logits,
                "input_image": x,
                "pga_stats": pga_stats,
            }
            return pooled, proto_features, bce_logits, pga_stats, viz_params

        return pooled, proto_features, bce_logits, pga_stats, feature_map


def create_model_and_bank(config: dict, device="cuda"):
    """从配置文件创建模型和原型库"""
    model_cfg = config.get("model", {})
    proto_cfg = config.get("prototype", {})
    pga_cfg = config.get("pga", {})
    ablation_cfg = config.get("ablation", {})
    
    use_proto = ablation_cfg.get("use_proto", True)
    use_pga = ablation_cfg.get("use_pga", True)
    
    model = MaxViTProtoNet(
        backbone_name=model_cfg.get("backbone", "maxvit_tiny_tf_224.in1k"),
        pretrained=model_cfg.get("pretrained", False),
        n_cls=model_cfg.get("num_classes", 14),
        dropout=config.get("training", {}).get("dropout", 0.4),
        feat_dim=proto_cfg.get("feat_dim", 256),
        use_pga=use_pga,
        pga_K=pga_cfg.get("K", 3),
        pga_temperature=pga_cfg.get("temperature", 0.1),
    ).to(device)
    
    if use_proto:
        bank = MultiPrototypeBank(
            n_cls=model_cfg.get("num_classes", 14),
            K=proto_cfg.get("K", 3),
            feat_dim=proto_cfg.get("feat_dim", 256),
            momentum=proto_cfg.get("ema_momentum", 0.99)
        ).to(device)
    else:
        bank = None
    
    return model, bank


# 兼容 baseline 导出
SimpleMaxViT = MaxViTProtoNet

def create_simple_model(config: dict) -> MaxViTProtoNet:
    """创建简单模型 (兼容 baseline 接口)"""
    model_cfg = config.get("model", {})
    return MaxViTProtoNet(
        backbone_name=model_cfg.get("backbone", "maxvit_tiny_tf_224.in1k"),
        pretrained=model_cfg.get("pretrained", False),
        n_cls=model_cfg.get("num_classes", 14),
        dropout=config.get("training", {}).get("dropout", 0.4),
        use_pga=False,
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = MaxViTProtoNet(
        backbone_name="maxvit_tiny_tf_224.in1k",
        pretrained=False,
        n_cls=14,
        dropout=0.4,
        use_pga=True,
    ).to(device)
    
    bank = MultiPrototypeBank(n_cls=14, K=3, feat_dim=256).to(device)
    
    x = torch.randn(2, 3, 224, 224).to(device)
    pooled, proto_feat, logits, stats, feature_map = model(x, proto_bank=bank)
    print(f"Pooled: {pooled.shape}")
    print(f"Proto features: {proto_feat.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Feature map: {feature_map.shape}")
    print(f"PGA stats: {stats}")
    
    # 示例：验证可用于原型相似度计算
    # prototypes: [K, C] -> 这里以类别0的原型为例
    prototypes = bank.prototypes[0]  # [K, feat_dim]
    sim_map = torch.einsum('bchw,kc->bkhw', feature_map, prototypes)
    print(f"Similarity map shape: {sim_map.shape}")  # 预期 [2, 3, H, W]
    
    print("\nTest passed!")