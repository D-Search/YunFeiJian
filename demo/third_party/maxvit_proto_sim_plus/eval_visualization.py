#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MaxViT Prototype Learning Evaluation Visualization Script (CLI)

独立评估可视化脚本，用于生成评估报告图表。
不依赖Django，不直接操作数据库。

使用方式：
    python eval_visualization.py --viz all --checkpoint checkpoint_best.pt

可视化功能:
1. t-SNE 特征可视化 (14类独立子图 + 合并图 + 聚类中心)
2. 原型库可视化
3. Grad-CAM 注意力热力图
4. 原型相似度热力图 + 病灶定位框 (20张)
5. Paper-style annotated 汇总图 (5x4 = 20样本)
6. 热力图汇总图 (5x4 = 20样本)
7. 14类疾病定位汇总图
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import json

from model import MaxViTProtoNet, create_model_and_bank
from data import prepare_data_splits, get_transforms, ChestXrayDataset, CLS_14

plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(r"demo/hello/mavit_proto_sim_plus/config.yaml", encoding='utf-8') as f:
    config = yaml.safe_load(f)
output_dir = Path(config["paths"]["output_dir"])

# ============================================================================
# 14类疾病的annotated颜色定义 (BGR格式，用于OpenCV)
# ============================================================================
_annotated_COLORS_BGR = [
    (255, 0, 0),       # 0  Atelectasis           - 红
    (0, 215, 255),     # 1  Cardiomegaly          - 金黄
    (0, 165, 255),     # 2  Consolidation         - 橙
    (0, 255, 127),     # 3  Edema                  - 翠绿
    (0, 255, 0),       # 4  Effusion               - 绿
    (255, 0, 255),     # 5  Emphysema             - 紫红
    (255, 140, 0),     # 6  Fibrosis              - 暗橙
    (0, 255, 255),     # 7  Hernia                - 黄
    (160, 32, 240),    # 8  Infiltration          - 紫
    (255, 191, 0),     # 9  Mass                  - 深黄
    (0, 128, 255),     # 10 Nodule                - 橙蓝
    (255, 127, 80),    # 11 Pneumonia             - 珊瑚
    (65, 117, 255),    # 12 Pneumothorax          - 蓝
    (128, 128, 255),   # 13 Pleural_Thickening   - 灰蓝
]
_annotated_COLORS_RGB = [(c[2] / 255.0, c[1] / 255.0, c[0] / 255.0) for c in _annotated_COLORS_BGR]


# ============================================================================
# Helper: 从注意力热力图计算Bounding Box
# ============================================================================

def _compute_annotatedes(attention_map, threshold_ratio=0.3, min_area=100):
    """从注意力热力图计算Bounding Box及其平均置信度"""
    if attention_map is None:
        return []

    attn = attention_map.copy()
    if attn.max() > attn.min():
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    else:
        attn_norm = attn

    threshold = threshold_ratio * attn_norm.max()
    binary = (attn_norm > threshold).astype(np.uint8)

    try:
        import cv2
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        return []

    annotatedes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            region_conf = float(attn_norm[y:y+h, x:x+w].mean())
            annotatedes.append((x, y, w, h, region_conf))

    return annotatedes


def _clamp_text_pos(x, y, text_w, text_h, W, H, margin=3):
    """将文本左上角约束在图像边界内"""
    x = max(margin, min(W - text_w - margin, x))
    y = max(margin, min(H - text_h - margin, y))
    return int(x), int(y)


# ============================================================================
# 1. Feature Extraction
# ============================================================================

def extract_features(model, dataset, proto_bank=None, n_samples=2000):
    """提取特征用于可视化"""
    model.eval()

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    all_features = []
    all_proto_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting features"):
            if len(all_labels) >= n_samples:
                break

            images = images.to(device)
            pooled, proto_feat, logits, stats, _ = model(images, proto_bank=proto_bank)

            all_features.append(pooled.cpu())
            all_proto_features.append(proto_feat.cpu())
            all_labels.append(labels.numpy())

    all_features = torch.cat(all_features, dim=0)[:n_samples].numpy()
    all_proto_features = torch.cat(all_proto_features, dim=0)[:n_samples].numpy()
    all_labels = np.concatenate(all_labels, axis=0)[:n_samples]

    return all_features, all_proto_features, all_labels


# ============================================================================
# 2. t-SNE Feature Visualization
# ============================================================================

def plot_tsne(all_features, all_proto_features, all_labels, n_samples=2000, perplexity=30, save_dir=None):
    """t-SNE特征可视化 (4x4布局: 14类独立 + 合并 + 聚类中心)"""
    if save_dir is None:
        save_dir = output_dir

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[Error] scikit-learn not installed")
        return

    n_samples = min(n_samples, len(all_features))
    np.random.seed(42)
    indices = np.random.choice(len(all_features), n_samples, replace=False)
    features_sub = all_features[indices]
    proto_features_sub = all_proto_features[indices]
    labels_sub = all_labels[indices]

    print(f"\nRunning t-SNE on {n_samples} samples...")

    tsne_global = TSNE(n_components=2, perplexity=perplexity, max_iter=1000,
                       random_state=42, learning_rate='auto', init='pca')
    features_2d = tsne_global.fit_transform(features_sub)

    tsne_proto = TSNE(n_components=2, perplexity=perplexity, max_iter=1000,
                      random_state=42, learning_rate='auto', init='pca')
    proto_features_2d = tsne_proto.fit_transform(proto_features_sub)

    colors = plt.cm.tab20(np.linspace(0, 1, 14))
    darker_colors = plt.cm.tab20(np.linspace(0, 1, 14)) * 0.8

    fig, axes = plt.subplots(4, 4, figsize=(24, 22))
    fig.suptitle("t-SNE Feature Visualization - 14 Classes", fontsize=18, fontweight='bold')
    axes = axes.flatten()

    for cls_idx in range(14):
        ax = axes[cls_idx]

        for other_idx in range(14):
            if other_idx != cls_idx:
                mask = labels_sub[:, other_idx] > 0.5
                if mask.sum() > 0:
                    ax.scatter(proto_features_2d[mask, 0], proto_features_2d[mask, 1],
                              c='dimgray', alpha=0.4, s=15, edgecolors='gray', linewidths=0.3)

        mask = labels_sub[:, cls_idx] > 0.5
        if mask.sum() > 0:
            ax.scatter(proto_features_2d[mask, 0], proto_features_2d[mask, 1],
                      c=[darker_colors[cls_idx]], alpha=0.9, s=30, edgecolors='black', linewidths=0.5,
                      label=f'{CLS_14[cls_idx]} (n={mask.sum()})')

        ax.set_xlabel('t-SNE Dim 1', fontsize=9)
        ax.set_ylabel('t-SNE Dim 2', fontsize=9)
        ax.set_title(f'{CLS_14[cls_idx]}', fontsize=11, fontweight='bold', color=colors[cls_idx])
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.4)
        ax.set_axisbelow(True)

    # 第15个子图: 14类合并
    ax_combined = axes[14]
    for cls_idx in range(14):
        mask = labels_sub[:, cls_idx] > 0.5
        if mask.sum() > 0:
            ax_combined.scatter(proto_features_2d[mask, 0], proto_features_2d[mask, 1],
                      c=[darker_colors[cls_idx]], alpha=0.7, s=25, edgecolors='black', linewidths=0.3,
                      label=f'{CLS_14[cls_idx]}')

    ax_combined.set_xlabel('t-SNE Dim 1', fontsize=10)
    ax_combined.set_ylabel('t-SNE Dim 2', fontsize=10)
    ax_combined.set_title('All 14 Classes Combined', fontsize=12, fontweight='bold')
    ax_combined.legend(loc='best', fontsize=7, ncol=2)
    ax_combined.grid(True, alpha=0.4)
    ax_combined.set_axisbelow(True)

    # 第16个子图: 聚类中心
    ax_centers = axes[15]

    ax_centers.scatter(proto_features_2d[:, 0], proto_features_2d[:, 1],
                      c='lightgray', alpha=0.3, s=10)

    centers = []
    for cls_idx in range(14):
        mask = labels_sub[:, cls_idx] > 0.5
        if mask.sum() > 0:
            center = proto_features_2d[mask].mean(axis=0)
        else:
            center = proto_features_2d.mean(axis=0)
        centers.append(center)
    centers = np.array(centers)

    for cls_idx in range(14):
        mask = labels_sub[:, cls_idx] > 0.5
        if mask.sum() > 0:
            ax_centers.scatter(proto_features_2d[mask, 0], proto_features_2d[mask, 1],
                             c=[darker_colors[cls_idx]], alpha=0.5, s=15, edgecolors='none')

        ax_centers.scatter(centers[cls_idx, 0], centers[cls_idx, 1],
                          c=[darker_colors[cls_idx]], s=400, marker='*', edgecolors='black', linewidths=2, zorder=10)
        ax_centers.annotate(CLS_14[cls_idx], centers[cls_idx],
                           xytext=(8, 8), textcoords='offset points', fontsize=9, fontweight='bold',
                           annotated=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))

    ax_centers.set_xlabel('t-SNE Dim 1', fontsize=10)
    ax_centers.set_ylabel('t-SNE Dim 2', fontsize=10)
    ax_centers.set_title('Cluster Centers', fontsize=12, fontweight='bold')
    ax_centers.grid(True, alpha=0.4)
    ax_centers.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_dir / "viz_tsne_all.png", dpi=150, annotated_inches='tight')
    plt.close()
    print(f"[OK] All t-SNE (16 subplots) saved: {save_dir / 'viz_tsne_all.png'}")


# ============================================================================
# 3. Prototype Similarity Visualization
# ============================================================================

def plot_prototype_similarity(proto_bank, save_dir=None):
    """可视化原型库相似度矩阵"""
    if save_dir is None:
        save_dir = output_dir

    if proto_bank is None or not proto_bank.initialized:
        print("[Info] Prototype bank not initialized, skipping similarity visualization")
        return

    prototypes = proto_bank.prototypes.cpu().numpy()
    n_cls, K, _ = prototypes.shape

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Prototype Similarity Visualization", fontsize=14, fontweight='bold')

    cls_centers = prototypes.mean(axis=1)
    cls_centers_norm = cls_centers / (np.linalg.norm(cls_centers, axis=-1, keepdims=True) + 1e-8)
    inter_cls_sim = np.matmul(cls_centers_norm, cls_centers_norm.T)

    ax = axes[0]
    im = ax.imshow(inter_cls_sim, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels([n[:6] for n in CLS_14], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([n[:6] for n in CLS_14], fontsize=8)
    ax.set_title('Inter-Class Prototype Similarity')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    intra_sims = []
    for c in range(n_cls):
        p_c = prototypes[c]
        p_c_norm = p_c / (np.linalg.norm(p_c, axis=-1, keepdims=True) + 1e-8)
        sim_matrix = np.matmul(p_c_norm, p_c_norm.T)
        sims = []
        for i in range(K):
            for j in range(i+1, K):
                sims.append(sim_matrix[i, j])
        intra_sims.append(sims if sims else [0])

    ax.boxplot(intra_sims, positions=range(n_cls), widths=0.6)
    ax.set_xticks(range(n_cls))
    ax.set_xticklabels([n[:6] for n in CLS_14], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Intra-Class Prototype Diversity')
    all_sims = [s for sims in intra_sims for s in sims]
    ax.axhline(y=np.mean(all_sims), color='red', linestyle='--', alpha=0.7,
               label=f'Mean: {np.mean(all_sims):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / "viz_prototypes.png", dpi=150, annotated_inches='tight')
    plt.close()
    print(f"[OK] Prototype similarity saved: {save_dir / 'viz_prototypes.png'}")


# ============================================================================
# 4. Grad-CAM Attention Heatmap
# ============================================================================

def plot_attention_heatmap(model, proto_bank, val_df, config, n_samples=6, save_dir=None):
    """使用Grad-CAM可视化注意力热力图"""
    if save_dir is None:
        save_dir = output_dir

    try:
        import cv2
    except ImportError:
        print("[Warning] OpenCV not available, skipping attention heatmap")
        return

    from data import get_transforms, ChestXrayDataset

    print("\n[Info] Generating Grad-CAM heatmaps...")

    val_transform = get_transforms(config["model"]["img_size"], is_train=False)
    dataset = ChestXrayDataset(val_df, config["paths"]["image_root"], val_transform)

    selected_samples = []
    seen_classes = set()

    for idx in range(len(dataset)):
        _, labels = dataset[idx]
        label_names = [CLS_14[i] for i in range(14) if labels[i] > 0.5]
        for cls_name in label_names:
            if cls_name not in seen_classes:
                selected_samples.append((idx, cls_name))
                seen_classes.add(cls_name)
                break
        if len(selected_samples) >= n_samples:
            break

    if len(selected_samples) == 0:
        print("[Warning] No positive samples found for attention visualization")
        return

    model.eval()

    try:
        target_layer = model.backbone.layers[-1].blocks[-1].drop_path
    except:
        try:
            target_layer = model.backbone.layers[-1]
        except:
            target_layer = model.backbone

    print(f"[Info] Using Grad-CAM target layer: {target_layer.__class__.__name__}")

    fig, axes = plt.subplots(len(selected_samples), 4, figsize=(16, 4 * len(selected_samples)))
    if len(selected_samples) == 1:
        axes = axes.reshape(1, -1)

    for row, (sample_idx, cls_name) in enumerate(selected_samples):
        image, labels = dataset[sample_idx]
        image_tensor = image.unsqueeze(0).to(device)
        image_tensor.requires_grad_(True)

        with torch.no_grad():
            pooled, proto_feat, logits, stats, _ = model(image_tensor, proto_bank=proto_bank)
            pred_class = logits.argmax(dim=1).item()
            pred_label = CLS_14[pred_class]
            confidence = torch.sigmoid(logits).max().item()

        activations = {}
        gradients = {}

        def forward_hook(module, input, output):
            activations['value'] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            gradients['value'] = grad_output[0].detach()

        handle1 = target_layer.register_forward_hook(forward_hook)
        handle2 = target_layer.register_full_backward_hook(backward_hook)

        output = model(image_tensor, proto_bank=proto_bank)[2]
        model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, pred_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        handle1.remove()
        handle2.remove()

        if 'value' in activations and 'value' in gradients:
            act = activations['value']
            grad = gradients['value']

            if act.dim() == 4:
                weights = torch.mean(grad, dim=[2, 3], keepdim=True)
                cam = torch.sum(weights * act, dim=1).squeeze()
            elif act.dim() == 3:
                weights = torch.mean(grad, dim=[1, 2], keepdim=True)
                cam = torch.sum(weights * act, dim=2).squeeze()
            elif act.dim() == 2:
                weights = grad.mean(dim=0, keepdim=True)
                cam = (weights * act).sum(dim=1).squeeze()
            else:
                cam = torch.ones(act.shape[0], act.shape[-1]).to(act.device) if act.dim() > 1 else torch.ones(act.shape[0]).to(act.device)
                for b in range(act.shape[0]):
                    cam[b] = (grad[b] * act[b]).sum()

            cam = torch.clamp(cam, min=0)
            cam = cam.cpu().numpy()
            if not isinstance(cam, np.ndarray) or cam.ndim == 0:
                cam = np.atleast_1d(np.array(cam.item() if hasattr(cam, 'item') else cam))
            elif cam.size == 1:
                cam = np.array([cam.item()])
            if cam.max() > 0:
                cam = cam / cam.max()
        else:
            cam = None

        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        ax_img = axes[row, 0]
        ax_img.imshow(img_np, cmap='gray')
        ax_img.set_title(f'Original\n{cls_name}', fontsize=10)
        ax_img.axis('off')

        ax_cam = axes[row, 1]
        if cam is not None and cam.size > 1:
            heatmap = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
            ax_cam.imshow(img_np, cmap='gray')
            ax_cam.imshow(heatmap, cmap='jet', alpha=0.7)
        else:
            ax_cam.text(0.5, 0.5, 'CAM unavailable', ha='center', va='center', transform=ax_cam.transAxes)
        ax_cam.set_title(f'Grad-CAM\nPred: {pred_label} ({confidence:.2f})', fontsize=10)
        ax_cam.axis('off')

        ax_overlay = axes[row, 2]
        if cam is not None and cam.size > 1:
            heatmap = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            img_uint8 = np.uint8(255 * img_np)
            overlay = cv2.addWeighted(img_uint8, 0.5, heatmap_color, 0.5, 0)
            ax_overlay.imshow(overlay)
        else:
            ax_overlay.imshow(img_np, cmap='gray')
        ax_overlay.set_title(f'Overlay\nPred: {pred_label}', fontsize=10)
        ax_overlay.axis('off')

        ax_heat = axes[row, 3]
        if cam is not None and cam.size > 1:
            heatmap = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
            ax_heat.imshow(heatmap, cmap='jet')
        else:
            ax_heat.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_heat.transAxes)
        ax_heat.set_title('Heatmap Only', fontsize=10)
        ax_heat.axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / "viz_attention_heatmap.png", dpi=150, annotated_inches='tight')
    plt.close()
    print(f"[OK] Grad-CAM heatmap saved: {save_dir / 'viz_attention_heatmap.png'}")


# ============================================================================
# 5. Proto Attention Heatmaps (20 samples)
# ============================================================================

def plot_proto_attention_heatmaps(model, proto_bank, val_df, config, n_samples=20, save_dir=None):
    """原型相似度热力图可视化（20张），每张4子图：原图 + Top-3类热力图"""
    if save_dir is None:
        save_dir = output_dir

    try:
        import cv2
    except ImportError:
        print("[Warning] OpenCV not available, skipping proto attention heatmap")
        return

    from data import get_transforms, ChestXrayDataset

    print(f"\n[Info] Generating Proto Attention Heatmaps ({n_samples} samples)...")

    val_transform = get_transforms(config["model"]["img_size"], is_train=False)
    dataset = ChestXrayDataset(val_df, config["paths"]["image_root"], val_transform)

    positive_indices = [idx for idx in range(len(dataset)) if dataset[idx][1].sum() > 0]

    if len(positive_indices) < n_samples:
        selected_indices = positive_indices
    else:
        np.random.seed(42)
        selected_indices = np.random.choice(positive_indices, n_samples, replace=False).tolist()

    if len(selected_indices) == 0:
        print("[Warning] No positive samples found for proto attention visualization")
        return

    model.eval()

    heatmap_dir = save_dir / "proto_attention_heatmaps"
    heatmap_dir.mkdir(exist_ok=True)

    for i, sample_idx in enumerate(tqdm(selected_indices, desc="Generating proto attention heatmaps")):
        image, labels = dataset[sample_idx]
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pooled, proto_feat, logits, pga_stats, viz_params = model(
                image_tensor, proto_bank=proto_bank, return_viz_params=True
            )
            pred_probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_class = logits.argmax(dim=1).item()
            pred_label = CLS_14[pred_class]

        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        H, W = img_np.shape[:2]

        similarity_map_from_model = viz_params.get("similarity_map")
        spatial_attention = viz_params.get("spatial_attention")

        if similarity_map_from_model is None:
            prototypes = proto_bank.prototypes.cpu().numpy()
            feat = viz_params["spatial_features"].squeeze().cpu().numpy()
            sim = np.einsum('dhw,ckd->ckhw', feat, prototypes)
            sim_max = sim.max(axis=1)
            for c in range(sim_max.shape[0]):
                if sim_max[c].max() > sim_max[c].min():
                    sim_max[c] = (sim_max[c] - sim_max[c].min()) / (sim_max[c].max() - sim_max[c].min() + 1e-8)
            sim_max = np.transpose(sim_max, (1, 2, 0))
        else:
            sim_raw = similarity_map_from_model.squeeze().cpu().numpy()
            sim_max = np.zeros_like(sim_raw)
            for c in range(sim_raw.shape[-1]):
                slice_c = sim_raw[..., c]
                if slice_c.max() > slice_c.min():
                    sim_max[..., c] = (slice_c - slice_c.min()) / (slice_c.max() - slice_c.min() + 1e-8)
                else:
                    sim_max[..., c] = slice_c

        top_k = 3
        top_classes = np.argsort(pred_probs)[-top_k:][::-1]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(img_np, cmap='gray')
        active_labels = [CLS_14[j] for j in range(14) if labels[j] > 0.5]
        axes[0].set_title(f'Original\nGT: {", ".join(active_labels[:3]) if active_labels else "None"}', fontsize=9)
        axes[0].axis('off')

        for k, cls_idx in enumerate(top_classes[:3]):
            sim_cls = sim_max[:, :, cls_idx]
            sim_cls_resized = cv2.resize(sim_cls, (W, H))

            axes[k + 1].imshow(img_np, cmap='gray')
            im = axes[k + 1].imshow(sim_cls_resized, cmap='hot', alpha=0.8)
            axes[k + 1].set_title(f'{CLS_14[cls_idx][:8]}\nProb: {pred_probs[cls_idx]:.3f}', fontsize=9)
            axes[k + 1].axis('off')
            plt.colorbar(im, ax=axes[k + 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(heatmap_dir / f"proto_attn_{i:03d}.png", dpi=120, annotated_inches='tight')
        plt.close()

    print(f"[OK] Proto attention heatmaps saved: {heatmap_dir}")
    return heatmap_dir


# ============================================================================
# 6. Lesion Bounding Boxes (20 samples)
# ============================================================================

def _draw_labelled_boxes(img_in, sim_cls_map, cls_idx, pred_probs, threshold_ratio,
                         min_area, W, H, font_scale=0.5, thickness=2):
    """在图像上绘制带标签的annotated，标签显示疾病名和概率，文本不越界"""
    import cv2

    img_out = img_in.copy()
    annotatedes = _compute_annotatedes(sim_cls_map, threshold_ratio=threshold_ratio, min_area=min_area)
    color_bgr = _annotated_COLORS_BGR[cls_idx]
    color_rgb = _annotated_COLORS_RGB[cls_idx]
    cls_name = CLS_14[cls_idx]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for rank, (x, y, bw, bh, region_conf) in enumerate(annotatedes[:6]):
        cv2.rectangle(img_out, (x, y), (x + bw, y + bh), color_bgr, thickness)
        label = f"{cls_name[:7]}_{pred_probs[cls_idx]:.2f}"
        conf_label = f"conf:{region_conf:.2f}"

        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        (cw, ch), _ = cv2.getTextSize(conf_label, font, font_scale * 0.82, 1)
        bg_w = max(tw, cw) + 4
        bg_h = th + ch + 5

        tx, ty = x, y - 2
        if x + bg_w > W:
            tx = x + bw - bg_w
        if ty - bg_h < 0:
            ty = y + bh
        tx, ty = _clamp_text_pos(tx, ty, bg_w, bg_h, W, H)

        overlay = img_out.copy()
        cv2.rectangle(overlay, (tx, ty - bg_h), (tx + bg_w, ty + 1), color_bgr, -1)
        cv2.addWeighted(overlay, 0.38, img_out, 0.62, 0, img_out)
        cv2.rectangle(img_out, (tx, ty - bg_h), (tx + bg_w, ty + 1), color_bgr, 1)
        cv2.putText(img_out, label, (tx + 2, ty - 1), font, font_scale, (255, 255, 255), 1)
        cv2.putText(img_out, conf_label, (tx + 2, ty + th - 2),
                    font, font_scale * 0.82, color_rgb, 1)

    return img_out, len(annotatedes)


def plot_lesion_bounding_boxes(model, proto_bank, val_df, config, n_samples=20, save_dir=None):
    """病灶定位框可视化（20张），基于原型相似度热力图，每类不同颜色"""
    if save_dir is None:
        save_dir = output_dir

    try:
        import cv2
    except ImportError:
        print("[Warning] OpenCV not available, skipping lesion bounding boxes")
        return

    from data import get_transforms, ChestXrayDataset

    print(f"\n[Info] Generating Lesion Bounding Boxes ({n_samples} samples)...")

    val_transform = get_transforms(config["model"]["img_size"], is_train=False)
    dataset = ChestXrayDataset(val_df, config["paths"]["image_root"], val_transform)

    positive_indices = [idx for idx in range(len(dataset)) if dataset[idx][1].sum() > 0]

    if len(positive_indices) < n_samples:
        selected_indices = positive_indices
    else:
        np.random.seed(123)
        selected_indices = np.random.choice(positive_indices, n_samples, replace=False).tolist()

    if len(selected_indices) == 0:
        print("[Warning] No positive samples found for lesion annotated visualization")
        return

    model.eval()

    annotated_dir = save_dir / "lesion_bounding_boxes"
    annotated_dir.mkdir(exist_ok=True)

    for i, sample_idx in enumerate(tqdm(selected_indices, desc="Generating lesion annotatedes")):
        image, labels = dataset[sample_idx]
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pooled, proto_feat, logits, pga_stats, viz_params = model(
                image_tensor, proto_bank=proto_bank, return_viz_params=True
            )
            pred_probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_class = logits.argmax(dim=1).item()
            pred_label = CLS_14[pred_class]

        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        H, W = img_np.shape[:2]
        img_uint8 = np.uint8(255 * img_np)
        if img_uint8.ndim == 2:
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        elif img_uint8.shape[-1] == 3:
            img_rgb = img_uint8.copy()
        elif img_uint8.shape[-1] == 1:
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

        sim_map = viz_params.get("similarity_map")
        if sim_map is None:
            prototypes = proto_bank.prototypes.cpu().numpy()
            feat = viz_params["spatial_features"].squeeze().cpu().numpy()
            sim = np.einsum('dhw,ckd->ckhw', feat, prototypes)
            sim_max = sim.max(axis=1)
            for c in range(sim_max.shape[0]):
                if sim_max[c].max() > sim_max[c].min():
                    sim_max[c] = (sim_max[c] - sim_max[c].min()) / (sim_max[c].max() - sim_max[c].min() + 1e-8)
            sim_max = np.transpose(sim_max, (1, 2, 0))
        else:
            sim_raw = sim_map.squeeze().cpu().numpy()
            sim_max = np.zeros_like(sim_raw)
            for c in range(sim_raw.shape[-1]):
                sc = sim_raw[..., c]
                if sc.max() > sc.min():
                    sim_max[..., c] = (sc - sc.min()) / (sc.max() - sc.min() + 1e-8)
                else:
                    sim_max[..., c] = sc

        top_classes = np.argsort(pred_probs)[-3:][::-1]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        axes[0].imshow(img_np, cmap='gray')
        active_labels = [f"{CLS_14[j]}" for j in range(14) if labels[j] > 0.5]
        axes[0].set_title(f'Original\nGT: {", ".join(active_labels[:4]) if active_labels else "None"}', fontsize=9)
        axes[0].axis('off')

        # 预测概率柱状图
        ax_prob = axes[1]
        top_n = min(7, 14)
        top_indices = np.argsort(pred_probs)[-top_n:][::-1]
        probs_to_plot = pred_probs[top_indices]
        names_to_plot = [CLS_14[idx][:6] for idx in top_indices]
        bar_colors = [_annotated_COLORS_RGB[idx] for idx in top_indices]
        ax_prob.barh(range(len(probs_to_plot)), probs_to_plot, color=bar_colors,
                     alpha=0.85, edgecolor='white', linewidth=0.5)
        ax_prob.set_yticks(range(len(probs_to_plot)))
        ax_prob.set_yticklabels(names_to_plot, fontsize=8)
        ax_prob.set_xlabel('Probability')
        ax_prob.set_title(f'Prediction\nPred: {pred_label} ({pred_probs[pred_class]:.3f})', fontsize=9)
        ax_prob.set_xlim([0, 1])
        ax_prob.invert_yaxis()
        ax_prob.grid(True, alpha=0.3, axis='x')

        # Top-3 综合annotated
        top_sim = np.mean([sim_max[:, :, c] for c in top_classes[:3]], axis=0)
        top_sim_resized = cv2.resize(top_sim, (W, H))
        img_top3, n_top3 = _draw_labelled_boxes(
            img_rgb, top_sim_resized, top_classes[0], pred_probs,
            threshold_ratio=0.4, min_area=200, W=W, H=H
        )
        axes[2].imshow(img_top3)
        axes[2].set_title(f'Top-3 Sim + Multi-Class Boxes\n({n_top3} regions)', fontsize=9)
        axes[2].axis('off')

        # Top-3 热力图
        axes[3].imshow(top_sim_resized, cmap='jet')
        axes[3].set_title('Top-3 Sim Heatmap', fontsize=9)
        axes[3].axis('off')

        # 每类独立annotated (3个Top类)
        for k, cls_idx in enumerate(top_classes[:3]):
            ax = axes[4 + k]
            sim_cls = sim_max[:, :, cls_idx]
            sim_cls_resized = cv2.resize(sim_cls, (W, H))
            img_cls, n_boxes = _draw_labelled_boxes(
                img_rgb, sim_cls_resized, cls_idx, pred_probs,
                threshold_ratio=0.5, min_area=150, W=W, H=H
            )
            ax.imshow(img_cls)
            gt_marker = '*' if labels[cls_idx] > 0.5 else ''
            ax.set_title(
                f'{CLS_14[cls_idx][:8]}{gt_marker} {pred_probs[cls_idx]:.2f}\n({n_boxes} regions)',
                fontsize=9, color=_annotated_COLORS_RGB[cls_idx]
            )
            ax.axis('off')

        # Top-1 热力图
        axes[7].imshow(sim_cls_resized, cmap='hot')
        axes[7].set_title(f'Top: {CLS_14[top_classes[0]][:8]}', fontsize=9)
        axes[7].axis('off')

        plt.tight_layout()
        plt.savefig(annotated_dir / f"lesion_annotated_{i:03d}.png", dpi=120, annotated_inches='tight')
        plt.close()

    print(f"[OK] Lesion bounding boxes saved: {annotated_dir}")
    return annotated_dir


# ============================================================================
# 7. Paper-style annotated Summary Grid (5x4 = 20 samples)
# ============================================================================

def plot_annotated_summary_grid(model, proto_bank, config,
                           threshold_json_path=None,
                           n_samples=20,
                           save_dir=None):
    """
    Paper-style annotated summary grid (5x4 = 20 samples)
    - Top-1 class only
    - Largest annotated only
    - Uses optimal thresholds (if provided)
    - Heatmap overlay
    """
    if save_dir is None:
        save_dir = output_dir

    try:
        import cv2
    except ImportError:
        print("[Warning] OpenCV not available")
        return

    from data import get_transforms, ChestXrayDataset

    print(f"\n[Info] Generating Paper-style annotated Grid ({n_samples} samples)...")

    # Load optimal thresholds if available
    threshold_array = np.ones(14) * 0.5
    if threshold_json_path and Path(threshold_json_path).exists():
        with open(threshold_json_path, 'r') as f:
            optimal_thresholds = json.load(f)
        threshold_array = np.array([
            optimal_thresholds.get(cls_name, 0.5)
            for cls_name in CLS_14
        ])

    val_transform = get_transforms(config["model"]["img_size"], is_train=False)
    _, val_df, _ = prepare_data_splits(config)
    dataset = ChestXrayDataset(val_df, config["paths"]["image_root"], val_transform)

    positive_indices = [idx for idx in range(len(dataset)) if dataset[idx][1].sum() > 0]

    if len(positive_indices) < n_samples:
        selected_indices = positive_indices
    else:
        np.random.seed(42)
        selected_indices = np.random.choice(positive_indices, n_samples, replace=False).tolist()

    model.eval()

    save_dir = Path(save_dir) / "paper_annotated_summary"
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    axes = axes.flatten()

    for i, sample_idx in enumerate(tqdm(selected_indices, desc="Paper-style annotated grid")):
        ax = axes[i]

        image, labels = dataset[sample_idx]
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pooled, proto_feat, logits, pga_stats, viz_params = model(
                image_tensor, proto_bank=proto_bank, return_viz_params=True
            )
            pred_probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        H, W = img_np.shape[:2]
        img_uint8 = np.uint8(255 * img_np)

        if img_uint8.shape[-1] == 1:
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_uint8.copy()

        sim_map = viz_params.get("similarity_map")

        if sim_map is None:
            prototypes = proto_bank.prototypes.cpu().numpy()
            feat = viz_params["spatial_features"].squeeze().cpu().numpy()
            sim = np.einsum('dhw,ckd->ckhw', feat, prototypes)
            sim_max = sim.max(axis=1)
            for c in range(sim_max.shape[0]):
                if sim_max[c].max() > sim_max[c].min():
                    sim_max[c] = (sim_max[c] - sim_max[c].min()) / (sim_max[c].max() - sim_max[c].min() + 1e-8)
            sim_max = np.transpose(sim_max, (1, 2, 0))
        else:
            sim_raw = sim_map.squeeze().cpu().numpy()
            sim_max = np.zeros_like(sim_raw)
            for c in range(sim_raw.shape[-1]):
                sc = sim_raw[..., c]
                if sc.max() > sc.min():
                    sim_max[..., c] = (sc - sc.min()) / (sc.max() - sc.min() + 1e-8)
                else:
                    sim_max[..., c] = sc

        # Apply thresholds
        valid_classes = [c for c in range(14) if pred_probs[c] >= threshold_array[c]]
        if len(valid_classes) == 0:
            top1_class = np.argmax(pred_probs)
        else:
            top1_class = valid_classes[np.argmax(pred_probs[valid_classes])]

        sim_cls = sim_max[:, :, top1_class]
        sim_cls = cv2.resize(sim_cls, (W, H))
        sim_cls = cv2.GaussianBlur(sim_cls, (11, 11), 0)
        sim_cls = (sim_cls - sim_cls.min()) / (sim_cls.max() - sim_cls.min() + 1e-8)

        # Binary mask & morphology
        binary = (sim_cls > 0.7).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_draw = img_rgb.copy()

        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area >= 800:
                x, y, bw, bh = cv2.boundingRect(largest)
                color = _annotated_COLORS_BGR[top1_class]
                cv2.rectangle(img_draw, (x, y), (x + bw, y + bh), color, 3)

        heatmap_uint8 = np.uint8(255 * sim_cls)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_draw, 0.72, heatmap_color, 0.28, 0)

        ax.imshow(overlay)
        ax.set_title(f"Top1: {CLS_14[top1_class]} ({pred_probs[top1_class]:.2f})", fontsize=9)
        ax.axis('off')

    for j in range(len(selected_indices), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    save_path = save_dir / "paper_annotated_summary_grid.png"
    plt.savefig(save_path, dpi=300, annotated_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")
    return save_dir


# ============================================================================
# 8. Top-3 Sim Heatmap Summary Grid (5x4 = 20 samples)
# ============================================================================

def plot_heatmap_summary_grid(model, proto_bank, config, n_samples=20, save_dir=None):
    """5x4 热力图汇总图：20个样本，每个子图显示Top-3 Sim热力图（均值），jet colormap"""
    if save_dir is None:
        save_dir = output_dir

    try:
        import cv2
    except ImportError:
        print("[Warning] OpenCV not available, skipping heatmap summary grid")
        return

    from data import get_transforms, ChestXrayDataset

    print(f"\n[Info] Generating Heatmap Summary Grid ({n_samples} samples)...")

    val_transform = get_transforms(config["model"]["img_size"], is_train=False)
    _, val_df, _ = prepare_data_splits(config)
    dataset = ChestXrayDataset(val_df, config["paths"]["image_root"], val_transform)

    positive_indices = [idx for idx in range(len(dataset)) if dataset[idx][1].sum() > 0]
    if len(positive_indices) < n_samples:
        selected_indices = positive_indices
    else:
        np.random.seed(42)
        selected_indices = np.random.choice(positive_indices, n_samples, replace=False).tolist()

    if len(selected_indices) == 0:
        print("[Warning] No positive samples found for heatmap summary grid")
        return

    model.eval()
    save_dir = Path(save_dir) / "heatmap_summary"
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.suptitle("Top-3 Sim Heatmap Summary (mean of Top-1, Top-2, Top-3 per sample)",
                 fontsize=14, fontweight='bold')

    im_ref = None

    for i, sample_idx in enumerate(tqdm(selected_indices, desc="Heatmap summary grid")):
        row, col = i // 4, i % 4
        ax = axes[row, col]

        image, labels = dataset[sample_idx]
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pooled, proto_feat, logits, pga_stats, viz_params = model(
                image_tensor, proto_bank=proto_bank, return_viz_params=True
            )
            pred_probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        H, W = img_np.shape[:2]

        sim_map = viz_params.get("similarity_map")
        if sim_map is None:
            prototypes = proto_bank.prototypes.cpu().numpy()
            feat = viz_params["spatial_features"].squeeze().cpu().numpy()
            sim = np.einsum('dhw,ckd->ckhw', feat, prototypes)
            sim_max = sim.max(axis=1)
            for c in range(sim_max.shape[0]):
                if sim_max[c].max() > sim_max[c].min():
                    sim_max[c] = (sim_max[c] - sim_max[c].min()) / (sim_max[c].max() - sim_max[c].min() + 1e-8)
            sim_max = np.transpose(sim_max, (1, 2, 0))
        else:
            sim_raw = sim_map.squeeze().cpu().numpy()
            sim_max = np.zeros_like(sim_raw)
            for c in range(sim_raw.shape[-1]):
                sc = sim_raw[..., c]
                if sc.max() > sc.min():
                    sim_max[..., c] = (sc - sc.min()) / (sc.max() - sc.min() + 1e-8)
                else:
                    sim_max[..., c] = sc

        top_classes = np.argsort(pred_probs)[-3:][::-1]
        top_sim = np.mean([sim_max[:, :, c] for c in top_classes[:3]], axis=0)
        top_sim_resized = cv2.resize(top_sim, (W, H))

        ax.imshow(img_np, cmap='gray')
        im = ax.imshow(top_sim_resized, cmap='jet', alpha=0.3)
        if im_ref is None:
            im_ref = im
        ax.set_title(f"#{i+1} Top: {CLS_14[top_classes[0]][:8]} / "
                     f"{CLS_14[top_classes[1]][:8]} / {CLS_14[top_classes[2]][:8]}",
                     fontsize=7)
        ax.axis('off')

    if im_ref is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
        fig.colorbar(im_ref, cax=cbar_ax, label='Top-3 Sim (normalized)')

    plt.tight_layout(rect=[0, 0, 0.88, 0.97])
    save_path = save_dir / "heatmap_summary_grid.png"
    plt.savefig(save_path, dpi=300, annotated_inches='tight')
    plt.close()
    print(f"[OK] Heatmap summary grid saved: {save_path}")
    return save_dir


# ============================================================================
# 9. Per-Class Localization Grid (14 diseases)
# ============================================================================

def plot_per_class_localization_grid(model, proto_bank, config,
                                      threshold_json_path=None,
                                      save_dir=None):
    """
    14类疾病定位可视化 (2x7布局)
    每类一张代表性样本，使用最优阈值，Top-1 annotated仅最大连通区域
    """
    if save_dir is None:
        save_dir = output_dir

    try:
        import cv2
    except ImportError:
        print("[Warning] OpenCV not installed")
        return

    print("\n[Info] Generating Per-Class Localization Grid...")

    threshold_array = np.ones(14) * 0.5
    if threshold_json_path and Path(threshold_json_path).exists():
        with open(threshold_json_path, 'r') as f:
            optimal_thresholds = json.load(f)
        threshold_array = np.array([
            optimal_thresholds.get(cls_name, 0.5)
            for cls_name in CLS_14
        ])

    val_transform = get_transforms(config["model"]["img_size"], is_train=False)
    _, val_df, _ = prepare_data_splits(config)
    dataset = ChestXrayDataset(val_df, config["paths"]["image_root"], val_transform)
    model.eval()

    print("[Info] Selecting representative samples...")

    best_samples = {}

    for cls_idx in range(14):
        best_prob = -1
        best_data = None

        for sample_idx in tqdm(range(len(dataset)), desc=f"{CLS_14[cls_idx]}", leave=False):
            image, labels = dataset[sample_idx]

            if labels[cls_idx] < 0.5:
                continue

            image_tensor = image.unsqueeze(0).to(device)

            with torch.no_grad():
                pooled, proto_feat, logits, pga_stats, viz_params = model(
                    image_tensor, proto_bank=proto_bank, return_viz_params=True
                )
                pred_probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            prob = pred_probs[cls_idx]

            if prob < threshold_array[cls_idx]:
                continue

            if prob > best_prob:
                best_prob = prob
                best_data = {
                    "image": image,
                    "labels": labels,
                    "pred_probs": pred_probs,
                    "viz_params": viz_params,
                    "sample_idx": sample_idx
                }

        best_samples[cls_idx] = best_data

    fig, axes = plt.subplots(2, 7, figsize=(24, 8))
    axes = axes.flatten()

    for cls_idx in range(14):
        ax = axes[cls_idx]
        data = best_samples.get(cls_idx)

        if data is None:
            ax.text(0.5, 0.5, f"No valid sample\n{CLS_14[cls_idx]}",
                    ha='center', va='center', fontsize=10)
            ax.axis('off')
            continue

        image = data["image"]
        pred_probs = data["pred_probs"]
        viz_params = data["viz_params"]

        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        H, W = img_np.shape[:2]
        img_uint8 = np.uint8(255 * img_np)

        if img_uint8.ndim == 2:
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        elif img_uint8.shape[-1] == 1:
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_uint8.copy()

        sim_map = viz_params.get("similarity_map")

        if sim_map is None:
            prototypes = proto_bank.prototypes.cpu().numpy()
            feat = viz_params["spatial_features"].squeeze().cpu().numpy()
            sim = np.einsum('dhw,ckd->ckhw', feat, prototypes)
            sim_max = sim.max(axis=1)
            for c in range(sim_max.shape[0]):
                if sim_max[c].max() > sim_max[c].min():
                    sim_max[c] = (sim_max[c] - sim_max[c].min()) / (sim_max[c].max() - sim_max[c].min() + 1e-8)
            sim_max = np.transpose(sim_max, (1, 2, 0))
        else:
            sim_raw = sim_map.squeeze().cpu().numpy()
            sim_max = np.zeros_like(sim_raw)
            for c in range(sim_raw.shape[-1]):
                sc = sim_raw[..., c]
                if sc.max() > sc.min():
                    sim_max[..., c] = (sc - sc.min()) / (sc.max() - sc.min() + 1e-8)
                else:
                    sim_max[..., c] = sc

        sim_cls = sim_max[:, :, cls_idx]
        sim_cls = cv2.resize(sim_cls, (W, H))
        sim_cls = cv2.GaussianBlur(sim_cls, (11, 11), 0)
        sim_cls = (sim_cls - sim_cls.min()) / (sim_cls.max() - sim_cls.min() + 1e-8)

        binary = (sim_cls > 0.7).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_draw = img_rgb.copy()

        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area >= 800:
                x, y, bw, bh = cv2.boundingRect(largest)
                color = _annotated_COLORS_BGR[cls_idx]
                cv2.rectangle(img_draw, (x, y), (x + bw, y + bh), color, 3)

        heatmap_uint8 = np.uint8(255 * sim_cls)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_draw, 0.60, heatmap_color, 0.40, 0)

        ax.imshow(overlay)
        ax.set_title(f"{CLS_14[cls_idx]}\nConf={pred_probs[cls_idx]:.3f}", fontsize=10)
        ax.axis('off')

    plt.tight_layout()

    save_dir = Path(save_dir) / "per_class_localization"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "per_class_localization_grid.png"
    plt.savefig(save_path, dpi=300, annotated_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")
    return save_dir


# ============================================================================
# 10. Class-wise annotated Summary Grid (14 diseases)
# ============================================================================

def plot_classwise_annotated_summary_grid(model, proto_bank, config,
                                      threshold_json_path=None,
                                      save_dir=None):
    """
    14类疾病定位汇总图 (4x4布局)
    每类一张代表性样本，Top-1 annotated，最佳阈值
    """
    if save_dir is None:
        save_dir = output_dir

    try:
        import cv2
    except ImportError:
        print("[Warning] OpenCV not available")
        return

    from data import get_transforms, ChestXrayDataset

    print("\n[Info] Generating classwise annotated summary grid...")

    threshold_dict = {}
    if threshold_json_path and Path(threshold_json_path).exists():
        with open(threshold_json_path, 'r') as f:
            threshold_dict = json.load(f)

    val_transform = get_transforms(config["model"]["img_size"], is_train=False)
    _, val_df, _ = prepare_data_splits(config)
    dataset = ChestXrayDataset(val_df, config["paths"]["image_root"], val_transform)
    model.eval()

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()

    for cls_idx in range(14):
        ax = axes[cls_idx]
        found = False

        for sample_idx in range(len(dataset)):
            image, labels = dataset[sample_idx]

            if labels[cls_idx] < 0.5:
                continue

            image_tensor = image.unsqueeze(0).to(device)

            with torch.no_grad():
                pooled, proto_feat, logits, pga_stats, viz_params = model(
                    image_tensor, proto_bank=proto_bank, return_viz_params=True
                )
                pred_probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            pred_prob = pred_probs[cls_idx]
            threshold = threshold_dict.get(CLS_14[cls_idx], 0.5)

            if pred_prob < threshold:
                continue

            img_np = image.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            H, W = img_np.shape[:2]
            img_uint8 = np.uint8(255 * img_np)

            if img_uint8.shape[-1] == 1:
                img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = img_uint8.copy()

            sim_map = viz_params.get("similarity_map")

            if sim_map is None:
                continue

            sim_raw = sim_map.squeeze().cpu().numpy()
            sim_cls = sim_raw[..., cls_idx]

            if sim_cls.max() > sim_cls.min():
                sim_cls = (sim_cls - sim_cls.min()) / (sim_cls.max() - sim_cls.min() + 1e-8)

            sim_cls = cv2.resize(sim_cls, (W, H))

            annotatedes = _compute_annotatedes(sim_cls, threshold_ratio=0.55, min_area=300)

            if len(annotatedes) > 0:
                annotatedes = sorted(annotatedes, key=lambda x: x[2] * x[3], reverse=True)
                x, y, bw, bh, conf = annotatedes[0]
                color = _annotated_COLORS_BGR[cls_idx]
                cv2.rectangle(img_rgb, (x, y), (x + bw, y + bh), color, 3)

            ax.imshow(img_rgb)
            ax.set_title(f"{CLS_14[cls_idx]}", fontsize=10)
            ax.axis('off')
            found = True
            break

        if not found:
            ax.text(0.5, 0.5, f"No sample\n{CLS_14[cls_idx]}",
                    ha='center', va='center', fontsize=10)
            ax.axis('off')

    for k in range(14, 16):
        axes[k].axis('off')

    plt.tight_layout()

    save_dir = Path(save_dir) / "classwise_annotated_summary"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "classwise_annotated_summary.png"
    plt.savefig(save_path, dpi=300, annotated_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='MaxViT Prototype Learning Evaluation Visualization')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path (default: checkpoint_best.pt)')
    parser.add_argument('--n_samples', type=int, default=2000,
                       help='Number of samples for t-SNE visualization')
    parser.add_argument('--viz', nargs='+', default=['all'],
                       choices=[
                           'all',
                           'tsne',
                           'prototypes',
                           'attention',
                           'proto_attn',
                           'lesion_annotated',
                           'annotated_summary',
                           'heatmap_summary',
                           'per_class',
                           'classwise_annotated_summary',
                       ],
                       help='Visualizations to generate')
    parser.add_argument('--n_heatmaps', type=int, default=6,
                       help='Number of Grad-CAM heatmaps')
    parser.add_argument('--n_proto_attn', type=int, default=20,
                       help='Number of proto attention heatmaps')
    parser.add_argument('--n_lesion_annotated', type=int, default=20,
                       help='Number of lesion annotated visualizations')
    parser.add_argument('--n_summary', type=int, default=20,
                       help='Number of samples in summary grids')
    parser.add_argument('--threshold_json', type=str, default=None,
                       help='Path to optimal thresholds JSON file')
    args = parser.parse_args()

    print("=" * 60)
    print("MaxViT Prototype Learning Evaluation Visualization")
    print("=" * 60)

    generate_all = 'all' in args.viz

    generate_tsne = generate_all or 'tsne' in args.viz
    generate_protos = generate_all or 'prototypes' in args.viz
    generate_attention = generate_all or 'attention' in args.viz
    generate_proto_attn = generate_all or 'proto_attn' in args.viz
    generate_lesion_annotated = generate_all or 'lesion_annotated' in args.viz
    generate_annotated_summary = generate_all or 'annotated_summary' in args.viz
    generate_heatmap_summary = generate_all or 'heatmap_summary' in args.viz
    generate_per_class = generate_all or 'per_class' in args.viz
    generate_classwise_annotated_summary = generate_all or 'classwise_annotated_summary' in args.viz

    # =========================================================================
    # Load Model (shared across all visualizations)
    # =========================================================================

    model = None
    proto_bank = None

    need_model = (
        generate_tsne
        or generate_protos
        or generate_attention
        or generate_proto_attn
        or generate_lesion_annotated
        or generate_annotated_summary
        or generate_heatmap_summary
        or generate_per_class
        or generate_classwise_annotated_summary
    )

    if need_model:
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            checkpoint_path = output_dir / "checkpoint_best.pt"

        if not checkpoint_path.exists():
            print(f"[Error] Checkpoint not found: {checkpoint_path}")
            return

        print(f"\nLoading model: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model, proto_bank = create_model_and_bank(config, device=device)
        model.load_state_dict(checkpoint["model"])

        if "proto_bank" in checkpoint and proto_bank is not None:
            proto_bank.load_state_dict(checkpoint["proto_bank"])
            print("Prototype bank loaded successfully")

        model.eval()
        if proto_bank is not None:
            proto_bank.eval()

    # =========================================================================
    # 1. t-SNE
    # =========================================================================

    if generate_tsne:
        print("Loading validation data...")
        _, val_df, _ = prepare_data_splits(config)
        val_transform = get_transforms(config["model"]["img_size"], is_train=False)
        val_dataset = ChestXrayDataset(val_df, config["paths"]["image_root"], val_transform)

        print("\n[1] Computing features and t-SNE...")
        all_features, all_proto_features, all_labels = extract_features(
            model, val_dataset, proto_bank=proto_bank, n_samples=args.n_samples
        )
        print("\n[2] Plotting t-SNE...")
        plot_tsne(all_features, all_proto_features, all_labels,
                  n_samples=min(args.n_samples, 2000))

    # =========================================================================
    # 2. Prototype Similarity
    # =========================================================================

    if generate_protos:
        print("\n[3] Plotting prototype similarity...")
        plot_prototype_similarity(proto_bank)

    # =========================================================================
    # 3. Grad-CAM
    # =========================================================================

    if generate_attention and model is not None:
        print("\n[4] Generating Grad-CAM attention heatmaps...")
        _, val_df, _ = prepare_data_splits(config)
        plot_attention_heatmap(model, proto_bank, val_df, config, n_samples=args.n_heatmaps)

    # =========================================================================
    # 4. Proto Attention Heatmaps
    # =========================================================================

    if generate_proto_attn and model is not None:
        print("\n[5] Generating proto attention heatmaps...")
        _, val_df, _ = prepare_data_splits(config)
        plot_proto_attention_heatmaps(
            model, proto_bank, val_df, config, n_samples=args.n_proto_attn
        )

    # =========================================================================
    # 5. Lesion Bounding Boxes
    # =========================================================================

    if generate_lesion_annotated and model is not None:
        print("\n[6] Generating lesion bounding box visualizations...")
        _, val_df, _ = prepare_data_splits(config)
        plot_lesion_bounding_boxes(
            model, proto_bank, val_df, config, n_samples=args.n_lesion_annotated
        )

    # =========================================================================
    # 6. Paper-style annotated Summary Grid
    # =========================================================================

    if generate_annotated_summary and model is not None:
        print("\n[7] Generating paper-style annotated summary grid...")
        plot_annotated_summary_grid(
            model, proto_bank, config,
            threshold_json_path=args.threshold_json,
            n_samples=args.n_summary
        )

    # =========================================================================
    # 7. Heatmap Summary Grid
    # =========================================================================

    if generate_heatmap_summary and model is not None:
        print("\n[8] Generating heatmap summary grid...")
        plot_heatmap_summary_grid(
            model, proto_bank, config, n_samples=args.n_summary
        )

    # =========================================================================
    # 8. Per-Class Localization Grid
    # =========================================================================

    if generate_per_class and model is not None:
        print("\n[9] Generating per-class localization grid...")
        plot_per_class_localization_grid(
            model, proto_bank, config,
            threshold_json_path=args.threshold_json
        )

    # =========================================================================
    # 9. Class-wise annotated Summary Grid
    # =========================================================================

    if generate_classwise_annotated_summary and model is not None:
        print("\n[10] Generating classwise annotated summary grid...")
        plot_classwise_annotated_summary_grid(
            model, proto_bank, config,
            threshold_json_path=args.threshold_json
        )

    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
