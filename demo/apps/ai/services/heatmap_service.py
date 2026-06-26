"""
热力图生成服务
基于MaxViT原型网络生成疾病热力图和标注图像
"""
import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from django.utils import timezone
from typing import Dict, Any, Optional, Tuple

from apps.ai.model_manager import get_model_manager
from apps.ai.image_processor import default_preprocessor
from apps.ai.thresholds import CLS_14_NAMES_EN
from apps.ai.thresholds import CLASS_NAME_MAP
from apps.ai.thresholds import OPTIMAL_THRESHOLDS


# 14种疾病对应的标注颜色 (BGR格式)
ANNOTATED_COLORS_BGR = {
    0: (255, 0, 0),      # 肺不张 - 蓝色
    1: (0, 0, 255),      # 心脏肥大 - 红色
    2: (0, 255, 255),    # 胸腔积液 - 黄色
    3: (0, 255, 0),      # 浸润 - 绿色
    4: (255, 0, 255),    # 肿块 - 紫色
    5: (255, 128, 0),    # 结节 - 橙色
    6: (128, 0, 255),    # 肺炎 - 粉紫色
    7: (0, 128, 255),    # 气胸 - 浅蓝色
    8: (255, 255, 0),    # 实变 - 青蓝色
    9: (255, 0, 128),    # 肺水肿 - 粉红色
    10: (128, 255, 0),   # 肺气肿 - 黄绿色
    11: (0, 128, 128),   # 肺纤维化 - 墨绿色
    12: (128, 0, 128),    # 胸膜增厚 - 深紫色
    13: (0, 128, 0),     # 疝 - 深绿色
}


class HeatmapGenerator:
    """
    基于MaxViT原型网络的热力图生成器
    
    功能：
    1. 生成疾病热力图
    2. 生成带标注的图像（病灶框）
    3. 支持多种可视化参数配置
    """
    
    def __init__(self, model=None, proto_bank=None, config_path: str = None):
        """
        初始化热力图生成器
        
        Args:
            model: MaxViT模型实例（可选，默认从ModelManager获取）
            proto_bank: 原型库实例（可选）
            config_path: 配置文件路径
        """
        self.model = model
        self.proto_bank = proto_bank
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载配置
        if config_path is None:
            base_dir = "/www/WXproject/demo"
            config_path = os.path.join(base_dir, "third_party", "maxvit_proto_sim_plus", "config.yaml")
        
        with open(config_path, encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.img_size = self.config.get("model", {}).get("img_size", 224)
        self.n_cls = self.config.get("model", {}).get("num_classes", 14)
    
    def _ensure_model_loaded(self):
        """确保模型已加载"""
        if self.model is None or self.proto_bank is None:
            manager = get_model_manager()
            self.model, self.proto_bank = manager.load_if_needed()
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image: PIL Image对象
        
        Returns:
            预处理后的张量 [3, H, W]
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_np = np.array(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        img_np = (img_np - mean) / std
        
        img_tensor = torch.from_numpy(img_np.astype(np.float32)).permute(2, 0, 1)
        return F.interpolate(
            img_tensor.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    
    def predict_with_viz(self, image_tensor: torch.Tensor) -> Tuple[np.ndarray, dict]:
        """
        带可视化参数的预测
        
        Args:
            image_tensor: 预处理后的图像张量
        
        Returns:
            (预测概率数组, 可视化参数字典)
        """
        self._ensure_model_loaded()
        
        img_tensor = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, bce_logits, _, viz_params = self.model(
                img_tensor,
                proto_bank=self.proto_bank,
                return_viz_params=True
            )
            pred_probs = torch.sigmoid(bce_logits).squeeze(0).cpu().numpy()
        
        return pred_probs, viz_params
    
    def generate_sim_heatmap(self, viz_params: dict) -> np.ndarray:
        """
        从可视化参数生成热力图
        
        Args:
            viz_params: predict_with_viz返回的可视化参数字典
        
        Returns:
            归一化后的热力图数组 [H, W, C]
        """
        if viz_params is None or viz_params.get("similarity_map") is None:
            raise ValueError("viz_params中缺少similarity_map")
        
        sim_raw = viz_params["similarity_map"].squeeze(0).cpu().numpy()
        
        # 对每个类别进行归一化，保持float32类型
        sim_max = np.zeros(sim_raw.shape, dtype=np.float32)
        for c in range(sim_raw.shape[-1]):
            sc = sim_raw[..., c]
            if sc.max() > sc.min():
                sim_max[..., c] = (sc - sc.min()) / (sc.max() - sc.min())
            else:
                sim_max[..., c] = sc
        
        return sim_max
    
    def compute_bounding_boxes(self, heatmap: np.ndarray,
                               threshold_ratio: float = 0.5,
                               min_area: int = 100) -> list:
        """
        从热力图计算边界框
        
        Args:
            heatmap: 单通道热力图
            threshold_ratio: 阈值比例
            min_area: 最小连通区域面积
        
        Returns:
            边界框列表 [{'x', 'y', 'w', 'h'}, ...]
        """
        attn_norm = heatmap.copy()
        heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
        if heatmap_max > heatmap_min:
            attn_norm = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        
        binary = (attn_norm > (threshold_ratio * attn_norm.max())).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append({'x': x, 'y': y, 'w': w, 'h': h})
        
        return boxes
    
    def render_heatmap(self, heatmap: np.ndarray,
                      target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        渲染热力图为彩色图像
        
        Args:
            heatmap: 热力图数组
            target_size: 目标尺寸 (width, height)
        
        Returns:
            RGB彩色热力图数组
        """
        hm = heatmap.copy()
        if target_size:
            hm = cv2.resize(hm, target_size)
        
        # 归一化到[0, 1]
        hm = np.clip(hm, 0, 1)
        
        # 应用JET颜色映射
        color = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
        return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    
    def render_annotated_image(self, original: np.ndarray,
                              boxes: list,
                              color: Tuple[int, int, int] = (0, 255, 0),
                              thickness: int = 2,
                              label: str = None) -> np.ndarray:
        """
        在原图上绘制边界框
        
        Args:
            original: 原图数组
            boxes: 边界框列表
            color: 框颜色 (BGR)
            thickness: 线条粗细
            label: 标签
        Returns:
            绘制后的图像
        """
        img = original.copy()
        if label:
            x, y, w, h = boxes[0]['x'], boxes[0]['y'], boxes[0]['w'], boxes[0]['h']
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            
            # 标签放在框内顶部中间
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # 水平居中，垂直位置固定在框顶部
            text_x = x + (w - text_w) // 2
            text_y = y + text_h + 5
            
            # 确保不超出图像边界
            text_x = max(0, min(text_x, img.shape[1] - text_w - 1))
            text_y = max(text_h + 1, min(text_y, img.shape[0] - 1))
            
            # 半透明黑色背景
            overlay = img.copy()
            cv2.rectangle(overlay, (text_x - 4, text_y - text_h - 4),
            (text_x + text_w + 4, text_y + 4), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
          
            cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return img
    
    def generate_all(self, image_path: str,
                    report_no: str = None,
                    user_id: str = None,
                    top_k: int = 3,
                    prob_threshold: float = 0.3) -> Dict[str, Any]:
        """
        一键生成热力图和标注图像
        
        Args:
            image_path: 输入图像路径
            report_no: 报告编号（用于文件命名）
            user_id: 用户ID（用于文件命名，防止不同用户间文件名冲突）
            top_k: 最多显示的疾病数量
            prob_threshold: 概率阈值，低于此值不显示
        
        Returns:
            {
                'heatmap_path': 热力图路径,
                'annotated_path': 标注图像路径,
                'top_diseases': [{'name', 'prob', 'boxes'}, ...]
            }
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"找不到输入的原始图像: {image_path}")
            
            # 加载图像（用于后续标注绘制）
            original_image = Image.open(image_path).convert('RGB')
            orig_w, orig_h = original_image.size
            img_array = np.array(original_image)
            
            # 使用与模型训练时一致的预处理，保证概率结果与 detail 接口一致
            image_tensor = default_preprocessor.preprocess(image_path)
            
            # 预测
            pred_probs, viz_params = self.predict_with_viz(image_tensor)
            sim_max = self.generate_sim_heatmap(viz_params)
            
            # 获取Top-K疾病
            top_indices = list(np.argsort(pred_probs)[-top_k:][::-1])
            
            # 生成热力图（取Top-K的平均）
            top_heatmaps = [sim_max[:, :, idx] for idx in top_indices]
            top_sim = np.mean(top_heatmaps, axis=0)
            top_sim_full = cv2.resize(top_sim.astype(np.float32), (orig_w, orig_h))
            
            # 渲染热力图
            heatmap_img = self.render_heatmap(top_sim_full)
            
            # 生成标注图像
            annotated_img = img_array.copy()
            top_diseases = []
            
            # 热力图尺寸（模型输入）
            input_h, input_w = sim_max.shape[:2]  # 例如 224, 224

            for cls_idx in top_indices:
                probs=pred_probs[cls_idx]
                threshold = OPTIMAL_THRESHOLDS.get(CLS_14_NAMES_EN[cls_idx], 0.5)
                if probs > threshold:
                    # 根据宽高比正确映射热力图到原图
                    orig_ratio = orig_w / orig_h
                    input_ratio = input_w / input_h  # 通常是 1.0 (224x224)

                    # 计算热力图在原图上的有效区域
                    # 热力图需要填充到原图完整宽度/高度，导致另一个方向超出
                    if orig_ratio > input_ratio:
                        # 原图更宽：热力图填充到原图宽度，高度方向会超出
                        effective_w = orig_w
                        effective_h = int(effective_w / input_ratio)  # 保持宽高比
                        offset_x = 0
                        offset_y = (effective_h - orig_h) // 2
                    else:
                        # 原图更高：热力图填充到原图高度，宽度方向会超出
                        effective_h = orig_h
                        effective_w = int(effective_h * input_ratio)  # 保持宽高比
                        offset_y = 0
                        offset_x = (effective_w - orig_w) // 2

                    # 先 resize 热力图到有效区域
                    heatmap_aligned = cv2.resize(
                        sim_max[:, :, cls_idx].astype(np.float32),
                        (effective_w, effective_h)
                    )
                    
                   
                    # 计算边界框
                    boxes = self.compute_bounding_boxes(
                        heatmap_aligned,
                        threshold_ratio=0.6,
                        min_area=300
                    )

                    # 将框坐标映射回原图坐标
                    for box in boxes:
                        box['x'] = int((box['x'] - offset_x) * orig_w / effective_w)
                        box['y'] = int((box['y'] - offset_y) * orig_h / effective_h)
                        box['w'] = int(box['w'] * orig_w / effective_w)
                        box['h'] = int(box['h'] * orig_h / effective_h)

                        # 确保框不超出原图边界
                        box['x'] = max(0, min(box['x'], orig_w - 1))
                        box['y'] = max(0, min(box['y'], orig_h - 1))
                        box['w'] = max(1, min(box['w'], orig_w - box['x']))
                        box['h'] = max(1, min(box['h'], orig_h - box['y']))
                    
                    
                    # 获取标签
                    label=f"{CLS_14_NAMES_EN[cls_idx]}: {probs:.2f}"
                    
                    # 获取颜色
                    color_bgr=ANNOTATED_COLORS_BGR.get(cls_idx, (0, 255, 0))
                    
                    # 绘制边界框
                    annotated_img = self.render_annotated_image(
                        annotated_img, boxes, color_bgr, thickness=2,label=label
                    )
                    
                    top_diseases.append({
                        'index': cls_idx,
                        'prob': float(pred_probs[cls_idx]),
                        'boxes': boxes
                    })
            
            # 文件命名加入 user_id，防止不同用户间 report_no 冲突覆盖
            timestamp = timezone.now().strftime('%Y%m%d%H%M%S')
            if user_id:
                prefix = f"{user_id}_{report_no}" if report_no else f"{user_id}_{timestamp}"
            else:
                prefix = report_no or timestamp
            
            heatmap_path = self._save_image(heatmap_img, f"{prefix}_heatmap.png", subdir='heatmaps')
            annotated_path = self._save_image(annotated_img, f"{prefix}_annotated.png", subdir='annotated_images')
            
            return {
                'heatmap_path': heatmap_path,
                'annotated_path': annotated_path,
                'top_diseases': top_diseases,
            }
        
        except Exception as e:
            raise RuntimeError(f"热力图生成失败: {str(e)}")
    
    def _save_image(self, image: np.ndarray, filename: str,subdir: str = 'heatmaps') -> str:
        """
        保存图像到媒体目录
        
        Args:
            image: 图像数组
            filename: 文件名
        
        Returns:
            保存后的相对路径
        """
        from django.conf import settings
        
        # 保存到 heatmaps 目录
        save_dir = os.path.join(settings.MEDIA_ROOT, subdir)
        os.makedirs(save_dir, exist_ok=True)
        
        filepath = os.path.join(save_dir, filename)
        
        # 确保图像是uint8格式
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        return f"{subdir}/{filename}"


# 全局热力图生成器（懒加载）
_heatmap_generator: Optional[HeatmapGenerator] = None


def get_heatmap_generator() -> HeatmapGenerator:
    """获取热力图生成器单例"""
    global _heatmap_generator
    if _heatmap_generator is None:
        _heatmap_generator = HeatmapGenerator()
    return _heatmap_generator
