"""
图像预处理器
提供医学影像的标准化预处理流程
"""
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typing import Union, Tuple


class ImagePreprocessor:
    """
    医学影像预处理器
    
    提供标准化的图像预处理流程：
    - RGB格式转换
    - 尺寸调整
    - 归一化（ImageNet标准）
    - 张量转换
    """
    
    # ImageNet 标准化参数
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    def __init__(self, img_size: int = 224):
        """
        初始化预处理器
        
        Args:
            img_size: 目标图像尺寸（正方形）
        """
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
        ])
    
    def load_image(self, source: Union[str, Image.Image]) -> Image.Image:
        """
        加载图像并确保RGB格式
        
        Args:
            source: 图像路径或PIL Image对象
        
        Returns:
            RGB格式的PIL Image
        """
        if isinstance(source, str):
            image = Image.open(source)
        else:
            image = source
        return image.convert('RGB')
    
    def preprocess(self, source: Union[str, Image.Image]) -> torch.Tensor:
        """
        预处理单张图像
        
        Args:
            source: 图像路径或PIL Image对象
        
        Returns:
            预处理后的张量 [3, H, W]
        """
        image = self.load_image(source)
        return self.transform(image)
    
    def preprocess_batch(self, sources: list) -> torch.Tensor:
        """
        批量预处理
        
        Args:
            sources: 图像路径或PIL Image对象列表
        
        Returns:
            批量张量 [B, 3, H, W]
        """
        tensors = [self.preprocess(src).unsqueeze(0) for src in sources]
        return torch.cat(tensors, dim=0)
    
    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        反归一化（用于可视化）
        
        Args:
            tensor: 归一化后的张量
        
        Returns:
            [0, 255]范围的numpy数组
        """
        mean = torch.tensor(self.MEAN).view(3, 1, 1)
        std = torch.tensor(self.STD).view(3, 1, 1)
        
        img = tensor.clone()
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


class HeatmapPreprocessor:
    """
    热力图专用预处理器
    支持numpy数组输入
    """
    
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    
    def __init__(self, img_size: int = 224):
        self.img_size = img_size
    
    def normalize(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        归一化图像到[-1, 1]范围
        
        Args:
            image: RGB图像数组 [H, W, 3]，值范围[0, 255]
        
        Returns:
            (归一化后的数组, 原图数组)
        """
        original = image.copy()
        
        img_float = image.astype(np.float32) / 255.0
        img_norm = (img_float - self.MEAN) / self.STD
        
        return img_norm, original
    
    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        numpy数组转torch张量
        
        Args:
            image: 归一化后的数组
        
        Returns:
            torch.Tensor [3, H, W]
        """
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        
        # 调整尺寸
        import torch.nn.functional as F
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return tensor


# 全局预处理器实例
default_preprocessor = ImagePreprocessor()
heatmap_preprocessor = HeatmapPreprocessor()
