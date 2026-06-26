"""
热力图服务兼容层
实际功能已迁移到 apps.ai.services.heatmap_service

本文件保留用于向后兼容
推荐使用新的 apps.ai.services.heatmap_service
"""
from apps.ai.services.heatmap_service import (
    HeatmapGenerator,
    get_heatmap_generator,
    ANNOTATED_COLORS_BGR,
)

__all__ = [
    'HeatmapGenerator',
    'get_heatmap_generator',
    'ANNOTATED_COLORS_BGR',
]
