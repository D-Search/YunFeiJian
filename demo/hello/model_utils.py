"""
模型工具模块 - 兼容层
实际功能已迁移到 apps.ai.model_manager

本文件保留用于向后兼容，新代码请直接使用 apps.ai.model_manager
"""
# 重新导出新模块的函数以保持向后兼容
from apps.ai.model_manager import (
    load_model_config,
    get_checkpoint_path,
    build_model_components,
    load_model_and_bank,
    get_model_and_bank,
    ModelManager,
)
from apps.medical.diseases import CLS_14

__all__ = [
    'load_model_config',
    'get_checkpoint_path', 
    'build_model_components',
    'load_model_and_bank',
    'get_model_and_bank',
    'ModelManager',
    'CLS_14',
]
