"""
LLM服务兼容层
实际功能已迁移到 apps.ai.services.llm_service

本文件保留用于向后兼容
推荐使用新的 apps.ai.services.llm_service
"""
from apps.ai.services.llm_service import (
    BaseLLMService,
    SiliconFlowService,
    DoubaoService,
    LLMServiceFactory,
    get_llm_service,
    reset_llm_service,
)

# 保留原有的便捷函数接口
QwenService = SiliconFlowService

def get_qwen_service():
    """获取Qwen服务实例（兼容旧接口）"""
    return get_llm_service()

__all__ = [
    'BaseLLMService',
    'SiliconFlowService', 
    'DoubaoService',
    'QwenService',
    'LLMServiceFactory',
    'get_llm_service',
    'get_qwen_service',
    'reset_llm_service',
]
