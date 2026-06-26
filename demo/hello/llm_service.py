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
    LLM_INTEGRATION_GUIDE,
)

# 向后兼容
QwenService = SiliconFlowService
KimiAPIService = None  # 保留类名但功能已整合
LLM_INTEGRATION_GUIDE = ""

def get_qwen_service():
    """获取LLM服务实例（兼容旧接口）"""
    return get_llm_service()

__all__ = [
    'BaseLLMService',
    'SiliconFlowService',
    'DoubaoService', 
    'QwenService',
    'KimiAPIService',
    'LLMServiceFactory',
    'get_llm_service',
    'get_qwen_service',
]
