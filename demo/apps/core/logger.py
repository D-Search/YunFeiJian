"""
日志配置模块
"""
import logging
import os
from pathlib import Path


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    创建配置好的logger
    
    Args:
        name: logger名称
        level: 日志级别
    
    Returns:
        配置好的Logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # 格式化
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger


# 全局logger
logger = setup_logger('apps')
