"""
模型管理器
单例模式管理AI模型，懒加载避免Django启动阻塞
"""
import os
import torch
import yaml
from typing import Tuple, Optional
from functools import lru_cache

from apps.core.exceptions import ModelLoadError
from apps.core.logger import logger


class ModelManager:
    """模型管理器（单例）"""
    
    _instance: Optional['ModelManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._model = None
        self._bank = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._config = None
        self._initialized = True
        self._lock = False  # 简单锁标志
        
        logger.info(f"[ModelManager] 初始化完成，设备: {self._device}")
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    def _load_config(self, config_path: str = None) -> dict:
        """加载配置文件"""
        if self._config is not None:
            return self._config
        
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            config_path = os.path.join(base_dir, "demo", "third_party", "maxvit_proto_sim_plus", "config.yaml")
        
        if not os.path.exists(config_path):
            raise ModelLoadError(f"配置文件不存在: {config_path}")
        
        with open(config_path, encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        return self._config
    
    def _get_checkpoint_path(self) -> str:
        """获取模型权重路径"""
        if self._config is None:
            self._load_config()
        
        output_dir = self._config.get("paths", {}).get("output_dir")
        if not output_dir:
            raise ModelLoadError("config.yaml 缺少 paths.output_dir")
        
        checkpoint = os.path.join(output_dir, "checkpoint_final.pt")
        
        # 如果相对路径，转换为绝对路径
        if not os.path.isabs(checkpoint):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            checkpoint = os.path.join(base_dir, output_dir, "checkpoint_final.pt")
        
        return checkpoint
    
    def load_if_needed(self) -> Tuple:
        """
        懒加载模型（线程安全）
        
        Returns:
            (model, bank) 元组
        """
        if self._model is not None and self._bank is not None:
            return self._model, self._bank
        
        if self._lock:
            # 等待模型加载完成
            import time
            while self._lock:
                time.sleep(0.1)
            return self._model, self._bank
        
        self._lock = True
        
        try:
            logger.info("==> [ModelManager] 首次加载AI模型...")
            
            # 加载配置
            config = self._load_config()
            
            # 动态导入避免循环依赖
            from third_party.maxvit_proto_sim_plus.model import (
                MaxViTProtoNet,
                MultiPrototypeBank,
                create_model_and_bank
            )
            
            # 创建模型
            model_cfg = config.get("model", {})
            proto_cfg = config.get("prototype", {})
            
            self._model = MaxViTProtoNet(
                backbone_name=model_cfg.get("backbone", "maxvit_tiny_tf_224.in1k"),
                pretrained=False,
                n_cls=model_cfg.get("num_classes", 14),
                dropout=0.0,
                use_pga=model_cfg.get("use_pga", True)
            ).to(self._device)
            
            # 创建原型库
            self._bank = MultiPrototypeBank(
                n_cls=model_cfg.get("num_classes", 14),
                K=proto_cfg.get("K", 3),
                feat_dim=proto_cfg.get("feat_dim", 256),
                momentum=proto_cfg.get("ema_momentum", 0.99)
            ).to(self._device)
            
            # 加载权重
            checkpoint_path = self._get_checkpoint_path()
            if os.path.exists(checkpoint_path):
                logger.info(f"==> [ModelManager] 加载权重: {checkpoint_path}")
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=self._device,
                    weights_only=False
                )
                
                if "model" in checkpoint:
                    self._model.load_state_dict(checkpoint["model"])
                else:
                    self._model.load_state_dict(checkpoint)
                
                if "bank" in checkpoint and self._bank is not None:
                    self._bank.load_state_dict(checkpoint["bank"])
                
                logger.info("==> [ModelManager] 权重加载成功")
            else:
                logger.warning(f"==> [ModelManager] 未找到权重文件: {checkpoint_path}")
            
            # 设置为评估模式
            self._model.eval()
            if self._bank is not None:
                self._bank.eval()
            
            logger.info("==> [ModelManager] 模型加载完成")
            
        except Exception as e:
            logger.error(f"==> [ModelManager] 模型加载失败: {str(e)}")
            raise ModelLoadError(f"模型加载失败: {str(e)}")
        
        finally:
            self._lock = False
        
        return self._model, self._bank
    
    def get_model(self):
        """获取模型实例"""
        model, _ = self.load_if_needed()
        return model
    
    def get_bank(self):
        """获取原型库实例"""
        _, bank = self.load_if_needed()
        return bank
    
    def reload(self):
        """重新加载模型"""
        self._model = None
        self._bank = None
        self._config = None
        return self.load_if_needed()


# 全局单例
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """获取模型管理器单例"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


@lru_cache(maxsize=1)
def get_model_and_bank() -> Tuple:
    """
    获取模型和原型库（带缓存）
    
    Returns:
        (model, bank) 元组
    """
    manager = get_model_manager()
    return manager.load_if_needed()


# ============================================================
# 兼容层函数（为了向后兼容 hello.model_utils）
# ============================================================

def load_model_config(config_path: str = None) -> dict:
    """
    加载模型配置文件
    
    Args:
        config_path: 配置文件路径，默认为 third_party/maxvit_proto_sim_plus/config.yaml
    
    Returns:
        配置字典
    """
    import yaml
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        config_path = os.path.join(base_dir, "third_party", "maxvit_proto_sim_plus", "config.yaml")
    
    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_checkpoint_path(config: dict) -> str:
    """
    从配置获取checkpoint路径
    
    Args:
        config: 配置字典
    
    Returns:
        checkpoint文件路径
    """
    output_dir = config.get("paths", {}).get("output_dir")
    if not output_dir:
        raise RuntimeError("config.yaml 缺少 paths.output_dir")
    
    checkpoint = os.path.join(output_dir, "checkpoint_final.pt")
    
    if not os.path.isabs(checkpoint):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        checkpoint = os.path.join(base_dir, output_dir, "checkpoint_final.pt")
    
    return checkpoint


def build_model_components(config: dict, device: torch.device):
    """
    从配置构建模型组件
    
    Args:
        config: 配置字典
        device: 计算设备
    
    Returns:
        (model, bank) 元组
    """
    from third_party.maxvit_proto_sim_plus.model import create_model_and_bank
    return create_model_and_bank(config, device=device)


def load_model_and_bank(config: dict, device: torch.device):
    """
    加载模型和权重
    
    Args:
        config: 配置字典
        device: 计算设备
    
    Returns:
        (model, bank) 元组
    """
    model, bank = build_model_components(config, device)
    checkpoint_path = get_checkpoint_path(config)
    
    if os.path.exists(checkpoint_path):
        print(f"--> [model_utils] 正在加载权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        if "bank" in checkpoint and bank is not None:
            bank.load_state_dict(checkpoint["bank"])
        print("--> [model_utils] 权重加载成功")
    else:
        print(f"WARNING: [model_utils] 未找到权重文件 {checkpoint_path}")
    
    model.eval()
    if bank is not None:
        bank.eval()
    return model, bank
