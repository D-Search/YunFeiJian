"""
配置管理模块
集中管理所有配置项，支持环境变量覆盖
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class AIConfig:
  """AI模型配置"""
  model_name: str = "maxvit_tiny_tf_224.in1k"
  num_classes: int = 14
  img_size: int = 224
  checkpoint_filename: str = "checkpoint_final.pt"
  
  @property
  def device(self) -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass 
class WeChatConfig:
  """微信配置"""
  app_id: str = "wx1f3563d2ea35fc62"
  secret: str = "91127926cb2c15147c735f677fabcded"


@dataclass
class LLMConfig:
  """LLM服务配置"""
  provider: str = "siliconflow"
  api_key: str = os.getenv("SILICONFLOW_API_KEY", "请在此处或环境变量中替换为全新的有效sk-...")
  api_key: str = "sk-lvlgahqezdshhdibadcvlqpqkrxragunirhjoklpzcjrokwt"
  model: str = "Qwen/Qwen2.5-7B-Instruct"
  api_url: str = "https://api.siliconflow.cn/v1/chat/completions"
  timeout: int = 120


@dataclass
class DatabaseConfig:
  """数据库配置"""
  name: str = "new_schema"
  user: str = "root"
  password: str = "Dong2004921."
  host: str = "127.0.0.1"
  port: int = 3306


@dataclass
class CeleryConfig:
  """Celery配置"""
  broker_url: str = "redis://127.0.0.1:6379/0"
  result_backend: str = "redis://127.0.0.1:6379/1"
  worker_concurrency: int = 1


@dataclass
class MediaConfig:
  """媒体文件配置"""
  url: str = "/media/"
  root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "media")


@dataclass
class AppConfig:
  """应用主配置"""
  ai: AIConfig = field(default_factory=AIConfig)
  wechat: WeChatConfig = field(default_factory=WeChatConfig)
  llm: LLMConfig = field(default_factory=LLMConfig)
  database: DatabaseConfig = field(default_factory=DatabaseConfig)
  celery: CeleryConfig = field(default_factory=CeleryConfig)
  media: MediaConfig = field(default_factory=MediaConfig)
  
  base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
  
  @property
  def model_config_path(self) -> Path:
    return self.base_dir  / "demo" / "third_party" / "maxvit_proto_sim_plus" / "config.yaml"
  
  @property
  def checkpoint_path(self) -> Path:
    return self.base_dir  / "demo" / "third_party" / "maxvit_proto_sim_plus" / "output" / self.ai.checkpoint_filename


# 全局配置实例
config = AppConfig()


def get_config() -> AppConfig:
  """获取配置实例"""
  return config