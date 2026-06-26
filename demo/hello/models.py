"""
模型兼容层
为了保持向后兼容，旧的导入路径仍然可用
"""
# 本文件用于保持向后兼容
# 实际模型已迁移到 apps.user.models 和 apps.report.models

# 重新导出新模块的模型
from apps.user.models import UserProfile
from apps.report.models import ReportRecord

__all__ = ['UserProfile', 'ReportRecord']
