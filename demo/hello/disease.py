"""
疾病定义兼容层
为了保持向后兼容，旧的导入路径仍然可用
"""
# 本文件用于保持向后兼容
# 实际疾病定义已迁移到 apps.medical.diseases

from apps.medical.diseases import (
    CLS_14,
    DISEASE_INDEX,
    DISEASE_TRANSLATIONS,
    DISEASE_SEVERITY,
    HIGH_RISK_DISEASES,
    get_disease_name,
    get_disease_index,
    get_english_name,
    is_high_risk,
    filter_high_risk,
    format_disease_summary,
)

__all__ = [
    'CLS_14',
    'DISEASE_INDEX',
    'DISEASE_TRANSLATIONS',
    'DISEASE_SEVERITY',
    'HIGH_RISK_DISEASES',
    'get_disease_name',
    'get_disease_index',
    'get_english_name',
    'is_high_risk',
    'filter_high_risk',
    'format_disease_summary',
]
