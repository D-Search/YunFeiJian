"""
疾病定义模块
定义医学影像分析相关的疾病标签和常量
"""

# 14种胸片疾病标签
CLS_14 = [
    "肺不张",      # 0
    "心脏肥大",    # 1
    "胸腔积液",    # 2
    "浸润",        # 3
    "肿块",        # 4
    "结节",        # 5
    "肺炎",        # 6
    "气胸",        # 7
    "实变",        # 8
    "肺水肿",      # 9
    "肺气肿",      # 10
    "肺纤维化",    # 11
    "胸膜增厚",    # 12
    "疝（膈疝）"   # 13
]


# 疾病索引映射
DISEASE_INDEX = {name: idx for idx, name in enumerate(CLS_14)}


# 疾病中英文对照
DISEASE_TRANSLATIONS = {
    "肺不张": "Atelectasis",
    "心脏肥大": "Cardiomegaly",
    "胸腔积液": "Pleural Effusion",
    "浸润": "Infiltration",
    "肿块": "Mass",
    "结节": "Nodule",
    "肺炎": "Pneumonia",
    "气胸": "Pneumothorax",
    "实变": "Consolidation",
    "肺水肿": "Pulmonary Edema",
    "肺气肿": "Emphysema",
    "肺纤维化": "Pulmonary Fibrosis",
    "胸膜增厚": "Pleural Thickening",
    "疝（膈疝）": "Hernia",
}


# 疾病严重程度分级
DISEASE_SEVERITY = {
    "肺不张": "medium",
    "心脏肥大": "high",
    "胸腔积液": "high",
    "浸润": "medium",
    "肿块": "high",
    "结节": "medium",
    "肺炎": "high",
    "气胸": "high",
    "实变": "medium",
    "肺水肿": "high",
    "肺气肿": "medium",
    "肺纤维化": "medium",
    "胸膜增厚": "low",
    "疝（膈疝）": "medium",
}


# 高风险疾病列表（需要优先关注）
HIGH_RISK_DISEASES = [
    "心脏肥大",
    "胸腔积液",
    "肿块",
    "肺炎",
    "气胸",
    "肺水肿",
]

from apps.ai.thresholds import OPTIMAL_THRESHOLDS

def get_disease_name(index: int) -> str:
    """根据索引获取疾病名称"""
    if 0 <= index < len(CLS_14):
        return CLS_14[index]
    return None


def get_disease_index(name: str) -> int:
    """根据疾病名称获取索引"""
    return DISEASE_INDEX.get(name)


def get_english_name(name: str) -> str:
    """获取疾病英文名"""
    return DISEASE_TRANSLATIONS.get(name, name)


def is_high_risk(name: str) -> bool:
    """判断是否为高风险疾病"""
    return name in HIGH_RISK_DISEASES


def filter_high_risk(disease_probs: dict) -> list:
    """
    从概率字典中筛选高风险疾病
    
    Args:
        disease_probs: {"疾病名": 概率值, ...}
    
    Returns:
        高风险疾病列表 [(name, prob), ...]
    """
    return [
        (name, prob) for name, prob in disease_probs.items()
        if prob > PROMPT_THRESHOLD[name]
    ]


def format_disease_summary(disease_probs: dict, top_n: int = 3) -> str:
    """
    格式化疾病摘要
    
    Args:
        disease_probs: 疾病概率字典
        top_n: 返回前N个
    
    Returns:
        格式化的摘要字符串
    """
    sorted_diseases = sorted(
        disease_probs.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    if not sorted_diseases:
        return "未检测到疾病"
    
    # 筛选高风险
    high_risk = [(n, p) for n, p in sorted_diseases if p > 0.5]
    
    if not high_risk:
        return "未检测到明显异常"
    
    # 取前N个高风险疾病
    top = high_risk[:top_n]
    return "，".join([f"{name}：{prob:.2f}" for name, prob in top])
