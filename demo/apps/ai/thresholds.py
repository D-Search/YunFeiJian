# 共享配置：/www/WXproject/demo/apps/ai/thresholds.py

OPTIMAL_THRESHOLDS = {
    "Atelectasis": 0.16,
    "Cardiomegaly": 0.15,
    "Effusion": 0.28,
    "Infiltration": 0.24,
    "Mass": 0.21,
    "Nodule": 0.15,
    "Pneumonia": 0.07,
    "Pneumothorax": 0.10,
    "Consolidation": 0.15,
    "Edema": 0.28,
    "Emphysema": 0.28,
    "Fibrosis": 0.10,
    "Pleural_Thickening": 0.10,
    "Hernia": 0.34
}

# 英文到中文的映射
CLS_14_NAMES_EN = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

CLS_14_NAMES_CN = [
    "肺不张", "心脏肥大", "胸腔积液", "浸润", "肿块", "结节", 
    "肺炎", "气胸", "实变", "肺水肿", "肺气肿", "肺纤维化", 
    "胸膜增厚", "疝（膈疝）"
]

CLASS_NAME_MAP = dict(zip(CLS_14_NAMES_CN, CLS_14_NAMES_EN))