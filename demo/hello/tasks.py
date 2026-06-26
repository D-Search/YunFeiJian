"""
Celery异步任务模块
使用新的模块化服务

本文件保留用于向后兼容，新代码请直接使用 apps.report.services
"""
import json
import torch
from celery import shared_task

from apps.ai.model_manager import get_model_and_bank
from apps.ai.image_processor import default_preprocessor
from apps.medical.diseases import CLS_14
from apps.report.models import ReportRecord


@shared_task
def async_analyze_chest_xray(report_id):
    """
    Celery后台异步任务：传入报告ID，进行AI推理并更新数据库
    
    Args:
        report_id: 报告记录的主键ID
    
    Returns:
        任务执行结果描述
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = default_preprocessor.transform
    
    try:
        # 获取数据库记录
        record = ReportRecord.objects.get(id=report_id)
        
        # 加载并预处理图像
        img_path = record.patient_image.path
        image = default_preprocessor.load_image(img_path)
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # 模型推理
        model, bank = get_model_and_bank()
        with torch.no_grad():
            _, _, bce_logits, _, _ = model(img_tensor, proto_bank=bank)
            probabilities = torch.sigmoid(bce_logits).squeeze(0).cpu().numpy()
        
        # 映射结果到疾病名称
        disease_probs = {
            disease: round(float(probabilities[idx]), 2)
            for idx, disease in enumerate(CLS_14)
        }
        
        # 更新数据库
        record.prediction_result = json.dumps(
            {"prediction_result": disease_probs},
            ensure_ascii=False
        )
        record.save()
        
        return f"Report {record.report_no} analyzed successfully."
    
    except ReportRecord.DoesNotExist:
        return f"Error: Report ID {report_id} not found."
    except Exception as e:
        # 错误信息写入数据库，避免前端一直卡在"正在处理..."
        try:
            record = ReportRecord.objects.get(id=report_id)
            record.prediction_result = json.dumps({"error": str(e)}, ensure_ascii=False)
            record.save()
        except Exception:
            pass
        return f"Error analyzing report {report_id}: {str(e)}"
