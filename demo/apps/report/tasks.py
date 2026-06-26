"""
Celery异步任务模块
处理报告的异步AI分析和热力图生成
"""
import json
import torch
from celery import shared_task

from apps.ai.model_manager import get_model_and_bank
from apps.ai.image_processor import default_preprocessor
from apps.ai.services.heatmap_service import get_heatmap_generator
from apps.ai.services.llm_service import get_llm_service
from apps.medical.diseases import CLS_14
from apps.report.models import ReportRecord
from apps.ai.thresholds import OPTIMAL_THRESHOLDS


@shared_task(bind=True, max_retries=3)
def async_full_analysis(self, report_id: int):
    """
    Celery异步任务：完整的分析流程（图像分析 + 热力图生成）
    
    Args:
        self: Celery任务实例
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
        
        # 1. 模型推理
        model, bank = get_model_and_bank()
        with torch.no_grad():
            _, _, bce_logits, _, _ = model(img_tensor, proto_bank=bank)
            probabilities = torch.sigmoid(bce_logits).squeeze(0).cpu().numpy()
        
        # 映射结果
        disease_probs = {
            disease: round(float(probabilities[idx]), 2)
            for idx, disease in enumerate(CLS_14)
        }
        
        # 2. 生成热力图
        heatmap_path = None
        annotated_path = None
        try:
            generator = get_heatmap_generator()
            result = generator.generate_all(
                img_path,
                report_no=record.report_no
            )
            heatmap_path = result.get('heatmap_path')
            annotated_path = result.get('annotated_path')
        except Exception as heatmap_err:
            print(f"热力图生成失败: {str(heatmap_err)}")
        
        # 3. 使用 update_report 统一更新数据库（保证三层结构一致性）
        from apps.report.services import ReportService
        report_service = ReportService()
        report_service.update_report(
            record,
            disease_probs={"probabilities": disease_probs, "positive_diseases": {k: v for k, v in disease_probs.items() if v > OPTIMAL_THRESHOLDS[k]}},
            heatmap_path=heatmap_path,
            annotated_path=annotated_path
        )
        
        return f"Report {record.report_no} fully analyzed successfully."
    
    except ReportRecord.DoesNotExist:
        return f"Error: Report ID {report_id} not found."
    
    except Exception as e:
        # 重试机制
        try:
            record = ReportRecord.objects.get(id=report_id)
            record.prediction_result = json.dumps(
                {"error": str(e), "status": "retrying"},
                ensure_ascii=False
            )
            record.save()
        except Exception:
            pass
        
        # 3次重试后放弃
        raise self.retry(exc=e, countdown=60)


@shared_task
def async_analyze_image_only(report_id: int):
    """
    Celery异步任务：仅执行图像分析（不含热力图）
    
    Args:
        report_id: 报告记录的主键ID
    
    Returns:
        任务执行结果描述
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = default_preprocessor.transform
    
    try:
        record = ReportRecord.objects.get(id=report_id)
        img_path = record.patient_image.path
        image = default_preprocessor.load_image(img_path)
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        model, bank = get_model_and_bank()
        with torch.no_grad():
            _, _, bce_logits, _, _ = model(img_tensor, proto_bank=bank)
            probabilities = torch.sigmoid(bce_logits).squeeze(0).cpu().numpy()
        
        disease_probs = {
            disease: round(float(probabilities[idx]), 2)
            for idx, disease in enumerate(CLS_14)
        }
        
        # 使用 update_report 统一更新数据库
        from apps.report.services import ReportService
        report_service = ReportService()
        report_service.update_report(
            record,
            disease_probs={"probabilities": disease_probs, "positive_diseases": {k: v for k, v in disease_probs.items() if v > OPTIMAL_THRESHOLDS[k]}}
        )
        
        return f"Report {record.report_no} image analysis completed."
    
    except ReportRecord.DoesNotExist:
        return f"Error: Report ID {report_id} not found."
    except Exception as e:
        try:
            record = ReportRecord.objects.get(id=report_id)
            record.prediction_result = json.dumps({"error": str(e)}, ensure_ascii=False)
            record.save()
        except Exception:
            pass
        return f"Error analyzing report {report_id}: {str(e)}"


@shared_task
def async_generate_heatmap(report_id: int):
    """
    Celery异步任务：仅生成热力图
    
    Args:
        report_id: 报告记录的主键ID
    
    Returns:
        任务执行结果描述
    """
    try:
        record = ReportRecord.objects.get(id=report_id)
        img_path = record.patient_image.path
        
        generator = get_heatmap_generator()
        result = generator.generate_all(img_path, report_no=record.report_no)
        
        heatmap_path = result.get('heatmap_path')
        annotated_path = result.get('annotated_path')
        
        if heatmap_path:
            record.heatmap_image = heatmap_path
        if annotated_path:
            record.annotated_image = annotated_path
        record.save()
        
        return f"Report {record.report_no} heatmap generated."
    
    except ReportRecord.DoesNotExist:
        return f"Error: Report ID {report_id} not found."
    except Exception as e:
        return f"Error generating heatmap for report {report_id}: {str(e)}"


@shared_task(bind=True, max_retries=3)
def async_llm_analysis(self, report_id: int, disease_probs: dict = None):
    """
    Celery异步任务：LLM疾病分析（不阻塞HTTP请求）

    Args:
        self: Celery任务实例
        report_id: 报告记录的主键ID
        disease_probs: 可选，疾病概率字典；如不传则从数据库读取

    Returns:
        任务执行结果
    """
    try:
        record = ReportRecord.objects.get(id=report_id)

        # 读取疾病概率
        if disease_probs is None:
            parsed = json.loads(record.prediction_result) if record.prediction_result else {}
            inner = parsed.get("prediction_result", {}) if isinstance(parsed, dict) else {}
            disease_probs = inner.get("probabilities", {}) if isinstance(inner, dict) else {}

        if not disease_probs:
            return f"Error: No disease probabilities for report {report_id}"

        # 调用LLM分析
        llm_service = get_llm_service()
        result = llm_service.analyze(disease_probs)

        # 更新数据库
        parsed = json.loads(record.prediction_result) if record.prediction_result else {}
        parsed["ai_analysis_text"] = result.get("analysis_text", "")
        parsed["suggestion_tags"] = result.get("suggestions", [])
        parsed["llm_model"] = result.get("model_used", "")
        record.prediction_result = json.dumps(parsed, ensure_ascii=False)
        record.save()

        return f"Report {record.report_no} LLM analysis completed."

    except ReportRecord.DoesNotExist:
        return f"Error: Report ID {report_id} not found."

    except Exception as e:
        try:
            record = ReportRecord.objects.get(id=report_id)
            parsed = json.loads(record.prediction_result) if record.prediction_result else {}
            parsed["ai_analysis_text"] = f"AI分析暂时不可用: {str(e)}"
            parsed["suggestion_tags"] = []
            record.prediction_result = json.dumps(parsed, ensure_ascii=False)
            record.save()
        except Exception:
            pass

        raise self.retry(exc=e, countdown=60)
