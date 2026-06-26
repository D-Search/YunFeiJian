"""
报告服务模块
处理报告创建、分析、查询等业务逻辑
"""
import json
from typing import Dict, Any, Optional, List

from django.db import IntegrityError, transaction
from django.utils import timezone

from apps.ai.model_manager import get_model_and_bank
from apps.ai.image_processor import default_preprocessor
from apps.ai.services.llm_service import get_llm_service
from apps.ai.services.heatmap_service import get_heatmap_generator
from apps.medical.diseases import CLS_14, format_disease_summary
from apps.ai.thresholds import OPTIMAL_THRESHOLDS
from apps.core.exceptions import (
    ValidationError, NotFoundError, PredictionError
)
from apps.ai.thresholds import CLS_14_NAMES_EN, CLS_14_NAMES_CN

from .models import ReportRecord


class ReportService:
    """
    报告服务
    
    核心业务逻辑：
    1. 创建报告记录
    2. 执行AI图像分析
    3. 生成热力图和标注
    4. 查询报告列表和详情
    """
    
    def __init__(self):
        self.transform_pipeline = default_preprocessor.transform
    
    def create_report(self, user_id: str, image_file) -> ReportRecord:
        """
        创建报告记录
        
        Args:
            user_id: 用户ID
            image_file: 上传的图像文件
        
        Returns:
            ReportRecord实例
        """

        # 生成报告编号
        report_no = self._generate_report_no()
        
        try:
            with transaction.atomic():
                record = ReportRecord.objects.create(
                    user_id=user_id,        # 字符串ID，方便查询
                    report_no=report_no,
                    patient_image=image_file,
                    prediction_result="正在处理..."
                )
            return record
            
        except IntegrityError:
            raise ValidationError("报告编号冲突，请重试")
    
    def analyze_image(self, record: ReportRecord) -> Dict[str, float]:
        """
        执行图像分析
        
        Args:
            record: ReportRecord实例
        
        Returns:
            疾病概率字典 {"疾病名"：概率}
        """
        try:
            # 获取模型
            model, bank = get_model_and_bank()
            device = model.device if hasattr(model, 'device') else 'cpu'
            
            # 预处理图像
            img_path = record.patient_image.path
            image = default_preprocessor.load_image(img_path)
            img_tensor = default_preprocessor.preprocess(image).unsqueeze(0).to(device)
            
            # 模型推理
            import torch
            with torch.no_grad():
                _, _, bce_logits, _, _ = model(img_tensor, proto_bank=bank)
                probabilities = torch.sigmoid(bce_logits).squeeze(0).cpu().numpy()
            
            all_probs={}
            positive_diseases ={}
            
            # 映射结果
        
            for idx, disease_name in enumerate(CLS_14):
                prob=round(float(probabilities[idx]), 2)
                all_probs[disease_name] = prob
                threshold = OPTIMAL_THRESHOLDS.get(CLS_14_NAMES_EN[idx], 0.5)
                if prob > threshold:
                    disease_name_zh = CLS_14_NAMES_CN[idx]
                    positive_diseases[disease_name_zh] = prob
            
            return{
                "probabilities": all_probs,
                "positive_diseases": positive_diseases,
                }
        
        except Exception as e:
            raise PredictionError(f"图像分析失败: {str(e)}")
    
            
    def generate_visualizations(self, record: ReportRecord) -> Dict[str, str]:
        """
        生成热力图和标注图像
        Args:
            record: ReportRecord实例
        
        Returns:
            {"heatmap_path": xxx, "annotated_path": xxx}
            失败时返回原图路径作为兜底
        """
        try:
            generator = get_heatmap_generator()
            result = generator.generate_all(
                record.patient_image.path,
                report_no=record.report_no,
                user_id=record.user_id
            )
            
            return {
                'heatmap_path': result.get('heatmap_path'),
                'annotated_path': result.get('annotated_path')
            }
                
        except Exception as e:
            import logging
            logging.warning(f"热力图生成失败: {str(e)}")
            # 失败时返回原图路径作为兜底，确保前端始终有图片可显示
            # 使用 .name 获取相对于 MEDIA_ROOT 的路径
            original_path = record.patient_image.name if record.patient_image else None
            return {
                'heatmap_path': original_path,
                'annotated_path': original_path
            }
    
    def update_report(self, record: ReportRecord,
                     disease_probs: Dict[str, Any],
                     heatmap_path: str = None,
                     annotated_path: str = None):
        """
        更新报告结果
        
        Args:
            record: ReportRecord实例
            disease_probs: 疾病概率结构，格式为 {"probabilities": {...}, "positive_diseases": {...}}
            heatmap_path: 热力图路径（相对于MEDIA_ROOT）
            annotated_path: 标注图像路径（相对于MEDIA_ROOT）
        """
        result_payload = {
            "prediction_result": disease_probs,
        }
        
        # 初始化更新字段
        update_fields = ["prediction_result"]
        
        record.prediction_result = json.dumps(result_payload, ensure_ascii=False)
        
        if heatmap_path:
            resolved_path = self._resolve_image_path(heatmap_path)
            # 直接设置字段值（存储相对于MEDIA_ROOT的路径）
            record.heatmap_image = resolved_path if resolved_path else ''
            update_fields.append("heatmap_image")
        if annotated_path:
            resolved_path = self._resolve_image_path(annotated_path)
            record.annotated_image = resolved_path if resolved_path else ''
            update_fields.append("annotated_image")
        
        record.save(update_fields=update_fields)
    
    def get_user_reports(self, user_id: str) -> List[dict]:
        """
        获取用户报告列表
        
        Args:
            user_id: 用户ID
        
        Returns:
            报告列表
        """
        reports = ReportRecord.objects.filter(user_id=user_id)
        return [self._format_summary(r) for r in reports]
    
    def get_report_detail(self, user_id: str, report_no: str) -> Optional[dict]:
        """
        获取报告详情
        
        Args:
            user_id: 用户ID
            report_no: 报告编号
        
        Returns:
            报告详情字典
        """
        record = ReportRecord.objects.filter(
            user_id=user_id,
            report_no=report_no
        ).first()
        
        if not record:
            return None
        
        return self._format_detail(record)
    
    def delete_reports(self, user_id: str, report_nos: List[str]) -> int:
        """
        批量删除报告
        
        Args:
            user_id: 用户ID
            report_nos: 报告编号列表
        
        Returns:
            删除数量
        """
        # 获取要删除的记录
        records = ReportRecord.objects.filter(
            user_id=user_id,
            report_no__in=report_nos
        )
        
        if not records.exists():
            raise NotFoundError("未找到匹配的报告记录")
        
        # 删除本地文件
        import os
        for record in records:
            try:
                if record.patient_image and os.path.exists(record.patient_image.path):
                    os.remove(record.patient_image.path)
                if record.heatmap_image and os.path.exists(record.heatmap_image.path):
                    os.remove(record.heatmap_image.path)
                if record.annotated_image and os.path.exists(record.annotated_image.path):
                    os.remove(record.annotated_image.path)
            except Exception:
                pass
        
        # 删除数据库记录
        deleted = records.delete()
        return deleted[0]
    
    def _generate_report_no(self) -> str:
        """生成唯一报告编号"""
        today_str = timezone.now().strftime('%Y%m%d')
        prefix = f"REP{today_str}"
        
        for _ in range(10):
            latest = ReportRecord.objects.filter(
                report_no__startswith=prefix
            ).order_by('-report_no').first()
            
            if latest and len(latest.report_no) >= len(prefix) + 4:
                try:
                    last_seq = int(latest.report_no[-4:])
                    return f"{prefix}{str(last_seq + 1).zfill(4)}"
                except ValueError:
                    pass
            
            return f"{prefix}0001"
        
        raise ValidationError("报告编号生成失败")
    
    def _format_summary(self, record: ReportRecord) -> dict:
        """格式化报告摘要"""
        parsed = record.parsed_prediction
        inner_result = parsed.get("prediction_result", {})
        probabilities = inner_result.get("probabilities", {}) if isinstance(inner_result, dict) else {}
        positive_diseases = inner_result.get("positive_diseases", {}) if isinstance(inner_result, dict) else {}

        return {
            'report_no': record.report_no,
            'patient_image_url': record.patient_image.url,
            'heatmap_image_url': record.heatmap_image.url if record.heatmap_image else '',
            'annotated_image_url': record.annotated_image.url if record.annotated_image else '',
            'generated_at': record.generated_at.strftime('%Y-%m-%d %H:%M'),
            'prediction_result': {
                'probabilities': probabilities,
                'positive_diseases': positive_diseases
            }
        }

    def _resolve_image_path(self, path: str):
        """
        解析图像路径，确保返回正确的ImageField值
        
        Args:
            path: 可能是相对路径、完整URL或None
        
        Returns:
            处理后的路径字符串（相对于MEDIA_ROOT）
        """
        if not path:
            return None
        
        # 如果是完整URL（包含/media/），提取相对路径部分
        if '/media/' in path:
            path = path.split('/media/')[-1]
        
        return path

    def _format_detail(self, record: ReportRecord) -> dict:
        """格式化报告详情"""
        try:
            parsed = json.loads(record.prediction_result) if record.prediction_result else {}
        except Exception:
            parsed = {}
            
        inner_result = parsed.get("prediction_result", {})
        probabilities = inner_result.get("probabilities", {}) if isinstance(inner_result, dict) else {}
        positive_diseases = inner_result.get("positive_diseases", {}) if isinstance(inner_result, dict) else {}
        
        return {
            'report_no': record.report_no,
            'patient_image_url': record.patient_image.url if record.patient_image else '',
            'heatmap_image_url': record.heatmap_image.url if record.heatmap_image else '',
            'annotated_image_url': record.annotated_image.url if record.annotated_image else '',
            'generated_at': record.generated_at.strftime('%Y-%m-%d %H:%M') if record.generated_at else '',
            
            # 与列表API保持一致的结构（嵌套格式，供前端 processReportData 解析）
            'prediction_result': {
                'probabilities': probabilities,
                'positive_diseases': positive_diseases
            },
            
            # LLM分析结果（异步生成）
            'analysis_text': parsed.get("ai_analysis_text", "AI医师正在努力分析中..."), 
            'suggestions': parsed.get("suggestion_tags", [])
        }


class AnalysisService:
    """
    分析服务
    
    处理AI分析和LLM解读
    """
    
    def __init__(self):
        self.report_service = ReportService()
        self.llm_service = get_llm_service()
    
    def full_analysis(self, user_id: str, image_file) -> Dict[str, Any]:
        """
        完整的分析流程（同步模式）
        
        Args:
            user_id: 用户ID
            image_file: 上传的图像
        
        Returns:
            完整分析结果
        """
        # 1. 创建报告
        record = self.report_service.create_report(user_id, image_file)
        
        # 2. 执行图像分析
        prediction_result = self.report_service.analyze_image(record)
        
        # 3. 生成可视化
        visualizations = self.report_service.generate_visualizations(record)
        
        # 4. 更新报告
        self.report_service.update_report(
            record,
            prediction_result,
            heatmap_path=visualizations.get('heatmap_path'),
            annotated_path=visualizations.get('annotated_path')
        )
        
        return {
            'report_no': record.report_no,
            'prediction_result': prediction_result,
            'heatmap_image_url': record.heatmap_image.url if record.heatmap_image else '',
            'annotated_image_url': record.annotated_image.url if record.annotated_image else ''
        }
    
    def full_analysis_sync(self, record) -> Dict[str, Any]:
        """
        同步分析已有报告记录
        
        Args:
            record: 已创建的ReportRecord实例
        
        Returns:
            完整分析结果
        """
        # 执行图像分析
        disease_probs = self.report_service.analyze_image(record)
        
        # 生成可视化  
        visualizations = self.report_service.generate_visualizations(record)
        
        # 更新报告
        self.report_service.update_report(
            record,
            disease_probs,
            heatmap_path=visualizations.get('heatmap_path'),
            annotated_path=visualizations.get('annotated_path')
        )
        
        return {
            'report_no': record.report_no,
            'prediction_result': disease_probs,
            'heatmap_image_url': record.heatmap_image.url if record.heatmap_image else '',
            'annotated_image_url': record.annotated_image.url if record.annotated_image else ''
        }
    
    def llm_analysis(self, disease_list: Dict[str, float]) -> Dict[str, Any]:
        """
        LLM分析
        
        Args:
            disease_list: 疾病概率字典
        
        Returns:
            LLM分析结果
        """
        return self.llm_service.analyze(disease_list)


# 全局服务实例
_report_service = None
_analysis_service = None


def get_report_service() -> ReportService:
    """获取报告服务单例"""
    global _report_service
    if _report_service is None:
        _report_service = ReportService()
    return _report_service


def get_analysis_service() -> AnalysisService:
    """获取分析服务单例"""
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = AnalysisService()
    return _analysis_service
