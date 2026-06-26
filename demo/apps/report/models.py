from django.db import models
from django.utils import timezone
import json


class ReportRecord(models.Model):
    """
    报告记录表
    
    记录用户历史产生的所有肺部X光报告信息
    """
    # 报告编号
    report_no = models.CharField(
        max_length=32,
        unique=True,
        verbose_name="报告编号"
    )
    
    # 外键关联用户（使用整数主键）
    user = models.ForeignKey(
        'user.UserProfile',
        on_delete=models.CASCADE,
        related_name='reports',
        verbose_name="所属用户"
    )
    
    
    # 图像字段
    patient_image = models.ImageField(
        upload_to='cxr_images/',
        verbose_name="病人胸片图像"
    )
    
    heatmap_image = models.ImageField(
        upload_to='heatmaps/',
        null=True,
        blank=True,
        verbose_name="热力图"
    )
    
    annotated_image = models.ImageField(
        upload_to='annotateds/',
        null=True,
        blank=True,
        verbose_name="标注图像"
    )
    
    # 预测结果（JSON字符串）
    prediction_result = models.TextField(
        verbose_name="疾病预测结果"
    )
    
    # 生成时间
    generated_at = models.DateTimeField(
        default=timezone.now,
        verbose_name="报告生成时间"
    )
    
    class Meta:
        db_table = 'report_record'
        ordering = ['-generated_at']
        verbose_name = '报告记录'
        verbose_name_plural = verbose_name
    
    def __str__(self):
        return f"{self.report_no} - {self.user_id}"
    
    def to_dict(self, include_details: bool = False) -> dict:
        """
        转换为字典
        
        Args:
            include_details: 是否包含详细信息
        """
        result = {
            'report_no': self.report_no,
            'user_id': self.user_id,
            'patient_image_url': self.patient_image.url,
            'heatmap_image_url': self.heatmap_image.url if self.heatmap_image else '',
            'annotated_image_url': self.annotated_image.url if self.annotated_image else '',
            'generated_at': self.generated_at.strftime('%Y-%m-%d %H:%M')
        }
        
        if include_details and self.prediction_result:
            try:
                parsed = json.loads(self.prediction_result)
                result['prediction_result'] = parsed.get("prediction_result", {})
            except json.JSONDecodeError:
                result['prediction_result'] = {}
        
        return result
    
    @property
    def parsed_prediction(self) -> dict:
        """解析预测结果"""
        if not self.prediction_result or self.prediction_result == "正在处理...":
            return {}
        try:
            return json.loads(self.prediction_result)
        except json.JSONDecodeError:
            return {}
    
    def get_summary(self) -> str:
        """获取报告摘要"""
        parsed = self.parsed_prediction
        disease_dict = parsed.get("prediction_result", {})
        
        if not disease_dict:
            return "正在处理..."
        
        # 兼容新旧格式
        if isinstance(disease_dict, dict) and "probabilities" in disease_dict:
            # 新格式：{"probabilities": {...}, "positive_diseases": {...}}
            positive = disease_dict.get("positive_diseases", {})
            if not positive:
                return "未检测到疾病"
            high_risk = [(n, p) for n, p in positive.items()]
        elif isinstance(disease_dict, dict):
            # 旧格式：直接是 {"疾病名": 0.85, ...}
            high_risk = [(n, p) for n, p in disease_dict.items() if p > 0.5]
        else:
            return "数据异常"
        
        if not high_risk:
            return "未检测到疾病"
        
        high_risk.sort(key=lambda x: x[1], reverse=True)
        top_3 = high_risk[:3]
        
        return "，".join([f"{n}：{p:.2f}" for n, p in top_3])
