"""
报告视图模块
处理报告相关的HTTP请求
"""
import json
from django.views.decorators.csrf import csrf_exempt

from apps.core.responses import APIResponse
from apps.core.exceptions import handle_service_exception, ServiceException
from .services import get_report_service, get_analysis_service
from apps.user.models import UserProfile

# 异步任务导入
try:
    from .tasks import async_full_analysis
    HAS_CELERY = True
except ImportError:
    HAS_CELERY = False


def _is_async_mode() -> bool:
    """
    判断是否使用异步模式
    
    可以通过环境变量 ASYNC_ANALYSIS=true 来启用异步模式
    """
    import os
    return os.environ.get('ASYNC_ANALYSIS', 'false').lower() == 'true'


@csrf_exempt
@handle_service_exception
def upload_and_analyze(request):
    """
    API: 上传胸片并分析
    
    POST /api/report/upload/
    FormData: user_id, image
    
    支持两种模式：
    - 同步模式（默认）：立即返回分析结果
    - 异步模式（需设置 ASYNC_ANALYSIS=true）：立即返回报告编号，
      实际分析在后台进行
    """
    if request.method != 'POST':
        return APIResponse.method_not_allowed("仅支持POST请求")
    
    user_id = request.POST.get('user_id')
    image_file = request.FILES.get('image')
    
    user_numeric_id= UserProfile.objects.filter(user_id=user_id).values_list('id', flat=True).first()
    
    if not user_numeric_id:
        return APIResponse.bad_request("缺少user_id参数")
    
    if not image_file:
        return APIResponse.bad_request("缺少图片文件")
    
    # 创建报告记录
    report_service = get_report_service()
    record = report_service.create_report(user_numeric_id, image_file)
    
    # 异步模式：启动后台任务
    if _is_async_mode() and HAS_CELERY:
        # 立即返回报告编号，不等待分析完成
        async_full_analysis.delay(record.id)
        
        return APIResponse.success({
            'report_no': record.report_no,
            'status': 'processing',
            'message': '报告已创建，分析任务已提交后台处理'
        }, msg="任务已提交")
    
    # 同步模式：执行完整分析
    analysis_service = get_analysis_service()
    result = analysis_service.full_analysis_sync(record)
    
    return APIResponse.success(result, msg="AI辅助诊断完成")


@csrf_exempt
@handle_service_exception
def get_user_reports(request):
    """
    API: 获取用户报告列表
    
    GET /api/report/list/?user_id=xxx
    """
    user_id = request.GET.get('user_id')
    
    if not user_id:
        return APIResponse.bad_request("缺少user_id参数")
    
    user_numeric_id= UserProfile.objects.filter(user_id=user_id).values_list('id', flat=True).first()
    
    if not user_numeric_id:
        return APIResponse.success([])

    report_service = get_report_service()
    reports = report_service.get_user_reports(user_numeric_id)
    
    return APIResponse.success(reports)


@csrf_exempt
@handle_service_exception
def get_report_detail(request):
    """
    API: 获取报告详情
    
    GET /api/report/detail/?user_id=xxx&report_no=xxx
    """
    user_id = request.GET.get('user_id')
    report_no = request.GET.get('report_no')
    
    if not user_id or not report_no:
        return APIResponse.bad_request("缺少必要参数")
    
    user_numeric_id = UserProfile.objects.filter(user_id=user_id).values_list('id', flat=True).first()
    
    if not user_numeric_id:
        return APIResponse.not_found("用户不存在")
    
    report_service = get_report_service()
    detail = report_service.get_report_detail(user_numeric_id, report_no)
    
    if detail is None:
        return APIResponse.not_found("报告不存在")
    
    return APIResponse.success(detail)


@csrf_exempt
@handle_service_exception
def delete_reports(request):
    """
    API: 批量删除报告
    
    POST /api/report/delete/
    Body: {"user_id": "xxx", "report_nos": ["xxx", "xxx"]}
    """
    if request.method != 'POST':
        return APIResponse.method_not_allowed("仅支持POST请求")
    
    try:
        data = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return APIResponse.bad_request("JSON格式解析失败")
    
    user_id = data.get('user_id')
    report_nos = data.get('report_nos', [])
    
    user_numeric_id= UserProfile.objects.filter(user_id=user_id).values_list('id', flat=True).first()
    
    if not user_numeric_id:
        return APIResponse.bad_request("缺少user_id参数")
    
    if not report_nos or not isinstance(report_nos, list):
        return APIResponse.bad_request("report_nos必须为非空数组")

    report_service = get_report_service()
    deleted_count = report_service.delete_reports(user_numeric_id, report_nos)
    
    return APIResponse.success({'deleted_count': deleted_count}, msg="删除成功")


@csrf_exempt
@handle_service_exception
def ai_analysis(request):
    """
    API: LLM分析
    
    POST /api/report/ai_analysis/
    Body: {"user_id": "xxx", "report_no": "xxx", "disease_list": {"疾病名": 0.5}}
    """
    if request.method != 'POST':
        return APIResponse.method_not_allowed("仅支持POST请求")
    
    try:
        data = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return APIResponse.bad_request("JSON格式解析失败")
    
    user_id = data.get('user_id')
    report_no = data.get('report_no')
    disease_list = data.get('disease_list', {})
    
    user_numeric_id= UserProfile.objects.filter(user_id=user_id).values_list('id', flat=True).first()
    
    # 如果没有传入疾病列表，尝试从报告获取
    if not disease_list and report_no and user_numeric_id:
        report_service = get_report_service()
        detail = report_service.get_report_detail(user_numeric_id, report_no)
        if detail:
            disease_list = detail.get('prediction_result', {})
    
    # 兼容新格式：{"probabilities": {...}, "positive_diseases": {...}}
    if isinstance(disease_list, dict) and 'probabilities' in disease_list:
        disease_list = disease_list.get('probabilities', {})
    
    if not disease_list:
        return APIResponse.bad_request("缺少疾病概率数据")
    
    # 转换格式（支持数组和字典两种格式）
    if isinstance(disease_list, list):
        disease_list = {
            item['name']: item['probability']
            for item in disease_list
            if 'name' in item
        }
    
    # 调用LLM分析
    analysis_service = get_analysis_service()
    result = analysis_service.llm_analysis(disease_list)
    
    return APIResponse.success(result, msg="AI分析完成")


@csrf_exempt
@handle_service_exception
def trigger_analysis(request):
    """
    API: 手动触发报告分析（用于异步模式下重试）
    
    POST /api/report/trigger_analysis/
    Body: {"report_id": 123}
    """
    if request.method != 'POST':
        return APIResponse.method_not_allowed("仅支持POST请求")
    
    if not HAS_CELERY:
        return APIResponse.server_error("Celery未配置，无法使用异步任务")
    
    try:
        data = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return APIResponse.bad_request("JSON格式解析失败")
    
    report_id = data.get('report_id')
    if not report_id:
        return APIResponse.bad_request("缺少report_id参数")
    
    async_full_analysis.delay(report_id)
    
    return APIResponse.success({'status': 'submitted'}, msg="分析任务已提交")
