"""
视图兼容层
为了保持向后兼容，旧接口仍然可用，但实际逻辑已迁移到 apps 模块
"""
# 本文件作为向后兼容的桥接层
# 所有API视图已迁移到 apps/user/views.py 和 apps/report/views.py
# 请使用新的模块结构

# 导入新模块的视图函数以保持URL配置兼容
from apps.user.views import wx_login as wx_login_new
from apps.user.views import update_user_info as update_user_info_new
from apps.user.views import get_user_profile as get_user_profile_new
from apps.report.views import upload_and_analyze as upload_and_analyze_new
from apps.report.views import get_user_reports as get_user_reports_new
from apps.report.views import get_report_detail as get_report_detail_new
from apps.report.views import delete_reports as delete_reports_new
from apps.report.views import ai_analysis as ai_analysis_new

# 为了兼容旧代码，提供别名
# 实际使用时请直接导入 apps.user.views 或 apps.report.views

def __getattr__(name):
    """动态导入旧版API"""
    mapping = {
        'wx_login': wx_login_new,
        'update_user_info': update_user_info_new,
        'get_user_profile': get_user_profile_new,
        'upload_and_analyze': upload_and_analyze_new,
        'get_user_reports': get_user_reports_new,
        'get_report_detail': get_report_detail_new,
        'delete_reports': delete_reports_new,
        'ai_analysis': ai_analysis_new,
    }
    if name in mapping:
        return mapping[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
