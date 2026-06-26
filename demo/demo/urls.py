"""
URL configuration for demo project.

URL路由配置
API路由已模块化，分布在各个apps中
"""

from django.contrib import admin
from django.urls import path, include

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Django Admin
    path("admin/", admin.site.urls),
    
    # === 新模块化API路径 ===
    
    # 用户相关API
    # POST /api/user/login/ - 微信登录
    # POST /api/user/update/ - 更新用户信息
    # GET  /api/user/profile/ - 获取用户资料
    path('api/user/', include('apps.user.urls')),
    
    # 报告相关API
    # POST /api/report/upload/ - 上传并分析
    # GET  /api/report/list/ - 获取报告列表
    # GET  /api/report/detail/ - 获取报告详情
    # POST /api/report/delete/ - 删除报告
    # POST /api/report/ai_analysis/ - LLM分析
    path('api/report/', include('apps.report.urls')),
    
    # === 保留旧接口兼容性（可选，逐步迁移）===
    # 旧路径会重定向到新的视图函数
    # 请优先使用新路径
    
]

# 媒体文件服务（开发环境）
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
