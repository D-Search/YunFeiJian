# 报告相关URL
from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_and_analyze, name='upload_analyze'),
    path('list/', views.get_user_reports, name='get_user_reports'),
    path('detail/', views.get_report_detail, name='get_report_detail'),
    path('delete/', views.delete_reports, name='delete_reports'),
    path('ai_analysis/', views.ai_analysis, name='ai_analysis'),
    path('trigger/', views.trigger_analysis, name='trigger_analysis'),
]
