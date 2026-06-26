"""
Django Admin 配置
"""
from django.contrib import admin
from .models import ReportRecord


@admin.register(ReportRecord)
class ReportRecordAdmin(admin.ModelAdmin):
    list_display = ('report_no', 'user', 'generated_at')
    search_fields = ('report_no', 'user__user_id')
    list_filter = ('generated_at',)
    raw_id_fields = ('user',)
