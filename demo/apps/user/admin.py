"""
Django Admin 配置
"""
from django.contrib import admin
from .models import UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user_id', 'nickname', 'avatar_url', 'created_at')
    search_fields = ('user_id', 'nickname')
    list_filter = ('created_at',)
