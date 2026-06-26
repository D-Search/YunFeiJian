# 用户相关URL
from django.urls import path
from apps.user import views

urlpatterns = [
    path('login/', views.wx_login, name='wx_login'),
    path('update/', views.update_user_info, name='update_user_info'),
    path('profile/', views.get_user_profile, name='get_user_profile'),
]
