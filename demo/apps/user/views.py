"""
用户视图模块
处理用户相关的HTTP请求
"""
import json
from django.views.decorators.csrf import csrf_exempt

from apps.core.responses import APIResponse
from apps.core.exceptions import handle_service_exception
from .services import get_user_service
from .models import UserProfile
import traceback


@csrf_exempt
@handle_service_exception
def wx_login(request):
    """
    API: 微信登录接口（新架构规范外壳 + 旧版核心逻辑）
    """
    if request.method != 'POST':
        return APIResponse.bad_request(msg='仅支持 POST 请求')
        
    try:
        # 1. 安全解析请求体
        try:
            data = json.loads(request.body.decode('utf-8'))
            code = data.get('code')
            user_id = data.get('user_id')
            front_nickname = data.get('nickname')
            front_avatar = data.get('avatar_url')
        except Exception:
            code = None
            user_id = None
            front_nickname = None
            front_avatar = None

        # 2. 调用融入了你旧版逻辑的 Service
        user_service = get_user_service()
        login_data = user_service.wx_login(
            code=code, 
            user_id=user_id,
            nickname=front_nickname, 
            avatar_url=front_avatar
        )

        # 3. 使用新架构的标准格式安全返回
        return APIResponse.success(data=login_data, msg='登录成功')

    except Exception as e:
        print("\n[wx_login 内部崩溃]：")
        traceback.print_exc()
        return APIResponse.server_error(msg=f'系统内部错误: {str(e)}')


@csrf_exempt
@handle_service_exception
def update_user_info(request):
    """
    API: 更新用户信息
    
    POST /api/user/update
    Body: {"user_id": "xxx", "nickname": "新昵称", "avatar_url": "新头像"}
    """
    if request.method != 'POST':
        return APIResponse.method_not_allowed("仅支持POST请求")
    
    try:
        data = json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return APIResponse.bad_request("JSON格式解析失败")
    
    user_id = data.get('user_id')
    nickname = data.get('nickname')
    avatar_url = data.get('avatar_url')
    
    if not user_id:
        return APIResponse.bad_request("缺少user_id参数")
    
    user_service = get_user_service()
    user = user_service.update_user(user_id, nickname, avatar_url)
    
    if user is None:
        return APIResponse.not_found("用户不存在")
    
    return APIResponse.success(user.to_dict(), msg="资料更新成功")


def get_user_profile(request):
    """
    API: 获取用户资料
    
    GET /api/user/profile?user_id=xxx
    """
    user_id = request.GET.get('user_id')
    
    if not user_id:
        return APIResponse.bad_request("缺少user_id参数")
    
    user_service = get_user_service()
    user = user_service.get_user(user_id)
    
    if user is None:
        return APIResponse.not_found("用户不存在")
    
    return APIResponse.success(user.to_dict())
