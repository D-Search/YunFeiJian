"""
用户服务模块
处理用户注册、登录、信息更新等业务逻辑
"""
import json
import uuid
import requests
import traceback
from typing import Dict, Any, Optional, Tuple
from django.db import IntegrityError
from .models import UserProfile

class WeChatService:
    """
    微信服务：纯粹负责换取 OpenID
    """
    def __init__(self):
        from apps.core.config import config
        self.appid = config.wechat.app_id
        self.secret = config.wechat.secret

    def code2session(self, code: str) -> Optional[str]:

        if not code:
            return None

        if str(code).startswith("mock_") or code == "mock_code_in_pc":
            print(f"检测到 Mock Code: {code}")
            return f"mock_openid_{code}" # 建议返回一个mock的openid，而不是None

        try:

            url = (
                f"https://api.weixin.qq.com/sns/jscode2session"
                f"?appid={self.appid}"
                f"&secret={self.secret}"
                f"&js_code={code}"
                f"&grant_type=authorization_code"
            )
            
            res = requests.get(url, timeout=5,verify=False  ).json()
            
            if "errcode" in res and res["errcode"] != 0:
                print(f"微信接口返回错误: {res.get('errmsg')}")
                return None

            return res.get("openid")
        except Exception as e:
            print(f"微信请求网络异常(连接超时或不通): {e}")
            return None
           

class UserService:
    """
    用户服务：完美融合你之前的旧版登录核心逻辑
    """
    def __init__(self):
        self.wechat_service = WeChatService()

    def wx_login(self, code: str, user_id: str = None, nickname: str = None, avatar_url: str = None):
        
        # 1. 如果传了 code，说明是尝试用微信登录
        if code:
            openid = self.wechat_service.code2session(code)
            if openid:
                final_user_id = openid
            else:
                # 微信换取失败，直接抛出异常，不要降级去用 user_id，否则会导致业务逻辑混乱
                raise Exception("微信身份换取失败，请检查服务器网络或 Code 是否过期")
        
        # 2. 如果没有 code，但是有 user_id（用于纯纯的纯前端本地 Mock 调试场景）
        elif user_id:
            final_user_id = user_id
            
        # 3. 彻底的全新用户（且没传 code）
        else:
            final_user_id = f"mock_user_{uuid.uuid4().hex[:8]}"

        user, created = UserProfile.objects.get_or_create(
            user_id=final_user_id,
            defaults={
                "nickname": nickname or "微信用户",
                "avatar_url": avatar_url or ""
            }
        )
        
        
        
        return {
            "user_id": user.user_id,
            "nickname": user.nickname,
            "avatar_url": user.avatar_url,
            "is_new": created
        }

    # === 补全你缺失的方法 ===
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        try:
            return UserProfile.objects.get(user_id=user_id)
        except UserProfile.DoesNotExist:
            return None
        
    def update_user(self, user_id: str, nickname: str = None, avatar_url: str = None):
        """
        核心业务：安全更新用户头像和昵称（对应你旧版的内部逻辑）
        """
        try:
            user = UserProfile.objects.get(user_id=user_id)
            if nickname:
                user.nickname = nickname
            if avatar_url:
                user.avatar_url = avatar_url
            user.save()
            return user  # 返回更新后的用户对象
        except UserProfile.DoesNotExist:
            return None

# 全局服务实例
_user_service = None


def get_user_service() -> UserService:
    """获取用户服务单例"""
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service
