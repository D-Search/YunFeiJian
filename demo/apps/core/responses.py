"""
统一API响应格式
提供标准化的HTTP响应接口
"""
from typing import Any, Optional
from django.http import JsonResponse


class APIResponse:
    """统一API响应类"""
    
    @staticmethod
    def success(data: Any = None, msg: str = "操作成功", **kwargs) -> JsonResponse:
        """
        成功响应
        
        Args:
            data: 响应数据
            msg: 响应消息
            **kwargs: 其他字段
        
        Returns:
            JsonResponse with code=200
        """
        response_data = {
            "code": 200,
            "msg": msg,
            "data": data,
        }
        response_data.update(kwargs)
        return JsonResponse(response_data, json_dumps_params={'ensure_ascii': False})
    
    @staticmethod
    def error(msg: str, code: int = 500, **kwargs) -> JsonResponse:
        """
        错误响应
        
        Args:
            msg: 错误消息
            code: 错误码
            **kwargs: 其他字段
        
        Returns:
            JsonResponse with specified code
        """
        response_data = {
            "code": code,
            "msg": msg,
        }
        response_data.update(kwargs)
        return JsonResponse(response_data, status=code)
    
    # 常用错误响应快捷方法
    @staticmethod
    def bad_request(msg: str = "请求参数错误") -> JsonResponse:
        """400 - 请求参数错误"""
        return APIResponse.error(msg, code=400)
    
    @staticmethod
    def unauthorized(msg: str = "未授权") -> JsonResponse:
        """401 - 未授权"""
        return APIResponse.error(msg, code=401)
    
    @staticmethod
    def forbidden(msg: str = "禁止访问") -> JsonResponse:
        """403 - 禁止访问"""
        return APIResponse.error(msg, code=403)
    
    @staticmethod
    def not_found(msg: str = "资源不存在") -> JsonResponse:
        """404 - 资源不存在"""
        return APIResponse.error(msg, code=404)
    
    @staticmethod
    def method_not_allowed(msg: str = "不支持的请求方法") -> JsonResponse:
        """405 - 请求方法不允许"""
        return APIResponse.error(msg, code=405)
    
    @staticmethod
    def server_error(msg: str = "服务器内部错误") -> JsonResponse:
        """500 - 服务器内部错误"""
        return APIResponse.error(msg, code=500)
    
    @staticmethod
    def service_unavailable(msg: str = "服务暂不可用") -> JsonResponse:
        """503 - 服务暂不可用"""
        return APIResponse.error(msg, code=503)


def success_response(data: Any = None, msg: str = "操作成功", **kwargs) -> JsonResponse:
    """成功响应快捷函数"""
    return APIResponse.success(data, msg, **kwargs)


def error_response(msg: str, code: int = 500, **kwargs) -> JsonResponse:
    """错误响应快捷函数"""
    return APIResponse.error(msg, code, **kwargs)
