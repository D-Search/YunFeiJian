"""
统一异常处理模块
定义业务异常和系统异常
"""


class ServiceException(Exception):
    """业务服务异常基类"""
    
    def __init__(self, message: str, code: int = 500):
        self.message = message
        self.code = code
        super().__init__(message)


class ValidationError(ServiceException):
    """参数验证错误"""
    def __init__(self, message: str = "参数验证失败"):
        super().__init__(message, code=400)


class UnauthorizedError(ServiceException):
    """未授权异常"""
    def __init__(self, message: str = "未授权访问"):
        super().__init__(message, code=401)


class NotFoundError(ServiceException):
    """资源不存在异常"""
    def __init__(self, message: str = "资源不存在"):
        super().__init__(message, code=404)


class ModelLoadError(ServiceException):
    """模型加载异常"""
    def __init__(self, message: str = "模型加载失败"):
        super().__init__(message, code=500)


class PredictionError(ServiceException):
    """预测执行异常"""
    def __init__(self, message: str = "预测执行失败"):
        super().__init__(message, code=500)


class LLMServiceError(ServiceException):
    """LLM服务异常"""
    def __init__(self, message: str = "LLM服务调用失败"):
        super().__init__(message, code=500)


class DatabaseError(ServiceException):
    """数据库操作异常"""
    def __init__(self, message: str = "数据库操作失败"):
        super().__init__(message, code=500)


class FileOperationError(ServiceException):
    """文件操作异常"""
    def __init__(self, message: str = "文件操作失败"):
        super().__init__(message, code=500)


def handle_service_exception(func):
    """
    异常处理装饰器
    将ServiceException转换为标准API响应
    """
    from functools import wraps
    from .responses import APIResponse
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            return APIResponse.bad_request(e.message)
        except UnauthorizedError as e:
            return APIResponse.unauthorized(e.message)
        except NotFoundError as e:
            return APIResponse.not_found(e.message)
        except ServiceException as e:
            return APIResponse.error(e.message, code=e.code)
        except Exception as e:
            return APIResponse.server_error(f"服务器异常: {str(e)}")
    
    return wrapper
