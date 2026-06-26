# WXproject 项目结构

## 概述

本项目已进行组件化和模块化重构，将原本臃肿的 `hello` 应用拆分为清晰的功能模块。

## 新目录结构

```
demo/
├── apps/                          # 核心业务模块
│   ├── __init__.py
│   ├── core/                     # 核心基础设施
│   │   ├── __init__.py
│   │   ├── config.py             # 配置管理
│   │   ├── responses.py          # 统一响应格式
│   │   ├── exceptions.py         # 统一异常处理
│   │   └── logger.py             # 日志配置
│   │
│   ├── ai/                       # AI相关模块
│   │   ├── __init__.py
│   │   ├── model_manager.py      # 模型加载管理（单例）
│   │   ├── image_processor.py    # 图像预处理
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── llm_service.py    # LLM服务（支持多后端）
│   │       └── heatmap_service.py # 热力图生成
│   │
│   ├── medical/                  # 医疗模块
│   │   ├── __init__.py
│   │   └── diseases.py           # 疾病定义
│   │
│   ├── user/                     # 用户模块
│   │   ├── __init__.py
│   │   ├── apps.py
│   │   ├── models.py             # UserProfile模型
│   │   ├── services.py           # 用户服务
│   │   ├── views.py              # 用户视图
│   │   ├── urls.py               # URL配置
│   │   └── admin.py              # Admin配置
│   │
│   └── report/                   # 报告模块
│       ├── __init__.py
│       ├── apps.py
│       ├── models.py             # ReportRecord模型
│       ├── services.py           # 报告服务
│       ├── views.py              # 报告视图
│       ├── urls.py               # URL配置
│       └── admin.py              # Admin配置
│
├── demo/                          # Django项目配置
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py                   # 主URL配置
│   ├── wsgi.py
│   └── asgi.py
│
├── third_party/                   # 第三方模型（符号链接到hello）
│   └── mavit_proto_sim_plus/
│       ├── model.py
│       ├── data.py
│       └── config.yaml
│
├── hello/                         # 保留用于兼容
│   ├── views.py                  # 兼容层
│   ├── models.py                 # 兼容层
│   ├── disease.py                # 兼容层
│   └── mavit_proto_sim_plus/     # 原始模型文件
│
└── media/                         # 用户上传文件
```

## API路由

### 用户API (`/api/user/`)

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/user/login/` | 微信登录 |
| POST | `/api/user/update/` | 更新用户信息 |
| GET | `/api/user/profile/` | 获取用户资料 |

### 报告API (`/api/report/`)

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/report/upload/` | 上传胸片并分析 |
| GET | `/api/report/list/` | 获取用户报告列表 |
| GET | `/api/report/detail/` | 获取报告详情 |
| POST | `/api/report/delete/` | 批量删除报告 |
| POST | `/api/report/ai_analysis/` | LLM疾病分析 |

## 使用示例

### 1. 使用模型管理器

```python
from apps.ai.model_manager import get_model_and_bank

model, bank = get_model_and_bank()
# 模型会在首次调用时懒加载
```

### 2. 使用图像预处理器

```python
from apps.ai.image_processor import default_preprocessor

image = default_preprocessor.load_image('/path/to/image.jpg')
tensor = default_preprocessor.preprocess(image)
```

### 3. 使用LLM服务

```python
from apps.ai.services.llm_service import get_llm_service

llm = get_llm_service()
result = llm.analyze({"肺不张": 0.45, "心脏肥大": 0.52})
```

### 4. 使用报告服务

```python
from apps.report.services import get_report_service

service = get_report_service()
reports = service.get_user_reports(user_id)
```

### 5. 使用统一响应

```python
from apps.core.responses import APIResponse

return APIResponse.success(data, msg="操作成功")
return APIResponse.bad_request("参数错误")
return APIResponse.not_found("资源不存在")
```

### 6. 使用统一异常

```python
from apps.core.exceptions import handle_service_exception

@handle_service_exception
def my_view(request):
    # 异常会自动转换为标准API响应
    raise ValidationError("参数验证失败")
```

## 兼容旧代码

原有的导入路径仍然可用：

```python
# 旧代码
from hello.models import UserProfile, ReportRecord
from hello.disease import CLS_14

# 新代码（推荐）
from apps.user.models import UserProfile
from apps.report.models import ReportRecord
from apps.medical.diseases import CLS_14
```

## 依赖关系

```
                    ┌─────────────────┐
                    │   小程序前端     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  apps.report    │
                    │  apps.user      │  ← 视图层
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ services │  │  models  │  │ services │
        └────┬─────┘  └──────────┘  └────┬─────┘
             │                            │
             ▼                            ▼
        ┌────────────────────────┐  ┌──────────┐
        │      apps.ai           │  │ apps.user│
        │  - model_manager       │  └──────────┘
        │  - image_processor     │
        │  - services/llm        │
        │  - services/heatmap    │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ third_party.mavit_...  │
        │ (MaxViT模型)          │
        └────────────────────────┘
```

## 迁移指南

### 从 hello.views 迁移

旧代码：
```python
from hello.views import upload_and_analyze
```

新代码：
```python
from apps.report.views import upload_and_analyze
```

### 从 hello.models 迁移

旧代码：
```python
from hello.models import ReportRecord
```

新代码：
```python
from apps.report.models import ReportRecord
```

### 从 hello.disease 迁移

旧代码：
```python
from hello.disease import CLS_14
```

新代码：
```python
from apps.medical.diseases import CLS_14
```

## 注意事项

1. **模型懒加载**：模型在首次请求时才加载，避免Django启动阻塞
2. **单例模式**：服务使用单例模式，避免重复创建
3. **向后兼容**：旧导入路径仍然可用，建议逐步迁移到新路径
4. **异常处理**：使用 `@handle_service_exception` 装饰器自动处理异常
