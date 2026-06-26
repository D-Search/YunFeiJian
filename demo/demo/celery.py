import os
from celery import Celery

# 设置 Django 的默认设置模块
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')

app = Celery('demo')

# 使用字符串格式，使 Worker 不必为子进程序列化配置对象
# 意思是所有跟 Celery 相关的配置都写在 settings.py 里，并以 CELERY_ 开头
app.config_from_object('django.conf:settings', namespace='CELERY')

# 自动从所有已注册的 Django app 中加载 tasks.py
app.autodiscover_tasks()