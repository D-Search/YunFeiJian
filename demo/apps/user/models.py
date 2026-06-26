from django.db import models


class UserProfile(models.Model):
    """
    用户信息表
    
    支持微信登录和电脑端测试：
    - user_id: 绑定微信OpenID或电脑端虚拟ID
    - nickname/avatar_url: 支持前端自定义头像和昵称
    """
    user_id = models.CharField(
        max_length=64,
        unique=True,
        db_index=True,
        verbose_name="用户ID/OpenID"
    )
    
    nickname = models.CharField(
        max_length=64,
        default="微信用户",
        blank=True,
        null=True,
        verbose_name="用户昵称"
    )
    
    avatar_url = models.CharField(
        max_length=500,
        default="",
        blank=True,
        null=True,
        verbose_name="头像链接/路径"
    )
    
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="创建时间"
    )
    
    class Meta:
        db_table = 'user_profile'
        verbose_name = '用户信息'
        verbose_name_plural = verbose_name
    
    def __str__(self):
        return self.user_id
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'user_id': self.user_id,
            'nickname': self.nickname or '微信用户',
            'avatar_url': self.avatar_url or '',
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
