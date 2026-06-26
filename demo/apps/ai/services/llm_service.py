"""
LLM服务模块
统一管理多种LLM后端（SiliconFlow、豆包、Kimi等）
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import json
import requests
from apps.ai.thresholds import OPTIMAL_THRESHOLDS,CLASS_NAME_MAP#OPTIMAL_THRESHOLDS的key是英文名称，需要使用CLASS_NAME_MAP将中文名称映射为英文名称
from apps.core.config import get_config

class BaseLLMService(ABC):
    """LLM服务抽象基类"""
    
    @abstractmethod
    def analyze(self, disease_list: Dict[str, float],
               patient_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        分析疾病概率并返回AI解读
        
        Args:
            disease_list: 疾病概率字典，如 {"肺不张": 0.45, "心脏肥大": 0.52}
            patient_info: 患者信息（可选）
        
        Returns:
            包含分析结果的字典:
            {
                "analysis_text": str,  # AI分析文本
                "suggestions": list,    # 建议列表
                "model_used": str,      # 使用的模型
                "source": str           # 来源标识
            }
        """
        pass
    
    def _format_disease_prompt(self, disease_list: Dict[str, float]) -> str:
        """
        格式化疾病信息为提示词
        
        Args:
            disease_list: 疾病概率字典
        
        Returns:
            格式化的提示词字符串
        """
    
        high_risk = []
        medium_risk = []
        low_risk = []
        
        for name, prob in disease_list.items():
            en_name = CLASS_NAME_MAP.get(name,name)
            threshold = OPTIMAL_THRESHOLDS.get(en_name, 0.5)

            ratio = prob / threshold

            if ratio >= 1.5:
                high_risk.append((name, prob))

            elif ratio >= 1.0:
                medium_risk.append((name, prob))

            else:
                low_risk.append((name, prob))

        prompt = "## 胸片AI检测结果\n\n"

        prompt += "### 高风险异常（明显超过诊断阈值）\n"

        if high_risk:
            prompt += "\n".join(
                f"- {n}: {p*100:.1f}%"
                for n, p in high_risk
            )
        else:
            prompt += "无\n"

        prompt += "\n\n### 中风险异常（超过诊断阈值）\n"

        if medium_risk:
            prompt += "\n".join(
                f"- {n}: {p*100:.1f}%"
                for n, p in medium_risk
            )
        else:
            prompt += "无\n"

        prompt += "\n\n### 低风险疾病\n"

        prompt += "\n".join(
            f"- {n}: {p*100:.1f}%"
            for n, p in sorted(
                low_risk,
                key=lambda x: x[1],
                reverse=True
            )[:5]
        )

        return prompt
    
    def _generate_suggestions(self, disease_list: Dict[str, float]) -> list:
        """
        根据疾病概率生成建议标签
        
        Args:
            disease_list: 疾病概率字典
        
        Returns:
            建议列表
        """
        high_risk = any( p > OPTIMAL_THRESHOLDS.get(n, 0.5) for n, p in disease_list.items())
        
        suggestions = []
        if high_risk:
            suggestions.append({"type": "warning", "text": "尽快就医"})
        else:
            suggestions.append({"type": "", "text": "保持健康"})
        
        suggestions.extend([
            {"type": "info", "text": "定期复查"},
            {"type": "info", "text": "咨询医生"}
        ])
        
        return suggestions


class SiliconFlowService(BaseLLMService):
    """
    SiliconFlow API服务
    
    使用 SiliconFlow 平台调用 Qwen 等模型
    文档: https://docs.siliconflow.cn/
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        config = get_config()
        self.api_key = config.llm.api_key
        self.api_url = config.llm.api_url
        self.model = model or config.llm.model
        self.timeout = config.llm.timeout
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return (
            "你是一位专业的医学影像分析AI助手，请根据胸片AI检测结果给出通俗、简洁、专业的解释和建议，"
            "强调仅供参考，最终诊断以医生为准。\n\n"
            "要求：\n"
            "1. 先简要说明检测结果\n"
            "2. 对每个高风险疾病进行解释（包括英文名称）\n"
            "3. 给出明确的建议\n"
            "4. 总字数控制在200-300字"
        )
    
    def analyze(self, disease_list: Dict[str, float],
               patient_info: Optional[Dict] = None) -> Dict[str, Any]:
        """使用SiliconFlow API分析"""
        print("disease_list =", disease_list) #使用中文名称
        print("Translation:", sorted(CLASS_NAME_MAP.values()))
        if not self.api_key:
            return self._fallback_analysis(disease_list)
        
        # 构建提示词
        disease_prompt = self._format_disease_prompt(disease_list)
        
        if patient_info:
            disease_prompt += f"\n\n## 患者基本信息\n"
            disease_prompt += f"- 年龄: {patient_info.get('age', '未知')}岁\n"
            disease_prompt += f"- 性别: {patient_info.get('gender', '未知')}\n"
        
        disease_prompt += "\n\n请基于以上检测结果进行分析和建议。"
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": disease_prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7,
            }
            
            # 👇 打印请求内容
            print(f"[LLM 请求] provider={self.__class__.__name__}, model={self.model}")
            print(f"[LLM prompt]\n{disease_prompt}")
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                verify=False
            )
            response.raise_for_status()
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            # 👇 打印原始返回
            print(f"[LLM 原始返回]\n{text}")
            
            return {
                "analysis_text": text,
                "suggestions": self._generate_suggestions(disease_list),
                "model_used": self.model,
                "source": "siliconflow_api",
            }
            
        except requests.exceptions.Timeout:
            print(f"[LLM 错误] API调用超时")
            return self._fallback_analysis(disease_list, "API调用超时")
        except requests.exceptions.RequestException as e:
            print(f"[LLM 错误] API调用失败: {str(e)}")
            return self._fallback_analysis(disease_list, f"API调用失败: {str(e)}")
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"[LLM 错误] 响应解析失败: {str(e)}")
            return self._fallback_analysis(disease_list, f"响应解析失败: {str(e)}")
    
    def _fallback_analysis(self, disease_list: Dict[str, float],
                          error_msg: str = "") -> Dict[str, Any]:
        """
        备用分析（API不可用时使用）
        """
        high_risk = [(n,p) for n,p in disease_list.items() if p > OPTIMAL_THRESHOLDS[CLASS_NAME_MAP.get(n,n)]]
        
        
        if not high_risk:
            analysis = (
                "根据AI检测结果，您的胸片未发现明显异常。请继续保持健康的生活方式，"
                "定期进行体检。如有不适，请及时就医。"
            )
        else:
            disease_name, prob = high_risk[0]
            analysis = (
                f"AI检测发现{disease_name}的可能性较高（{prob*100:.1f}%）。"
                f"建议您尽快到正规医院进行进一步检查，以获得更准确的诊断结果。"
                f"请不要过度担心，AI辅助检测仅供参考，最终诊断需以专业医生的判断为准。"
            )
        
        return {
            "analysis_text": analysis,
            "suggestions": self._generate_suggestions(disease_list),
            "model_used": self.model,
            "source": "fallback" if error_msg else "local_fallback",
            "error": error_msg if error_msg else None,
        }


class DoubaoService(BaseLLMService):
    """
    豆包API服务（字节跳动）
    
    优势：国内直连、价格便宜、响应快
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        config = get_config()
        self.api_key = api_key or config.llm.api_key or os.environ.get("DOUBAO_API_KEY", "")
        self.api_base = "https://ark.cn-beijing.volcengineapi.com"
        self.model = model or config.llm.model
        self.timeout = config.llm.timeout
    
    def analyze(self, disease_list: Dict[str, float],
               patient_info: Optional[Dict] = None) -> Dict[str, Any]:
        """使用豆包API分析"""
        
        if not self.api_key:
            return self._fallback_analysis(disease_list)
        
        system_prompt = (
            "你是一位专业的医学影像分析AI助手。请用简洁专业的语言分析胸片检测结果，"
            "给出通俗易懂的解释和就医建议。强调：这是AI辅助诊断，最终诊断需以医生为准。"
        )
        
        disease_prompt = self._format_disease_prompt(disease_list)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": disease_prompt}
        ]
        
        try:
            import aiohttp
            import asyncio
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            # 👇 打印请求内容
            print(f"[LLM 请求] provider={self.__class__.__name__}, model={self.model}")
            print(f"[LLM prompt]\n{disease_prompt}")
            
            # 同步版本
            import urllib.request
            import urllib.error
            
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                f"{self.api_base}/v1/chat/completions",
                data=data,
                headers=headers,
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                text = result["choices"][0]["message"]["content"]
            
            # 👇 打印原始返回
            print(f"[LLM 原始返回]\n{text}")
            
            return {
                "analysis_text": text,
                "suggestions": self._generate_suggestions(disease_list),
                "model_used": self.model,
                "source": "doubao_api",
            }
            
        except Exception as e:
            print(f"[LLM 错误] DoubaoService调用失败: {str(e)}")
            return self._fallback_analysis(disease_list)
    
    def _fallback_analysis(self, disease_list: Dict[str, float]) -> Dict[str, Any]:
        return SiliconFlowService()._fallback_analysis(disease_list)


class LLMServiceFactory:
    """
    LLM服务工厂
    
    根据配置创建相应的LLM服务实例
    """
    
    _services = {
        'siliconflow': SiliconFlowService,
        'doubao': DoubaoService,
    }
    
    @classmethod
    def create(cls, provider: str = None, **kwargs) -> BaseLLMService:
        """
        创建LLM服务实例
        
        Args:
            provider: 服务提供商标识
                - 'siliconflow': SiliconFlow（默认）
                - 'doubao': 豆包API
            **kwargs: 传递给服务构造函数的参数
        
        Returns:
            LLM服务实例
        
        Raises:
            ValueError: 不支持的服务类型
        """
        if provider is None:
            provider = os.environ.get('LLM_PROVIDER', 'siliconflow')
        
        if provider not in cls._services:
            available = ', '.join(cls._services.keys())
            raise ValueError(f"不支持的LLM服务: {provider}，可用: {available}")
        
        return cls._services[provider](**kwargs)
    
    @classmethod
    def register(cls, name: str, service_class: type):
        """注册新的LLM服务"""
        if not issubclass(service_class, BaseLLMService):
            raise TypeError("服务类必须继承自BaseLLMService")
        cls._services[name] = service_class


# 全局单例
_llm_service: Optional[BaseLLMService] = None


def get_llm_service() -> BaseLLMService:
    """
    获取LLM服务单例
    
    Returns:
        LLM服务实例
    """
    global _llm_service
    if _llm_service is None:
        provider = os.environ.get('LLM_PROVIDER', 'siliconflow')
        _llm_service = LLMServiceFactory.create(provider)
    return _llm_service


def reset_llm_service():
    """重置LLM服务（用于切换服务类型）"""
    global _llm_service
    _llm_service = None
