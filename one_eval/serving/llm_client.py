import httpx
from typing import List, Dict, Any, Optional
import asyncio
import os
from huggingface_hub import snapshot_download


class BaseLLMClient:
    """统一接口基类"""

    async def achat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        return asyncio.run(self.achat(messages, **kwargs))


# ===  API 模式（OpenAI / Qwen / Spark / Claude 都通用） ===
class APILLMClient(BaseLLMClient):
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._sync = httpx.Client(timeout=timeout, headers=self._headers())
        self._async = httpx.AsyncClient(timeout=timeout, headers=self._headers())

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages: List[Dict[str, str]], **extra):
        return {"model": self.model, "messages": messages, **extra}

    def _parse(self, data: Dict[str, Any]) -> str:
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(f"Bad response schema: {data}")

    async def achat(self, messages: List[Dict[str, str]], **extra) -> str:
        r = await self._async.post(self.base_url, json=self._payload(messages, **extra))
        r.raise_for_status()
        return self._parse(r.json())


# === vLLM 本地推理（直接部署模型） ===
class VLLMLocalClient(BaseLLMClient):
    """
    直接使用vLLM部署和启动模型，而不是仅仅访问vLLM的API接口
    """

    def __init__(self, 
                 model_path: str,
                 tensor_parallel_size: int = 1,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 max_tokens: int = 1024,
                 top_k: int = 40,
                 repetition_penalty: float = 1.0,
                 seed: int = None,
                 max_model_len: int = None,
                 gpu_memory_utilization: float = 0.9,
                 hf_cache_dir: str = None,
                 hf_local_dir: str = None):
        """
        初始化vLLM本地模型客户端
        
        Args:
            model_path: HuggingFace模型名称或本地模型路径
            tensor_parallel_size: 张量并行大小
            temperature: 温度参数
            top_p: Top-p采样参数
            max_tokens: 最大生成token数
            top_k: Top-k采样参数
            repetition_penalty: 重复惩罚
            seed: 随机种子
            max_model_len: 最大模型长度
            gpu_memory_utilization: GPU内存利用率
            hf_cache_dir: HuggingFace缓存目录
            hf_local_dir: HuggingFace本地目录
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.hf_cache_dir = hf_cache_dir
        self.hf_local_dir = hf_local_dir
        
        # 初始化vLLM引擎
        self._init_vllm_engine()

    def _init_vllm_engine(self):
        """初始化vLLM引擎"""
        # 设置环境变量
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = "spawn"
        
        # 下载模型（如果需要）
        if os.path.exists(self.model_path):
            self.real_model_path = self.model_path
        else:
            self.real_model_path = snapshot_download(
                repo_id=self.model_path,
                cache_dir=self.hf_cache_dir,
                local_dir=self.hf_local_dir,
            )
        
        # 导入vLLM
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vllm is not installed. Please install it by running:\n"
                "    pip install vllm\n"
            )
        
        # 创建采样参数
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed
        )
        
        # 初始化LLM引擎
        self.llm = LLM(
            model=self.real_model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        
        # 延迟初始化tokenizer，只在需要时才导入transformers
        self._tokenizer = None

    @property
    def tokenizer(self):
        """延迟加载tokenizer"""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.real_model_path, cache_dir=self.hf_cache_dir)
            except ImportError:
                raise ImportError(
                    "transformers is not installed. Please install it by running:\n"
                    "    pip install transformers\n"
                )
        return self._tokenizer

    async def achat(self, messages: List[Dict[str, str]], **extra) -> str:
        """异步聊天接口"""
        # 应用聊天模板
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # 如果tokenizer不支持apply_chat_template，则使用简单的格式
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            prompt += "\nassistant:"
        
        # 生成响应
        outputs = self.llm.generate(prompt, self.sampling_params)
        return outputs[0].outputs[0].text

    def chat(self, messages: List[Dict[str, str]], **extra) -> str:
        """同步聊天接口"""
        # 应用聊天模板
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # 如果tokenizer不支持apply_chat_template，则使用简单的格式
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            prompt += "\nassistant:"
        
        # 生成响应
        outputs = self.llm.generate(prompt, self.sampling_params)
        return outputs[0].outputs[0].text


# === 工厂函数 ===
def get_llm_client(kind: str = "api", **kwargs) -> BaseLLMClient:
    """
    kind ∈ {"api", "vllm"}
    Example:
        get_llm_client("api", base_url="https://api.openai.com/v1/chat/completions", api_key="sk-xxx", model="gpt-4o")
        get_llm_client("vllm", model_path="Qwen/Qwen2.5-7B", tensor_parallel_size=2)
    """
    if kind == "api":
        return APILLMClient(**kwargs)
    elif kind == "vllm":
        return VLLMLocalClient(**kwargs)
    else:
        raise ValueError(f"Unknown LLM client type: {kind}")
