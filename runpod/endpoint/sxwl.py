import json
import time
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


# 全局变量用于存储需要清理的资源
resources_to_cleanup = {
    'inference_services': set(),
    'finetune_jobs': set()
}


@dataclass
class APIConfig:
    base_url: str
    token: str
    headers: Dict[str, str]

    @classmethod
    def create_default(cls) -> 'APIConfig':
        from runpod import (  # pylint: disable=import-outside-toplevel, cyclic-import
            api_key,
            endpoint_url_base,
        )
        token = api_key
        if token and isinstance(token, bytes):
            # 如果是 bytes，解码为字符串
            token = token.decode('utf-8')
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Origin': endpoint_url_base,
            'Connection': 'keep-alive',
            'User-Agent': 'github-actions'
        }
        return cls(endpoint_url_base, token, headers)

class SXWLClient:
    def __init__(self, config: APIConfig):
        self.config = config

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = f"{self.config.base_url}/api{endpoint}"
        response = requests.request(method, url, headers=self.config.headers, **kwargs)
        response.raise_for_status()
        return response

    def get_models(self) -> List[Dict[str, Any]]:
        """获取可用的模型列表"""
        try:
            response = self._make_request('GET', '/resource/models')
            data = response.json()
            models = data.get('public_list', []) + data.get('user_list', [])
            print(f"获取到 {len(models)} 个模型", flush=True)
            return models
        except Exception as e:
            print(f"获取模型列表失败: {str(e)}", flush=True)
            return []

    def delete_inference_service(self, service_name: str) -> None:
        try:
            self._make_request('DELETE', '/job/inference', params={'service_name': service_name})
            print("推理服务删除成功", flush=True)
            resources_to_cleanup['inference_services'].discard(service_name)
        except Exception as e:
            print(f"删除推理服务失败: {str(e)}", flush=True)

    def delete_finetune_job(self, finetune_id: str) -> None:
        try:
            self._make_request('POST', '/userJob/job_del', json={'job_id': finetune_id})
            print("微调任务删除成功", flush=True)
            resources_to_cleanup['finetune_jobs'].discard(finetune_id)
        except Exception as e:
            print(f"删除微调任务失败: {str(e)}", flush=True)

class InferenceService:
    def __init__(self, client: SXWLClient):
        self.client = client
        self.service_name: Optional[str] = None
        self.api_endpoint: Optional[str] = None

    def deploy(self, model_config: Dict[str, Any]) -> 'InferenceService':
        response = self.client._make_request('POST', '/job/inference', json=model_config)
        self.service_name = response.json()['service_name']
        print(f"服务名称: {self.service_name}", flush=True)
        # 添加到需要清理的资源列表
        resources_to_cleanup['inference_services'].add(self.service_name)
        return self
    
    def wait_until_complete(self) -> Dict[str, Any]:
        """等待服务部署完成并返回结果"""
        self._wait_for_ready()
        return {
            "service_name": self.service_name,
            "api_endpoint": self.api_endpoint,
            "status": "running"
        }

    def status(self) -> str:
        response = self.client._make_request('GET', '/job/inference')
        status_json = response.json()
        for item in status_json.get('data', []):
            if item['service_name'] == self.service_name:
                return item['status']
        return 'unknown'

    def output(self) -> Dict[str, Any]:
        response = self.client._make_request('GET', '/job/inference')
        result = {}
        status_json = response.json()
        for item in status_json.get('data', []):
            if item['service_name'] == self.service_name:
                if item['status'] == 'running':
                    result['chat_url'] = item['api']
                    return result
                return {'status': item['status']}
        return {'status': 'unknown'}

    def _wait_for_ready(self, max_retries: int = 60, retry_interval: int = 30) -> None:
        for attempt in range(max_retries):
            response = self.client._make_request('GET', '/job/inference')
            status_json = response.json()
            
            for item in status_json.get('data', []):
                if item['service_name'] == self.service_name:
                    if item['status'] == 'running':
                        self.api_endpoint = item['api']
                        print(f"服务已就绪: {item}", flush=True)
                        return
                    break
            
            print(f"服务启动中... ({attempt + 1}/{max_retries})", flush=True)
            time.sleep(retry_interval)
        
        raise TimeoutError("服务启动超时")

    def chat(self, messages: list) -> Dict[str, Any]:
        if not self.api_endpoint:
            raise RuntimeError("服务尚未就绪")
            
        chat_url = self.api_endpoint
        headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
        data = {"model": "/mnt/models", "messages": messages}
        
        response = requests.post(chat_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

class FinetuneJob:
    def __init__(self, client: SXWLClient):
        self.client = client
        self.job_id: Optional[str] = None
        self.adapter_id: Optional[str] = None

    def start(self, finetune_config: Dict[str, Any]) -> 'FinetuneJob':
        response = self.client._make_request('POST', '/job/finetune', json=finetune_config)
        self.job_id = response.json()['job_id']
        print(f"微调任务ID: {self.job_id}", flush=True)
        # 添加到需要清理的资源列表
        resources_to_cleanup['finetune_jobs'].add(self.job_id)
        return self

    def wait_until_complete(self) -> Dict[str, Any]:
        """等待任务完成并返回结果"""
        self._wait_for_completion()
        self._get_adapter_id()
        return {
            "job_id": self.job_id,
            "adapter_id": self.adapter_id,
            "status": "succeeded"
        }

    def _wait_for_completion(self, max_retries: int = 60, retry_interval: int = 30) -> None:
        for _ in range(max_retries):
            print(f"正在检查微调任务状态... (第 {_ + 1}/{max_retries} 次尝试)", flush=True)
            response = self.client._make_request('GET', '/job/training', 
                                               params={'current': 1, 'size': 1000})
            
            print(f"API响应: {response.json()}", flush=True)
            for job in response.json().get('content', []):
                if job['jobName'] == self.job_id:
                    status = job['status']
                    print(f"微调状态: {status}", flush=True)
                    
                    if status == 'succeeded':
                        return
                    elif status in ['failed', 'error']:
                        raise RuntimeError("微调任务失败")
                    break
            
            time.sleep(retry_interval)
        raise TimeoutError("微调任务超时")

    def _get_adapter_id(self) -> None:
        response = self.client._make_request('GET', '/resource/adapters')
        
        for adapter in response.json().get('user_list', []):
            try:
                meta = json.loads(adapter.get('meta', '{}'))
                if meta.get('finetune_id') == self.job_id:
                    self.adapter_id = adapter['id']
                    print(f"适配器ID: {self.adapter_id}", flush=True)
                    return
            except json.JSONDecodeError:
                continue
        
        raise ValueError(f"未找到对应的适配器")

# ---------------------------------------------------------------------------- #
#                                   Endpoint                                   #
# ---------------------------------------------------------------------------- #
class Endpoint:
    """Manages an endpoint to run jobs on the sxwl."""

    def __init__(self, endpoint_id: str):
        """
        Initialize an Endpoint instance with the given endpoint ID.

        Args:
            endpoint_id: The identifier for the endpoint.

        Example:
            >>> endpoint = runpod.Endpoint("INFERENCE")
            >>> run_request = endpoint.run({"your_model_input_key": "your_model_input_value"})
            >>> print(run_request.status())
            >>> print(run_request.output())
        """
        self.endpoint_id = endpoint_id
        self.rp_client = SXWLClient(APIConfig.create_default())

    def run(self, request_input: Dict[str, Any]) -> InferenceService:
        """
        Run the endpoint with the given input.

        Args:
            request_input: The input to pass into the endpoint.

        Returns:
            An InferenceService instance for the run request.
        """
        if not request_input.get("input"):
            request_input = {"input": request_input}
        
        
        inference = InferenceService(self.rp_client)
        return inference.deploy(request_input)

    def run_sync(
        self, request_input: Dict[str, Any], timeout: int = 86400
    ) -> Dict[str, Any]:
        """
        Run the endpoint with the given input synchronously.

        Args:
            request_input: The input to pass into the endpoint.
            timeout: The maximum time to wait for the result in seconds.

        Returns:
            The output of the completed job.
        """
        if not request_input.get("input"):
            request_input = {"input": request_input}

        config = APIConfig.create_default()
        client = SXWLClient(config)
        
        inference = InferenceService(client)
        # 启动任务
        inference = inference.deploy(request_input)
        # 等待任务完成
        return inference.wait_until_complete()

    def health(self, timeout: int = 3) -> Dict[str, Any]:
        """
        Check the health of the endpoint (number/state of workers, number/state of requests).

        Args:
            timeout: The number of seconds to wait for the server to respond before giving up.
        """
        config = APIConfig.create_default()
        client = SXWLClient(config)
        try:
            response = client._make_request('GET', '/job/health', timeout=timeout)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def purge_queue(self, timeout: int = 3) -> Dict[str, Any]:
        """
        Purges the endpoint's job queue and returns the result of the purge request.

        Args:
            timeout: The number of seconds to wait for the server to respond before giving up.
        """
        config = APIConfig.create_default()
        client = SXWLClient(config)
        try:
            response = client._make_request('POST', '/job/purge-queue', timeout=timeout)
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}