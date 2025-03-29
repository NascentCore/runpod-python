import runpod

# 设置runpod的api key
runpod.api_key = "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI4NzVhYzI0ZTQ0NGM0YzE4OGI4OWM2YTNiYjU3ZTJkOSIsInN1YiI6InBsYXlncm91bmRAc3h3bC5haSIsInVzZXJfaWQiOiJ1c2VyLTdlNjg3ZWEwLTg0NGItNDJjMy05MDA1LWVjOWRkZjRhZTg2MyIsInVzZXJpZCI6MjU0LCJ1c2VybmFtZSI6InBsYXlncm91bmRAc3h3bC5haSJ9.Y9XGk2zshcxxy6VZFCeZBhbce9KACbz3U8q7cze-subIfCNaEIgTP_R_GWXBINuWPWDxmfQfVlHujli0Am35LQ"

endpoint = runpod.Endpoint("INFERENCE")

# 异步方法
# run_request = endpoint.run(
#     {
#         "input": {
#            "gpu_model": "NVIDIA-GeForce-RTX-3090",
#            "model_category": "chat",
#            "gpu_count": 1,
#            "model_id": "model-storage-0ce92f029254ff34",
#            "model_name":"google/gemma-2b-it",
#            "model_size": 15065904829,
#            "model_is_public": True,
#            "model_template": "gemma",
#            "min_instances": 1,
#            "model_meta": "{\"template\":\"gemma\",\"category\":\"chat\", \"can_finetune\":true,\"can_inference\":true}",
#            "max_instances": 1
#         }
#     }
# )

# 打印推理的状态
# print(run_request.status())

# 打印推理的结果,结果返回推理接口的地址
# print(run_request.output())



run_request = endpoint.run_sync(
    {
        "input": {
           "gpu_model": "NVIDIA-GeForce-RTX-3090",
           "model_category": "chat",
           "gpu_count": 1,
           "model_id": "model-storage-0ce92f029254ff34",
           "model_name":"google/gemma-2b-it",
           "model_size": 15065904829,
           "model_is_public": True,
           "model_template": "gemma",
           "min_instances": 1,
           "model_meta": "{\"template\":\"gemma\",\"category\":\"chat\", \"can_finetune\":true,\"can_inference\":true}",
           "max_instances": 1
        }
    }
)

print(run_request.output())