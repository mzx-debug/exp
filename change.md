修改了：1. batch_scaling_experiment.py 第 309 行：
  tensor_parallel_size=1,
  改为：
  tensor_parallel_size=3,

  
  2. batch_scaling_config.yaml 第 106 行：
  vllm_gpu_memory_utilization: 0.85
  改为：
  vllm_gpu_memory_utilization: 0.50
  
  3. batch_scaling_config.yaml 第 130 行：

  gpu_id: "0,5,6"