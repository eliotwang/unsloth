# Support QLoRA of LLaMA and Qwen-MoE on gfx1201 with ROCm 7.1.1

## 目录

- [内容总结](#内容总结)
- [使用方法](#使用方法)
- [跑出正确结果](#跑出正确结果)


## 内容总结


1.支持**unsloth微调后**的**unsloth_merged_16bit**模型在**vllm**跑通
---

## 使用方法


### 1. 准备unsloth微调后unsloth_merged_16bit模型 放在/home/heyi/models下
### 2. 创建 Docker 容器，该容器镜像为rocm_vllm镜像，与执行unsloth镜像不同

```bash
sudo docker run -it -d \
  --device /dev/dri \
  --device /dev/kfd \
  --network host \
  --ipc host \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  --shm-size 32G \
  -v /home/heyi/models:/models \
  -v /home/heyi/share:/share \
  -v /home/heyi/workspace/pr:/workspace \
  --name unsloth_pr \
  rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103 /bin/bash
```

---

### 2. 卸载原有vllm

```bash
pip uninstall vllm
```

---

### 3. 拉取并编译 `rocm_vllm`

#### 3.1 拉取代码

```bash
git clone https://github.com/ROCm/vllm.git
cd vllm
```

#### 3.2 安装依赖

```bash
pip install -r requirements/rocm.txt
```

#### 3.3 编译安装

```bash
python setup.py develop
```

---

### 4. 拉取unsloth_amd_radeon_vllm并在vllm上跑通

#### 4.1 拉取unsloth

模型名称示例：

```text
git clone -b amd_radeon_vllm --single-branch https://github.com/eliotwang/unsloth.git
cd unsloth
```

#### 4.2 执行

```bash
python tests/qlora/test_vllm.py
```

#### 4.3 执行日志（可折叠展示）

<details>
<summary><strong>点击展开查看vllm执行日志</strong></summary>

```text

INFO 12-18 09:44:30 [__init__.py:216] Automatically detected platform rocm.
INFO 12-18 09:44:31 [utils.py:328] non-default args: {'max_model_len': 8192, 'disable_log_stats': True, 'model': '/models/unsloth_merged_16bit/'}
INFO 12-18 09:44:35 [__init__.py:744] Resolved architecture: LlamaForCausalLM
`torch_dtype` is deprecated! Use `dtype` instead!
INFO 12-18 09:44:35 [__init__.py:1798] Using max model len 8192
INFO 12-18 09:44:35 [scheduler.py:222] Chunked prefill is enabled with max_num_batched_tokens=8192.
WARNING 12-18 09:44:36 [__init__.py:2959] We must use the `spawn` multiprocessing start method. Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing for more information. Reasons: CUDA is initialized
INFO 12-18 09:44:38 [__init__.py:216] Automatically detected platform rocm.
(EngineCore_DP0 pid=5552) INFO 12-18 09:44:38 [core.py:654] Waiting for init message from front-end.
(EngineCore_DP0 pid=5552) INFO 12-18 09:44:38 [core.py:76] Initializing a V1 LLM engine (v0.9.2rc2.dev1802+geb9d4de9e) with config: model='/models/unsloth_merged_16bit/', speculative_config=None, tokenizer='/models/unsloth_merged_16bit/', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=True, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=/models/unsloth_merged_16bit/, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, pooler_config=None, compilation_config={"level":3,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.mamba_mixer2","vllm.mamba_mixer","vllm.short_conv","vllm.linear_attention","vllm.plamo2_mamba_mixer"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"cudagraph_mode":1,"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"pass_config":{},"max_capture_size":512,"local_cache_dir":null}
[W1218 09:44:40.279573542 ProcessGroupNCCL.cpp:915] Warning: TORCH_NCCL_AVOID_RECORD_STREAMS is the default now, this environment variable is thus deprecated. (function operator())
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
(EngineCore_DP0 pid=5552) INFO 12-18 09:44:40 [parallel_state.py:1164] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
(EngineCore_DP0 pid=5552) INFO 12-18 09:44:40 [gpu_model_runner.py:2178] Starting to load model /models/unsloth_merged_16bit/...
(EngineCore_DP0 pid=5552) INFO 12-18 09:44:40 [gpu_model_runner.py:2210] Loading model from scratch...
(EngineCore_DP0 pid=5552) INFO 12-18 09:44:41 [rocm.py:245] Using Triton Attention backend on V1 engine.
(EngineCore_DP0 pid=5552) INFO 12-18 09:44:41 [triton_attn.py:261] Using vllm unified attention for TritonAttentionImpl
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.07it/s]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.69it/s]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:02<00:00,  1.33it/s]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.19it/s]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:03<00:00,  1.25it/s]
(EngineCore_DP0 pid=5552)
(EngineCore_DP0 pid=5552) INFO 12-18 09:44:44 [default_loader.py:266] Loading weights took 3.32 seconds
(EngineCore_DP0 pid=5552) INFO 12-18 09:44:44 [gpu_model_runner.py:2232] Model loading took 15.0547 GiB and 3.651670 seconds
(EngineCore_DP0 pid=5552) INFO 12-18 09:44:47 [backends.py:538] Using cache directory: /root/.cache/vllm/torch_compile_cache/ccfc716f7f/rank_0_0/backbone for vLLM's torch.compile
(EngineCore_DP0 pid=5552) INFO 12-18 09:44:47 [backends.py:549] Dynamo bytecode transform time: 2.10 s
(EngineCore_DP0 pid=5552) INFO 12-18 09:44:50 [backends.py:161] Directly load the compiled graph(s) for dynamic shape from the cache, took 2.715 s
(EngineCore_DP0 pid=5552) /workspace/vllm/vllm/model_executor/layers/utils.py:106: UserWarning: Failed validator: GCN_ARCH_NAME (Triggered internally at /app/pytorch/aten/src/ATen/hip/tunable/Tunable.cpp:366.)
(EngineCore_DP0 pid=5552)   return torch.nn.functional.linear(x, weight, bias)
(EngineCore_DP0 pid=5552) INFO 12-18 09:44:56 [monitor.py:34] torch.compile takes 2.10 s in total
(EngineCore_DP0 pid=5552) INFO 12-18 09:44:57 [gpu_worker.py:276] Available KV cache memory: 9.62 GiB
(EngineCore_DP0 pid=5552) INFO 12-18 09:44:57 [kv_cache_utils.py:864] GPU KV cache size: 78,800 tokens
(EngineCore_DP0 pid=5552) INFO 12-18 09:44:57 [kv_cache_utils.py:868] Maximum concurrency for 8,192 tokens per request: 9.62x
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|███████████████████████████| 67/67 [00:04<00:00, 15.28it/s]
(EngineCore_DP0 pid=5552) INFO 12-18 09:45:02 [gpu_model_runner.py:2934] Graph capturing finished in 5 secs, took 0.54 GiB
(EngineCore_DP0 pid=5552) INFO 12-18 09:45:02 [core.py:218] init engine (profile, create kv cache, warmup model) took 17.84 seconds
INFO 12-18 09:45:03 [llm.py:285] Supported_tasks: ['generate']
INFO 12-18 09:45:03 [__init__.py:36] No IOProcessor plugins requested by the model
Adding requests: 100%|███████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 4777.11it/s]
Processed prompts: 100%|█████████████| 5/5 [00:01<00:00,  2.80it/s, est. speed input: 35.23 toks/s, output: 37.47 toks/s]

Generated Outputs:
------------------------------------------------------------
Prompt:    'Hello, my name is'
Output:    ' Bastian Scholtz.\nI was born in 1984 in Stuttgart,'
------------------------------------------------------------
Prompt:    'The president of the United States is'
Output:    ' the leader of the free world. In other words, the president of the United'
------------------------------------------------------------
Prompt:    'The capital of France is'
Output:    ' Paris, which is located in the northern part of France. The Paris region is'
------------------------------------------------------------
Prompt:    'The future of AI is'
Output:    ' the eternal AI\n\nEarly Adopters Drive Industry Innovation\n\nThe growth of AI technology'
------------------------------------------------------------
Prompt:    '<|begin_of_text|><|start_header_id|>system<|end_header_id|>Cutting Knowledge Date: December 2023Today Date: 26 Jul 2024<|eot_id|><|start_header_id|>user<|end_header_id|>What day was I born?<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
Output:    'Date Date'
------------------------------------------------------------
[rank0]:[W1218 09:45:05.975432556 ProcessGroupNCCL.cpp:1522] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())




```
## 跑出正确结果


###  选择以下unsloth-zoo的安装方式
###  使用以下unsloth-zoo安装方式微调出的unsloth_merged_16bit模型重新执行以上vllm步骤

### 5. 拉取unsloth-zoo并编译

### 5.1 进入unsloth目录 并拉取unsloth-zoo代码
```bash
cd unsloth
git clone -b amd_radeon --single-branch  https://github.com/eliotwang/unsloth-zoo.git
cd unsloth-zoo
pip install -e .
```
### 重新微调出新的unsloth_merged_16bit模型重新执行以上vllm步骤
#### 5.2 执行日志（可折叠展示）

<details>
<summary><strong>点击展开查看vllm执行正确日志</strong></summary>

```text
INFO 12-22 08:02:10 [__init__.py:216] Automatically detected platform rocm.
INFO 12-22 08:02:11 [utils.py:328] non-default args: {'max_model_len': 8192, 'disable_log_stats': True, 'model': '/models/unsloth_merged_16bit/'}
INFO 12-22 08:02:15 [__init__.py:744] Resolved architecture: LlamaForCausalLM
`torch_dtype` is deprecated! Use `dtype` instead!
INFO 12-22 08:02:15 [__init__.py:1798] Using max model len 8192
INFO 12-22 08:02:16 [scheduler.py:222] Chunked prefill is enabled with max_num_batched_tokens=8192.
WARNING 12-22 08:02:16 [__init__.py:2959] We must use the `spawn` multiprocessing start method. Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing for more information. Reasons: CUDA is initialized
INFO 12-22 08:02:18 [__init__.py:216] Automatically detected platform rocm.
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:19 [core.py:654] Waiting for init message from front-end.
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:19 [core.py:76] Initializing a V1 LLM engine (v0.9.2rc2.dev1802+geb9d4de9e) with config: model='/models/unsloth_merged_16bit/', speculative_config=None, tokenizer='/models/unsloth_merged_16bit/', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=True, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=/models/unsloth_merged_16bit/, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, pooler_config=None, compilation_config={"level":3,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.mamba_mixer2","vllm.mamba_mixer","vllm.short_conv","vllm.linear_attention","vllm.plamo2_mamba_mixer"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"cudagraph_mode":1,"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"pass_config":{},"max_capture_size":512,"local_cache_dir":null}
[W1222 08:02:21.129507581 ProcessGroupNCCL.cpp:915] Warning: TORCH_NCCL_AVOID_RECORD_STREAMS is the default now, this environment variable is thus deprecated. (function operator())
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:21 [parallel_state.py:1164] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:21 [gpu_model_runner.py:2178] Starting to load model /models/unsloth_merged_16bit/...
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:21 [gpu_model_runner.py:2210] Loading model from scratch...
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:21 [rocm.py:245] Using Triton Attention backend on V1 engine.
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:21 [triton_attn.py:261] Using vllm unified attention for TritonAttentionImpl
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:02,  1.27it/s]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:00,  2.11it/s]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  1.68it/s]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.48it/s]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.55it/s]
(EngineCore_DP0 pid=7089)
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:24 [default_loader.py:266] Loading weights took 2.66 seconds
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:24 [gpu_model_runner.py:2232] Model loading took 15.0547 GiB and 2.938731 seconds
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:27 [backends.py:538] Using cache directory: /root/.cache/vllm/torch_compile_cache/ccfc716f7f/rank_0_0/backbone for vLLM's torch.compile
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:27 [backends.py:549] Dynamo bytecode transform time: 2.15 s
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:29 [backends.py:161] Directly load the compiled graph(s) for dynamic shape from the cache, took 1.589 s
(EngineCore_DP0 pid=7089) /workspace/vllm/vllm/model_executor/layers/utils.py:106: UserWarning: Failed validator: GCN_ARCH_NAME (Triggered internally at /app/pytorch/aten/src/ATen/hip/tunable/Tunable.cpp:366.)
(EngineCore_DP0 pid=7089)   return torch.nn.functional.linear(x, weight, bias)
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:35 [monitor.py:34] torch.compile takes 2.15 s in total
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:36 [gpu_worker.py:276] Available KV cache memory: 9.62 GiB
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:36 [kv_cache_utils.py:864] GPU KV cache size: 78,800 tokens
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:36 [kv_cache_utils.py:868] Maximum concurrency for 8,192 tokens per request: 9.62x
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|███████████████████████████| 67/67 [00:04<00:00, 15.25it/s]
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:41 [gpu_model_runner.py:2934] Graph capturing finished in 5 secs, took 0.54 GiB
(EngineCore_DP0 pid=7089) INFO 12-22 08:02:41 [core.py:218] init engine (profile, create kv cache, warmup model) took 16.60 seconds
INFO 12-22 08:02:42 [llm.py:285] Supported_tasks: ['generate']
INFO 12-22 08:02:42 [__init__.py:36] No IOProcessor plugins requested by the model
Adding requests: 100%|███████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 5023.12it/s]
Processed prompts: 100%|█████████████| 5/5 [00:01<00:00,  4.63it/s, est. speed input: 58.33 toks/s, output: 66.66 toks/s]

Generated Outputs:
------------------------------------------------------------
Prompt:    'Hello, my name is'
Output:    " Helen and I'm a primary school teacher in South Africa. I'm also a"
------------------------------------------------------------
Prompt:    'The president of the United States is'
Output:    ' the head of government and head of state of the United States, and is elected'
------------------------------------------------------------
Prompt:    'The capital of France is'
Output:    ' Paris, which is located in the north-central part of the country. Paris is'
------------------------------------------------------------
Prompt:    'The future of AI is'
Output:    " bright, and it's essential to focus on the ethical implications of AI development,"
------------------------------------------------------------
Prompt:    '<|begin_of_text|><|start_header_id|>system<|end_header_id|>Cutting Knowledge Date: December 2023Today Date: 26 Jul 2024<|eot_id|><|start_header_id|>user<|end_header_id|>What day was I born?<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
Output:    'January 1, 2058'
------------------------------------------------------------
[rank0]:[W1222 08:02:43.154996547 ProcessGroupNCCL.cpp:1522] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())


```

</details>

---


