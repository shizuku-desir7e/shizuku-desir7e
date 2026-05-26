# 紫霄 GCU vLLM-Omni 适配 Patch 完整说明

> **环境**: Pod `cgame-partner-llm-zx-c200-debug-66f5bdfc79-fbvk5` (namespace: `ieg-gztechtke-zixiao-c200-nj`)  
> **日期**: 2026-05-22  
> **模型**: Qwen3-Omni-30B-A3B-Instruct / Qwen2.5-Omni-7B  
> **硬件**: 紫霄 GCU C200 × 4 卡 (40.87 GiB/卡)  
> **软件栈**: Python 3.10, torch_zx, vllm 0.19.x, vllm_omni, vllm_zx, triton_zx

---

## 一、Patch 总览

### A. 平台层 Patch

| # | Patch 名称 | 修改文件 | 目的 |
|---|-----------|---------|------|
| 1 | ZXOmniPlatform 平台类 | `vllm_omni/platforms/zx_platform.py` (新建) | 让 vllm_omni 正确识别紫霄 ZX 设备 |
| 2 | 平台插件注册 | `vllm_omni/platforms/__init__.py` | 将 ZX 平台加入自动检测链 |

### B. API / Serving 层 Patch

| # | Patch 名称 | 修改文件 | 目的 |
|---|-----------|---------|------|
| 3 | OpenAIServingRender stub | `vllm/entrypoints/serve/render/serving.py` (新建) | 提供 `OpenAIServingRender` 空类供 api_server 导入 |
| 4 | Registry 兼容 shim | `vllm/entrypoints/openai/models/serving.py` | `ZX_REGISTRY_PATCH`: 添加 `self.registry = None` 属性 |
| 5 | api_server io_processor 初始化 | `vllm_omni/entrypoints/openai/api_server.py` | 初始化 `renderer`/`io_processor`/`openai_serving_render` |
| 6 | ServingChat __init__ 适配 | `vllm_omni/entrypoints/openai/serving_chat.py` | `OmniOpenAIServingChat.__init__` 增加 `openai_serving_render` 参数 |
| 7 | Qwen3-Omni Chat Template | `/opt/qwen3_omni_chat_tmpl.jinja` (新建) | 支持 `input_audio` 类型 + 正确的 audio placeholder token |

### C. 模型层 Patch

| # | Patch 名称 | 修改文件 | 目的 |
|---|-----------|---------|------|
| 8 | kaiser_window ZX fallback | `vllm_omni/.../qwen2_5_omni/qwen2_5_omni_token2wav.py` | `torch.kaiser_window` 缺 ZX kernel，CPU-first fallback |
| 9 | sampling_metadata None 检查 | `vllm_omni/.../qwen2_5_omni/qwen2_5_omni.py` | Talker forward 时 `prompt_token_ids` 为 None |
| 10 | Qwen2.5-Omni Thinker 适配 | `vllm_omni/.../qwen2_5_omni/qwen2_5_omni_thinker.py` | 多模态 prompt 处理/audio_feature 兼容 |
| 11 | Qwen3-Omni MoE Thinker 适配 | `vllm_omni/.../qwen3_omni/qwen3_omni_moe_thinker.py` | audio_token 注册 + prompt_updates 兼容 |

### D. MTP (Code Predictor) Patch

| # | Patch 名称 | 修改文件 | 目的 |
|---|-----------|---------|------|
| 12 | Code Predictor MTP 替换 | `vllm_omni/.../qwen3_omni/qwen3_omni_moe_code_predictor_mtp.py` | KV Cache + One-shot Parallel 优化版 Code Predictor |

### E. Worker / Engine 层 Patch

| # | Patch 名称 | 修改文件 | 目的 |
|---|-----------|---------|------|
| 13 | 移除 defer_finalize 参数 | `vllm_omni/worker/gpu_generation_model_runner.py` + `gpu_ar_model_runner.py` | vllm 版本 API 不匹配 |
| 14 | gpu_model_runner 适配 | `vllm_omni/worker/gpu_model_runner.py` | 紫霄环境下 model runner 兼容调整 |
| 15 | MoE OOT Kernel Patch | `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` | 紫霄 OOT 平台 MoE kernel=None 修复 |

---

## 二、各 Patch 详细说明

### Patch 1: ZXOmniPlatform 平台类

**文件**: `/root/.conda/envs/py310/lib/python3.10/site-packages/vllm_omni/platforms/zx_platform.py` (新建)

**问题**: vllm_omni 只有 CUDA/ROCm/NPU/XPU 四种平台，紫霄被识别为 `UnspecifiedOmniPlatform`，导致：
- `get_torch_device()` 返回 `cuda`（错误）
- `device_control_env_var` 为空字符串（stage 无法隔离多卡）
- `get_device_count()` 调用 `torch.cuda.device_count()`（返回 0）

**解决方案**: 创建 `ZXOmniPlatform` 类，继承 `vllm_zx.zx.ZXPlatform`（获得 `device_type="zx"`、`_enum=PlatformEnum.OOT`、`device_control_env_var="TOPS_VISIBLE_DEVICES"` 等属性）和 `OmniPlatform`（获得 omni 特定接口）。

```python
class ZXOmniPlatform(ZXPlatform, OmniPlatform):
    _omni_enum = OmniPlatformEnum.UNSPECIFIED

    @classmethod
    def is_cuda_alike(cls) -> bool:
        return False  # 关键：避免 diffusion envs 调用 pyzxml.zxmlInit() 崩溃

    @classmethod
    def get_device_count(cls) -> int:
        return torch.zx.device_count()

    @classmethod
    def get_torch_device(cls, local_rank=None):
        if local_rank is None:
            return torch.device("zx")
        return torch.device("zx", local_rank)

    @classmethod
    def get_free_memory(cls, device=None) -> int:
        free, _ = torch.zx.mem_get_info(device)
        return free
    # ... 其他方法
```

**关键设计决策**:
- MRO 顺序 `ZXPlatform` 在前，确保 `device_type`/`_enum`/`dispatch_key` 等从 ZXPlatform 继承
- `is_cuda_alike()` 返回 False：ZXPlatform 原始返回 True（为了复用 CUDA 代码路径），但 omni 的 diffusion 模块会据此调用 `get_device_name()` → `pyzxml.zxmlInit()` 崩溃
- 显式实现 `get_device_count`/`get_torch_device`/`synchronize`/`get_free_memory`，避免 MRO 找到 OmniPlatform 的抽象方法

---

### Patch 2: 平台插件注册

**文件**: `/root/.conda/envs/py310/lib/python3.10/site-packages/vllm_omni/platforms/__init__.py`

**修改内容**: 在 `builtin_omni_platform_plugins` 字典中添加 ZX 检测函数：

```python
# ZX_OMNI_PLATFORM_PATCH_V1
def zx_omni_platform_plugin() -> str | None:
    """Check if ZiXiao/ZX OmniPlatform should be activated."""
    try:
        import torch
        if hasattr(torch, "zx") and torch.zx.is_available():
            return "vllm_omni.platforms.zx_platform.ZXOmniPlatform"
    except Exception:
        pass
    return None

builtin_omni_platform_plugins = {
    "zx": zx_omni_platform_plugin,   # ← 新增，优先级最高
    "cuda": cuda_omni_platform_plugin,
    "rocm": rocm_omni_platform_plugin,
    "npu": npu_omni_platform_plugin,
    "xpu": xpu_omni_platform_plugin,
}
```

**原理**: vllm_omni 启动时遍历 `builtin_omni_platform_plugins`，第一个返回非 None 的插件被激活。ZX 放在最前面确保优先匹配。

---

### Patch 3: OpenAIServingRender Stub

**文件**: `/root/.conda/envs/py310/lib/python3.10/site-packages/vllm/entrypoints/serve/render/serving.py` (新建)

**问题**: `vllm_omni/entrypoints/openai/api_server.py` 在初始化时 import `OpenAIServingRender`，但当前 vllm 版本没有这个类。

**修改**: 创建一个空 stub 类：
```python
class OpenAIServingRender:
    def __init__(self, *args, **kwargs):
        pass
```

---

### Patch 4: Registry 兼容 Shim

**文件**: `/root/.conda/envs/py310/lib/python3.10/site-packages/vllm/entrypoints/openai/models/serving.py`

**问题**: vllm_omni 的 `api_server.py` 在初始化 `OpenAIServingRender` 时传入 `model_registry=state.openai_serving_models.registry`，但 `OpenAIServingModels` 没有 `.registry` 属性。

**修改** (line 67):
```python
# ZX_REGISTRY_PATCH
# Compatibility shim: vllm_omni expects .registry attribute
self.registry = None
```

---

### Patch 5: api_server io_processor 初始化

**文件**: `/root/.conda/envs/py310/lib/python3.10/site-packages/vllm_omni/entrypoints/openai/api_server.py`

**问题**: AsyncOmni engine 在紫霄环境初始化时 `engine_client` 缺少 `renderer` 和 `io_processor` 属性，导致后续 OpenAI serving 层无法处理多模态输入。

**修改** (line 580-615):
```python
# 初始化 renderer（如果 engine_client 没有）
renderer = getattr(engine_client, "renderer", None)
if renderer is None:
    from vllm.renderers import renderer_from_config
    renderer = renderer_from_config(vllm_config)
    engine_client.renderer = renderer
engine_client.io_processor = get_io_processor(vllm_config, renderer, io_processor_plugin)

# 初始化 openai_serving_render
state.openai_serving_render = OpenAIServingRender(
    model_config=engine_client.model_config,
    renderer=engine_client.renderer,
    io_processor=engine_client.io_processor,
    model_registry=state.openai_serving_models.registry,
    ...
)
```

---

### Patch 10: Qwen2.5-Omni Thinker 适配

**文件**: `/root/.conda/envs/py310/lib/python3.10/site-packages/vllm_omni/model_executor/models/qwen2_5_omni/qwen2_5_omni_thinker.py`

**修改**: 多模态 `_maybe_apply_prompt_updates` 方法适配，处理紫霄环境下 audio/video token 的 prompt replacement 逻辑。确保在 ZX 平台上正确处理 `audio_feature_lengths` 和 `input_audio_features`。

---

### Patch 11: Qwen3-Omni MoE Thinker 适配

**文件**: `/root/.conda/envs/py310/lib/python3.10/site-packages/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py`

**修改**:
- Line 591-592: 确保 processor 有 `audio_token` 属性：
  ```python
  if not hasattr(processor, "audio_token"):
      processor.audio_token = "<|audio_pad|>"
  ```
- 修正 `mm_max_tokens` 计算中的 audio tokens 获取逻辑
- 适配 `_maybe_apply_prompt_updates` 方法处理 `use_audio_in_video` 场景

---

### Patch 12: Code Predictor MTP 替换（One-shot Parallel）

**文件**: `/root/.conda/envs/py310/lib/python3.10/site-packages/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_code_predictor_mtp.py`

**修改**: 整个文件替换为 KV Cache + One-shot Parallel 优化版实现。

**优化内容**:
1. **KV Cache in attention layers** — 增量 decode，不再 re-prefill
2. **One-shot parallel prediction** — 单次 prefill [hidden, codec_0]，然后 31 个 head 并行预测
3. **Fallback AR mode** — 保留 `forward_ar()` 作为回退
4. **torch.compile on inner transformer** — 内部 transformer 支持编译加速
5. **Inline sampling (top-k + top-p)** — 内联采样避免额外开销

**核心结构**:
```python
class Qwen3OmniMoeTalkerCodePredictor(nn.Module):
    """Two modes: forward() = one-shot parallel, forward_ar() = KV-cached AR."""
    
    def forward(self, ...):  # One-shot parallel: 31 heads predict in parallel
    def forward_ar(self, ...):  # KV-cached AR: incremental decode
```

---

### Patch 14: gpu_model_runner 适配

**文件**: `/root/.conda/envs/py310/lib/python3.10/site-packages/vllm_omni/worker/gpu_model_runner.py`

**修改**: 适配紫霄环境下的 model runner 行为，包括 CUDA graph 相关逻辑在 eager 模式下的正确跳过。

---

### Patch 3: kaiser_window ZX CPU-first Fallback

**文件**: `/root/.conda/envs/py310/lib/python3.10/site-packages/vllm_omni/model_executor/models/qwen2_5_omni/qwen2_5_omni_token2wav.py`

**问题**: Code2Wav 的 BigVGAN 模型在 `__init__` 时调用 `kaiser_sinc_filter1d()` → `torch.kaiser_window(...)`，而紫霄缺少该算子的 ZX dispatch kernel：
```
RuntimeError: DispatchStub: missing kernel for zx
```

**原始代码** (line ~744):
```python
if current_omni_platform.is_npu():
    kaiser_window = torch.kaiser_window(..., device="cpu").to("npu")
elif current_omni_platform.is_xpu():
    kaiser_window = torch.kaiser_window(..., device="cpu").to("xpu")
else:
    kaiser_window = torch.kaiser_window(...)  # 直接在当前设备创建 → ZX 崩溃
```

**修改后**:
```python
if current_omni_platform.is_npu():
    kaiser_window = torch.kaiser_window(..., device="cpu").to("npu")
elif current_omni_platform.is_xpu():
    kaiser_window = torch.kaiser_window(..., device="cpu").to("xpu")
elif hasattr(torch, "zx") and torch.zx.is_available():
    # ZiXiao/ZX: avoid direct torch.kaiser_window on zx, which may miss kernel.
    kaiser_window = torch.kaiser_window(..., device="cpu").to("zx")
else:
    kaiser_window = torch.kaiser_window(...)
```

**原理**: `torch.kaiser_window` 是 CPU 端的窗口函数，在 CPU 上计算后 `.to("zx")` 搬运到 GCU 即可。该 filter 仅在模型初始化时计算一次，不影响推理性能。

---

### Patch 4: sampling_metadata.prompt_token_ids None 检查

**文件**: `/root/.conda/envs/py310/lib/python3.10/site-packages/vllm_omni/model_executor/models/qwen2_5_omni/qwen2_5_omni.py`

**问题**: Talker forward 第 335 行：
```python
if sampling_metadata is not None:
    sampling_metadata.prompt_token_ids[sampling_metadata.prompt_token_ids == 152064] = 8448
```
在 decode 阶段 `sampling_metadata.prompt_token_ids` 可能为 None，导致 `TypeError: 'NoneType' object does not support item assignment`。

**修改后**:
```python
if sampling_metadata is not None and sampling_metadata.prompt_token_ids is not None:
    sampling_metadata.prompt_token_ids[sampling_metadata.prompt_token_ids == 152064] = 8448
```

**说明**: 这不是紫霄特有问题，是 vllm_omni 版本兼容性 bug。H20 上可能因为不同的代码路径/版本而未触发。

---

### Patch 5: 移除 defer_finalize 参数

**文件**:
- `/root/.conda/envs/py310/lib/python3.10/site-packages/vllm_omni/worker/gpu_generation_model_runner.py` (line 285)
- `/root/.conda/envs/py310/lib/python3.10/site-packages/vllm_omni/worker/gpu_ar_model_runner.py`

**问题**: vllm_omni 调用 `self.maybe_get_kv_connector_output(scheduler_output, defer_finalize=...)` 但当前 vllm 底层的 `KVConnectorModelRunnerMixin.maybe_get_kv_connector_output()` 不接受 `defer_finalize` 关键字参数：
```
TypeError: maybe_get_kv_connector_output() got an unexpected keyword argument 'defer_finalize'
```

**修改**: 注释掉 `defer_finalize=defer_kv_connector_finalize,` 这行参数传递。

**影响**: KV connector 的延迟 finalize 功能不可用（该功能仅在分布式 KV transfer 场景需要，当前单节点部署无影响）。

---

### Patch 6: MoE OOT Kernel Patch

**文件**: `/root/.conda/envs/py310/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`

**脚本**: `/tmp/patch_zx_moe_kernel.py` (来源: `/data/workspace/patch_zx_moe_kernel.py`)

**问题**: Qwen3-Omni 的 Talker 是 MoE 架构。vllm 0.19.x 中 `UnquantizedFusedMoEMethod.forward_cuda` 需要 `self.kernel`，但在 OOT 平台（紫霄）：
1. `select_unquantized_moe_backend` 返回 `OOT`
2. `OOT in UNSUPPORTED_BACKEND` → `make_unquantized_moe_kernel` 返回 None
3. `self.kernel` 保持 None
4. `forward_cuda` 时 `assert self.kernel is not None` 报错

**修改**: 在 `process_weights_after_loading` 中增加 OOT 分支：
```python
elif current_platform.is_out_of_tree():
    # ZX_OOT_MOE_KERNEL_PATCH_V1
    try:
        self._setup_kernel_oot(layer=layer, w13=layer.w13_weight, w2=layer.w2_weight)
    except Exception as e:
        logger.warning('OOT MoE kernel setup failed: %s', e)
```

新增 `_setup_kernel_oot` 方法使用 `vllm_zx_utils.kernels.modular_experts.TritonExpertsPad` 构建 kernel。

---

### Patch 7: Qwen3-Omni Chat Template

**文件**: `/opt/qwen3_omni_chat_tmpl.jinja` (新建)

**问题**: 
- 模型自带的 `chat_template.json` 不处理 `input_audio` 类型（OpenAI 兼容接口发送的格式）
- H20 的 `chat_tmpl.jinja` 使用 DeepSeek 风格的 `<|audio_bos|><|AUDIO|><|audio_eos|>` placeholder，与 Qwen3-Omni 不兼容
- Qwen3-Omni 期望的 placeholder 是 `<|audio_start|><|audio_pad|><|audio_end|>`（对应 token_id 151669/151675/151670）

**Template 内容**:
```jinja
{%- for message in messages -%}
{%- if message['content'] is string -%}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% else -%}
<|im_start|>{{ message['role'] }}
{% for content in message['content'] -%}
{%- if content['type'] == 'text' -%}
{{ content['text'] }}
{%- elif content['type'] == 'audio' or content['type'] == 'input_audio' or ... -%}
<|audio_start|><|audio_pad|><|audio_end|>
{%- elif content['type'] == 'image' or ... -%}
<|vision_start|><|image_pad|><|vision_end|>
{%- endif -%}
{%- endfor -%}
<|im_end|>
{% endif -%}
{%- endfor -%}
<|im_start|>assistant
```

**启动参数**: `--chat-template /opt/qwen3_omni_chat_tmpl.jinja --chat-template-content-format openai`

---

### Patch 8: ServingChat __init__ 参数兼容

**文件**: `/root/.conda/envs/py310/lib/python3.10/site-packages/vllm_omni/entrypoints/openai/serving_chat.py`

**问题**: `OmniOpenAIServingChat` 的 `__init__` 在紫霄环境启动时因缺少 `openai_serving_render` 参数导致初始化失败。

**修改** (line 98):
```python
class OmniOpenAIServingChat(OpenAIServingChat, AudioMixin):
    # ZX_SERVING_CHAT_INIT_PATCH
    def __init__(self, *args, openai_serving_render=None, **kwargs):
        self.openai_serving_render = openai_serving_render
        super().__init__(*args, **kwargs)
```

**原理**: vllm_omni 的 API server 在构造 `OmniOpenAIServingChat` 时可能传入 `openai_serving_render` 关键字参数（取决于 vllm 底层版本），但原始类未声明该参数。加上默认值 `None` 的声明后，兼容有无该参数的两种调用方式。

---

## 三、环境变量

启动服务时需设置：

```bash
export DS_IGNORE_CUDA_DETECTION=1      # DeepSpeed 跳过 CUDA 检测
export DISABLE_VERSION_CHECK=1          # 跳过非关键版本检查
```

注意：**不需要** `TENCENT_TORCH_ZX_UTILS_ENABLE_AUTO_MIGRATION=True`。vllm_zx 插件会自动完成 CUDA→ZX 的迁移。

---

## 四、Stage 配置 (Qwen3-Omni 全链路)

**文件**: `/tmp/qwen3_omni_full_zx.yaml`

```yaml
stage_args:
  - stage_id: 0          # Thinker (理解 + 文本生成)
    devices: "0,1"       # 2 卡 TP
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.90
    hf_config_name: thinker_config
    max_model_len: 4096

  - stage_id: 1          # Talker (文本 → codec tokens)
    devices: "2"         # 单卡
    gpu_memory_utilization: 0.60
    hf_config_name: talker_config

  - stage_id: 2          # Code2Wav (codec → 音频)
    devices: "3"         # 单卡
    gpu_memory_utilization: 0.10
    hf_config_name: thinker_config  # Code2Wav 用 thinker_config 加载
```

---

## 五、验证结果

### Qwen2.5-Omni-7B (全链路)

| 测试 | 结果 | 耗时 |
|------|------|------|
| 文本输入 → 文本+语音输出 | ✅ | ~3s |
| 音频输入 → 文本+语音回复 | ✅ | ~3.13s |

### Qwen3-Omni-30B-A3B (全链路)

| 测试 | 结果 | 耗时 |
|------|------|------|
| 文本输入 → 文本+语音输出 | ✅ | ~15s |
| 音频输入 → 语音回复 | ✅ | ~109s |

> 30B 模型慢的原因：MoE 在单卡 eager 模式无 CUDA Graph，Talker decode 每 token 需完整 forward。

---

## 六、已知限制

1. **`enforce_eager=true` 必须开启**: 紫霄暂不支持 vllm 的 CUDA Graph capture
2. **torch.compile 不可用**: ZX Inductor backend 尚未成熟
3. **is_cuda_alike=False 的副作用**: diffusion 模块的 flash_attn 检测被跳过（不影响 Omni 推理）
4. **30B MoE 全链路较慢**: Talker 单卡 decode ~0.5s/token，建议用 7B 做快速验证
5. **test_single_pcm.py 不直接兼容**: 该脚本使用 DeepSeek 风格全角角色名，需改为标准 `user/assistant/system`

---

## 七、文件清单（全部 15 个修改文件）

| # | 文件路径 (Pod 内，省略 `/root/.conda/envs/py310/lib/python3.10/site-packages/` 前缀) | 类型 | 修改时间 |
|---|---|---|---|
| 1 | `vllm_omni/platforms/zx_platform.py` | 新建 | 14:39 |
| 2 | `vllm_omni/platforms/__init__.py` | 修改 | 14:32 |
| 3 | `vllm/entrypoints/serve/render/serving.py` | 新建 | 11:23 |
| 4 | `vllm/entrypoints/openai/models/serving.py` | 修改 | 11:20 |
| 5 | `vllm_omni/entrypoints/openai/api_server.py` | 修改 | 11:43 |
| 6 | `vllm_omni/entrypoints/openai/serving_chat.py` | 修改 | 12:04 |
| 7 | `/opt/qwen3_omni_chat_tmpl.jinja` | 新建 | 15:31 |
| 8 | `vllm_omni/model_executor/models/qwen2_5_omni/qwen2_5_omni_token2wav.py` | 修改 | 14:24 |
| 9 | `vllm_omni/model_executor/models/qwen2_5_omni/qwen2_5_omni.py` | 修改 | 14:44 |
| 10 | `vllm_omni/model_executor/models/qwen2_5_omni/qwen2_5_omni_thinker.py` | 修改 | 11:56 |
| 11 | `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py` | 修改 | 11:56 |
| 12 | `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_code_predictor_mtp.py` | 替换 | 11:43 |
| 13 | `vllm_omni/worker/gpu_generation_model_runner.py` + `gpu_ar_model_runner.py` | 修改 | 14:44 |
| 14 | `vllm_omni/worker/gpu_model_runner.py` | 修改 | 11:53 |
| 15 | `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` | 修改 | 14:57 |

**辅助文件**:
| 文件 | 说明 |
|---|---|
| `/tmp/qwen3_omni_full_zx.yaml` | 3-stage 全链路配置 |
| `/tmp/patch_zx_moe_kernel.py` | MoE patch 自动化脚本 |
| `/tmp/test_audio_reply.py` | 音频回复测试脚本 |

---

## 八、复现步骤

```bash
# 1. 进入 Pod
kubectl exec -it cgame-partner-llm-zx-c200-debug-66f5bdfc79-fbvk5 \
  -n ieg-gztechtke-zixiao-c200-nj -- /bin/bash

# 2. 应用 MoE patch (首次)
python3 /tmp/patch_zx_moe_kernel.py

# 3. 启动 Qwen3-Omni 全链路
MODEL=/root/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695
export DS_IGNORE_CUDA_DETECTION=1 DISABLE_VERSION_CHECK=1

vllm-omni serve $MODEL \
  --omni \
  --stage-configs-path /tmp/qwen3_omni_full_zx.yaml \
  --served-model-name qwen3-omni \
  --host 0.0.0.0 --port 8081 \
  --generation-config vllm \
  --chat-template /opt/qwen3_omni_chat_tmpl.jinja \
  --chat-template-content-format openai

# 4. 测试 (等 health=200 后)
python3 /tmp/test_audio_reply.py
```
