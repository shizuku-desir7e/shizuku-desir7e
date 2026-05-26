# Talker MTP Speculative Decoding 改造方案

## Pod 环境

```bash
kubectl exec -it cgame-partner-vllm-multimodal-omni-debug-79cc6d5c5c-dplsb \
  -n ieg-gztechtke-aigc-h20-nj -- /bin/bash
```

---

## 1. MTP 结构分析

这个模型的 Code Predictor **本身就是 MTP（Multi-Token Prediction）结构**：

```
Code Predictor 结构:
├── model: 5层 Decoder (共享 backbone)
├── codec_embedding: 31 × Embedding(2048, 1024)  ← 每组单独 embedding
├── lm_head: 31 × Linear(1024, 2048)             ← 每组单独 head
├── small_to_mtp_projection: Linear(1024, 1024)   ← 输入投影
└── generation_steps: 控制当前预测第几组
```

**MTP 核心逻辑**（`forward` 方法）：

```python
# Prefill 阶段: inputs_embeds.shape[1] > 1
if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
    generation_steps = inputs_embeds.shape[1] - 2  # hidden & layer 0

# Generation 阶段: 逐步预测
else:
    # 用 generation_steps 对应的 embedding 层
    inputs_embeds = self.model.get_input_embeddings()[generation_steps - 1](input_ids)

# 共享 backbone 前向
outputs = self.model(inputs_embeds=self.small_to_mtp_projection(inputs_embeds))

# 用 generation_steps 对应的 head 输出
logits = self.lm_head[generation_steps](outputs.last_hidden_state)

# 返回 generation_steps + 1 (下一步)
return ..., generation_steps=generation_steps + 1
```

**关键洞察**：Code Predictor 在每个时间步 t 的行为是：
```
输入: [talker_hidden_t, codec_0_embed, codec_1_embed, ..., codec_i_embed]
       ↓
    5层 Decoder (共享参数)
       ↓
    lm_head[i] → 预测 codec_{i+1}
```

这**天然就是 Speculative Decoding 的 draft model**！

---

## 2. 改造方案：MTP-based Speculative Decoding for Talker

### 核心思路

```
现状 (Talker 生成 codec_0):
  Step 1: Talker(20层) → codec_0[1]
  Step 2: Talker(20层) → codec_0[2]
  Step 3: Talker(20层) → codec_0[3]
  ...
  每步 15ms, T步 = 15T ms

改造后 (Speculative):
  Draft:   CodePredictor(5层) → 快速猜 codec_0[2], codec_0[3], codec_0[4]
  Verify:  Talker(20层) 一次验证 [codec_0[1], codec_0[2], codec_0[3], codec_0[4]]
  结果:    一次前向产出 3-4 个有效 token
```

但注意：Code Predictor 原本是预测 **不同组**（codec_1~31），不是预测同一组的**下一时间步**。

所以真正的方案是：

### 方案 A：Early Exit Speculative（推荐，最直接）

```python
# 用 Talker 的前 N 层做 draft，完整 20 层做 verify
# 无需改动 Code Predictor
```

### 方案 B：将 Code Predictor 适配为时间步 Draft（需微调）

```python
# 复用 Code Predictor 的 5层 backbone
# 加一个新的 draft_head 预测 codec_0 的下一步
# 需要少量微调
```

---

## 3. 方案 A 实现代码（Early Exit，即刻可用）

在 Pod 上创建以下文件：

```bash
# 进入 pod 后
cd /path/to/deepseek_omni
vim talker_speculative.py
```

```python
"""
talker_speculative.py - Talker Speculative Decoding (Early Exit)

原理:
  - Draft: 用 Talker 前 5 层快速生成 gamma 个候选 codec_0 token
  - Verify: 完整 20 层一次性验证
  - 加速: 每次验证产出多个 token，减少前向次数
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.cache_utils import DynamicCache


class TalkerSpeculativeDecoder:
    """
    Early Exit Speculative Decoding for Talker
    
    利用 Talker 20 层中的前 N 层作为 draft model，
    完整 20 层作为 target model 验证。
    """
    
    def __init__(
        self,
        talker,                   # Qwen3TTSTalkerForConditionalGeneration
        draft_layers: int = 5,    # Draft 使用前几层
        gamma: int = 4,           # 每次投机猜测的 token 数
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
    ):
        self.talker = talker
        self.draft_layers = draft_layers
        self.gamma = gamma
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        self.model = talker.model      # Qwen3TTSTalkerModel (20层)
        self.codec_head = talker.codec_head  # Linear(1024, 3072)
        self.eos_token_id = talker.config.codec_eos_token_id  # 4198
    
    def _draft_forward_single(self, hidden_states, position_embeddings, 
                               causal_mask, past_key_values, cache_position):
        """
        只跑前 draft_layers 层，快速预测一个 token
        """
        for layer_idx in range(self.draft_layers):
            layer = self.model.layers[layer_idx]
            layer_output = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=None,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_output[0]
        
        # Norm + Head
        normed = self.model.norm(hidden_states)
        logits = self.codec_head(normed[:, -1:])
        return logits, hidden_states
    
    def _full_forward(self, inputs_embeds, past_key_values, cache_position, 
                      attention_mask, position_ids):
        """
        完整 20 层前向，一次处理多个 token（验证阶段）
        """
        outputs = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            output_hidden_states=True,
        )
        
        logits = self.codec_head(outputs.last_hidden_state)  # [1, seq_len, vocab]
        return logits, outputs.past_key_values, outputs.hidden_states
    
    def _sample_token(self, logits):
        """采样一个 token"""
        if self.temperature == 0:
            return logits.argmax(dim=-1)
        
        logits = logits / self.temperature
        
        # Top-K
        if self.top_k > 0:
            top_k_values, _ = torch.topk(logits, self.top_k, dim=-1)
            logits[logits < top_k_values[..., -1:]] = float('-inf')
        
        # Top-P
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cum_probs > self.top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            indices_to_remove = sorted_indices[remove_mask]
            logits.view(-1, logits.size(-1))[:, indices_to_remove] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(logits.shape[:-1])
    
    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Speculative Decoding 主循环
        
        Returns:
            generated_tokens: [T] codec_0 token 序列
            stats: 统计信息 (acceptance rate, etc.)
        """
        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        assert batch_size == 1, "Speculative decoding only supports batch_size=1"
        
        # === Prefill ===
        past_key_values = DynamicCache()
        seq_len = inputs_embeds.shape[1]
        cache_position = torch.arange(seq_len, device=device)
        
        # 完整前向做 prefill
        prefill_out = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        past_key_values = prefill_out.past_key_values
        
        # 第一个 token
        first_logits = self.codec_head(prefill_out.last_hidden_state[:, -1:])
        first_token = self._sample_token(first_logits.squeeze(1))
        
        generated = [first_token]
        current_pos = seq_len
        
        # Stats
        total_draft = 0
        total_accepted = 0
        
        # === Decode Loop ===
        while len(generated) < max_new_tokens:
            if generated[-1].item() == self.eos_token_id:
                break
            
            # --- Draft Phase ---
            # 保存 draft 前的 KV Cache 状态（用于回滚）
            draft_past_kv = past_key_values.clone() if hasattr(past_key_values, 'clone') else None
            
            draft_tokens = []
            draft_input = self.model.codec_embedding(generated[-1].unsqueeze(0))  # [1, 1, 1024]
            
            # 使用独立的 draft KV Cache (仅前 draft_layers 层)
            for step in range(self.gamma):
                cache_pos = torch.tensor([current_pos + step], device=device)
                
                # 前 5 层快速前向
                h = draft_input
                for layer_idx in range(self.draft_layers):
                    layer = self.model.layers[layer_idx]
                    layer_out = layer(
                        h,
                        attention_mask=None,  # causal by default
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=cache_pos,
                    )
                    h = layer_out[0]
                
                # Norm + sample
                normed = self.model.norm(h)
                logits = self.codec_head(normed[:, -1:])
                token = self._sample_token(logits.squeeze(1))
                draft_tokens.append(token)
                
                if token.item() == self.eos_token_id:
                    break
                
                draft_input = self.model.codec_embedding(token.unsqueeze(0))
            
            total_draft += len(draft_tokens)
            
            # --- Verify Phase ---
            # 回滚 KV Cache 到 draft 前的状态
            # 然后用完整 20 层一次性处理所有 draft tokens
            
            # 构造验证输入: [last_accepted_token] + [draft_tokens]
            verify_tokens = torch.cat([generated[-1].unsqueeze(0)] + 
                                      [t.unsqueeze(0) for t in draft_tokens], dim=0)
            verify_embeds = self.model.codec_embedding(verify_tokens).unsqueeze(0)  # [1, gamma+1, 1024]
            
            verify_cache_pos = torch.arange(
                current_pos - 1, current_pos - 1 + verify_embeds.shape[1], device=device
            )
            
            # 完整 20 层验证 (需要从 draft 前的 KV 状态开始)
            # 这里简化处理: 重新跑剩余层
            verify_out = self.model(
                input_ids=None,
                inputs_embeds=verify_embeds,
                past_key_values=past_key_values,  # 注意: 这里需要正确的 KV
                use_cache=True,
                cache_position=verify_cache_pos,
            )
            verify_logits = self.codec_head(verify_out.last_hidden_state)  # [1, gamma+1, vocab]
            
            # --- Accept/Reject ---
            n_accepted = 0
            for i, draft_t in enumerate(draft_tokens):
                # verify_logits[:, i] 是对 position i 的预测 (预测 i+1 的 token)
                target_token = self._sample_token(verify_logits[:, i + 1])
                
                if target_token.item() == draft_t.item():
                    generated.append(draft_t)
                    n_accepted += 1
                else:
                    # 拒绝 draft，使用 target 的结果
                    generated.append(target_token)
                    n_accepted += 1
                    break
            
            total_accepted += n_accepted
            current_pos += n_accepted
            
            # 更新 KV Cache (截断到 accepted 位置)
            past_key_values = verify_out.past_key_values
            # TODO: 需要截断到正确位置
        
        # 移除 EOS
        result = torch.cat([t.view(1) for t in generated], dim=0)
        if result[-1].item() == self.eos_token_id:
            result = result[:-1]
        
        stats = {
            "total_tokens": len(generated),
            "total_draft": total_draft,
            "total_accepted": total_accepted,
            "acceptance_rate": total_accepted / max(total_draft, 1),
            "speedup_estimate": total_accepted / max(len(generated) - total_accepted + total_draft, 1),
        }
        
        return result, stats


class TalkerMTPSpeculativeDecoder:
    """
    利用 MTP Code Predictor 结构做 Speculative Decoding
    
    核心思想:
      Code Predictor 本身是 MTP 结构 (5层, 31个 head)
      它习惯于接收 talker_hidden 然后预测多组 codec
      我们可以复用它的 backbone 来预测 codec_0 的未来时间步
      
    注意: 这需要微调 Code Predictor，让它除了预测不同组，
          也能预测同一组的未来步。或者利用它作为 "能力相近" 的 draft。
    """
    
    def __init__(
        self,
        talker,
        gamma: int = 3,
        temperature: float = 0.9,
    ):
        self.talker = talker
        self.gamma = gamma
        self.temperature = temperature
        
        # Code Predictor 的 backbone (5层) 作为 draft
        self.draft_model = talker.code_predictor.model
        self.draft_projection = talker.code_predictor.small_to_mtp_projection
        
        # Target: Talker 主模型 (20层)
        self.target_model = talker.model
        self.codec_head = talker.codec_head
        self.codec_embedding = talker.model.codec_embedding
        
        self.eos_token_id = talker.config.codec_eos_token_id
    
    @torch.no_grad()
    def draft_step(self, hidden, past_kv):
        """用 Code Predictor 的 5 层 backbone 做快速预测"""
        projected = self.draft_projection(hidden)
        out = self.draft_model(
            inputs_embeds=projected,
            past_key_values=past_kv,
            use_cache=True,
        )
        # 用 codec_head (共享) 预测 codec_0
        logits = self.codec_head(out.last_hidden_state[:, -1:])
        return logits, out.past_key_values
    
    @torch.no_grad()
    def generate(self, inputs_embeds, max_new_tokens=200, **kwargs):
        """MTP-based Speculative Generate"""
        # ... 类似上面的逻辑，但 draft 用 Code Predictor backbone
        pass
```

---

## 4. 在 Pod 上的操作步骤

```bash
# 1. 进入 Pod
kubectl exec -it cgame-partner-vllm-multimodal-omni-debug-79cc6d5c5c-dplsb \
  -n ieg-gztechtke-aigc-h20-nj -- /bin/bash

# 2. 找到代码目录
find / -name "modeling_deepseek_omni.py" 2>/dev/null
# 或
find / -name "deepseek_omni_model" -type d 2>/dev/null

# 3. 查看 Python 环境
which python3
pip list | grep -E "torch|transformers|vllm"

# 4. 创建 speculative decoding 文件
cd /path/to/project
cat > talker_speculative.py << 'EOF'
# ... (上面的代码)
EOF

# 5. 创建测试脚本
cat > test_speculative.py << 'EOF'
"""测试 Speculative Decoding 加速效果"""
import time
import torch
from deepseek_omni_model.register import register_deepseek_omni
register_deepseek_omni()
from deepseek_omni_model import Qwen3OmniForConditionalGeneration, CustomQwen3OmniAudioProcessor
from talker_speculative import TalkerSpeculativeDecoder

MODEL_PATH = "/path/to/checkpoint"  # 替换为实际路径

# 加载模型
model = Qwen3OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype='auto', attn_implementation='flash_attention_2', device_map="auto"
)

# 获取 Talker
talker = model.talker

# 初始化 Speculative Decoder
spec_decoder = TalkerSpeculativeDecoder(
    talker=talker,
    draft_layers=5,
    gamma=4,
    temperature=0.9,
)

# 构造测试输入 (模拟 Talker 的 prefill 输入)
# 实际使用时从 Thinker 的输出构造
test_embed = torch.randn(1, 50, 1024, device=talker.device, dtype=talker.dtype)

# Baseline: 原始自回归
print("=== Baseline (Original AR) ===")
t0 = time.time()
# ... 原始 generate 调用
baseline_time = time.time() - t0

# Speculative
print("\n=== Speculative Decoding ===")
t0 = time.time()
tokens, stats = spec_decoder.generate(test_embed, max_new_tokens=100)
spec_time = time.time() - t0

print(f"  Generated tokens: {stats['total_tokens']}")
print(f"  Acceptance rate:  {stats['acceptance_rate']:.2%}")
print(f"  Time: {spec_time*1000:.1f}ms")
print(f"  Speedup estimate: {stats['speedup_estimate']:.2f}x")
EOF

# 6. 运行测试
python3 test_speculative.py
```

---

## 5. MTP 机制的正确利用方式

Code Predictor 的 MTP 结构**不是**用来加速 codec_0 生成的，它是用来**并行预测 codec_1~31**的。但我们可以这样利用：

### 方式 1：利用 MTP 的训练前知识做 codec_0 的 Draft

```python
# Code Predictor 见过大量 talker_hidden → codec 的映射
# 虽然它训练时预测的是不同组，但它的 backbone 对音频 token 的模式有"理解"
# 可以直接复用它的 5 层 backbone + 共享 codec_head 来猜 codec_0 的下一步

# 关键: Code Predictor 的输入是 talker_hidden_t
# Talker 每步的 hidden_state 就是 Code Predictor 的输入
# 所以 Code Predictor 天然能"看到" Talker 在想什么
```

### 方式 2：MTP 并行 + Speculative 组合

```
优化后的完整流程 (每个时间步 t):

1. Talker(20层) → codec_0[t]  (自回归，或用 Speculative 加速)
2. Code Predictor:
   - Prefill: [talker_hidden_t, codec_0[t]_embed]
   - MTP step 1: → codec_1[t]   (head[0])
   - MTP step 2: → codec_2[t]   (head[1])
   - ...
   - MTP step 31: → codec_31[t] (head[30])
   
   这 31 步可以用 forward_finetune 一次并行出来！
```

### 方式 3：MTP 的 `forward_finetune` 并行化

```python
# 当前: Code Predictor 逐步自回归预测 31 组
# forward_finetune 实际上已经支持一次性并行:

def forward_finetune(self, inputs_embeds, labels):
    """
    inputs_embeds: [batch, 32, hidden]
      = [talker_hidden, codec_0_embed, codec_1_embed, ..., codec_30_embed]
    
    一次前向，输出所有 31 个位置的 logits
    logits[i] 是用 lm_head[i] 预测 codec_{i+1}
    """
    projected = self.small_to_mtp_projection(inputs_embeds)
    outputs = self.model(inputs_embeds=projected)
    hidden = outputs.last_hidden_state
    
    # 并行输出所有组！
    logits = []
    for i in range(1, self.config.num_code_groups):
        logits.append(self.lm_head[i-1](hidden[:, i]))
    return torch.stack(logits, dim=1)  # [batch, 31, vocab]
```

**这意味着**：推理时 Code Predictor 部分可以**一次前向出 31 组结果**，而不是逐步自回归！

---

## 6. 即刻可执行的优化（不需要训练）

```python
# optimize_code_predictor.py
"""
将 Code Predictor 从逐步自回归改为一次并行预测
这是最简单且最确定的优化
"""

def parallel_predict_residual_codes(talker, talker_hidden_t, codec_0_t):
    """
    一次前向预测所有 31 组残余 codec
    替代原来的逐步自回归 generate
    
    Args:
        talker_hidden_t: [1, 1024] - Talker 在时间步 t 的 hidden state
        codec_0_t: int - 时间步 t 的 codec_0 token
    
    Returns:
        all_codes: [32] - 完整的 32 组 codec
    """
    code_predictor = talker.code_predictor
    
    # 构造输入: [hidden, codec_0, codec_1_pred, codec_2_pred, ...]
    # 第一步: 准备好 hidden 和 codec_0
    inputs = [talker_hidden_t.unsqueeze(0)]  # [1, 1, 1024]
    inputs.append(talker.model.codec_embedding(
        torch.tensor([[codec_0_t]], device=talker_hidden_t.device)
    ))  # [1, 1, 1024]
    
    # 贪心自回归 (但在 Code Predictor 5层内，很快)
    all_codes = [codec_0_t]
    
    for step in range(31):
        # 拼接当前所有已知的 embedding
        current_embeds = torch.cat(inputs, dim=1)  # [1, step+2, 1024]
        
        # Code Predictor 前向
        projected = code_predictor.small_to_mtp_projection(current_embeds)
        outputs = code_predictor.model(inputs_embeds=projected)
        
        # 用对应的 head 预测下一组
        logits = code_predictor.lm_head[step](outputs.last_hidden_state[:, -1])
        next_code = logits.argmax(dim=-1).item()
        all_codes.append(next_code)
        
        # 用对应的 embedding 层编码
        if step < 30:
            next_embed = code_predictor.model.codec_embedding[step](
                torch.tensor([[next_code]], device=talker_hidden_t.device)
            )
            inputs.append(next_embed)
    
    return torch.tensor(all_codes)


def parallel_predict_all_codes_oneshot(talker, talker_hidden_t, codec_0_t):
    """
    终极并行版: 利用 forward_finetune 的并行特性
    
    训练时 forward_finetune 接收完整的 [hidden, c0, c1, ..., c30]
    推理时我们可以:
      1. 先用 codec_0 做第一步
      2. 然后把预测的 codec_1 放入，再预测 codec_2
      3. ... 但这还是串行的
      
    真正的 one-shot 需要:
      假设所有组独立 (近似), 直接从 [hidden, codec_0] 预测所有组
    """
    code_predictor = talker.code_predictor
    
    # 只用 hidden + codec_0 做预测 (忽略组间依赖)
    inputs = torch.cat([
        talker_hidden_t.unsqueeze(0),  # [1, 1, 1024]
        talker.model.codec_embedding(
            torch.tensor([[codec_0_t]], device=talker_hidden_t.device)
        )
    ], dim=1)  # [1, 2, 1024]
    
    projected = code_predictor.small_to_mtp_projection(inputs)
    outputs = code_predictor.model(inputs_embeds=projected)
    hidden = outputs.last_hidden_state  # [1, 2, 1024]
    
    # 用最后位置的 hidden 并行预测所有组
    all_codes = [codec_0_t]
    for i in range(31):
        logits = code_predictor.lm_head[i](hidden[:, -1])
        all_codes.append(logits.argmax(dim=-1).item())
    
    return torch.tensor(all_codes)
```

---

## 7. 总结：现在能做什么

| 优化 | 能否立刻做 | 预期加速 | 改动量 |
|------|----------|---------|--------|
| **Code Predictor 一次并行出31组** | ✅ 立刻 | Code Predictor 部分 5-10x | 替换 generate → forward_finetune |
| **Early Exit Speculative (Talker)** | ✅ 立刻 | Talker 部分 1.5-2x | 新增 speculative 逻辑 |
| **MTP backbone 做 draft** | ⚠️ 需验证 acceptance rate | 如果 >70% 则 1.5x | 复用 Code Predictor |
| **微调 draft head** | ❌ 需训练 | 2-3x | 训练 + 部署 |

**第一优先级**：把 Code Predictor 的 31 组预测从自回归改为并行——这是最确定、最容易的加速。
