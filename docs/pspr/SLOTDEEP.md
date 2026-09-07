# SlotDeep — 当前最优 PSPR 头：架构与训练

> 代码：`specforge/modeling/draft/pspr_slotdeep.py`（283 行）
> 类：`_SlotBlock` → `SlotDeepCorrector(ClozeCorrector)` → `PSPRSlotDeepDraftModel(PSPRClozeDraftModel)`
> 总览与家族对比见 [`README.md`](./README.md)

---

## 1. 它为什么存在

SlotDeep 是对 cloze 的一次**结构性简化 + 定向增容**，动因来自三个推理期消融
（同一份 cloze@7115 权重，参数量、FLOPs、候选格完全不变，只改 attention mask）：

| 消融 | macro | Δ |
|---|---:|---:|
| full bidirectional | 6.0438 | — |
| causal（去掉未来） | 6.0427 | −0.0012 |
| anchor_self（去掉**所有** cross-slot） | 5.9993 | −0.0445 |
| zero_context（encoder 输出置零） | 5.5352 | −0.5086 |

读法：encoder 那条路径总共值 ~0.509，其中 cross-slot 信息只值 **0.045**，未来信息只值 **0.001**，
剩下 **~91% 是一个 per-slot 的非线性变换**。

用一个 6 层双向 Transformer 跑 `H×(H+1)` 个 cell 来买一个 per-slot MLP 是极贵的：
cloze 每 block 8.63 ms，v2 只要 6.16 ms。

**所以 SlotDeep 把 cross-slot attention 整个拆掉，换成 9 层 per-slot 残差塔。**

### 1.1 为什么这不是"调参失败"而是结构性的

- 骨干在 block 内是全 attention（`dflash.py:245`, `is_causal=False`）⟹ `h_i` 早已聚合整个 block。
- 候选格从**冻结的** tied `lm_head` 读出（`dflash.py:785`）⟹ 每个 seed token / log-prob / margin
  都是 `{h_1..h_H}` 的确定性函数。
- 所以 cloze 序列携带的信息**不超过** `{h_j}` 本身，cross-slot attention 在重新组装
  骨干已经组装过的东西。
- 更糟：`direct_hidden=True` 时 query 行仍拿到全宽 `h_i`，而 `argmax(lm_head(h_i))` 就是被 mask
  的那个 token —— **query 位置从来没有被真正 mask 过，它就不是一个 cloze**。

> **不能过度解读**：以上消融**不**构成"91%/8.4% 因果贡献分解"，也**不**证明双向信息无用。
> 它只说明在这个具体的骨干/格/预算下，attention 不划算。接受长度增益仍是实验性的。

---

## 2. 架构

### 2.1 全景

```
                     ┌───────────────────────────────────────┐
hidden_states h ────▶│ h_in(h_ln(h))                         │
[.., H, 2560]        │       +                               │
log_probs, scalars ─▶│ conf_in(conf_ln(confidence(lp, sc)))  │──▶ base [.., H, 512]
                     └───────────────────────────────────────┘
                                      │
anchor_ids ─▶ tok_in(tok_ln(E(a))) + anchor_feat            │
      + role_emb[ANCHOR] + pos_emb[0]  ──▶ anchor [.., 1, 512]
                                      │
                    ┌─────────────────▼──────────────────┐
        anchor_fusion == "concat":  anchor_fuse([base ; anchor.expand])
                    │  == "sum" :  base + anchor         │
                    └─────────────────┬──────────────────┘
                                      │
                        + mask_emb + role_emb[QUERY] + pos_emb[1:H+1]
                                      │
                                 cell_ln(·)
                                      │
                   ┌──────────────────▼──────────────────┐
                   │  9 × _SlotBlock（pre-norm 残差 FFN）│   z = z + block(z)
                   │  d=512, ff=2048, GELU, dropout 0.05 │
                   └──────────────────┬──────────────────┘
                                 slot_out_ln(·)
                                      │
                                    z [.., H, 512]
                                      │
   ┌──────────────────────────────────┼──────────────────────────────────┐
   ▼                       ▼            ▼
delta = W2·silu(W1[LN_h(h);LN_z(z);LN_s(S)])   err_logits(...)     candidate rank residual
   │  (W2 零初始化)                    (frontier 检测器)             (rank_out 零初始化)
   ▼
score_k = log_probs_k + <E(c_k), delta>  ────────────────────▶  + (residual - residual[0])
                                          ▲
              S = GRU(E(已提交前缀))  ─────┘
```

### 2.2 `_SlotBlock`：pre-norm 残差 FFN

```python
class _SlotBlock(nn.Module):
    def __init__(self, d, ff, dropout):
        self.ln  = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ff)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff, d)
    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(self.ln(x)))))
```

三个刻意的设计：

1. **命名子层，不用 `nn.Sequential`。**
   `Optimizer._parameter_groups` 用 `name.startswith(prefix) or f".{prefix}" in name` 选权重衰减
   （`specforge/optimizer.py:94`），而配方的规则表就是用这套词汇写的（`linear1.weight`,
   `err_head.1.weight`, ...）。用 `nn.Sequential` 会产生 `slot_blocks.0.1.weight`，只能被
   `1.weight` 这种规则命中 —— 而那条规则**同时**会命中 `err_head.1.weight` 和
   `err_state_head.1.weight`。命名子层让规则表只有两行、与深度无关、且不可能串味。

2. **pre-norm 而非 post-norm。**
   9 层深、`initializer_range=0.02` 下 post-norm 条件数很差；而且它替换的那个 encoder
   本来就是 `norm_first=True`。

3. **block 输出不做零初始化。**
   `delta_w2` 已经保证了 step-0 no-op；再把每个 block 的第二个矩阵清零，只会让第一批矩阵在
   step 0 拿不到梯度，没有任何好处。

**`slot_out_ln`**：父类的 `nn.TransformerEncoder` 没有 final norm，但它也从未把这么深的残差流
交给三个各自归一化的消费者。一个输出 norm 让 `z` 的尺度与 `slot_layers` 无关 ——
否则深度扫描同时也变成了学习率扫描。

### 2.3 `anchor_fusion: concat`

anchor 是"block 在任何 slot 被决定之前就已知的那一个已提交 token"。
父类里它是一个独立的序列 cell，每一行都 attend 它；attention 拆掉之后，它被加进**每一个** slot ——
同样的信息，位置数少了 16 倍。

```python
if self.anchor_fusion == "concat":
    base = self.anchor_fuse(torch.cat([base, anchor.expand_as(base)], dim=-1))   # 2d → d
else:  # "sum"
    base = base + anchor
```

`sum` 是**不可逆**的：塔无法再分离"这是我自己的 h"和"这是 anchor"。`concat` 保留这个区分，
代价是一个 `2d→d` 的线性层（0.525M）。这是 2026-09-05 版本相对原始 SlotDeep 的改动之一。

### 2.4 候选条件打分残差（`candidate_rank_dim: 256`）

**这是 SlotDeep 唯一一处真正的"候选级"计算。** cloze/v2 的整个打分路径都是
"算一个 `delta`，然后所有候选共享地用 `<E(c_k), delta>` 打分" —— 每个候选只通过它自己的
embedding 与一个 slot 级向量的内积参与。这里加一个**读得到具体候选 + 具体分数特征**的小非线性项：

```python
context   = [ fuse_h_ln(h) ; fuse_z_ln(z) ; fuse_s_ln(S) ]        # 2560 + 512 + 512
query     = rank_query_out(silu(rank_query_in(context)))          # → 1024 → 256
candidates= rank_candidate_in(rank_candidate_ln(E(c)))            # 2560 → 256, no bias
features  = [ lp/10, (lp - lp0)/10, (s - s0)/10, is_base ]        # 4 维，全部 detach
joint     = query.unsqueeze(-2) + candidates + rank_features_in(features)
residual  = rank_out(gelu(joint)).squeeze(-1)                     # rank_out 零初始化
return scores + (residual - residual[..., :1])                    # 以候选 0 为中心
```

四个要点：

- **分数特征 detach。** `lp`、`gap = s − s0` 都取 `.detach()`。否则模型能通过"改变主打分头"来
  操纵这个残差自己的输入，制造一条辅助的作弊路径。主分数和 `h/z/S` 仍然从最终的多类 CE 拿联合梯度。
- **`rank_out` 零初始化。** step 0 时 `residual ≡ 0`，SlotDeep 的选择与 plain DFlash 逐位一致。
- **以候选 0 为中心（`residual - residual[..., :1]`）。** 平移不改变 softmax 也不改变分数差，
  但它把参考分数的数值契约写明了：候选 0 是被区分出来的 KEEP。
- **K-restricted，全词表被显式拒绝。**
  ```python
  self.supports_full_vocab = not bool(self.candidate_rank_dim)
  def full_vocab_logits(self, ...):
      if self.candidate_rank_dim:
          raise ValueError("SlotDeep candidate residual is top-K only; full_vocab would drop it")
  ```
  开着这个残差时全词表 serving 会**默默丢掉它的贡献** —— 所以直接报错，而不是静默降级。
  同理，`_init_draft_head` 里强制 `selector_decision_mode == "margin_gate"`。

### 2.5 拒绝无效消融

SlotDeep 对两类"看起来能跑但语义已失效"的开关直接抛错，而不是静默 no-op：

```python
if not kwargs.get("bidirectional", True):
    raise ValueError("SlotDeep has no attention; bidirectional=False is not an ablation")

if pattern != "bidirectional":
    raise ValueError("SlotDeepCorrector has no cross-slot attention, so attention_pattern="
                     f"{pattern!r} cannot be applied; it removes information that is already absent")
```

理由：一个静默 no-op 的诊断比一个失败的诊断更糟 —— `causal` / `anchor_self` 两条 arm 会对一个
**本来就没有 cross-slot attention** 的模型报告"无影响"。

### 2.6 丢掉父类的 encoder

```python
kwargs = dict(kwargs, n_layers=1)   # 构造最便宜的 encoder
super().__init__(**kwargs)
del self.encoder            # 立刻删掉
...
self._config = dict(self._config, n_layers=requested, ...)   # 但把原始 n_layers 记进 config
```

父类无条件构建 dense encoder。构造 6 层只为删除会白白浪费 ~19M 的初始化；而留着不删会把
19M 死参数塞进 optimizer state、checkpoint 和每一个参数量门禁。`n_layers` 仍记录在
`reference_config` 里，所以这个 arm 从存档的 config 依然可辨识。

`PSPRSlotDeepDraftModel._init_draft_head` 则是先让父类建一个 `ClozeCorrector`，读它的
`reference_config()` 拿到全部旋钮名，再用这些值构造 `SlotDeepCorrector` 并替换掉 ——
这样父类是旋钮名的**唯一定义处**，父类新增的旋钮不可能在这里被静默丢掉。

### 2.7 参数量

`outputs/slotdeep_joint_init/model.safetensors` 实测：

| 组件 | 张量数 | 参数 |
|---|---:|---:|
| `slot_blocks`（9 × `_SlotBlock`） | 54 | **18.907M** |
| `delta_w1`（3584 → 2048） | 2 | 7.342M |
| `delta_w2`（2048 → 2560，零初始化） | 1 | 5.243M |
| `gru`（2560 → 512） | 4 | 4.722M |
| `rank_query_in`（3584 → 1024） | 2 | 3.671M |
| `err_head` | 6 | 1.845M |
| `err_seed_in` / `h_in` / `tok_in`（各 2560→512） | 3 | 3.933M |
| `rank_candidate_in`（2560 → 256） | 1 | 0.655M |
| `anchor_fuse`（1024 → 512） | 2 | 0.525M |
| `rank_query_out` / 其余 LN / 嵌入 / `rank_out` | 33 | 0.310M |
| **selector 合计** | **104** | **47.153M** |
| backbone（官方 DFlash-b16，bf16） | 58 | 537.427M |
| **总计** | **162** | **584.58M** |

对比 cloze selector：**42.039M / 108 张量**。SlotDeep 多出的 5.1M 主要是
`anchor_fuse` + rank 分支，而 9 层塔（18.9M）替代了 6 层 dense encoder。

### 2.8 完整配置

`configs/qwen3-4b-pspr-slotdeep.json` 的 `dflash_config`：

```json
{
  "mask_token_id": 151669,
  "target_layer_ids": [1, 9, 17, 25, 33],

  "selector_top_k": 16,           // 候选格宽度 K
  "selector_dim": 512,            // d，塔的宽度
  "selector_layers": 6,           // 惰性：仅记录，encoder 已删
  "selector_heads": 8,            // 惰性
  "selector_delta_hidden": 2048,  // delta MLP 隐层
  "selector_max_slots": 32,
  "selector_dropout": 0.05,
  "selector_state_dim": 512,      // GRU 状态维度
  "selector_direct_hidden": true, // h 全宽进入 delta / err
  "selector_use_state": true,     // 开启 GRU 因果流
  "selector_bidirectional": true, // 必须 true，否则构造期报错

  "selector_slot_layers": 9,      // ★ 塔深度
  "selector_slot_hidden": 0,      // 0 ⟹ ff = 4d = 2048
  "selector_anchor_fusion": "concat",       // ★
  "selector_candidate_rank_dim": 256,       // ★ 0 = 关闭候选残差
  "selector_candidate_query_hidden": 1024,

  "selector_decision_mode": "margin_gate",  // rank_dim>0 时强制
  "selector_gate_rho": 3.0,
  "selector_gate_tau": 0.0,
  "selector_gate_theta": 0.0,     // 检测器不作为活跃门
  "selector_compute_dtype": "float32",      // selector 全程 fp32
  "freeze_backbone": true         // ← Stage 2 唯一改动：false
}
```

`qwen3-4b-pspr-slotdeep-joint.json` 与上面**唯一的语义差异就是 `freeze_backbone: true → false`**
（其余是 JSON 数值格式差异，`3.0` vs `3` 等）。

---

## 3. 训练

### 3.1 时间线（已完成的实际运行）

```
官方 Qwen3-4B-DFlash-b16 (58 张量, bf16)
   │
   │ Stage 1: freeze_backbone=true, selector 从零, lr 5e-4, 1 pass = 7115 步
   ▼
qwen3-4b-pspr-slotdeep-s1-20260905 @step7115     macro 6.0945 / pooled 5.5039
   │
   │ 续训 1000 步，骨干仍冻结，lr 2e-5，加辅助 loss（T0..T3 四组消融）
   ▼
qwen3-4b-pspr-slotdeep-loss-t3-20260905 @step1000   macro 6.1180 / pooled 5.5165
   │
   │ Stage 2: freeze_backbone=false, lr 1e-4, backbone scale 0.3, turnwise 数据, 9052 步
   ▼
qwen3-4b-pspr-slotdeep-stage2-20260906 @step9052    macro 6.4386 / pooled 5.7764
```

**平行对照 arm（均从零起、同 7115 步、均未胜出）**：

| arm | 变量 | macro | pooled |
|---|---|---:|---:|
| `slotdeep-s1-20260905` | 基准配方 | **6.0945** | **5.5039** |
| `turn-s1` | 数据换成 turnwise 切分 | 6.0697 | 5.4876 |
| `distill-s1` | `candidate_distill α=0.5, T=1.0` | 6.0570 | 5.4718 |

### 3.2 Stage 1

配方：`examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-slotdeep.yaml`

```yaml
model:
  draft_model_config: configs/qwen3-4b-pspr-slotdeep.json
  draft_checkpoint_path: <.../Qwen3-4B-DFlash-b16>   # 58 骨干张量，无 selector 张量
  torch_dtype: bfloat16
  sglang_context_length: 3200

data:
  train_data_path: ./cache/dataset/perfectblend_qwen3-4b_regen.jsonl
  max_length: 3072
  chat_template: qwen-nosys          # ← 硬性要求

training:
  strategy: dflash
  num_epochs: 1                      # 7115 步
  batch_size: 2
  accumulation_steps: 2              # global batch = 7 × 2 × 2 = 28
  fsdp_sharding: "NO_SHARD"

  learning_rate: 5.0e-4
  warmup_ratio: 0.06
  max_grad_norm: 1.0
  weight_decay: 0.0
  weight_decay_rules: { ... 只枚举 2-D 权重矩阵，无 catch-all ... }

  attention_backend: flex_attention
  num_anchors: 512
  objective_chunk_blocks: 64         # cloze 家族每 block 跑 H 行，激活峰值是 v2 的 ~7×

  loss_type: dpace
  dpace_alpha: 0.5
  lk_loss_type: lambda
  kl_scale: 1.0
  kl_decay: 1.0

  dflash2_selector_objective: multiclass
  dflash2_selector_weight_mode: uniform_frontier_boost
  dflash2_selector_frontier_boost: 3.0
  dflash2_selector_loss_alpha: 1.0
  dflash2_selector_err_loss_alpha: 1.0
  dflash2_selector_own_denominator: false    # true 已被代码硬拒
  dflash2_selector_warmup_ratio: 0.0
  dflash2_selector_ramp_ratio: 0.0
  dflash2_selector_stop_gradient: false
  dflash2_selector_target_greedy_labels: false

  save_interval: 1000
  seed: 42

deployment:
  trainer: { nnodes: 1, nproc_per_node: 7 }        # GPU 1-7
  disaggregated.managed_local.capture_servers:
    - { port: 30000, cuda_visible_devices: ["0"], tp_size: 1, mem_fraction_static: 0.5 }
```

启动：

```bash
PYTHONPATH=/path/to/sglang-patched-0.5.15:. \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 HF_HUB_OFFLINE=1 \
no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 \
python -u -m specforge.cli train \
  -c examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-slotdeep.yaml
```

**注意：骨干冻结时 `loss_type: dpace` 这一项不产生梯度**，保留它只是因为候选格是从
`objective_logits` 读的，而它的 telemetry 用来确认冻结的特征流与 stage-1 测量时一致。

### 3.3 辅助 loss 消融（T0–T3，均在冻结骨干上从 S1@7115 续训 1000 步，lr 2e-5）

实现见 `dflash_family_model.py:257-316`（`selector_training_auxiliaries`）：

```python
repairable    = covered & (target_index != 0)
base_correct  = covered & (target_index == 0)
reachable     = frontier_mask(weight_mask > 0, policy_ok)      # 解码实际能到达的 slot

alt_ce      = -log_softmax(scores[..., 1:])[target_index - 1]  # 只在 15 个"替代"上做 CE
safe_hinge  = relu(best_alt - base - log(rho) + safe_margin)   # 基座对时，别让替代冒头
repair_hinge= relu(base + log(rho) + repair_margin - truth)    # 基座错时，让真值越过门限
```

三项分别对应三种失败模式：
- `alt_ce` —— "在替代里选谁"这个子问题，绕开 16 路 softmax 里 77.4% 的 class-0 先验
- `safe_hinge` —— **防破坏**：base 是对的时候，最佳替代不许越过 `rho` 门限
- `repair_hinge` —— **促修复**：base 是错的时候，真值必须越过 `rho` 门限

| arm | `alt_α` | `safe_α` | `repair_α` | macro | pooled |
|---|---:|---:|---:|---:|---:|
| T0 | 0 | 0 | 0 | 6.1095 | 5.5041 |
| T1 | 0.5 | 0 | 0 | 6.1157 | 5.5108 |
| T2 | 0.5 | 0.1 | 0.1 | 6.1106 | 5.5108 |
| **T3** | **0.5** | **1.0** | **0.1** | **6.1180** | **5.5165** |

（`safe_margin = repair_margin = 0.1`，全部 `--max-samples 20`）

> **诚实结论：这四组之间无法区分。** 极差 0.0085 macro，而 n=20 分辨不了 0.04 量级。
> T3 被选为 Stage 2 起点是流程上的选择，**不是**统计上被证明的最优。
> 注意所有辅助项都要求 `selector_preserve_fp32=true`，而它又要求
> `selector_compute_dtype='float32'`（`dflash_family_model.py:589-593`）。

### 3.4 Stage 2（已完成的那次，step 9052）

**实际启动命令**（`outputs/SLOTDEEP_stage2_20260906_PROVENANCE/COMMAND.txt`）：

```bash
PYTHONPATH=/path/to/sglang-patched-0.5.15:. \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 HF_HUB_OFFLINE=1 \
no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 \
python -u -m specforge.cli train \
  -c examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-slotdeep.yaml \
  model.draft_model_config=configs/qwen3-4b-pspr-slotdeep-joint.json \
  model.draft_checkpoint_path=<.../slotdeep-loss-t3-20260905-step1000/training_state.pt> \
  data.train_data_path=./cache/dataset/perfectblend_qwen3-4b_regen_turnwise_20260905.jsonl \
  training.num_epochs=1 training.max_steps=9052 training.save_interval=500 \
  training.learning_rate=1.0e-4 \
  'training.lr_scale_rules={"candidate_selector.":1.0,"":0.3}' \
  training.warmup_ratio=0.06 training.prompt_seed=42 \
  runtime.producer_concurrency=1 \
  training.dflash2_selector_preserve_fp32=true \
  training.dflash2_selector_alt_loss_alpha=0.5 \
  training.dflash2_selector_safe_loss_alpha=1.0 \
  training.dflash2_selector_repair_loss_alpha=0.1 \
  training.dflash2_selector_safe_margin=0.1 \
  training.dflash2_selector_repair_margin=0.1 \
  run_id=qwen3-4b-pspr-slotdeep-stage2-20260906 \
  output_dir=./outputs/qwen3-4b-pspr-slotdeep-stage2-20260906 ...
```

即：selector lr `1e-4`、backbone lr `1e-4 × 0.3 = 3e-5`、turnwise 数据、带 T3 的三个辅助项。

结果（n=20/域）：

| mode | macro | pooled |
|---|---:|---:|
| oneshot（骨干-only） | 5.7523 | 5.2223 |
| **latgate（完整）** | **6.4386** | **5.7764** |
| oracle@16 | 10.8688 | 10.3043 |

**两个关键读数**：
1. 骨干-only 从 5.3046 → 5.7523（**+0.448**）—— 解冻确实在改善骨干本身。
2. oracle@16 从 10.4424 → 10.8688（**+0.426**）—— 解冻**没有毁掉候选集**，反而扩大了它的上限。
 这是 stage 2 最需要盯的安全指标（frontier recall 的 stage-1 基线是 **0.8432**）。

### 3.5 Stage 2 规范化配方（`qwen3-4b-pspr-slotdeep-joint.yaml`，待跑）

与 3.4 那次不同，这是一份**从 stage-1 直接派生**、只改必要项的干净配方：

```yaml
model:
  draft_model_config: configs/qwen3-4b-pspr-slotdeep-joint.json   # freeze_backbone: false
  draft_checkpoint_path: ./outputs/slotdeep_joint_init            # ← 见 3.6

training:
  learning_rate: 5.0e-4              # selector 保持 stage-1 基准率
  lr_scale_rules:
    "candidate_selector.": 1.0
    "": 0.06                         # backbone = 5e-4 × 0.06 = 3e-5，比值 16.67×
  save_interval: 200                 # 不是 stage-1 的 1000
  # 其余全部与 stage 1 逐字相同
```

两处设计说明：

- **`lr_scale_rules` 的书写顺序不是 load-bearing。** `assembly.py:275` 会按 key 长度降序排序，
  `""` 这条 catch-all 永远最后评估。
- **`save_interval: 200`。** 1 pass 只有 7115 步，而解冻的骨干**可能在早期破坏 selector 的候选集**，
  所以第一个可回滚的 checkpoint 必须也早。

**Stage 2 的诚实边界（预启动审计结论，写进了 YAML 头部）**：
这是一次带全部已配置 loss 路径的热启动联合续训，**不是**干净的"解冻骨干的效应"测量。
相对 stage 1 它同时还：

1. 重置 Adam moments（warm start 只搬权重）
2. 重启 cosine schedule —— 把一个在 **lr 6.89e-10** 结束的 selector 重新加热回 `5e-4`
3. `max_grad_norm` 跨 selector + backbone 全局生效，骨干梯度触发裁剪时也改变 selector 更新

要把 delta 归因给"解冻"本身，需要一个从同一 joined checkpoint 出发、同样 fresh optimizer /
schedule / 数据序 / 步数、**只有 `freeze_backbone` 不同**的对照。该对照不在本轮范围内。

另外：六域 n=20 的解码**分辨不了这个效应量级**。stage-1 eval 的 bootstrap 给出
S1-vs-baseline macro `+0.0444`，95% CI `[−0.0251, +0.1111]`（跨零）。0.05 macro 的 80% power
需要约 80 prompt/域。同 120 条 prompt 上的 stage-2 数字应视为 **developmental，不是 confirmatory**。

### 3.6 构建 Stage 2 的 warm start

现成的 `scripts/import_pspr_selector.py` **不能用**：它硬断言只认 `PSPRDraftModel` / `LatticeSelector`，
而 cloze 家族的 `model_type` 是 `SlotDeepCorrector`。

因此新增 `scripts/import_pspr_cloze_selector.py`（对称于 `export_pspr_cloze_for_decode.py`，
用 `resolve_draft` registry 构建 probe）：

```bash
PYTHONPATH=. python scripts/import_pspr_cloze_selector.py \
  --selector  outputs/qwen3-4b-pspr-slotdeep-s1-20260905_step7115_decode/selector.pt \
  --draft-config configs/qwen3-4b-pspr-slotdeep-joint.json \
  --output-dir   outputs/slotdeep_joint_init
```

产物 `outputs/slotdeep_joint_init/`（584.58M）已实测：

- selector **104 张量** strict-load OK（fp32）
- backbone **58 张量**与 stage-1 导出**逐位一致**（bf16）
- roundtrip `max|delta| = 0`
- 加载 `unexpected = 0 / missing = 0`

> **与 stage 1 的关键差异**：这里 repairer **不在**零初始化 no-op 上 —— `delta ≠ 0`，
> 所以 step-0 的预测等于 stage-1 arm，**而不是** plain DFlash。

### 3.7 门禁

开训前必须全绿。

**`scripts/gate_slotdeep.py`（G1–G13 ALL PASS）** —— 架构本身：
含 step-0 与 plain DFlash 逐位一致、`selector_slot_layers=9` 下的形状/dtype 契约、
无效消融确实抛错、`full_vocab` 在 `rank_dim>0` 时确实被拒。

**`scripts/gate_slotdeep_joint_groups.py`（G1–G6 ALL PASS）** —— Stage 2 专用，含 9 项 provenance 断言：

| 断言 | 检查 |
|---|---|
| SHA 锁定 | selector 来源 checkpoint 的 sha256 与记录一致（`4bffa768...`，step 7115，`objective=multiclass`，`backbone=frozen`） |
| `torch.equal` 查 dtype | selector fp32 / backbone bf16 均未漂移 |
| lr 分组 | selector **5e-4**（与 stage 1 相同）、backbone **3e-5**（scale 0.06） |
| wd 规则未泄漏 | stage-1 的后缀式 wd 规则**未泄漏**到 backbone（`leaked=[]`） |
| 排序行为 | 按 `assembly.py:275` 排序后的**真实**分组，而非 YAML 顺序 |

> 门禁本身也被审查修正过：原 G2 测的是"YAML 顺序是 load-bearing"这个**不存在的风险**，
> 因为 `assembly.py:275` 会按最长前缀排序 —— 已改成测排序后的真实行为。

**`--plan` 干跑**（不启动 worker，确认进程计划）：

```bash
PYTHONPATH=. python -m specforge.cli train \
  -c examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-slotdeep-joint.yaml --plan
```
应解析出：producer（CPU）+ 7 卡 trainer（GPU 1-7）+ GPU 0 上的 capture server。

**冒烟**：`scripts/run_train_slotdeep_joint_smoke.sh`

### 3.8 导出与评测

```bash
PYTHONPATH=. python scripts/export_pspr_cloze_for_decode.py \
  --checkpoint outputs/qwen3-4b-pspr-slotdeep-stage2-20260906/…-step9052 \
  --output outputs/qwen3-4b-pspr-slotdeep-stage2-20260906_step9052_decode

cd TAPS-SP && python scripts/decode_lattice.py \
  --target-model models/Qwen3-4B \
  --draft-model  <…_decode>/backbone \
  --lattice-head <…_decode>/selector.pt \
  --datasets gsm8k,math500,humaneval,mbpp,alpaca,mt-bench \
  --modes oneshot,latgate,oracle16 \
  --gate-tau 0 --gate-rho 3 --gate-theta 0 \
  --eval-reserved --max-samples 20 --max-new-tokens 256 --shuffle-seed 2026 \
  --json-output <out>.json
```

---

## 4. 已知问题与未动的杠杆

### 4.1 结构死区

ρ=3 的 margin gate 下，**41.7% 的"truth 是最佳替代"决策不可触发**（slot 0 高达 **90.9%**）；
slot 0 的 `captured` 仅 **0.71%**。也就是说 SlotDeep 在 slot 0 上基本不起作用 ——
而 slot 0 是每个 block 的第一个决策点。

### 4.2 SlotDeep 换掉 encoder 没有带来可分辨的接受长度增益

同一 seed 2026 manifest、n=20/域：**slotdeep-s1 6.0945 vs cloze 6.0501**，差 **+0.044** ——
恰好落在 n=20 分辨不了的量级上（已有反例：latfront t0.6 在两台机器上 +0.0415 / −0.344 完全反转）。
**这个差不能当作"SlotDeep 更强"的结论**，只能说与"诚实预期持平"一致。

换 encoder 的真实收益在**延迟**（cloze 每 block 8.63 ms vs v2 6.16 ms），不在接受长度。
真正拉平与官方 Domino 差距的是 **stage 2 解冻骨干**（n=50 配对 bootstrap：
+0.3444，CI `[+0.2922, +0.3996]`，显著）。

### 4.3 最大未动杠杆：解码器工程（零模型改动）

head 延迟的 **72.5% 在 15 步串行 Python 循环 + 每槽 2 次 device→host 同步**，两臂完全相同。
换 encoder 只省 **0.767 ms**（tok/s +2.7）。

消除 host sync + CUDA-graph 化 `TAPS-SP/scripts/decode_lattice.py` 可达 **~166 tok/s**
（超过 Domino 的 160.4），**不需要任何模型改动**。这是目前 ROI 最高的方向。

### 4.4 未完成项

- 真 LOO 消融从未实现
- 三套独立集合（train / dev / test）未拆
- `_sample_anchor_positions` 的均匀采样偏移未修（`dflash_family_model.py:832`）
- Stage 2 的 `freeze_backbone`-only 对照未跑（见 3.5 的诚实边界）
