# PSPR 训练方案设计：路线 A 与路线 B

> 适用仓库：`wzyyy-lab/specforge-pspr`  
> 目标模型：`Qwen/Qwen3-4B`  
> Draft backbone：`z-lab/Qwen3-4B-DFlash-b16`  
> 最终结构：**DFlash backbone + PSPR `LatticePathSelector`**  
> 明确不使用：DFlash2 grouped dynamic convolution、DFlash2 vocab transition codebook（`selector_trans_rank=0`）  
> 分析基准：仓库 `main`，2026-09-02

---

## 0. 文档结论

本文比较两条可执行路线：

- **路线 A：保留 selector 预训练，再联合微调**
  1. 加载官方已训练好的 DFlash backbone；
  2. 冻结 DFlash，只训练 PSPR selector；
  3. 将 selector 权重合并到 DFlash checkpoint；
  4. 解冻 DFlash 与 PSPR，进行端到端联合微调。

- **路线 B：像 DFlash2 一样直接联合训练**
  1. 加载官方已训练好的 DFlash backbone；
  2. 新建 PSPR selector，并使用严格 no-op 的零输出初始化；
  3. 从同一个训练任务的第一轮开始，同时优化 DFlash 与 PSPR；
  4. 通过 selector loss ramp、差分学习率和早停，保护已经收敛的 DFlash。

两条路线都**不要求重新从随机初始化训练 DFlash backbone**。所谓路线 B 的“直接联合训练”，指的是：

```text
官方 DFlash checkpoint + fresh PSPR selector -> 直接 joint training
```

而不是：

```text
随机初始化 DFlash + 随机初始化 PSPR -> 全部从头训练
```

### 推荐排序

当前最稳妥的工程主线是：

1. **先完成路线 A 的正确联合微调**，因为 Stage 1 已经实测把同一冻结 backbone 的 macro acceptance length 从 `5.2842` 提升到 `5.7217`，说明 selector 本身已经有效；
2. 同时保留路线 B 作为等计算预算的关键对照；
3. 最终根据 held-out speculative decode 的真实接受长度选择路线，而不是根据训练 CE 选择。

路线 A 的优势是优化风险低、能复用现有 `best.pt`；路线 B 的优势是训练过程更统一，backbone 与 selector 从一开始共同适配。现阶段没有实验依据可以预先断言路线 B 一定超过路线 A，因此二者必须在相同数据、相同 joint token 数和相同评测协议下比较。

---

## 1. 当前模型与代码语义

当前模型类是：

```python
class PSPRDraftModel(DFlashDraftModel):
    ...
```

因此实际架构是：

```text
Qwen3-4B target lm_head / embedding
              ↑
DFlash 5-layer block-parallel backbone
              ↓
每个 slot 的 Top-K logits、hidden state、不确定性统计
              ↓
PSPR LatticePathSelector
  ├─ 每个 slot 的候选集合摘要
  ├─ 双向 Transformer
  ├─ 已提交正确前缀的 GRU state
  ├─ candidate residual scorer
  ├─ hidden-space delta scorer
  └─ err/frontier head
```

当前固定结构参数如下：

| 参数 | 当前值 | 第一轮是否建议修改 |
|---|---:|---|
| `block_size` | 16 | 否 |
| `selector_top_k` | 16 | 否 |
| `selector_dim` | 512 | 否 |
| `selector_layers` | 3 | 否 |
| `selector_heads` | 8 | 否 |
| `selector_state_dim` | 512 | 否 |
| `selector_delta_hidden` | 2048 | 否 |
| `selector_max_slots` | 32 | 否 |
| `selector_dropout` | 0.1 | 否 |
| `selector_trans_rank` | 0 | 保持 0 |
| grouped convolution | 无 | 保持无 |

在训练路线对比完成前，不应同时修改 selector 的层数、宽度、Top-K 或候选格编码方式。否则无法判断提升来自训练范式还是结构变化。

---

## 2. Backbone 来源：两条路线都直接使用官方 DFlash

### 2.1 推荐 checkpoint

```text
z-lab/Qwen3-4B-DFlash-b16
```

本地目录可对应为：

```text
TAPS-SP/models/Qwen3-4B-DFlash-b16
```

它必须与以下 target 严格配套：

```text
Qwen/Qwen3-4B
```

并保持：

```text
block_size = 16
thinking = disabled
temperature = 0
tokenizer revision 一致
chat template 与评测逐字节一致
```

官方 model card 明确该 draft 与 `Qwen/Qwen3-4B` 配套，并用于 `enable_thinking=False`。

### 2.2 当前不需要自己重训 DFlash 的原因

PSPR 的研究问题是：

> 在 DFlash 一次并行前向产生的 Top-K 候选格中，是否能通过更强的全局 selector 找到更长的可接受前缀？

直接使用官方 DFlash 有三个好处：

1. 先隔离 selector 的真实贡献；
2. 与 Domino、纯 DFlash 和当前 Stage 1 结果保持可比；
3. 避免把 DFlash 从头训练质量的方差混入 PSPR 结论。

### 2.3 只有以下情况需要重训或深度适配 DFlash

- target 权重不再是同一 `Qwen/Qwen3-4B`；
- target 做过 SFT、RLHF、领域微调或量化变化；
- thinking 模式发生变化；
- block size 从 16 改为其他值；
- tokenizer、chat template 或 target layer feature 发生变化；
- 论文需要严格的 from-scratch 公平对照；
- 官方 DFlash 在目标数据上的 one-shot acceptance 明显异常。

即使出现领域迁移，也优先采用：

```text
官方 DFlash warm start -> 继续联合微调
```

而不是随机初始化 0.5B DFlash。

---

## 3. 两条路线共同的数据要求

## 3.1 训练标签必须和 target greedy 接受判据一致

投机解码验证的核心条件是：

```text
draft token == target 在当前真实前缀下的 greedy argmax token
```

原始 PerfectBlend、ShareGPT 或人工 assistant 回复并不等价于 target 自己会生成的 greedy token。仓库记录的原始 open-perfectblend 一致率约为 `0.7302`，意味着大量 corpus token 永远不会被 target greedy 验证接受。

因此，两条路线都应使用 target 重生成后的数据：

```bash
PYTHONPATH=. python -u scripts/regenerate_train_data.py \
  --model /path/to/Qwen3-4B \
  --server-address 127.0.0.1:30000 127.0.0.1:30010 \
                   127.0.0.1:30020 127.0.0.1:30030 \
                   127.0.0.1:30040 127.0.0.1:30050 \
                   127.0.0.1:30060 127.0.0.1:30070 \
  --concurrency 64 \
  --max-tokens 2048 \
  --temperature 0 \
  --reasoning disable \
  --input-file-path cache/dataset/perfectblend_200k.jsonl \
  --output-file-path cache/dataset/perfectblend_qwen3-4b_regen.jsonl \
  --resume
```

验证：

```bash
PYTHONPATH=. python scripts/measure_label_mismatch.py \
  --target-model /path/to/Qwen3-4B \
  --data cache/dataset/perfectblend_qwen3-4b_regen.jsonl \
  --num-samples 1000 \
  --chat-template qwen-nosys
```

仓库当前期望重生成后达到约 `0.97`。最终建议继续排查剩余不一致，并将目标提高到 `>99.5%`。需要检查：

- target checkpoint revision；
- tokenizer revision；
- `enable_thinking=False`；
- system prompt；
- BF16/FP16 与推理 backend；
- EOS/stop token；
- greedy tie-breaking；
- SGLang 与本地 Transformers 的 logits 差异。

## 3.2 推荐直接携带 exact target-greedy sidecar

最佳监督不是“target 曾经生成的一整段 response”，而是：

> 对每一个训练 anchor 的真实前缀，重新计算 target 当前 checkpoint 的 greedy argmax。

推荐在 FeatureContract 中增加：

```text
target_greedy_ids
base_top1_correct
target_in_topk
target_topk_rank
```

当前 online 配置设置：

```yaml
dflash2_selector_target_greedy_labels: false
```

原因是现有流式 capture contract 没有传递该张量，而不是 exact greedy 标签不重要。第一版可先使用 target 重生成数据；最终正式结果应补齐 sidecar。

## 3.3 Prompt 模板和 thinking 设置必须统一

建议训练、数据重生成、Stage 1 trace、Stage 2 joint、最终评测全部使用：

```yaml
chat_template: qwen-nosys
```

并保持：

```text
enable_thinking=False
reasoning=disable
temperature=0
```

仓库的 Stage 1 offline YAML 和 Stage 2 offline YAML 当前仍写有 `chat_template: qwen`，应修改为 `qwen-nosys`。否则训练序列会多出评测时不存在的 system turn。

## 3.4 数据规模与划分

第一轮主实验可使用：

```text
200,000 条 target-regenerated PerfectBlend
max_length = 3072
```

划分必须按 prompt，而不是按 anchor 或 token：

```text
train       90%
calibration 5%
final test  5%
```

更稳妥的做法是三套完全独立集合：

1. checkpoint selection validation；
2. gate calibration validation；
3. untouched final test。

不能在同一批每域 40 条样本上同时选 checkpoint、调 gate、报告最终结果。

---

## 4. 共同的基础 loss

联合训练阶段建议保留 DFlash backbone 的目标：

```yaml
loss_type: dpace
lk_loss_type: lambda
num_anchors: 512
attention_backend: flex_attention
max_grad_norm: 1.0
```

总目标为：

\[
L_{total}=L_{DFlash}+\lambda_{sel}L_{PSPR}+\lambda_{err}L_{err}
\]

其中：

- `L_DFlash`：改善 unary top-1、Top-K coverage 和 block 后段质量；
- `L_PSPR`：在 target 位于 Top-K 时选择正确候选；
- `L_err`：预测 base top-1 是否错误，用于 calibrated gate。

### 4.1 PSPR selector CE

\[
L_{PSPR}=
-\frac{1}{|\mathcal C|}
\sum_{i\in\mathcal C}
\log p^{sel}_i(y_i),
\qquad
\mathcal C=\{i:y_i\in TopK_i\}
\]

当 target 不在 Top-K 时，该位置不是 selector 分类失败，而是 backbone coverage 失败，不应计算 candidate CE。

### 4.2 当前代码中的重要行为

当前 `dflash_family_model.py` 使用：

```python
selector_loss_weights = loss_weights * target_is_candidate.float()
```

当 `loss_type: dpace` 时，`loss_weights` 已经是 backbone D-PACE 权重。因此当前 joint selector CE 并不是纯 uniform covered CE。

这对路线 A 尤其不利，因为 Stage 1 当前最佳 selector 是在 uniform covered CE 下训练的，进入 Stage 2 后会突然切换目标分布。

### 4.3 建议新增 selector 独立权重模式

建议增加：

```yaml
dflash2_selector_weight_mode: uniform
```

核心逻辑：

```python
if self.selector_weight_mode == "uniform":
    selector_loss_weights = weight_mask * target_is_candidate.float()
elif self.selector_weight_mode == "base_dpace":
    selector_loss_weights = loss_weights * target_is_candidate.float()
else:
    raise ValueError(...)
```

建议：

- 路线 A 主配置：`uniform`；
- 路线 B 主配置：先用 `uniform`；
- 额外跑一组 `base_dpace`，作为更接近官方 DFlash2 objective 的对照。

不要通过把 backbone `loss_type` 改回 `dflash` 来间接获得 uniform selector CE。backbone 和 selector 的权重应显式解耦。

---

# 5. 路线 A：selector 预训练 + 联合微调

## 5.1 路线 A 的完整流程

```text
Stage A0：准备 target-regenerated 数据和 exact lattice traces
      ↓
Stage A1：加载官方 DFlash，冻结 backbone，训练 PSPR selector
      ↓
Stage A2：将 best selector 合并到官方 DFlash checkpoint
      ↓
Stage A3：解冻 DFlash + PSPR，低学习率联合微调
      ↓
Stage A4：独立 gate calibration、导出和最终评测
```

## 5.2 路线 A 的适用场景

路线 A 更适合当前项目，原因是：

- 已经存在有效的 Stage 1 `best.pt`；
- selector 约 27M，明显比 DFlash2 的局部低秩 selector 更复杂；
- selector 在固定候选格上先学会基本重排，可降低 joint 初期破坏 backbone 的风险；
- Stage 1 本身提供清晰基线：若 joint 后低于 Stage 1，可直接判断 joint 设置不合理。

它的主要风险是：

- Stage 1 适配固定 DFlash 候选分布；
- Stage 2 中 backbone 变化后，候选格分布漂移；
- 如果 LR 和 loss 不连续，已训练 selector 会被迅速破坏。

因此路线 A 的关键不是“是否要 joint”，而是**如何让 Stage 1 到 Stage 2 连续过渡**。

---

## 6. 路线 A / Stage A1：冻结 backbone 训练 selector

## 6.1 Backbone 与 trace

Stage A1 使用：

```text
backbone = z-lab/Qwen3-4B-DFlash-b16
backbone requires_grad = False
selector requires_grad = True
```

优先使用 `scripts/train_pspr_accept_selector.py` 的 anchor-level lattice trace 路径，因为它已经复现当前最佳结果，而且 checkpoint 选择直接基于 calibration macro acceptance gain。

Stage A1 的 trace 应尽量来自和 Stage A3 相同的数据分布。推荐：

```text
target-regenerated PerfectBlend
+ 数学
+ 代码
+ 通用对话
```

如果继续使用当前六域 trace，需要把它定义为复现基线，而不是最终训练分布。

## 6.2 Stage A1 推荐结构

```text
K=16
d=512
layers=3
heads=8
state_dim=512
delta_hidden=2048
dropout=0.1
trans_rank=0
```

第一轮直接复用当前最佳初始化：

```text
output_zero_init = false
gamma starts at 0
gamma uses 5x LR and no weight decay
```

这是为了严格复现已经达到 `5.7217` 的路线。

`--output-zero-init` 可作为单独消融，但不能和当前 best 的训练结论混在一起。开启该选项后：

```text
gamma = 1
correction output layers = 0
```

此时 gamma 不再需要 5 倍学习率。

## 6.3 Stage A1 推荐 loss

主配置：

```text
CE auxiliary        = 1.0
err BCE             = 1.0
accept expectation  = 0.0
D-PACE selector CE  = off
reach weighting     = off
wrong weighting     = 1.0
```

即：

```bash
--ce-aux 1.0 \
--err-w 1.0 \
--accept-w 0.0 \
--reach-w 1.0 \
--wrong-w 1.0
```

原因是仓库记录的同 backbone 消融中，uniform CE 优于 selector D-PACE、固定 frontier weighting 和直接 expected-accept loss。

不过 `err_w=1` 是否适合 joint 不能由 Stage A1 直接推出。Stage A1 保持它是为了复现当前 best；Stage A3 应重新调低。

## 6.4 Stage A1 的 global batch 与 LR

训练脚本内部执行：

\[
LR_{effective}=LR_{arg}\sqrt{world\_size}
\]

为避免 GPU 数改变后有效 LR 被意外放大，建议保持：

```text
anchor-level global batch ≈ 256
effective selector LR ≈ 5.5e-4 ～ 5.8e-4
```

| GPU 数 | 每卡 batch | `--lr` 参数 | 脚本中的有效 LR |
|---:|---:|---:|---:|
| 2 | 128 | `4.0e-4` | `5.66e-4` |
| 4 | 64 | `2.8e-4` | `5.60e-4` |
| 8 | 32 | `2.0e-4` | `5.66e-4` |

不要在 8 卡上继续使用：

```text
per-rank batch=128, --lr=4e-4
```

否则 global batch 和有效 LR 都会显著超过已验证配置。

## 6.5 Stage A1 推荐命令

8 卡、保持 global anchor batch 256：

```bash
torchrun --nproc_per_node=8 scripts/train_pspr_accept_selector.py \
  --trace-dir /path/to/lattice_traces \
  --cal-trace-dir /path/to/onpolicy_cal_traces \
  --target-model /path/to/Qwen3-4B \
  --output outputs/pspr_route_a_stage1/best.pt \
  --epochs 8 \
  --batch-size 32 \
  --lr 2.0e-4 \
  --weight-decay 2.0e-3 \
  --warmup-ratio 0.0 \
  --dim 512 \
  --n-layers 3 \
  --n-heads 8 \
  --delta-h 2048 \
  --dstate 512 \
  --dropout 0.1 \
  --trans-rank 0 \
  --ce-aux 1.0 \
  --err-w 1.0 \
  --accept-w 0.0 \
  --reach-w 1.0 \
  --wrong-w 1.0 \
  --eval-every 200 \
  --val-frac 0.10 \
  --cal-per-ds 1200 \
  --seed 2026
```

## 6.6 Stage A1 停止规则

脚本允许 `epochs=8`，但不应默认使用最后一个 epoch。当前参考结果的峰值出现在较早 epoch。

应按 calibration macro acceptance gain 保存 best：

```text
每 200～400 step 评估
每个 epoch 结束评估
连续 3 次评估无提升则停止
```

Stage A1 选择标准：

1. calibrated PSPR acceptance gain；
2. recovery rate；
3. destruction rate；
4. 不使用训练 CE 作为最终 checkpoint 标准。

---

## 7. 路线 A / Stage A2：合并 checkpoint

使用：

```bash
PYTHONPATH=. python scripts/import_pspr_selector.py \
  --selector outputs/pspr_route_a_stage1/best.pt \
  --backbone /path/to/Qwen3-4B-DFlash-b16 \
  --draft-config configs/qwen3-4b-pspr-joint.json \
  --output-dir outputs/pspr_route_a_joint_init
```

随后执行：

```bash
PYTHONPATH=. python scripts/gate_pspr_warm_start.py --dtype bfloat16
```

Stage A2 的关键要求：

- backbone tensor 必须来自官方 DFlash；
- selector tensor 必须完整加载；
- 不允许只加载一部分 `candidate_selector.*`；
- FP32 读取后只做一次 BF16 cast，避免双重舍入；
- step 0 的 joint 模型必须与 Stage A1 selector 逐位一致。

Stage A3 的模型配置中应保持：

```json
"selector_output_zero_init": false,
"selector_trans_rank": 0,
"freeze_backbone": false
```

这里不能改成 `selector_output_zero_init=true`，因为 Stage A3 加载的是已经训练好的 selector，而不是 fresh selector。

---

## 8. 路线 A / Stage A3：联合微调

## 8.1 Joint 的必要条件

```yaml
dflash2_selector_stop_gradient: false
```

这意味着 PSPR loss 会通过：

```text
selector score
  -> unary top-K logits
  -> DFlash hidden states
  -> DFlash backbone
```

反向传播。Stage A3 是严格的端到端 joint training。

## 8.2 Stage A3 推荐学习率

Stage A3 同时加载：

- 已收敛的官方 DFlash；
- 已收敛的 Stage A1 selector。

因此不应直接使用全模型统一 `6e-4`。推荐中心配置：

| 参数组 | 推荐 LR |
|---|---:|
| DFlash backbone | `5e-5` |
| PSPR selector 主体 | `1e-4` |
| `candidate_selector.gamma` | `5e-5` |
| `err_head` | `1e-4`，仅启用 gate 时 |

推荐最小 sweep：

| Run | Backbone LR | Selector LR | 适用目的 |
|---|---:|---:|---|
| A-J1 | `3e-5` | `1e-4` | 最保守，防止 backbone 退化 |
| A-J2 | `5e-5` | `1e-4` | 主推荐 |
| A-J3 | `5e-5` | `2e-4` | selector 跟不上候选格漂移时 |

不建议第一轮测试超过：

```text
backbone 1e-4
selector 3e-4
```

### 推荐 optimizer 规则

```yaml
learning_rate: 5.0e-5
lr_scale_rules:
  "candidate_selector.gamma": 1.0
  "candidate_selector.err_head.": 2.0
  "candidate_selector.": 2.0
```

注意当前 optimizer 是**第一个匹配的 prefix 生效**，不是自动最长前缀匹配，因此具体规则必须写在通用规则前面。

## 8.3 Stage A3 推荐 selector loss

```yaml
dflash2_selector_weight_mode: uniform
dflash2_selector_own_denominator: true
dflash2_selector_loss_alpha: 0.5
dflash2_selector_stop_gradient: false
```

理由：

- `uniform` 保持与 Stage A1 目标连续；
- `own_denominator=true` 保持 selector CE 是 covered slot 的完整均值；
- `alpha=0.5` 避免 selector CE 与 backbone mean loss、err BCE 三项同时以 1.0 强度竞争。

建议 sweep：

```text
alpha = 0.25 / 0.5 / 0.75
```

不建议第一轮直接使用 `1.0`。

## 8.4 Stage A3 的 selector alpha 调度

对于 warm-start selector，不建议使用一段较长的 `alpha=0` warmup，因为此时 backbone 已经在变化，而 selector 没有同步更新，会加剧分布失配。

### 无代码修改的推荐

```yaml
dflash2_selector_warmup_ratio: 0.0
dflash2_selector_ramp_ratio: 0.0
dflash2_selector_loss_alpha: 0.5
```

### 更好的代码修改

增加：

```yaml
selector_alpha_start: 0.25
selector_alpha_end: 0.50
selector_alpha_ramp_steps: 750
```

即：

\[
\lambda_{sel}(t)=0.25+0.25\min(t/750,1)
\]

这样从 step 1 起 selector 就能跟随 backbone，同时避免突然以完整强度改变 backbone。

## 8.5 Stage A3 的 err head

当前 gate 条件包含：

\[
\sigma(err\_logit)>\theta
\]

当 `theta=0` 时，该条件对任何有限 logit 都成立，err head 实际不参与过滤。

因此必须二选一：

### 使用 err gate

```yaml
dflash2_selector_err_loss_alpha: 0.1
```

并为每个 checkpoint 独立校准：

```text
theta ∈ {0.2, 0.3, 0.35, 0.4, 0.5}
rho   单独 sweep
tau   单独 sweep
```

### 不使用 err gate

```yaml
dflash2_selector_err_loss_alpha: 0.0
```

同时不要把最终结果归因于 err head。

主推荐先跑两组：

```text
A-J2a：err alpha = 0
A-J2b：err alpha = 0.1 + 独立 gate calibration
```

不建议在 Stage A3 继续使用 `err alpha=1.0`。

## 8.6 Stage A3 训练长度

当前 online 配置：

```text
200,000 rows
DP=7
per-device batch=2
accumulation=2
global batch=28
```

每个数据 pass 约为：

\[
200000/28\approx7143\text{ optimizer steps}
\]

推荐：

```yaml
num_epochs: 1
```

先完成约 7143 steps，不要直接 6 pass、约 42857 steps。

评测点：

```text
step 100
step 250
step 500
step 1000
step 2000
step 4000
step 6000
step 7143
```

只有 6000～7143 step 的 held-out acceptance length 仍持续上升时，才继续第二个 pass。第二个 pass 建议将 LR 降为第一轮的 0.3～0.5 倍。

## 8.7 Stage A3 推荐 YAML 骨架

> `dflash2_selector_weight_mode` 需要按前文增加到 schema 和 model。

```yaml
model:
  target_model_path: /path/to/Qwen3-4B
  draft_model_config: configs/qwen3-4b-pspr-joint.json
  draft_checkpoint_path: ./outputs/pspr_route_a_joint_init
  target_backend: sglang
  embedding_key: model.embed_tokens.weight
  torch_dtype: bfloat16
  mask_token_id: 151669
  sglang_context_length: 3200

data:
  train_data_path: ./cache/dataset/perfectblend_qwen3-4b_regen.jsonl
  max_length: 3072
  chat_template: qwen-nosys
  cache_dir: ./cache
  build_dataset_num_proc: 32

training:
  strategy: dflash
  num_epochs: 1
  batch_size: 2
  accumulation_steps: 2
  fsdp_sharding: "NO_SHARD"

  learning_rate: 5.0e-5
  lr_scale_rules:
    "candidate_selector.gamma": 1.0
    "candidate_selector.err_head.": 2.0
    "candidate_selector.": 2.0

  warmup_ratio: 0.02
  max_grad_norm: 1.0
  weight_decay: 0.0
  attention_backend: flex_attention
  num_anchors: 512

  loss_type: dpace
  lk_loss_type: lambda
  loss_decay_gamma: 7.0

  dflash2_selector_weight_mode: uniform
  dflash2_selector_loss_alpha: 0.5
  dflash2_selector_err_loss_alpha: 0.1
  dflash2_selector_own_denominator: true
  dflash2_selector_warmup_ratio: 0.0
  dflash2_selector_ramp_ratio: 0.0
  dflash2_selector_stop_gradient: false
  dflash2_selector_target_greedy_labels: false

  save_interval: 1000
  log_interval: 10
  dist_timeout: 30
  seed: 42
```

---

# 9. 路线 B：fresh PSPR 直接联合训练

## 9.1 “像 DFlash2”在这里的准确含义

路线 B 只复用 DFlash2 的训练思想：

```text
新增 selector 以 no-op 状态初始化
+ 从同一个训练任务开始联合优化
+ selector loss 逐渐加入
```

路线 B 不使用：

```text
DFlash2 grouped convolution
DFlash2 CandidateSelector
DFlash2 selector_rank=256 codebook
```

最终模型仍然是：

```text
DFlashDraftModel + PSPR LatticePathSelector
```

## 9.2 路线 B 的完整流程

```text
Stage B0：准备 target-regenerated 数据
      ↓
Stage B1：加载官方 DFlash backbone
      + 新建 fresh PSPR
      + PSPR 严格 no-op 初始化
      ↓
Stage B2：同一 optimizer、同一训练任务直接 joint
      ↓
Stage B3：独立 gate calibration、导出和评测
```

不存在：

```text
冻结 backbone -> 单独保存 selector -> 再重新加载
```

## 9.3 路线 B 的关键初始化

模型配置必须使用：

```json
"selector_output_zero_init": true,
"selector_trans_rank": 0,
"freeze_backbone": false
```

其含义是：

```text
gamma = 1
mlp 最后一层 = 0
delta_mlp 最后一层 = 0
初始 PSPR correction = 0
初始 candidate score = DFlash unary log-prob
```

因此 step 0 的输出必须与纯 DFlash top-1 完全一致，但 correction 输出层从第一步就能收到梯度。

不要在路线 B 使用默认的：

```text
gamma=0 + correction network 随机初始化
```

因为此时约 27M correction 参数的梯度会被 gamma=0 阻断，最初主要只有一个 gamma 标量移动，fresh joint 的启动会很慢。

## 9.4 路线 B 的 checkpoint 加载

```yaml
draft_checkpoint_path: /path/to/Qwen3-4B-DFlash-b16
```

`PSPRDraftModel.warm_start_optional_prefixes = ("candidate_selector.",)` 允许 DFlash checkpoint 缺少 selector 张量，因此：

```text
backbone 从官方 DFlash 加载
selector 使用 fresh 初始化
```

必须新增一个 Route B step-0 gate：

```text
missing keys 只能是 candidate_selector.* 和 target embedding
unexpected keys = 0
PSPR output == DFlash unary output
selected token 逐位一致
```

## 9.5 路线 B 推荐学习率

路线 B 的两部分状态不同：

- DFlash backbone：已经训练好；
- PSPR selector：fresh。

因此路线 B 必须采用差分 LR。

推荐中心配置：

| 参数组 | 推荐 LR |
|---|---:|
| DFlash backbone | `5e-5` |
| PSPR selector 主体 | `2e-4` |
| `gamma` | `5e-5` |
| err head | 第一轮关闭，或 `1e-4` |

最小 sweep：

| Run | Backbone LR | Selector LR | 说明 |
|---|---:|---:|---|
| B-J1 | `3e-5` | `2e-4` | 最强 backbone 保护 |
| B-J2 | `5e-5` | `2e-4` | 主推荐 |
| B-J3 | `5e-5` | `3e-4` | selector 学习不足时 |
| B-J4 | `1e-4` | `3e-4` | 更积极，仅在 base 稳定时 |

推荐规则：

```yaml
learning_rate: 5.0e-5
lr_scale_rules:
  "candidate_selector.gamma": 1.0
  "candidate_selector.": 4.0
```

路线 B 中 gamma 从 1 开始，不需要 Stage A1 的 5 倍 LR。

## 9.6 路线 B 的 selector alpha ramp

fresh PSPR 比官方 DFlash2 selector 大得多。官方 DFlash2 的 selector 是局部低秩 transition scorer，而 PSPR 包含双向 Transformer、GRU 和多个 scorer，因此不建议机械照搬极短的 0.0005/0.0005 调度。

推荐：

```yaml
dflash2_selector_loss_alpha: 0.5
dflash2_selector_warmup_ratio: 0.0
dflash2_selector_ramp_ratio: 0.08
dflash2_selector_stop_gradient: false
```

若一轮为 7143 steps，前约 571 steps：

\[
\lambda_{sel}:0\rightarrow0.5
\]

这仍然是 joint training：

- backbone 从第一步通过自身 loss 更新；
- selector 从早期小权重开始更新；
- selector loss 从非零开始就能反向进入 backbone；
- 不存在 checkpoint 切换或 freeze/unfreeze 阶段。

如果 early metrics 显示非常稳定，可将 ramp 缩短到 `0.04`；如果 base/oracle 明显下降，可延长到 `0.12～0.15`。

## 9.7 路线 B 推荐 loss

主配置：

```yaml
loss_type: dpace
lk_loss_type: lambda

dflash2_selector_weight_mode: uniform
dflash2_selector_own_denominator: true
dflash2_selector_loss_alpha: 0.5
dflash2_selector_err_loss_alpha: 0.0
```

第一轮先关闭 fresh err head，以便回答一个干净问题：

> PSPR selector 本体在 direct joint 下能否提高接受长度？

然后再运行：

```text
B-J2-gate：err alpha=0.1，并校准非零 theta
```

### 必须保留的官方风格对照

为了判断 PSPR 是否需要独立 loss 设计，建议增加：

```text
B-official-like
  selector_weight_mode = base_dpace
  own_denominator       = false
  selector_alpha        = 1.0
  err_alpha             = 0
```

这组更接近官方 DFlash2 的 selector objective 语义，但仍使用 PSPR 架构。它是消融，不是默认推荐。

## 9.8 路线 B 推荐训练长度

fresh selector 需要比路线 A 的 joint 部分更长的适配，但不需要直接训练 100K steps。

推荐：

```text
第一轮：1 pass，约 7143 steps
若 6K～7K 仍增长：继续到 10K～14K steps
```

第二个 pass 建议降低 LR：

```text
backbone LR × 0.3～0.5
selector LR × 0.5
```

## 9.9 路线 B 模型配置

建议新增：

```json
{
  "architectures": ["PSPRDraftModel"],
  "block_size": 16,
  "dflash_config": {
    "mask_token_id": 151669,
    "target_layer_ids": [1, 9, 17, 25, 33],
    "selector_top_k": 16,
    "selector_dim": 512,
    "selector_layers": 3,
    "selector_heads": 8,
    "selector_state_dim": 512,
    "selector_delta_hidden": 2048,
    "selector_max_slots": 32,
    "selector_dropout": 0.1,
    "selector_output_zero_init": true,
    "selector_trans_rank": 0,
    "freeze_backbone": false
  }
}
```

## 9.10 路线 B 推荐 YAML 骨架

```yaml
model:
  target_model_path: /path/to/Qwen3-4B
  draft_model_config: configs/qwen3-4b-pspr-direct-joint.json
  draft_checkpoint_path: /path/to/Qwen3-4B-DFlash-b16
  target_backend: sglang
  embedding_key: model.embed_tokens.weight
  torch_dtype: bfloat16
  mask_token_id: 151669
  sglang_context_length: 3200

data:
  train_data_path: ./cache/dataset/perfectblend_qwen3-4b_regen.jsonl
  max_length: 3072
  chat_template: qwen-nosys
  cache_dir: ./cache
  build_dataset_num_proc: 32

training:
  strategy: dflash
  num_epochs: 1
  batch_size: 2
  accumulation_steps: 2
  fsdp_sharding: "NO_SHARD"

  learning_rate: 5.0e-5
  lr_scale_rules:
    "candidate_selector.gamma": 1.0
    "candidate_selector.": 4.0

  warmup_ratio: 0.04
  max_grad_norm: 1.0
  weight_decay: 0.0
  attention_backend: flex_attention
  num_anchors: 512

  loss_type: dpace
  lk_loss_type: lambda
  loss_decay_gamma: 7.0

  dflash2_selector_weight_mode: uniform
  dflash2_selector_loss_alpha: 0.5
  dflash2_selector_err_loss_alpha: 0.0
  dflash2_selector_own_denominator: true
  dflash2_selector_warmup_ratio: 0.0
  dflash2_selector_ramp_ratio: 0.08
  dflash2_selector_stop_gradient: false
  dflash2_selector_target_greedy_labels: false

  save_interval: 1000
  log_interval: 10
  dist_timeout: 30
  seed: 42
```

---

# 10. 路线 A 与路线 B 参数差异总表

| 项目 | 路线 A：预训练后 joint | 路线 B：fresh 直接 joint |
|---|---|---|
| 初始 backbone | 官方 DFlash | 官方 DFlash |
| 初始 selector | Stage A1 `best.pt` | fresh PSPR |
| `selector_output_zero_init` | Stage A3 为 `false` | `true` |
| Stage 1 freeze | 有 | 无 |
| joint `freeze_backbone` | `false` | `false` |
| joint `selector_stop_gradient` | `false` | `false` |
| selector alpha | 固定 0.25～0.5，或 0.25→0.5 | 0→0.5 ramp |
| selector ramp | 通常 0；建议非零起点 ramp | 推荐 4%～10% |
| Backbone LR | `3e-5～5e-5` | `3e-5～5e-5` |
| Selector LR | `1e-4～2e-4` | `2e-4～3e-4` |
| Gamma LR | 与 backbone 接近 | 与 backbone 接近 |
| err loss | 0 或 0.1 | 第一轮 0，之后 0.1 |
| selector weight | 主推 uniform | 主推 uniform；另做 base-DPACE 对照 |
| `own_denominator` | `true` | 主推 `true` |
| 第一轮 joint 步数 | 约 7K | 约 7K，必要时扩到 10K～14K |
| 风险 | Stage 1/2 分布漂移 | fresh 大 head 干扰 backbone |
| 优势 | 稳定、可复用已验证 selector | 训练统一、共同适配更充分 |

---

# 11. 统一评测与 checkpoint 选择

## 11.1 必须记录的指标

### Backbone 质量

```text
oneshot acceptance length
base top-1 accuracy by depth
target-in-TopK coverage by depth
oracle@16 acceptance length
```

### Selector 质量

```text
PSPR ungated acceptance length
PSPR gated acceptance length
selector accuracy on covered slots
selector target probability
recovery rate
destruction rate
```

定义：

\[
Recovery=P(PSPR\ correct\mid base\ wrong,target\in TopK)
\]

\[
Destruction=P(PSPR\ wrong\mid base\ correct)
\]

### Gate 质量

```text
gate fire rate
precision / recall
Brier score
ECE
不同 theta/rho/tau 下的真实 acceptance length
```

### 优化稳定性

```text
backbone grad norm
selector grad norm
err-head grad norm
global clipping coefficient
gamma
correction RMS / unary-score RMS
参数更新范数 / 参数范数
```

## 11.2 成功标准

一次 joint run 只有同时满足以下条件才算成功：

```text
PSPR acceptance length 上升
oneshot acceptance 基本稳定或上升
oracle@16 基本稳定或上升
recovery 上升
destruction 没有抵消 recovery
```

推荐保护线：

```text
oneshot 相对 step 0 下降 > 0.03～0.05：警告
连续两次评估下降并伴随 oracle@16 下降：停止该 run
```

如果评测集仍是每域 40 条，应只用于 quick check。正式 checkpoint 选择建议每域至少 200～500 条，并运行多个 seed。

## 11.3 失败诊断

### 情况 1：oneshot 与 oracle@16 同时下降

说明 joint 在破坏 backbone 候选格：

```text
backbone LR × 0.5
selector alpha 0.5 -> 0.25
Route B ramp 8% -> 12%～15%
```

### 情况 2：oneshot 稳定，但 PSPR 提升很慢

说明 selector 更新不足：

```text
selector LR 1e-4 -> 2e-4（路线 A）
selector LR 2e-4 -> 3e-4（路线 B）
```

不要同时提高 backbone LR。

### 情况 3：selector CE 下降，但 acceptance 不升

优先检查：

- corpus label 与 target greedy 是否一致；
- destruction 是否增加；
- gate 是否阻断正确 recovery；
- 推理是否真的调用 PSPR；
- selector loss 是否被 D-PACE 隐式重权；
- checkpoint 的 gate 参数是否来自同一个 checkpoint。

### 情况 4：大量 step 被 global clipping

当前 optimizer 对全部参数一起计算 global norm。PSPR 的大梯度可能把 backbone 梯度一起缩小。

若超过约 20%～30% steps 的 clip coefficient 明显小于 1，可增加分组 norm 日志，并考虑：

```text
backbone clip = 1.0
selector clip = 0.5～1.0
```

在得到日志证据前，不要先修改 clipping 实现。

---

# 12. 最小实验矩阵

## 12.1 先做 1K～2K step pilot

所有实验保持：

```text
同一 DFlash checkpoint
同一 200K regen 数据或同一固定子集
同一 global batch 28
同一 seed
同一 PSPR 架构
同一 quick validation
```

### 路线 A pilot

| Run | Backbone LR | Selector LR | Selector alpha | Err alpha | Weight mode |
|---|---:|---:|---:|---:|---|
| A1 | `3e-5` | `1e-4` | 0.25 | 0 | uniform |
| A2 | `5e-5` | `1e-4` | 0.5 | 0 | uniform |
| A3 | `5e-5` | `2e-4` | 0.5 | 0 | uniform |
| A4 | `5e-5` | `1e-4` | 0.5 | 0.1 | uniform |

### 路线 B pilot

| Run | Backbone LR | Selector LR | Selector alpha | Ramp | Err alpha |
|---|---:|---:|---:|---:|---:|
| B1 | `3e-5` | `2e-4` | 0.5 | 8% | 0 |
| B2 | `5e-5` | `2e-4` | 0.5 | 8% | 0 |
| B3 | `5e-5` | `3e-4` | 0.5 | 8% | 0 |
| B4 | `5e-5` | `2e-4` | 1.0 | 官方风格短 ramp | 0，base-DPACE/共享分母 |

每条路线选 1～2 个最优 pilot 进入完整 7K-step run。

## 12.2 完整对比必须固定 joint 预算

推荐主要论文对比：

```text
Route A：使用固定 Stage A1 best.pt + 7K joint updates
Route B：fresh selector + 7K joint updates
```

Stage A1 的便宜 head-only 计算需单独报告，不应隐藏。另可报告 practical wall-clock-to-quality，体现路线 A 是否用少量额外成本换来更高稳定性。

## 12.3 最终结果运行多个 seed

pilot 可使用一个 seed。最终两条路线至少使用：

```text
seed = 42, 2026, 3407
```

报告：

```text
mean ± std
每域结果
macro acceptance
end-to-end selector latency
tokens/s
```

---

# 13. 建议优先实现的代码修改

## P0：selector 独立 weight mode

必须实现。否则 `loss_type=dpace` 会同时改变 backbone 与 selector 的样本权重，路线 A 无法保持 Stage 1/2 objective 连续。

新增：

```text
dflash2_selector_weight_mode = uniform | base_dpace
```

## P0：路线 B 的独立 config 与 step-0 等价门禁

新增：

```text
configs/qwen3-4b-pspr-direct-joint.json
examples/configs/online/.../qwen3-4b-pspr-direct-joint-online.yaml
scripts/gate_pspr_direct_joint_init.py
```

门禁验证：

```text
fresh PSPR score == DFlash unary score
argmax 逐位一致
gamma == 1
correction output == 0
```

## P1：支持非零起点的 selector alpha schedule

当前 schedule 只能：

```text
0 -> target alpha
```

路线 A 更适合：

```text
0.25 -> 0.5
```

建议增加：

```yaml
selector_alpha_start
selector_alpha_end
selector_alpha_ramp_steps
```

## P1：exact target-greedy online FeatureContract

将 exact per-prefix greedy 标签透传到 online objective，避免剩余 3% 左右标签错位。

## P1：分组梯度与更新范数日志

至少记录：

```text
backbone_grad_norm
selector_grad_norm
err_grad_norm
clip_coefficient
backbone_update_ratio
selector_update_ratio
```

## P2：分组 clipping

只有日志确认 PSPR 经常触发全局 clip 后再实现。

---

# 14. 最终执行顺序

## 第一阶段：先修现有路线 A

1. 将所有训练、regen、trace 和评测统一为 `qwen-nosys`、thinking disabled；
2. 实现 `selector_weight_mode=uniform`；
3. 复用当前 Stage 1 `best.pt`；
4. 用 `backbone LR=5e-5`、`selector LR=1e-4`、`alpha=0.5`、`err=0/0.1` 跑 1K～2K pilot；
5. 选择稳定配置跑完整 1 pass；
6. 不直接跑当前的 `6e-4 × 6 pass`。

## 第二阶段：建立路线 B

1. 新建 `selector_output_zero_init=true` 的 direct-joint config；
2. 直接加载官方 DFlash；
3. `stop_gradient=false`；
4. `backbone LR=5e-5`、`selector LR=2e-4`；
5. selector alpha 在前 8% steps 从 0 增加到 0.5；
6. 第一轮关闭 err loss；
7. 跑与路线 A 相同的 joint steps 和评测。

## 第三阶段：选择主路线

优先级为：

```text
真实 PSPR acceptance length
> oneshot/oracle 稳定性
> recovery-destruction 净收益
> 训练方差
> selector 延迟
> 训练成本
```

如果路线 A 在低 LR joint 后仍无法超过 Stage 1 的 `5.7217`，而路线 B 能稳定提升，则切换路线 B 为主方法。若路线 B 在多个 LR/ramp 设置下持续破坏 backbone，而路线 A 稳定，则路线 A 应作为论文和工程默认方案。

---

# 15. 推荐的第一组正式配置

## 路线 A

```text
Backbone              official Qwen3-4B-DFlash-b16
Selector init          current Stage 1 best.pt
Backbone LR            5e-5
Selector LR            1e-4
Gamma LR               5e-5
Selector alpha         0.5
Err alpha              0 与 0.1 各一组
Selector weight        uniform
Own denominator        true
Stop gradient          false
Optimizer warmup       2%
Joint duration         1 pass ≈ 7143 steps
```

## 路线 B

```text
Backbone              official Qwen3-4B-DFlash-b16
Selector init          fresh, output-zero-init
Backbone LR            5e-5
Selector LR            2e-4
Gamma LR               5e-5
Selector alpha         0 -> 0.5 over first 8%
Err alpha              0
Selector weight        uniform
Own denominator        true
Stop gradient          false
Optimizer warmup       4%
Joint duration         1 pass ≈ 7143 steps
```

这两组是最应优先比较的核心实验。

---

## 参考来源

1. [PSPR_README.md](https://github.com/wzyyy-lab/specforge-pspr/blob/main/PSPR_README.md)
2. [PSPR model implementation](https://github.com/wzyyy-lab/specforge-pspr/blob/main/specforge/modeling/draft/pspr.py)
3. [DFlash-family joint objective](https://github.com/wzyyy-lab/specforge-pspr/blob/main/specforge/algorithms/common/dflash_family_model.py)
4. [Stage-1 selector training script](https://github.com/wzyyy-lab/specforge-pspr/blob/main/scripts/train_pspr_accept_selector.py)
5. [Stage-2 online config](https://github.com/wzyyy-lab/specforge-pspr/blob/main/examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-joint-online.yaml)
6. [Official SpecForge DFlash2 config](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/managed-local/qwen3.6-27b-dflash2-disaggregated.yaml)
7. [Official DFlash2 implementation](https://github.com/sgl-project/SpecForge/blob/main/specforge/modeling/draft/dflash2.py)
8. [Qwen3-4B-DFlash-b16 model card](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16)
