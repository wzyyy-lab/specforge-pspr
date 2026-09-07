# PSPR — 冻结 DFlash 骨干上的候选格选择头家族

> 上游基座：`sgl-project/SpecForge` @ `2fc9930`
> Target：`Qwen3-4B`（`tie_word_embeddings=True`）
> Backbone：官方 `Qwen3-4B-DFlash-b16`（5 层，537.4M，`block_size=16`）

本文档是 PSPR 全家族的总览：有哪些版本、各自在哪、架构长什么样、怎么训练。
当前最优版本 **SlotDeep** 的详细说明在 [`SLOTDEEP.md`](./SLOTDEEP.md)。

---

## 0. 一分钟版本

DFlash 骨干一次 all-MASK 并行前向，为 block 内 15 个 proposal slot 各产出一个 hidden state。
经 target 的 `lm_head` 读出后每个 slot 得到 top-16 候选，构成一个 `[15, 16]` 的**候选格**。

PSPR 干的事：**在这个格上加一个选择头，重排每个 slot 的 16 个候选**，从而把接受长度顶上去。
骨干不改（stage 1 冻结），词表矩阵不学（复用 tied embedding 做读出）。

```
        ┌─ slot 1: [c1 c2 ... c16] + logprobs
all-MASK 并行前向 ──▶ ├─ slot 2: [c1 c2 ... c16] + logprobs   ──▶ selector ──▶ 重排 ──▶ margin gate ──▶ 提交路径
   (DFlash 骨干)      └─ ...   (共 15 slot) (PSPR)
         ▲
        GRU(已提交前缀) ┘
```

六域实测（同一冻结官方 DFlash-b16 骨干，n=20/域，`--gate-rho 3`）：

| 头 | macro accept length | 相对 plain DFlash |
|---|---:|---:|
| 无（plain DFlash one-shot） | 5.3046 | — |
| Domino projector head（复现） | 5.8868 | +0.582 |
| PSPR-cloze @7115 | 6.0501 | +0.746 |
| **PSPR-SlotDeep S1 @7115** | **6.0945** | **+0.790** |
| SlotDeep Stage 2 @9052（骨干解冻） | **6.4386** | +1.134 |
| oracle@16（理论上限） | 10.4424 | +5.138 |

> 除 Domino projector head 外，全部出自同 seed 2026 manifest 的三份 JSON
> （`EVAL_SLOTDEEP_BASELINE_/_S1_STEP7115_/_STAGE2_STEP9052_*.json`）。
> 5.8868 出自另一次 n=20 运行（`NEWHW_oneshot_domino.log`），该次 oneshot 复现为 5.3043
> （此处 5.3046），故可比但**不是同一个文件**。
> **这里的 Domino 是挂在同一冻结骨干上的 projector head 复现，不是官方 Domino 完整模型**，见 §4.3。
> n=20 分辨不了 0.04 量级 —— cloze/SlotDeep 之间那 0.044 不构成结论。

---

## 1. 版本地图

全部 head 都在 `specforge/modeling/draft/`，都用 `@register_draft` 注册，都跑在
`strategy: dflash` 下（共享 DFlash 家族的目标函数与数据面）。

| # | 模块 | 类 | 继承自 | 行数 | 一句话 | 状态 |
|---|---|---|---|---:|---|---|
| 1 | `pspr.py` | `LatticePathSelector` / `PSPRDraftModel` | `DFlashDraftModel` | 809 | **原始版**：slot 候选压成 centroid，双向 transformer 混 slot | 已冻结，勿改 |
| 2 | `pspr_v2.py` | `LatticeCorrector` / `PSPRv2DraftModel` | `DFlashDraftModel` | 667 | 把 `h` 全宽直连修正器，编码器降为纯加性 cross-slot 项 | 已冻结，勿改 |
| 3 | `pspr_cloze.py` | `ClozeCorrector` / `PSPRClozeDraftModel` | `DFlashDraftModel` | 826 | **cloze**：mask 掉查询 slot，读 seed path 上下文 | 已冻结，勿改 |
| 4 | `pspr_slotdeep.py` | `SlotDeepCorrector` / `PSPRSlotDeepDraftModel` | `PSPRClozeDraftModel` | 283 | **当前最优**：拆掉 attention，换 9 层 per-slot 残差塔 + 候选打分残差 | **主线** |
| 5 | `pspr_decision.py` | `DecisionCorrector` / `PSPRDecisionDraftModel` | `DFlashDraftModel` | 762 | 把 16 路 softmax 分解成 `p(需修) × r(修成谁)` | 已关闭 |
| 6 | `pspr_cascade.py` | `CascadeCorrector` / `PSPRCascadeDraftModel` | `PSPRClozeDraftModel` | 505 | 让 KEEP/REPAIR 门读得到 ranker 自己的分数 | 已关闭 |
| 7 | `pspr_memory.py` | `MemoryCorrector` / `PSPRMemoryDraftModel` | `PSPRClozeDraftModel` | 293 | 读已验证 target 特征作残差（含 blind 对照） | 已关闭 |
| 8 | `pspr_oracle_probe.py` | `OracleFutureProbeSelector` | `LatticePathSelector` | 165 | **诊断专用**：喂 ground-truth 未来 token，测上界 | 诊断 |

> `pspr.py` / `pspr_v2.py` / `pspr_cloze.py` 三个文件各自绑着已发布的 checkpoint 和已报告的数字，
> 且有逐位等价性门禁锁着（`gate_pspr_reference_equivalence.py` 断言 `max|Δ| = 0`）。
> **任何新想法都走子类，不改这三个文件。** SlotDeep / Cascade / Memory 都是 `PSPRClozeDraftModel` 的子类。

### 1.1 演进路线与每一步的动因

```
pspr (v1)     slot 候选 → softmax 加权 centroid → 双向 transformer
   │   问题：候选身份在 attention 之前就被抹平了；h 只能从 2560→512 的挤压里挤过去
   ▼
pspr_v2     h 全宽直连修正器，encoder 退化成纯加性 cross-slot 项
   │         实测：delta-only (5.9329) ≥ full (5.9043)，pointwise MLP 是噪声
   │       问题：query slot 从来没被真正 mask 过——它自己的 top-1 就是主导项
   ▼
pspr_cloze   mask 查询 slot，上下文换成 seed path d0[j] = cand[j,0]
   │                   关键设计：seed path 在训练与serving 完全一致，无 teacher forcing 分布偏移
   │         H 行一次 batched encoder，因果信息由 GRU 单独补
   │
   │  ── 三个推理期消融（同一份 cloze@7115 权重，参数/FLOPs/格完全不变）──
   │     full bidirectional  6.0438
   │     causal        6.0427   −0.0012   ← 未来信息几乎零价值
   │     anchor_self         5.9993   −0.0445   ← 所有 cross-slot 信息只值 0.045
   │     zero_context        5.5352   −0.5086   ← encoder 总价值
   │     ⟹ encoder 那 0.509 里 ~91% 是一个 per-slot 非线性变换，双向 attention 是奢侈品
   ▼
pspr_slotdeep  拆掉 H×(H+1) 的 attention，换成 9 层 per-slot 残差 FFN 塔
 另加 concat anchor 融合 + 候选条件打分残差
```

**为什么双向 encoder 注定回本困难（结构性原因，不是调参失败）**：
骨干在 block 内是全 attention（`dflash.py:245`, `is_causal=False`），所以 `h_i` 早就聚合了整个
block；而候选格是从冻结的 tied `lm_head` 读出的（`dflash.py:785`），每个 seed token / log-prob /
margin 都是 `{h_1..h_H}` 的确定性函数。cloze 序列携带的信息**不超过** `{h_j}` 本身。
更糟的是 `direct_hidden=True` 时 query 行仍拿到全宽 `h_i`，而 `argmax(lm_head(h_i))` 就是被
mask 的那个 token —— 所以它**从来就不是一个真正的 cloze**。

---

## 2. 共享架构组件

以下组件由 cloze 家族（含 SlotDeep）共用，定义在 `pspr_cloze.py`。

### 2.1 候选格提取

```python
extract_lattice = LatticePathSelector.extract_lattice   # pspr_cloze.py:296
```
从 `lm_head(hidden)` 取 top-K=16，返回 `(unary_logits, candidate_ids, lattice_scalars)`。
`transform_unary_logits` 是 fp32 identity —— 格直接读原始 logits，不做任何变换。
三个 arm 共用**逐字节相同**的格提取与决策规则，所以任何 cloze-vs-v2 对比都只隔离头本身。

### 2.2 双流设计：一次双向 + 逐槽因果

| 流 | 回答什么问题 | 成本 |
|---|---|---|
| cloze rows（batched，每 block 一次） | "整个 block 长什么样？" | 1 次 encoder / slot 塔 |
| GRU over committed prefix（因果，逐 slot） | "我们实际提交了什么？" | 1 次小 matmul / slot |

**为什么不能把已提交前缀塞进 cloze 行**：那会让第 `i` 行依赖第 `0..i-1` 行，serving 必须把
encoder 跑 `H` 次；`H` 个串行 launch-bound kernel 的墙钟开销比它省下的 target token 还贵。
只在 FLOPs 上便宜的投机头不叫便宜。

### 2.3 晚融合修正器 + 冻结 tied 读出

```
delta_i = W2 · silu( W1 [ LN_h(h_i) ; LN_z(z_i) ; LN_s(S_i) ] )      # 三路各自归一化后 concat
score_k = <E(c_k), h_i + delta_i> - logZ = log_probs_k + <E(c_k), delta_i>
```

- `W2`（`delta_w2`）**零初始化** ⟹ `delta == 0` ⟹ step 0 与 plain DFlash **逐位一致**，
  任何增益都可归因；同时所有参数从 step 1 就有梯度（不同于 v1 的 `gamma=0` 标量门）。
- 读出用冻结的 tied embedding，**不学任何 `[vocab, *]` 矩阵**（Domino 在这上面花掉 50.82M 中的 38.9M）。
  `E(h + delta)` 天然是全词表分布，`full_vocab_logits` 免费。
- `restore_zero_init_contract()` 必须用 `nn.init.zeros_` 而非裸 `.zero_()`：HF 在 `_init_weights`
  期间会给 `nn.init.*` 打 `_is_hf_initialized` 守卫跳过已加载张量，裸 in-place 会绕过守卫，
  在 `post_init` 里把热启动权重悄悄抹成 0，而且 HF 仍报告 `missing=0 / unexpected=0`。

### 2.4 frontier 检测器（err head）

`err_logits()` 输出每个 slot 的 KEEP/REPAIR 门 logit。
标签是 **"骨干 top-1 在此处是错的"**：`err_target = ~(target_is_candidate & target_index == 0)`
（`dflash_family_model.py:1675`）。**故意把 top-k miss 也算作正类** —— 它们修不好，selector CE 会丢掉
它们，但检测器仍必须标记，因为它们照样终止接受串。

注意：serving 时 `--gate-theta 0`，检测器**不是活跃的门**，只作表示监督用。

### 2.5 决策规则（serving）

```
margin_gate:  repair iff  p_alt > tau + rho * p0        部署值 (tau=0, rho=3, theta=0)
keep_repair:  factorized，P(keep)=σ(-e), P(repair j)=σ(e)·softmax(alt)[j]
full_vocab:   argmax over 151936（训练可用，serving 一般不用）
```

`selector_decision_mode` 与 `dflash2_selector_objective` 在 `__init__` 里做了强一致性校验
（`dflash_family_model.py:612-644`）：`margin_gate` ⇔ `{multiclass, candidate_distill,
profitable_action, covered_conditional}`，不匹配直接 `ValueError`，防止训练目标与 serving 规则脱节。

---

## 3. 训练流程

### 3.0 阶段 0：数据重生成（**必需前置**）

draft 的任务不是"预测人类会怎么回答"，而是"猜中 target 的 greedy argmax"。
原始 open-perfectblend 上 `P(target greedy == corpus next token) = 0.7302`
⟹ **27% 的监督信号在教 draft 输出永远不可能被接受的 token**。

```bash
# 1) 分层抽样 20 万条（按可用行而非总行分配配额）
python scripts/sample_perfectblend.py \
  --source-dir /path/to/open-perfectblend/data \
  --total 200000 --output cache/dataset/perfectblend_200k.jsonl

# 2) 起 8 个 target server（每卡一个，端口间隔 10）
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
  --model-path /path/to/Qwen3-4B --dtype bfloat16 --tp-size 1 \
    --port $((30000 + i*10)) --mem-fraction-static 0.85 &
done

# 3) 重生成：丢弃全部原 assistant 轮，用已重生成的历史逐轮重问，完全 on-policy
PYTHONPATH=. python -u scripts/regenerate_train_data.py \
  --model /path/to/Qwen3-4B \
  --server-address 127.0.0.1:30000 ... 127.0.0.1:30070 \
  --concurrency 64 --max-tokens 2048 \
  --temperature 0 --reasoning disable \
  --input-file-path  cache/dataset/perfectblend_200k.jsonl \
  --output-file-path cache/dataset/perfectblend_qwen3-4b_regen.jsonl --resume

# 4) 验证一致率（应从 0.7302 升到 ~0.97）
PYTHONPATH=. python scripts/measure_label_mismatch.py \
  --target-model /path/to/Qwen3-4B \
  --data cache/dataset/perfectblend_qwen3-4b_regen.jsonl \
  --num-samples 200 --chat-template qwen-nosys
```

产物：199,293 行 / 794 MB。

**`chat_template: qwen-nosys` 是硬性要求。** `specforge/data/parse.py:211` 会把模板的
`system_prompt` 作为 system turn 前置到每条训练序列，而 `qwen` 模板带
`"You are a helpful assistant."`；评测端用 `apply_chat_template` 处理裸 user 消息，
**不产生任何 system turn**。用 `qwen` 训练 = draft 每条序列都看到一个部署时不存在的前缀。
`qwen-nosys`（`specforge/data/template.py:130`）与 `qwen` 唯一差别是 `system_prompt=None`，
渲染结果与 tokenizer 自带模板逐字节一致。thinking 也须三端一致：regen 必须带 `--reasoning disable`。

### 3.1 两阶段训练策略

```
┌──────────────── Stage 1 ────────────────┐  ┌───────── Stage 2 ─────────┐
backbone      官方 DFlash-b16，freeze_backbone: true         freeze_backbone: false
      (58 张量，bf16，零梯度)        lr × 0.06 = 3e-5
selector      从零初始化，delta_w2 = 0 ⟹ step0 ≡ plain DFlash   热启动 stage1 权重
              lr 5e-4，1 pass = 7115 步           lr 5e-4（重新加热），Adam moments 重置
起点          Qwen3-4B-DFlash-b16          outputs/slotdeep_joint_init
```

**Stage 1 为什么冻结**：骨干冻结时任何接受长度变化都能唯一归因到头上，且 27M~47M 的头能在
1 pass 内收敛。代价是 D-PACE 权重由固定骨干产生因而是静态的 —— 所以
**stage-1 上对 `dpace` / `lambda` 做的消融结论不适用于 stage 2**。

**Stage 2 的诚实边界**：这是一次热启动联合续训，**不是**干净的"解冻骨干的效应"测量。
相对 stage 1 它同时还：重置 Adam moments、重启 cosine schedule（把一个在 `lr 6.89e-10` 结束的
selector 重新加热回 `5e-4`）、`max_grad_norm` 跨 selector+backbone 全局生效（骨干梯度触发裁剪时
也会改变 selector 更新）。要把 delta 归因给"解冻"本身，需要一个从同一 joined checkpoint 出发、
同样 fresh optimizer / schedule / 数据序 / 步数、只有 `freeze_backbone` 不同的对照 —— 这个对照
**不在本轮范围内**。

### 3.2 损失函数

总损失（每个 objective chunk 累加**分子**，trainer 施加一个全局 all-reduce 的分母）：

```
L =  [ w · Σ(-log q_t · W_t)  +  (1-w) · Σ((1-q_t) · W_t) ]    ← 骨干 (lk_loss_type: lambda)
   +  α_sel · Σ( CE_16way,i · Wsel_i )            ← selector 主目标
   +  α_err · Σ( BCE(err_i, ~base_top1_ok_i) · reachable_i )← 辅助 frontier 检测器
   [+ α_alt · alt_CE + α_safe · safe_hinge + α_rep · repair_hinge]  ← SlotDeep 专用辅助项
   ────────────────────────────────────────────────────────────────
   /  Σ W_t     （跨 rank 与梯度累积全局 all-reduce）

W_t = weight_mask · dpace_weights
w   = kl_scale · exp(-kl_decay · acceptance)
```

| 配置项 | 值 | 语义 |
|---|---|---|
| `loss_type: dpace` | `dpace_alpha=0.5` | `smooth=(1-α)p+α`；`prefix=cumprod(smooth)`；`suffix=reverse_cumsum(prefix·mask)`。suffix 恰是该 slot 对 `E[L]=Σprefix[i]` 的**边际贡献**，随骨干变强自适应 |
| `lk_loss_type: lambda` | `kl_scale=1, kl_decay=1` | `tv_num = Σ(1−q(target))·w` **字面就是 1−接受概率**。CE 的梯度在 `q→1` 处消失，TV 直接顶接受率。按当前接受率自适应混合：弱时 CE 主导（梯度好），强时 TV 主导 |
| `dflash2_selector_objective: multiclass` | — | top-16 上的 K 路 CE。**top-k miss 不产生标签**（`selector_supervised = target_is_candidate`） |
| `dflash2_selector_weight_mode: uniform_frontier_boost` | `boost=3.0` | 保留全部 uniform 覆盖，**外加**对"当前首错 slot"乘 `(1+boost)`。`boost=0` 与 `uniform` 逐位一致。15 slot 下 frontier 占约 22% 权重（uniform 下 6.7%） |
| `dflash2_selector_err_loss_alpha: 1.0` | — | 辅助 BCE，标签 = "骨干 top-1 错了"，只在 frontier-reachable slot 上加权 |
| `dflash2_selector_stop_gradient: false` | — | **必须 false**：截断 unary/骨干边界的梯度正是让 selector 目标对骨干不可见，这个开关就是"joint"的含义 |
| `dflash2_selector_own_denominator: false` | — | **`true` 已被代码硬拒**（`dflash_family_model.py:527`）：局部 selector 均值在 DDP/梯度累积下非分区不变 |
| `dflash2_selector_target_greedy_labels: false` | — | online 下必须 false：流式 FeatureContract 只透传 `input_ids`/`loss_mask` |

**为什么 `uniform_frontier_boost` 不是 `teacher_forced_frontier`**：接受长度只在**一个** slot 上被
截断（selector 第一次搞错的那个），但 `teacher_forced_frontier` 把该 slot 之后全部清零，恰好饿死
cloze 行当作上下文来读的非 frontier 表示。`uniform_frontier_boost` 是两者之间：没有任何 slot 丢监督。

`frontier_mask`（`dflash_family_model.py:209`）= 前导正确串 **+** 第一个错配，即解码实际能到达的 slot。

### 3.3 优化器分组规则

两层，必须一起读：

```python
# 层 1：排序 —— specforge/training/assembly.py:275
lr_scale_rules = tuple(sorted(rules.items(), key=lambda kv: -len(kv[0])))   # 按 key 长度降序

# 层 2：匹配 —— specforge/optimizer.py:94
for prefix, value in rules:
    if name.startswith(prefix) or f".{prefix}" in name:   # 注意第二个条件：任意点分路径段都能匹配
        return float(value)
```

**首个命中生效，因为排过序所以等价于最长匹配。** YAML 里的书写顺序**不是** load-bearing。
`f".{prefix}" in name` 让 `"in_proj_weight"` / `"weight_ih_l0"` / `"err_head.1.weight"`
这类裸片段也能匹配嵌套参数名。

Stage 2 的 `{"candidate_selector.": 1.0, "": 0.06}`：`""` 长度 0 永远最后评估
⟹ 骨干 `5e-4 × 0.06 = 3e-5`，selector 保持 `5e-4`，比值 16.67×。

> **历史踩坑**：`weight_decay_rules` 里放一条 `"candidate_selector."`（19 字符）catch-all 会
> **遮蔽所有更短的规则**，PSPR-v2 就是这样把全部 74 个 LayerNorm 的 weight/bias 都按 2e-3 衰减了。
> 现在的配方**不设 catch-all**，只逐条枚举 2-D 权重矩阵，让 `training.weight_decay: 0.0` 兜底。

### 3.4 热启动加载语义

`specforge/training/model_loading.py:410` — **只加载 draft 权重，绝不加载 optimizer/counters/RNG。**

- `_load_pretrained_draft_state` 必须 pin `dtype=torch.float32` 载入，只在最终 `load_state_dict`
  时单次 cast，否则会遵循 `config.dtype=bf16` 造成**双次舍入**。
- `warm_start_optional_prefixes`（如 `candidate_selector.`）是 **all-or-nothing**：
  若 checkpoint 里已有该前缀的**任一**键，就不再豁免缺失张量（`model_loading.py:463`）。
  半个头的 checkpoint 是"部分应用的热启动"，不是"全新的头"。
- `warm_start_optional_key_groups` 是精确的全有全无键集合，**写了一半仍是硬错误**。
- 任何 `unexpected_keys`、或加载 0 个键，都是硬错误。

**推论**：stage 2 的 warm start 只搬权重，Adam moments 是全新的、cosine schedule 会重启。
真正的续训要走 `training.resume_from`（`assembly.py:649`），那条路才会恢复 optimizer state。

### 3.5 启动方式（online disaggregated / managed_local）

```bash
# 0) 环境（patched sglang 必须在 PYTHONPATH 最前）
export PYTHONPATH=/path/to/sglang-patched-0.5.15:.
export OMP_NUM_THREADS=4 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 HF_HUB_OFFLINE=1
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

# 1) 先看解析出的进程计划，不启动 worker
python -m specforge.cli train -c <config>.yaml --plan

# 2) 正式训练
python -m specforge.cli train -c <config>.yaml

# 支持点号覆盖，便于扫参
python -m specforge.cli train -c <config>.yaml training.learning_rate=2e-4
```

`managed_local` 下 supervisor 按 phase 拉起：

```
phase 0  mooncake_master （GPU 全部隐藏）
phase 1  sglang.launch_server --enable-spec-capture --spec-capture-method ...   GPU 0
  └─ health check http://127.0.0.1:30000/health
then     producer  （GPU 全部隐藏，纯 CPU）
         consumer  torchrun nproc_per_node=7      GPU 1-7
```

启动前 preflight（`launch_plan.py:946`）：`control_dir` 必须不存在、`mooncake_master` 在 PATH、
mooncake python 包可导入、**patched SGLang 带 spec capture**、所有端口空闲。

> capture server 必须是打了 spec-capture patch 的 sglang 构建。
> `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` 只在这条 capture 路径上验证过，不代表所有 SGLang 特性。

### 3.6 导出与评测

```bash
# 导出（cloze 家族用 cloze 版脚本；PSPRDraftModel 用 export_pspr_for_decode.py）
PYTHONPATH=. python scripts/export_pspr_cloze_for_decode.py \
  --checkpoint outputs/<run>/<run>-step7115 --output <decode_ckpt>

# 评测：六域，同 manifest，greedy，非 thinking
cd TAPS-SP && python scripts/decode_lattice.py \
  --datasets gsm8k,math500,humaneval,mbpp,alpaca,mt-bench \
  --modes oneshot,latgate,oracle16 \
  --gate-tau 0 --gate-rho 3 --gate-theta 0 \
  --eval-reserved --max-samples 20 --max-new-tokens 256 --shuffle-seed 2026 \
  --json-output <out>.json
```

`--modes` 三个值：

- `oneshot` — `lm_head(draft(...)).argmax()`，**纯骨干，完全不过头与门**，用于隔离骨干本身
- `latgate` — 完整 PSPR：候选格 + 选择头 + 标定门
- `oracle16` — 每 slot 若 target 在 top-16 就算命中，block=16 的理论上限

---

## 4. 实测结果全表

### 4.1 SlotDeep 各 arm（n=20/域，同一 seed 2026 manifest，`--gate-rho 3`）

| arm | 训练配置 | macro | pooled |
|---|---|---:|---:|
| plain DFlash（oneshot） | — | 5.3046 | 4.8734 |
| cloze @7115 | 参考（同 manifest 对照） | 6.0501 | 5.4676 |
| **slotdeep-s1 @7115** | 默认配方 | **6.0945** | **5.5039** |
| turn-s1 @7115 | turnwise 数据 | 6.0697 | 5.4876 |
| distill-s1 @7115 | `candidate_distill α=0.5` | 6.0570 | 5.4718 |
| loss-T0（S1 续 1000 步） | `alt=0, safe=0, rep=0` | 6.1095 | 5.5041 |
| loss-T1 | `alt=0.5` | 6.1157 | 5.5108 |
| loss-T2 | `alt=0.5, safe=0.1, rep=0.1` | 6.1106 | 5.5108 |
| **loss-T3** | `alt=0.5, safe=1.0, rep=0.1` | **6.1180** | **5.5165** |
| **Stage 2 @9052** | T3 起点，骨干解冻 | **6.4386** | **5.7764** |
| oracle@16 | — | 10.4424 | 9.8926 |

**turnwise 数据与 candidate_distill 两个改动都没帮上忙。** T0→T3 四组辅助 loss 的差异在
0.009 量级，而 n=20 分辨不了 0.04 量级 —— **这四组之间无法区分**，T3 被选为 stage 2 起点更多
是流程上的选择而非统计结论。

### 4.2 与官方 Domino 的对比（n=50/域，同 300 条 manifest，H=15，带配对 bootstrap）

来自 `outputs/STAGE1_STAGE2_DOMINO_H15_EVAL50_20260907/COMPARISON.md`：

| 数据集 | Stage1 T3 | 官方 Domino H15 | Stage2 @9052 |
|---|---:|---:|---:|
| gsm8k | 7.5304 | 8.8457 | 8.3377 |
| math500 | 8.1037 | 7.9820 | 8.5699 |
| humaneval | 7.0882 | 6.6384 | 7.3384 |
| mbpp | 6.6884 | 6.7986 | 6.9525 |
| alpaca | 3.0162 | 3.4863 | 3.1303 |
| mt-bench | 3.6204 | 3.9231 | 3.7847 |
| **macro** | **6.0079** | **6.2790** | **6.3523** |
| micro | 5.4787 | 5.8140 | 5.7550 |

提示级配对 bootstrap（10000 次，seed 2026）：

| 对比 | macro 差 | 95% CI | 结论 |
|---|---:|---|---|
| Stage1 − Domino | −0.2711 | [−0.3767, −0.1704] | **显著落后** |
| Stage2 − Domino | +0.0733 | [−0.0200, +0.1615] | **跨零 ⟹ 追平，非赢** |
| Stage2 − Stage1 | +0.3444 | [+0.2922, +0.3996] | **显著提升** |

### 4.3 口径警告（务必读）

1. **两个"domino"是完全不同的东西。**
   - `outputs/domino/best.pt`（203MB）= 挂在同一冻结 DFlash 骨干上的 **domino projector head 复现**，
     `decode_lattice.py --modes domino` 走这条。macro 5.8868（n=20）。这才是 PSPR 的**直接对照组**。
   - `models/Qwen3-4B-Domino-b16`（1.18GB）= **官方 Domino 完整 draft 模型**，自己的骨干、自己的训练预算，
     `bench_official_domino.py` 走这条。macro 6.2790（n=50, H15）。这是**另一条路线**，
     只有解冻骨干后的 Stage 2 才与它自由度对等。

2. **绝不跨场次对比。** 同一个 cloze checkpoint 在两次运行里给出 6.0498 / 6.0438，
   而 Domino 侧在不同场次相差 0.574 —— 那个差来自实现路径不同，不是噪声。

3. **n=20 分辨不了 0.04 量级。** 已有反例：latfront t0.6 在两台机器上分别是 +0.0415 和 −0.344，
   完全反转。0.05 macro 的 80% power 需要约 80 prompt/域。

4. `--max-samples 40` 与 `--max-samples 20` 的历史结果**不可混用**（"老卡 Domino 6.4255" 是
   n=40/20 混合口径的产物）。

5. MATH500 已知有 index 7 训练重叠；这是已被反复使用的开发 benchmark，不声称是全新测试集。

6. 本表只衡量**相对 target 逐 token 一致的接受长度**，不代表 GSM8K 答题正确率或 HumanEval pass@1；
 也没有做延迟/吞吐测量，不据此宣称更快。

---

## 5. 已关闭的方向（避免重复投入）

以下都在同一冻结骨干与数据上单独测量过，均劣于或等于当前配置：

| 方向 | 结论 |
|---|---|
| `pspr_decision`（分解 keep/repair） | held-out 一律降 2–3pp |
| `pspr_cascade`（门读 ranker 分数） | latrepair macro 5.8085，低于 latgate |
| `pspr_memory`（读已验证 target 特征） | verified 6.0509 vs **blind 对照 6.0570** — 盲对照反而更高，上下文无价值 |
| `candidate_distill` | 6.0570 < 6.0945 |
| turnwise 数据 | 6.0697 < 6.0945 |
| gate / 阈值 / 加打分头容量 | 三条路 held-out 一律降 2–3pp |
| `selector_trans_rank`（vocab codebook, 77.8M） | 诊断显示需修正 slot 上 `cos(top1_emb, truth_emb)=0.1382`（近乎正交），候选可区分性不是瓶颈 |
| `tok_rank` / `pair_dim` / `thidden_dim` | +0.315 / +0.388 / — ，均劣于基线 +0.440 |
| `err_w = 0`、depth-aware gate、`dropout 0.2`、`accept_w > 0`、conv 分支 | 均更差 |

**结构死区（未解决）**：ρ=3 时 41.7% 的"truth 是最佳替代"决策**不可触发**，slot 0 高达 90.9%；
slot 0 的 `captured` 仅 0.71%。

**最大未动杠杆（零模型改动）**：head 延迟的 **72.5% 在 15 步串行 Python 循环 + 每槽 2 次
device→host 同步**，两臂相同。换 encoder 只省 0.767ms（tok/s +2.7）。消除 host sync +
CUDA-graph 化 `TAPS-SP/scripts/decode_lattice.py` 可达 ~166 tok/s（超 Domino 160.4）。

---

## 6. 门禁（等价性验证）

每个门禁验证一处"移植后必须逐位等价"的契约。改动落地前必须全绿。

| 门禁 | 验证什么 |
|---|---|
| `gate_pspr_reference_equivalence.py` | `LatticePathSelector` 与参考实现 7 个分量 `max\|Δ\|=0`，argmax 60/60 |
| `gate_pspr_joint_equivalence.py` | stage-2 联合路径与 stage-1 在 step 0 等价（23/23） |
| `gate_pspr_warm_start.py` | 热启动张量在真实 trainer 的 bf16 路径下精确（71 selector + 58 骨干） |
| `gate_slotdeep.py` | SlotDeep G1–G13，含 step-0 与 plain DFlash 逐位一致 |
| `gate_slotdeep_joint_groups.py` | Stage-2 G1–G6，含 9 项 provenance 断言（SHA 锁定、dtype 未漂移、wd 规则未泄漏） |
| `gate_pspr_objective.py` / `gate_pspr_selector.py` | 目标函数 / 选择头 |
| `gate_pspr_selector_weight_mode.py` | 各 weight_mode 语义 |
| `gate_spec_capture_fidelity.py` | online capture 的 hidden states 与本地参考前向一致 |
| `scripts/run_pspr_reproduction_gate.sh` | 复现门禁总入口 |

---

## 7. 文件索引

### 模型
```
specforge/modeling/draft/
├── pspr.py    LatticePathSelector    / PSPRDraftModel
├── pspr_v2.py      LatticeCorrector       / PSPRv2DraftModel
├── pspr_cloze.py      ClozeCorrector   / PSPRClozeDraftModel
├── pspr_slotdeep.py         SlotDeepCorrector   / PSPRSlotDeepDraftModel   ← 主线
├── pspr_decision.py         DecisionCorrector      / PSPRDecisionDraftModel
├── pspr_cascade.py       CascadeCorrector       / PSPRCascadeDraftModel
├── pspr_memory.py   MemoryCorrector        / PSPRMemoryDraftModel
├── pspr_oracle_probe.py     OracleFutureProbeSelector（诊断）
└── __init__.py              全部经 @register_draft 注册
```

### 配置
```
configs/qwen3-4b-pspr-slotdeep.json         Stage 1（freeze_backbone: true）
configs/qwen3-4b-pspr-slotdeep-joint.json   Stage 2（freeze_backbone: false，其余完全相同）
examples/configs/online/disaggregated/managed-local/
├── qwen3-4b-pspr-slotdeep.yaml Stage 1 训练配方
└── qwen3-4b-pspr-slotdeep-joint.yaml Stage 2 训练配方（+ lr_scale_rules, save_interval=200）
```

### 关键代码位置
| 关注点 | 位置 |
|---|---|
| CLI / `--plan` | `specforge/cli.py:177-268` |
| 进程计划 / managed_local | `specforge/launch_plan.py:426-551, 686` |
| **全部损失数学** | `specforge/algorithms/common/dflash_family_model.py` |
| selector chunk 项 | 同上 `:1024-1690` |
| frontier_mask | 同上 `:209` |
| 辅助 alt/safe/repair | 同上 `:257-316` |
| `apply_backbone_freeze` 调用点 | `specforge/algorithms/model_providers.py:379` |
| 优化器分组 + 规则匹配 | `specforge/optimizer.py:71-113` |
| 规则排序 | `specforge/training/assembly.py:275` |
| 热启动 | `specforge/training/model_loading.py:410-500` |
| anchor 采样 | `dflash_family_model.py:832-872` |
| chat template | `specforge/data/template.py:130` |

---

## 8. 硬性约定

1. **不改 `pspr.py` / `pspr_v2.py` / `pspr_cloze.py`** —— 它们绑着已发布 checkpoint 和逐位等价门禁。新想法走子类。
2. **重要改动必须先过 codex 审查再开训** —— 但审查结论要逐条核实，不全信。
3. **macro 口径 = block 池化（macroB）**，报告时须同时标注 macroA。
4. **每步记入 `TAPS-SP/PSPR_DECISION.md`**（append-only）。
5. **绝不跨场次对比两个 arm**。
6. `nohup` 必须用绝对路径脚本 + 显式重定向。
7. 冒烟脚本里裸 `PYTHONPATH=.` 会覆盖掉 patched sglang 包 —— 必须写全 `PYTHONPATH=.:/path/to/sglang-patched-0.5.15`。
