# PSPR — 面向 draft block top-K 候选格的双向路径选择头

PSPR 是一个投机解码 draft 头，构建在 DFlash 的 block-diffusion 骨干之上：*a bidirectional path
selector over the draft block's top-K candidate lattice*。本文档说明它在本仓库中的代码位置、
训练方式和参数配置依据。

上游基座：`sgl-project/SpecForge` @ `2fc9930`。

---

## 1. 设计定位

一次 all-MASK 并行前向之后，DFlash 骨干为 block 内每个 slot 产出一份 hidden state；经 target 的
`lm_head` 读出后，每个 slot 得到一个 top-K 候选集（K=16）。这个 **[block_size, K] 的候选格**就是
PSPR 的输入。

与本仓库中两个最近邻的区别：

| | 每个 slot 的头能看到什么 | 决策规则 |
|---|---|---|
| `DominoDraftModel` | 只有自己 slot 的 `[z_n ; causal_GRU(已提交前缀)_n]` | per-slot argmax |
| `DFlash2DraftModel.CandidateSelector` | 自己 slot + **前驱 token**（严格局部低秩 bigram `unary + <pred·Wh, succ>`） | 左到右贪心 walk |
| **PSPR `LatticePathSelector`** | **整个 block 所有 slot 的候选集、log-probs 与不确定性统计** | 双向 transformer 编码后打分 |

核心是那个双向 transformer：在任何 token 被提交之前，每个 slot 的 summary 都 attend 到其他所有
slot 的候选集。这在解码时是合法的，因为候选格来自**一次**并行 all-MASK pass，整个 block 内固定不变
（与"已提交 token"不同，后者是逐步产生的）。

**为什么候选格不是骨干双向性的冗余**：骨干确实混合了 slot（block 内 attention 无限制），但它混合的是
*hidden states*。候选格是 `lm_head` 读出的结果——一个 `[vocab, hidden]` 投影加 top-K argsort——
5 层骨干从不计算这个量。而它的消费者是头，在 Domino 里头只能看到自己那一个 slot。

参数量 27.06M，含一个 `err_head` frontier 检测器（判断"骨干 top-1 在此处是错的"）及其辅助 BCE 目标。

参考实现中另有三个分支**未移植**，每个都在同一冻结骨干与数据上单独测量为更差：`tok_rank`
（逐 token 学习表，+0.315 vs +0.440）、`pair_dim`（LiLiCorr 式 bigram 兼容性，+0.388）、
`thidden_dim`（把 target 的 anchor context 喂给编码器）。

---

## 2. 当前结果

评测协议：`TAPS-SP/scripts/decode_lattice.py`，六域 `gsm8k,math500,humaneval,mbpp,alpaca,mt-bench`，
每域 `--max-samples 40 --max-new-tokens 256 --eval-reserved`，报 macro 平均接受长度。

| 配置 | macro accept length |
|---|---|
| 纯 one-shot（DFlash 骨干，不过任何头） | 5.2842 |
| 官方 Domino 头（同一冻结骨干） | 5.594 |
| PSPR stage-1，早期 global batch 128 | 5.6923 |
| **PSPR stage-1，global batch 对齐后** | **5.7217** |
| 参考 dh2048 实现 | 5.7352（复现率 99.76%，3/6 域超过参考） |
| 官方 Domino released（唯一合法对手） | 6.4255 |
| oracle@16（block 16 的理论上限） | 9.917 |

分域数据（`--eval-reserved --max-samples 40 --max-new-tokens 256`）：

| 域 | oneshot | PSPR stage-1 | dh2048 参考 |
|---|---|---|---|
| gsm8k | 6.040 | 6.813 | 6.907 |
| math500 | 7.474 | **8.139** | 8.088 |
| humaneval | 6.130 | **6.676** | 6.628 |
| mbpp | 6.082 | 6.399 | 6.497 |
| alpaca | 2.801 | **2.959** | 2.947 |
| mt-bench | 3.178 | 3.344 | 3.344 |
| **macro** | **5.2842** | **5.7217** | **5.7352** |

stage-2（解冻骨干的联合训练）**尚未产出可用结果**，进度见 `TAPS-SP/EXPERIMENT_LOG.md` 迭代 AG/AH。

---

## 3. 代码地图

### 3.1 新增文件（PSPR 本体）

| 路径 | 行数 | 作用 |
|---|---|---|
| `specforge/modeling/draft/pspr.py` | 578 | **核心**。`LatticePathSelector`（选择头）+ `PSPRDraftModel`（`DFlashDraftModel` 子类），经 `@register_draft` 注册 |
| `specforge/algorithms/pspr/accept_selector.py` | 249 | stage-1 的选择器目标函数与**标定门（calibrated gate）**评测：`slot_probs` `frontier_mask` `expected_accept` `dpace_weights` `eval_gate` `report_gate` |
| `specforge/algorithms/pspr/__init__.py` | 27 | 上述符号的导出面 |
| `specforge/data/lattice_trace_data.py` | 192 | anchor 级候选格 trace 的数据平面（逐字节复刻参考实现的 reader） |
| `specforge/algorithms/common/lattice_trace_data.py` | 183 | 预算候选格特征的 FeatureContract 接入 |

`pspr.py` 中的关键方法：

- `extract_lattice`（:266）从 `lm_head` logits 取 top-K 并组装格
- `encode`（:311）双向 transformer 编码整个 block 的格
- `causal_states`（:332）已提交前缀的因果状态
- `delta_term` / `transition_term`（:339 / :377）候选分数的两个修正项
- `score`（:394）合成最终打分；`score_candidates`（:459）训练侧接口
- `err_logits`（:430）frontier 检测器
- `restore_zero_init_contract`（:349）**零初始化契约**：保证新初始化的选择头在数值上精确等于纯 unary DFlash
- `PSPRDraftModel.transform_unary_logits`（:553）identity（格直接取自 `lm_head` 原始 logits）

### 3.2 配置文件

| 路径 | 用途 |
|---|---|
| `configs/qwen3-4b-pspr.json` | draft 架构（stage-1，冻结骨干） |
| `configs/qwen3-4b-pspr-joint.json` | draft 架构（stage-2，联合训练，`freeze_backbone: false`） |
| `examples/configs/offline/colocated/qwen3-4b-pspr-offline.yaml` | stage-1 离线训练 |
| `examples/configs/offline/colocated/qwen3-4b-pspr-joint-offline.yaml` | stage-2 离线训练 |
| `examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-joint-online.yaml` | **stage-2 online 联合训练（主配置）** |

`dflash_config` 中的 PSPR 专有键：

```json
"selector_top_k": 16,          // 候选格宽度 K，与 DFlash2 一致
"selector_dim": 512,           // 双向 transformer 宽度
"selector_layers": 3,          // 层数
"selector_heads": 8,
"selector_state_dim": 512,     // 因果前缀状态维度
"selector_delta_hidden": 2048, // delta MLP 隐层（即 "dh2048"）
"selector_max_slots": 32,
"selector_dropout": 0.1,
"selector_output_zero_init": false,
"selector_trans_rank": 0,      // 0 = 关闭 codebook（见 §5.3）
"freeze_backbone": false
```

### 3.3 工具脚本

| 脚本 | 作用 |
|---|---|
| `scripts/train_pspr_accept_selector.py` | stage-1 选择器训练（移植自 `TAPS-SP/scripts/train_accept_selector.py`，后者是真源） |
| `scripts/import_pspr_selector.py` | 把 stage-1 的 `best.pt` 融合到 released DFlash 骨干上，产出 stage-2 的热启动目录 |
| `scripts/export_pspr_for_decode.py` | 把训练 checkpoint 导出为参考解码器可读格式（含 `PORT_TO_REFERENCE` 名字映射） |
| `scripts/sample_perfectblend.py` | 从 open-perfectblend 按**可用行**分层抽样（见 §5.4） |
| `scripts/measure_label_mismatch.py` | 测 `P(target greedy == corpus next token)`，即训练标签与接受判据的一致率 |
| `scripts/sweep_depth_gate.py` | 门参数扫描 |

### 3.4 门禁（等价性验证）

每个门禁都验证一处"移植后必须逐位等价"的契约：

| 门禁 | 验证什么 |
|---|---|
| `gate_pspr_reference_equivalence.py` | SpecForge 的 `LatticePathSelector` 与参考实现在 dh2048 权重下 7 个分量 `max|Δ| = 0`，argmax 60/60 一致 |
| `gate_accept_selector_equivalence.py` | 移植的选择器目标与参考训练脚本一致（20/20） |
| `gate_pspr_joint_equivalence.py` | stage-2 联合路径与 stage-1 在 step 0 等价（23/23） |
| `gate_pspr_warm_start.py` | 热启动张量在真实 trainer 的 bf16 路径下精确（71 selector + 58 骨干） |
| `gate_spec_capture_fidelity.py` | online capture 的 hidden states 与本地参考前向一致（生产 argv，20/20 slot） |
| `gate_onpolicy_collect_equivalence.py` | on-policy trace 采集等价（12/12） |
| `gate_pspr_objective.py` / `gate_pspr_selector.py` / `gate_target_greedy_labels.py` | 目标函数 / 选择头 / greedy 标签旁路 |
| `scripts/run_pspr_reproduction_gate.sh` | 复现门禁总入口 |

### 3.5 对上游的修改（17 个文件，585 插入 / 50 删除）

| 文件 | 改动性质 |
|---|---|
| `specforge/algorithms/common/dflash_family_model.py` | +250。选择器目标接入 DFlash 家族目标：`_selector_chunk_terms`、err objective、`selector_*` 度量 |
| `specforge/optimizer.py` | +58。`lr_scale_rules` / `weight_decay_rules`（按参数名前缀分组）|
| `specforge/runtime/data_plane/offline_reader.py` | +61。候选格 trace 的离线读取 |
| `specforge/algorithms/common/hidden_states_data.py` | +58。hidden states 数据面扩展 |
| `specforge/config/schema.py` | +29。`dflash2_selector_*`、`lr_scale_rules` 等键 |
| `specforge/algorithms/dflash/providers.py` | +33。把 selector 配置传给模型 |
| `specforge/algorithms/common/collation.py`、`contracts.py` | 各 +24。FeatureContract 支持格特征 |
| `specforge/training/model_loading.py` | +21。热启动加载（fp32 pin + all-or-nothing 前缀校验，见 §5.5）|
| `specforge/data/template.py` | +18。注册 `qwen-nosys` 模板（见 §5.2）|
| `specforge/data/preprocessing.py` | +15。预处理接入 |
| `specforge/algorithms/model_providers.py` | +11。模型构建分派 |
| `specforge/runtime/data_plane/feature_store.py`、`training/strategies/base.py`、`training/assembly.py`、`algorithms/domino/providers.py`、`modeling/draft/__init__.py` | 小改：注册与接线 |

---

## 4. 训练流程

### 阶段 0：数据重生成（**必需前置**）

draft 的任务不是"预测人类会怎么回答"，而是"猜中 target 下一个 token"——接受判据是
`draft_token == target 的 greedy argmax`。原始语料的 assistant 回复由人或其他模型写成，
Qwen3-4B 自己不会那样输出。实测原始 open-perfectblend 上
`P(target greedy == corpus next token) = 0.7302`，即 **27% 的监督信号在教 draft 输出永远不可能被接受
的 token**。官方同 target、同 online 模式的配方
（`examples/configs/online/disaggregated/external/qwen3-4b-dflash-online.yaml`）数据路径也正是
`perfectblend_qwen3-4b_regen.jsonl`。

```bash
# 1) 起 8 个 target server（每卡一个，端口间隔 10）
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
    --model-path /path/to/Qwen3-4B --dtype bfloat16 --tp-size 1 \
    --port $((30000 + i*10)) --mem-fraction-static 0.85 &
done

# 2) 重生成（temperature 0 对齐 greedy 评测协议；--reasoning disable 三端一致）
PYTHONPATH=. python -u scripts/regenerate_train_data.py \
  --model /path/to/Qwen3-4B \
  --server-address 127.0.0.1:30000 127.0.0.1:30010 127.0.0.1:30020 127.0.0.1:30030 \
                   127.0.0.1:30040 127.0.0.1:30050 127.0.0.1:30060 127.0.0.1:30070 \
  --concurrency 64 --max-tokens 2048 \
  --temperature 0 --reasoning disable \
  --input-file-path cache/dataset/perfectblend_200k.jsonl \
  --output-file-path cache/dataset/perfectblend_qwen3-4b_regen.jsonl \
  --resume

# 3) 验证一致率（应从 0.7302 升到 ~0.97）
PYTHONPATH=. python scripts/measure_label_mismatch.py \
  --target-model /path/to/Qwen3-4B \
  --data cache/dataset/perfectblend_qwen3-4b_regen.jsonl \
  --num-samples 200 --chat-template qwen-nosys
```

多轮对话是完全 on-policy 的：原 assistant 轮全部丢弃，每个 user 轮用**已重生成的历史**
（含 target 自己前几轮的输出）再调 target，无残留 off-policy。

### 阶段 1：冻结骨干训练选择头

真源是 `TAPS-SP/scripts/train_accept_selector.py`。产物 `outputs/pspr_dh2048_exact/best.pt`
（macro 5.7217，`gamma = 0.6901515`）。

### 阶段 2：联合训练（骨干 + 选择头）

```bash
# 1) 把 stage-1 选择头融合到 released DFlash 骨干，产出热启动目录
PYTHONPATH=. python scripts/import_pspr_selector.py \
  --selector outputs/pspr_dh2048_exact/best.pt \
  --draft-config configs/qwen3-4b-pspr-joint.json \
  --output-dir outputs/pspr_joint_init

# 2) 校验热启动张量（bf16 路径下逐张量精确）
PYTHONPATH=. python scripts/gate_pspr_warm_start.py --dtype bfloat16

# 3) 先看解析出的进程计划，不启动 worker
PYTHONPATH=. python -m specforge.cli train \
  -c examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-joint-online.yaml --plan

# 4) 正式训练（GPU0 跑 capture server，GPU1-7 训练）
PYTHONPATH=. python -m specforge.cli train \
  -c examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-joint-online.yaml
```

支持点号覆盖，便于扫参：

```bash
PYTHONPATH=. python -m specforge.cli train -c <config> training.learning_rate=2e-4
```

online 模式下 capture server 必须是打了 spec-capture patch 的 sglang 构建；
`specforge/launch_plan.py` 负责拼 server argv。

### 阶段 3：导出与评测

```bash
PYTHONPATH=. python scripts/export_pspr_for_decode.py \
  --checkpoint outputs/qwen3-4b-pspr-joint-online/step-XXXX \
  --output <decode_ckpt>

cd TAPS-SP && python scripts/decode_lattice.py \
  --datasets gsm8k,math500,humaneval,mbpp,alpaca,mt-bench \
  --modes latgate --gate-tau 0 --gate-rho 3 --gate-theta 0 \
  --eval-reserved --max-samples 40 --max-new-tokens 256
```

`--modes` 的三个值含义：

- `oneshot` — `lm_head(draft(...)).argmax()`，**纯 DFlash 骨干，完全不过选择头与门**。用于隔离
  骨干本身的好坏
- `latgate` — 完整 PSPR：候选格 + 选择头 + 标定门
- gate 元组 `(tau, rho, skip0, theta)`，stage-1 使用 `(0, 3.0, False, 0)`

---

## 5. 参数配置依据

### 5.1 骨干训练：照搬官方 dflash / dflash2

联合训练里骨干与选择头是两套策略。骨干这一侧不自创，直接对齐官方：

| 参数 | 值 | 出处 |
|---|---|---|
| `loss_type` | `dpace` | 官方 dflash2 |
| `lk_loss_type` | `lambda` | 官方 dflash2 |
| `num_anchors` | 512 | 官方 dflash 4b / 27b |
| `loss_decay_gamma` | 7.0 | 官方（注意 `dpace` 下不被读取，见下） |
| `max_grad_norm` / `warmup_ratio` | 1.0 / 0.04 | 官方 |
| `attention_backend` | `flex_attention` | 官方 |
| `weight_decay` | 0（默认） | 官方均未设 |
| lr 分组 | **单组** | 官方全部单组 |
| global batch | 28（`dp7 × bs2 × accum2`） | 对齐官方 4b 的 32 |

**为什么这两个目标函数直接决定接受长度**：

- `lk_loss_type: lambda` — `tv_num = Σ(1 − q(target))·w`。在 DFlash 的 hard target 下
  `q(target) = exp(-neg_log_q)`，所以这一项**字面就是 1 − 接受概率**。CE 优化 `log q`，其梯度在
  `q → 1` 处消失；而接受长度是 `q` 本身的函数。`lambda` 按当前平均接受率自适应混合：
  `kl_weight = exp(−acceptance)`，draft 弱时 CE 主导（梯度好），强时 TV 主导（直接顶接受率）。
  两项都只读 `target_ids`，**不需要额外监督信号**。
- `loss_type: dpace` — `prefix = cumprod((1−α)q + α)` 是"前 i 个 slot 全被接受"的概率，
  `suffix = reverse_cumsum(prefix)` 因此恰是**该 slot 对 `E[L] = Σ prefix[i]` 的边际贡献**，
  且随骨干变强自适应。相比之下 `loss_decay_gamma` 的 `exp(−i/7)` 是与模型当前表现无关的固定衰减。
  注意 `loss_decay_gamma` **只在 `loss_type == "dflash"` 分支被读取**
  （`dflash_family_model.py:673-682`），切到 `dpace` 后它是惰性 no-op，保留只为一行回退。

两者同开是安全的：`lambda` 的 `acceptance` 分母取的是 `accuracy_denom = weight_mask.sum()`
（`dflash_family_model.py:891`），**不是** D-PACE 加权的 `loss_den`，所以 D-PACE 权重不会稀释 TV 项。

### 5.2 训练 / 评测 prompt 必须逐字节一致

`specforge/data/parse.py:211-212` 会把模板的 `system_prompt` 作为 system turn 前置到每条训练序列，
而 `template.py` 的 `qwen` 模板带 `system_prompt="You are a helpful assistant."`。评测端
（`TAPS-SP/scripts/decode_lattice.py:347`）用 `apply_chat_template` 处理一条裸 user 消息，
**不产生任何 system turn**。用 `qwen` 训练意味着 draft 每条序列都看到一个部署时不存在的前缀。

因此本仓库注册了 `qwen-nosys`（与 `qwen` 唯一差别是 `system_prompt=None`），实测渲染结果与
tokenizer 自带模板**逐字节一致**且 loss mask 不变。**配置必须用 `chat_template: qwen-nosys`。**

三端 thinking 也须一致：评测端与训练端都是 `enable_thinking=False`，所以 regen 必须加
`--reasoning disable`。released DFlash-b16 的 model card 亦明确
`# Note: this draft model is used for thinking mode disabled`。

### 5.3 选择头参数：针对与 DFlash2 的差异做的适配

| 参数 | 本仓库 | 官方 dflash2 | 依据 |
|---|---|---|---|
| `dflash2_selector_warmup_ratio` / `ramp_ratio` | **0 / 0** | 0.0005 / 0.0005 | dflash2 的选择器从零初始化（`successor_codebook` 零初始化），需要骨干先稳再引入梯度。本仓库选择头是 stage-1 热启动。更强的理由：`training/strategies/base.py:479-482` 显示 warmup 期间 `alpha = 0` 即选择头完全不训练，而骨干正在改变，会让热启动权重与新骨干**失配加剧** |
| `dflash2_selector_loss_alpha` | 1.0 | 1.0 | 一致 |
| `dflash2_selector_err_loss_alpha` | 1.0 | 无此模块 | PSPR 专有的 err head；stage-1 消融证明 `err_w = 0` 更差 |
| `dflash2_selector_own_denominator` | `true` | `false` | 热启动权重就是在 `true` 这个目标下训出的，切 `false` 会改变 loss 尺度、与热启动不连续 |
| `dflash2_selector_stop_gradient` | `false` | `false` | **必须 false**：在 unary/骨干边界截断梯度正是会让选择器目标对骨干不可见，这个开关就是"joint"的含义 |
| `dflash2_selector_target_greedy_labels` | `false` | — | online 下必须 false：流式 FeatureContract 不声明可选张量，capture 布局只透传 `input_ids`/`loss_mask`，`target_greedy` 永远到不了，开启会在 `dflash_family_model.py:783-789` 抛错 |
| `selector_trans_rank` | 0（关闭） | `selector_rank: 256` | dflash2 的 vocab 级 codebook（77.8M）已证否：诊断显示需修正的 slot 上 `cos(top-1 emb, truth emb) = 0.1382`（近乎正交），候选可区分性不是瓶颈 |

### 5.4 数据抽样

`scripts/sample_perfectblend.py` 按 source 分层抽样 20 万条。注意配额按**可用行**而非总行分配：
`ultrafeedback_binarized` 约 37% 的行只有 prompt 没有 assistant 回复，按含废行的总体配额会系统性
欠采样（该源占比 8.815% → 实际可用 4.646%）。

### 5.5 热启动加载的两个正确性要求

`specforge/training/model_loading.py`：

- `_load_pretrained_draft_state` 必须 pin `dtype=torch.float32` 载入，只在最终 `load_state_dict`
  时做单次 cast。否则会遵循 `config.dtype = bf16` 造成**双次舍入**。
- `warm_start_optional_prefixes` 对 `candidate_selector.` 必须是 all-or-nothing：若 checkpoint 里
  已有该前缀的任一键，就不再豁免缺失张量。否则从联合 checkpoint 热启动时丢张量会静默通过。

另有一个已修的静默数据损坏：`pspr.py::restore_zero_init_contract()` 必须用 `nn.init.zeros_/ones_`
而非裸的 in-place `gamma.zero_()`。后者绕过 HF 在 `_init_weights` 期间对 `nn.init.*` 打的
`_is_hf_initialized` 守卫，而该调用发生在**加载权重之后**，会把 `gamma` 从 0.6914 抹成 0.0，
且 HF 仍报告 `missing=0 / unexpected=0 / mismatched=0`。

---

## 6. 已证否的方向（避免重复投入）

以下都在同一冻结骨干与数据上单独测量过，均劣于当前配置：`err_w = 0`、depth-aware gate、
`dropout 0.2`、`accept_w > 0`、conv 分支、`tok_rank`、`pair_dim`、`thidden`、
`selector_trans_rank`（codebook）。

`--data-frac 0.5` 相比全量 **+0.303**，说明 stage-1 在数据上是受限的。

需要注意的一条方法论：`loss_type: dpace` 与 `lk_loss_type: lambda` 在 stage-1 曾被测为更差，
但那是**冻结骨干**条件下的结论——骨干冻结时 D-PACE 权重由固定骨干产生因而是静态的，无法自适应。
联合训练下这两项才第一次真正生效，故 stage-1 的消融对它们不适用。
