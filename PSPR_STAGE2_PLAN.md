# PSPR Stage-2 联合训练：定稿方案

> 定稿日期：2026-09-02 · 配置文件：`examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-joint-online.yaml`
> 前置文档：`PSPR_Training_Route_A_vs_Route_B.md`（参考稿，本文列出全部分歧与纠错）
> 状态：**配置已就绪、门禁全绿、等待批准启动**

---

## 0. 一句话方案

在 released `Qwen3-4B-DFlash-b16` 骨干 + stage-1 最优 selector（macro **5.7217**）的基础上，用 on-policy 重生成语料做 **1 pass（7117 步）** 联合训练，**backbone 3e-5 / selector 6e-4** 差分学习率，selector 目标切成 `uniform` 以消除 stage-1→2 的目标突变。先用 400 步 pilot 判定"上一轮崩塌是数据问题还是学习率问题"，再决定是否放大 backbone 学习率。

对标：自己的 **5.7217**，唯一合法对手官方 Domino **6.4255**。

---

## 1. 为什么是这个方案：上一轮（迭代 AG）失败的归因

AG 跑到 step 3370 主动停止。训练侧健康（loss 5.333→2.644、无 NaN、grad_norm 0.22、`gamma` 0.6914→0.8320），但 gsm8k 解码崩塌：

| ckpt | oneshot（纯骨干） | latgate（过 selector） |
|---|---|---|
| joint_init | **6.040** | **6.813** |
| step 200 | 4.374 | 4.688 |
| step 3000 | 4.396 | 4.770 |

`oneshot = lm_head(draft(...)).argmax()`，**不经过 selector/gate** ⇒ 骨干本身坏了。两条排除性证据：① oneshot 也掉 1.67 ⇒ 非测量假象；② 骨干漂移涨 5.4 倍（0.034→0.184）而损害持平 ⇒ **不是"lr 太大、损害逐步累积"**。决定性反向指标：语料标签 `train/acc` 0.284→0.458 **上升**，而目标贪心 `oneshot` 6.040→4.396 **下降** —— 模型在学语料，而语料是错的。

**AG 的 backbone 学习率是 3e-5**（`2.8e-4 × 0.05`），已是官方 `6e-4` 的 **1/20**，仍然在 step 200 就崩。这是"根因不在学习率"的最强证据。

迭代 AH 修掉三件事，全部与此相关：

| 修复 | 前 | 后 |
|---|---|---|
| 语料 off-policy（`regenerate_train_data.py` 重生成） | 一致率 0.7302 | **0.9876**（n=1000，466859 监督位） |
| `system_prompt` 注入（`parse.py:211-212`，评测端没有 ⇒ draft 永远看到部署时不存在的前缀） | `chat_template: qwen` | `qwen-nosys`（与 tokenizer jinja 逐字节一致，loss_mask 不变） |
| selector 目标突变（本轮新发现，见 §3） | 继承 D-PACE 权重 | `uniform` |

---

## 2. 定稿参数表

全部值均已通过 schema 解析、optimizer 分组与 `--plan` 校验。

### 2.1 训练规模

| 参数 | 值 | 依据 |
|---|---|---|
| `num_epochs` | **1** | warm start 不需要 6 pass。**更关键**：`warmup_ratio` 是 `total_steps` 的比例，6 pass（42705 步）下 warmup 达 1708 步，400 步的 pilot 届时 lr 只爬到 23%，**测不到目标学习率**。1 pass ⇒ warmup 284 步，pilot 有效。 |
| `total_steps` | **7117** | `199293 × 1 / (dp7 × bs2 × accum2 = 28)`。注意是 199293 而非 200000（regen 有 597 条超长输入失败、110 条跳过，损失 0.30%）。 |
| `batch_size` / `accumulation_steps` | 2 / 2 | global batch 28，贴近官方 4b 的 8×4×1 = 32 |
| `warmup_ratio` | 0.04 | 官方 dflash recipe |
| `save_interval` | **200** | 证据驱动：AG 的崩塌在 step 200 已完全可见（6.040→4.374），这是 pilot 决定性测量点。35 个 ckpt ≈ 262 GB，磁盘余 32 TB。 |
| `fsdp_sharding` | `NO_SHARD` | 官方 dp7 recipe 都锁这个值 |

### 2.2 优化器（本轮最主要的改动）

```yaml
learning_rate: 6.0e-4          # = selector 的率
lr_scale_rules:
  "candidate_selector.gamma": 1.0
  "candidate_selector.": 1.0
  "": 0.05                     # catch-all = backbone
weight_decay: 0.0              # backbone，照官方
weight_decay_rules:
  "candidate_selector.gamma": 0.0
  "candidate_selector.": 2.0e-3
```

**实测分组**（`base_lrs`，非 warmup 当前值）：

| 组 | 张量数 | base_lr | weight_decay |
|---|---|---|---|
| backbone | 58 | **3.0e-5** | 0 |
| selector | 70 | **6.0e-4** | 2e-3 |
| gamma | 1 | 6.0e-4 | 0 |

三点说明：

1. **为什么要差分。** 我上一轮删掉了 `lr_scale_rules`，理由是 `optimizer.py:75-77` 的前提是 *"pretrained backbone together with a freshly initialised head"* 而我们的 head 是 warm start ⇒ 前提不成立。这个推理只否掉了**那条 5× gamma 规则**，不该顺手否掉差分本身。差分有独立且更强的理由：**backbone 是 released 官方权重，它的 one-shot 6.040 是整个项目的立足资产；selector 是我们在冻结骨干分布上训的 27M 头，而联合训练正要作废那个分布。两者风险不对称，就不该取相同的率。**
2. **为什么 backbone 恰好是 3e-5。** 这**精确复现 AG 的两个率**（backbone 3e-5 / selector 6e-4）。故意如此：AG 在这个率下崩了，而它已是官方值的 1/20 ⇒ 保持率不变，pilot 就成为对 §1 三项修复的干净检验。只有它仍然崩，学习率才被牵连。
3. **gamma 不给 5×。** 那个倍率的理由是"gamma 从 0 出发、必须走到 ~1 才有效果"，而我们的 gamma warm-start 在 **0.6902**。

> **规则顺序不是承重的。** `assembly.py:276-277` 在传给 optimizer 前按前缀长度**降序排序**，所以有效语义是 longest-prefix-wins。`optimizer.py` 内部的 `match()` 确实是 first-match —— 那正是那个排序存在的原因。任何直接构造 `BF16Optimizer` 的代码（包括门禁）必须复制该排序。

### 2.3 目标函数

```yaml
loss_type: dpace               # backbone
lk_loss_type: lambda
loss_decay_gamma: 7.0          # dpace 下惰性（只在 loss_type=="dflash" 分支被读）
num_anchors: 512
max_grad_norm: 1.0

dflash2_selector_weight_mode: uniform      # 本轮新增，见 §3
dflash2_selector_loss_alpha: 1.0
dflash2_selector_err_loss_alpha: 1.0
dflash2_selector_own_denominator: true
dflash2_selector_warmup_ratio: 0.0
dflash2_selector_ramp_ratio: 0.0
dflash2_selector_stop_gradient: false
dflash2_selector_target_greedy_labels: false
```

| 参数 | 为什么 |
|---|---|
| `loss_type: dpace` + `lk_loss_type: lambda` | 官方 dflash2 recipe。两者都直接面向接受长度而非代理。交互风险已证伪：`lambda` 的 acceptance 实参是 `accuracy_denom = weight_mask.sum()`，与 D-PACE 权重解耦。 |
| `selector_loss_alpha: 1.0` | 官方 dflash2 值。参考稿建议 0.5，但那是无证据的猜测；保持官方值，pilot 里可另跑 0.5 对照。 |
| `err_loss_alpha: 1.0` | 参考稿建议降到 0/0.1。不采纳：`err_w=0` 在 stage-1 已实测更差。注意 `err_head` 的收益**不来自 gate**——我们的 gate 元组是 `theta=0`，而 `sigmoid(x)>0` 恒真，所以 err_head 从未参与过滤；收益来自它作为辅助任务的正则效应。 |
| `own_denominator: true` | warm-start 权重就是在"covered slot 均值"这个分母下训出来的 |
| selector `warmup/ramp: 0/0` | dflash2 用 0.0005/0.0005（42857 步下仅 21 步，是防止 selector 在骨干随机时吃梯度的数值保护）。我们不能照抄：warmup 期 `alpha=0` ⇒ selector 冻结而 backbone 正在改，**会加剧 stage-1 warm start 的失配**。 |
| `stop_gradient: false` | 这个 flag 就是"joint"本身 |
| `target_greedy_labels: false` | 物理不可能为 true：流式 FeatureContract 不声明可选张量，`target_greedy` 到不了，开启会在 `dflash_family_model.py:783-789` 抛错 |

### 2.4 数据

| 参数 | 值 |
|---|---|
| `train_data_path` | `cache/dataset/perfectblend_qwen3-4b_regen.jsonl`（**199293** 行，794 MB） |
| `chat_template` | **`qwen-nosys`** |
| `max_length` | 3072 |

质量核实：199293 行 JSON 全有效；264530 个 assistant 轮，**含 `<think>` = 0**、空回复 = 0；一致率 **0.9876**（n=1000）。

---

## 3. 本轮唯一的代码改动：`selector_weight_mode`

### 问题

`dflash_family_model.py` 原本是：

```python
selector_loss_weights = loss_weights * target_is_candidate.float()
```

而 `loss_weights` 在任何 D-PACE `loss_type` 下都是 `weight_mask * dpace_weights`（`:687-693`）。stage-1（`scripts/train_pspr_accept_selector.py`）训的是 **covered slot 上的 uniform 均值 CE**。

⇒ **stage-1 权重在加载进 stage-2 的那一刻，它被优化的目标就变了** —— 而这恰恰是 warm start 存在的目的所要防止的唯一一件事。且 `EXPERIMENT_LOG.md:1767` 已记录 `dpace selector CE +0.286 < uniform +0.440`，差距很大不是噪声。

### 改法

新增 `selector_weight_mode: base_dpace | uniform`，默认 `base_dpace`（**保持原行为不变**）：

```python
selector_base_weights = (
    weight_mask if self.selector_weight_mode == "uniform" else loss_weights
)
selector_loss_weights = selector_base_weights * target_is_candidate.float()
```

只改 **selector** 的权重，backbone 仍是完整 D-PACE —— 这正是要一个独立开关、而不是把 `loss_type` 退回 `dflash` 的原因。

改动文件：`dflash_family_model.py`、`config/schema.py`、`algorithms/model_providers.py`、`algorithms/dflash/providers.py`。

### 门禁 `scripts/gate_pspr_selector_weight_mode.py`（4/4 PASS）

| 检查 | 结果 |
|---|---|
| W1 `base_dpace` 与改前逐元素相同（回归保护） | `\|delta\| = 0.000e+00` |
| W2 `uniform` 按 `weight_mask` 加权（独立重推，不比对实现自己的中间量） | `\|delta\| = 0.000e+00` |
| W3 **`uniform` + `own_denominator` 精确等于 stage-1 的 mean CE** | `2.75682211` vs `2.75682211`，`\|delta\| = 0.00e+00` |
| W4 两模式在 `loss_weights ≠ weight_mask` 下必须不同（防死开关） | `\|delta\| = 4.241e+00` |

**副作用（须留痕）**：`correct_num` 也用 `selector_loss_weights` 加权，所以 `train/acc` 口径从 D-PACE 加权变为 uniform。这使它更可解释（每个 covered slot 同权），但**与 AG 的 `train/acc 0.284→0.458` 不可跨轮比较**（本来也受约束 4 禁止）。

---

## 4. Pilot：先分离"数据 vs 学习率"，再决定放大

利用一个关键事实：**AG 的崩塌在 step 200 就完全可见**。所以 pilot 只需约 400 步，非常便宜。

### 判据（AG 已验证的检测器）

| 指标 | step-0 基线 | 通过条件 |
|---|---|---|
| **`oneshot`** gsm8k（纯骨干，不过 selector） | **6.040** | 下降 > 0.05 → 警告；连续两次下降 → **停** |
| **top-K recall @ frontier** | **0.9192** | 下降 > 0.02 → **停**（见下） |
| `latgate` gsm8k（过 selector + gate） | **6.813** | 应上升 |
| recover / destroy | 781 / 75（NET +706） | NET 不应变负 |

**为什么必须有 top-K recall 这一条**：`oneshot` 只看 backbone 的 **top-1**。存在一个 `oneshot` 完全看不见的失败模式——联合训练把 top-1 磨得更准，同时 **top-16 候选集塌缩**，于是 selector 没有候选可选，PSPR 的价值基础被掏空。该指标在 `decode_lattice.py` 的 frontier 处统计：那是唯一既可信（前缀正确 ⇒ `truth` 是真目标）又有信息（`i < accept` 时 `truth` 必在候选集内，恒 1）的采样点。

> **不是 oracle@16。** trace 口径的 `oracle@16 = 9.917`（macro）/ `12.809`（gsm8k）与 decode 口径的 5.7217 / 6.4255 **population 不同，不可并排比较** —— 迭代 AC 已因混用人造上限栽过一次（"吃到 51%"作废，真实 9.0%）。EVAL 口径的精确 oracle@16 需要 frontier 之后的目标续写，单次 teacher-forced pass 拿不到，仍是独立待办。

step-0 基线由 `pspr_dh2048_exact/best.pt` 实测（`--eval-reserved --max-samples 40 --gate-tau 0 --gate-rho 3 --gate-theta 0 --gate-stats`），并已验证开关 `--gate-stats` 不改变任何解码数字。

评测口径：`--eval-reserved --max-samples 40 --max-new-tokens 256 --datasets gsm8k`，gate `--gate-tau 0 --gate-rho 3 --gate-theta 0`。40 条/域**仅作 quick check**；选型与最终结果用六域全量。

### 三个 arm（自适应，预计只需跑 2 个）

| arm | `learning_rate` | scale | backbone | selector | 何时跑 |
|---|---|---|---|---|---|
| **S1** | 6e-4 | `"": 0.05` | **3e-5** | 6e-4 | **先跑。** = AG 同率 ⇒ 唯一变量是 §1 三项修复 |
| **S2** | 6e-4 | 无规则（单组） | **6e-4** | 6e-4 | S1 不崩则跑：探 backbone 学习能力上限 |
| **S3** | 1e-4 | `"": 0.5` | **5e-5** | 2e-4 | S1 崩则跑：参考稿主推的保守档 |

判读：

- **S1 不崩** ⇒ 根因确认是数据。继续 S2，因为 backbone 是 **+0.83 量级**的杠杆（Domino 从冻结 5.594 到联合 6.4255），不该无谓压低它。
- **S1 仍崩** ⇒ 学习率也是因素，走 S3，并补 2e-4 中间档。

### 全量与产出

pilot 定档后跑满 7117 步 → `export_pspr_for_decode.py` → 六域 decode（`gsm8k,math500,humaneval,mbpp,alpaca,mt-bench`）→ 对标 5.7217 与 6.4255。

---

## 5. 与参考稿 `Route_A_vs_Route_B.md` 的分歧

参考稿技术质量高，**6 条关键断言我逐一验证全部为真**：dpace 重加权 selector CE、stage-1 `lr = args.lr × √world`、`theta=0` 时 err gate 恒真、`gamma=0` 阻断 correction 梯度、两个 offline YAML 仍是 `chat_template: qwen`、以及 `optimizer.py` 内部的 first-match。

### 采纳（3）

1. `num_epochs: 1` 而非 6
2. 差分学习率（含我对自己上一轮判断的推翻）
3. `selector_weight_mode: uniform`（P0）

### 不采纳（3）

| 项 | 理由 |
|---|---|
| `err_loss_alpha` 降到 0/0.1 | "三项以 1.0 竞争"是猜测；`err_w=0` 有实测反证 |
| exact target-greedy sidecar（P1） | 一致率已 98.76%，剩余 1.24% 是 sglang batched bf16 vs HF 单条 forward 的**数值噪声**，非标签错误。改 FeatureContract + capture 通路成本高、收益近零。同理不追其建议的 ">99.5%"。 |
| 平行建立路线 B | 成本翻倍且丢掉 stage-1 的诊断基线 —— AG 能被快速定位到骨干，全靠 `oneshot` 这个分解 |

### 纠错（3）

1. **"DFlash2 的局部低秩 selector"暗示它比 PSPR 小/简单** —— 实际 DFlash2 的 vocab 级 codebook 是 **77.8M**，比 PSPR 的 **27M 大**。"低秩"指打分形式，不是规模。
2. **7143 步** → 应为 **7117**（199293 而非 200000）。
3. **"不是最长前缀匹配，具体规则必须写在通用规则前面"** —— 只看了 `optimizer.py` 内部，漏了唯一的生产调用方 `assembly.py:277` 的降序排序。**有效语义确实是 longest-prefix。** 我据此错改了 schema 注释，已回滚（见 §7）。

### 盲点（1，改变了 pilot 设计）

参考稿**没有把 off-policy 认定为 AG 的根因**，而是把降学习率/降 alpha/加保护线当主要药方 —— 那些在治症状。它也不知道 AG 的 backbone 率已是 3e-5，所以建议的 5e-5 实际是**提高**。这正是本方案把 S1 定为"AG 同率"而非"更保守"的原因。

---

## 6. 顺带修掉的两个陷阱

1. `examples/configs/offline/colocated/qwen3-4b-pspr-offline.yaml` 与 `...-joint-offline.yaml` 的 `chat_template: qwen` → `qwen-nosys`（本轮跑的是 online，但留着就是给未来的地雷）。
2. `gate_pspr_joint_equivalence.py` 只校验 offline YAML，**不覆盖实际要跑的 online YAML** —— 已知盲区，本轮用独立的 optimizer 分组核验补上（结果见 §2.2），尚未固化成脚本。

---

## 7. 本轮我自己犯并纠正的两个错误（留痕）

1. **差点把正确的 schema 注释改错。** 只读了 `optimizer.py` 的 first-match 就断定"顺序承重"，漏了 `assembly.py:277` 的降序排序。三处注释已回滚为准确的两层描述。教训：判断配置语义必须追到生产调用方，不能只看被调方。
2. **探针门禁的空洞通过。** `gamma` 零初始化会让 `scores = log_probs + 0·correction`，导致泄漏门禁 G1/G2 **在什么都没连通的情况下也 PASS**、G3 与真实断线不可区分。打开 `gamma` 后才是有效检验。

---

## 8. 启动清单（等批准）

```bash
cd /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge

# S1（先跑这个）
PYTHONPATH=/home/wangzhuoyu/sglang-patched:. nohup python -u -m specforge.cli train \
  -c examples/configs/online/disaggregated/managed-local/qwen3-4b-pspr-joint-online.yaml \
  > outputs/STAGE2_S1.log 2>&1 &
```

前置条件（**全部已满足**）：

- [x] 数据 199293 行、一致率 0.9876、`<think>` = 0
- [x] `outputs/pspr_joint_init/` 129 张量，step-0 六域 macro 精确 = 5.7217
- [x] optimizer 分组实测 = backbone 58@3e-5/wd0、selector 70@6e-4/wd2e-3、gamma 1@6e-4/wd0
- [x] `gate_pspr_selector_weight_mode.py` 4/4、`gate_pspr_joint_equivalence.py` 23/23
- [x] `--plan` 通过且不残留 `control_dir`
- [x] 8 卡空闲（旧 AG 产物已归档为 `outputs/qwen3-4b-pspr-joint-online.iter-AG-failed`，120 GB 保留作证据）
