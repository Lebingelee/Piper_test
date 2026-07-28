# CPIQL Critic 阶段总结与后续讨论交接

本文用于配合 `gpt_log/Plan/CPIQL.md` 阅读，帮助后续继续分析当前 CPIQL critic 的问题、已经完成的改造，以及下一轮值得优先讨论和验证的方向。

当前代码分支：`test_new_cpiql`

## 1. 当前任务背景

本轮工作的目标不是直接改 actor，而是优先把 CPIQL critic 改造成一个真正能够通过 `k` 表达不同分布的 value / Q family 的结构。

希望的语义是：

- `V(k=1)`：偏成功条件的 progress critic，尽量少被失败数据污染。
- `V(k=0)`：现实分布 critic，建模 success + 全部 failure 的价值期望。
- `0 < k < 1`：随着 `k` 增大，逐步剔除更旧的 failure/risk 数据，得到不同 failure mixture 下的 critic。

项目中的一个核心痛点是：原始 CPIQL 虽然把 `k` 作为输入，但单模型在训练中很可能直接忽略 `k embedding`，学习到一个 success / failure 混合后的“平均分布”，导致：

- `V(k=1)` 估计偏低；
- `V(k=0)` 与 `V(k=1)` 深度绑定；
- `critic_gap = V(k=1) - V(k=0)` 很小；
- 无法通过同一个 critic 家族稳定表达不同 `k` 下的分布差异。

---

## 2. 最初实验观察与主要痛点

### 2.1 与 ITQC 的对比现象

用户最初的实验对比对象主要包括：

- `itqc_anchor1:1`
- `cpiql_apro_0.1`
- `cpiql_1.0`

初步现象可以概括为：

1. 成功轨迹上，ITQC 的 `V(k=1)` 更容易学出平滑上升的 progress 曲线，而原 CPIQL 很难。
2. 失败轨迹上，ITQC 虽然容易把失败整体压低，但也暴露出对失败数据过敏、过度依赖 MC supervision 的问题。
3. 原 CPIQL 在单模型共享条件输入时，`V(k=0)` 和 `V(k=1)` 常常几乎不分离，表现为：
   - 成功轨迹上 `V(k=1)` 上升不理想；
   - 失败轨迹上 `V(k=0)` 也没有明显低于 `V(k=1)`；
   - 两者更像学到了一个平均分布。

### 2.2 新架构第一轮训练后的异常

在引入新 critic 架构后，用户对训练集中的失败轨迹做了一轮评估，发现：

- `V(k=1)` 的曲线比之前平稳，也更接近递增；
- 但 `V(k=0)` 与 `V(k=1)` 的输出差距依旧很小；
- 日志中 `critic_gap_mean` 很小，例如约 `0.003`；
- `loss_v_progress`（界面里显示为 `vp`）下降很快，容易让人误判模型已经把失败分布学好了。

这导致了一个关键问题：

> 为什么 `vp` 已经很低，但 `V(k=0)` 仍然没有学出明显更现实的 failure-aware 分布？

---

## 3. 对问题根因的重新定位

### 3.1 `vp` 低并不代表 failure 建模成功

当前 CPIQL 中的 `loss_v_progress` 本质上是 progress / MC anchor loss，而不是“所有样本统一的 MC loss”。

之前数据集中的关键实现为：

```python
progress_mask = np.ones(length, dtype=np.float32) if success_side else np.zeros(length, dtype=np.float32)
```

这意味着：

- success-side 样本有 progress supervision；
- failure-side 样本完全没有 progress supervision；
- failure 样本只能依赖 IQL expectile 和 Q 的 TD 自举信号间接学习。

因此：

- `vp` 下降很快，只能说明 success progress 拟合得不错；
- 它不能说明 `V(k=0)` 已经学会 failure-aware critic。

### 3.2 稀疏奖励下 failure 的监督被切断，是核心设计失配

在当前任务中，reward 基本是稀疏的：

- success 终点 reward 为 `1`
- natural failure 终点 reward 为 `0`
- 中间大部分帧 reward 都是 `0`

如果 failure 轨迹再被 `progress_mask=0` 屏蔽掉，那么 failure 与 success 的主要区别几乎只剩末端，而中间绝大多数 failure state 没有明确的逐帧标签。

结果就是：

- `V(k=1)` 会因为 success supervision 强而变平稳；
- `V(k=0)` 会因为缺少直接 failure supervision，而贴着 `V(k=1)` 微调；
- `critic_gap` 很难拉开。

这也是为什么 ITQC 更容易把 failure / success 区分开：

- ITQC 没有把 failure 的 MC 通路完全切断；
- 即使 failure 的 MC target 比较保守，它至少提供了逐帧可传播的监督。

### 3.3 联合训练集的 success / failure 严重失衡

对当前训练集做静态统计后，得到：

- expert dataset：`27980` 个 slice，全部来自 success
- replay dataset：`12163` 个 slice
  - success slice：`6373`
  - failure slice：`5790`
- 合并训练集总计：`40143` 个 slice
- 其中 failure slice 只占约 `14.4%`

这说明即使不考虑 `progress_mask` 问题，联合训练本身也会让 success 信号天然占优。

更进一步：

- `k=0` 时 active set 中包含全部 success + 全部 failure；
- `k=1` 时 active set 只剩 success；
- 从全局样本分布看，`k=0 -> k=1` 真正被拿掉的只是一小部分 failure slice；
- 如果不显式做 class balance，critic 非常容易偏向 success 主干，penalty 分支学不起来。

### 3.4 `num_k_samples=1` 会削弱“同一步内学整条 family”的效果

为了省显存，当前训练中启用了：

- `use_per_k_backward = true`
- `num_k_samples = 1`

这样每个 critic step 只随机采一个 `k` 来更新，而不是对整组 `k_grid` 同时施加约束。

这解决了显存问题，但也带来一个副作用：

- 每一步里真正参与的 failure 分布更稀疏；
- 不同 `k` 端点间的一致性与分离性更难在同一步中同时学出来。

因此，目前训练是一个“显存友好版本”，但并不完全等价于最初计划中的 full-grid family training。

---

## 4. 已经完成的核心代码改造

### 4.1 Critic 主体结构改为 success base - failure penalty

当前 critic 的核心结构改为：

```text
V(o, k) = V_success(o) - (1 - k) * P_fail(o, k)
Q(o, a, k) = Q_success(o, a) - (1 - k) * P_fail_Q(o, a, k)
```

对应文件：

- `agent_factory/modules/critics/cpiql_critic.py`
- `agent_factory/agents/mixins/critic/CPIQL.py`

这样做的核心动机是：

- 给 `k=1` 一个更强的 success-side 归纳偏置；
- 给 `V(k=1) - V(k=0)` 一个结构上可解释的来源；
- 避免简单 concat-k 网络把 `k` 当成可被忽略的弱条件。

### 4.2 数据集不再随机返回单个 `k`

原来 dataset 会随机采一个 `k` 给 critic。现在已经改为：

- dataset 负责返回 `failure_rank`
- critic 内部根据 `k_grid` 和 `failure_rank >= k` 决定当前 step 的 active distribution

这样可以让：

- `k=0` 对应 success + 全部 failure
- `k>0` 对应 success + 一部分更新的 failure
- `k=1` 对应 success-only

### 4.3 引入 failure rank 排序语义

数据集目前会优先按文件名时间戳排序 failure segment，失败时回退到 `traj_x`。

对于 failure/risk segment：

- 最老设为 `0`
- 最新设为 `1`

这使得 `failure_rank >= k` 可以近似表达：

- `k` 越高，只保留越新的 failure 数据；
- `k` 越低，保留更多历史 failure 数据。

### 4.4 用固定 `k=0` 统计做 class balance

当前已经引入固定 class balance 权重：

```text
A0 = k=0 时 active success-side slice 数
B0 = k=0 时 active failure-side slice 数

w_success = (A0 + B0) / (2 * A0)
w_failure = (A0 + B0) / (2 * B0)
```

在当前训练集统计下，实际得到：

- `k0_success_count = 34353`
- `k0_failure_count = 5790`
- `success_class_weight ≈ 0.5843`
- `failure_class_weight ≈ 3.4666`

这组权重的语义是：

- `k=0` 时 success 与 failure 的总 loss 质量接近 `1:1`
- `k>0` 时 failure 不再重新 balance，而是随着 active failure 数减少，自然衰减
- `k=1` 时 failure 全被 mask 掉，因此 `V(k=1)` 仍是 success-only

代码位置：

- `agent_factory/agents/mixins/critic/CPIQL.py`
  - `_collect_k0_balance_counts(...)`
  - `_configure_k0_balance_weights(...)`
  - `_class_balanced_weights(...)`

### 4.5 恢复 failure 的 progress / MC supervision

这是本轮最关键的一次改动。

目前数据集中的 progress 构造已改为：

```python
progress_mask = (
    np.ones(length, dtype=np.float32)
    if success_side
    else np.full(length, self.failure_progress_weight, dtype=np.float32)
)
```

并且把 `progress_weight` 的语义理顺成额外缩放项，而不是和 `progress_mask` 重复叠乘。

当前默认配置：

- `success_progress_weight = 1.0`
- `failure_progress_weight = 1.0`
- `intervention_progress_weight = 0.3`

这意味着：

- 普通 success 段的 progress supervision 开启，基线权重为 `1.0`
- 普通 failure 段的 progress supervision 现在也开启，基线权重也为 `1.0`
- intervention-boundary pseudo segment 如果启用 anchor，则额外乘 `0.3`

代码位置：

- `agent_factory/data/impl/cpiql/common.py:629-643`

### 4.6 已在 critic mixin 中补充权重语义注释

当前 `CPIQLCriticMixin` 的类注释中已经补了中英双语说明，清楚解释了：

- `distribution_mask`
- `class_balance_weight`
- `progress_mask`
- `progress_weight`

分别如何进入：

- V 的 IQL / expectile loss
- Q 的 TD loss
- V 的 progress / MC anchor loss

代码位置：

- `agent_factory/agents/mixins/critic/CPIQL.py`

### 4.7 修复 critic 扩宽后的 encoder 维度联动问题

当尝试把：

- `critic.encoder.visual.out_dim`
- `critic.encoder.out_dim`
- `critic.hidden_dims`

从 `256` 放大到 `512` 时，曾出现：

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (40x1574 and 806x256)
```

根因是：

- `VisualEncoder` 的实际输出已经变成 `512`
- 但 `BaseStateEncoder.projector` 构造时仍按旧默认 `visual_feature_dim=256` 计算融合输入维度

目前已在 `BaseStateEncoder` 中修正为：

- `visual_feature_dim` 缺省时自动从 `visual_encoder.out_dim` 推断

这解决了 critic encoder 扩宽时 projector 输入维度不同步的问题。

代码位置：

- `agent_factory/modules/encoders/state_encoder.py`

---

## 5. 当前代码语义下，不同数据类型的权重处理方式

当前 critic 训练中，总体有四层权重：

1. `distribution_mask`
   - 决定样本在当前 `k` 下是否属于 `D_k`
2. `class_balance_weight`
   - 固定按 `k=0` 的 success/failure slice 比例统计
3. `progress_mask`
   - 决定该样本参与 progress / MC supervision 的比例
4. `progress_weight`
   - 额外缩放 progress / MC supervision 的系数

最终进入 loss 的方式：

- V 的 IQL / expectile loss:
  - `distribution_mask * class_balance_weight`
- Q 的 TD loss:
  - `distribution_mask * class_balance_weight`
- V 的 progress / MC anchor loss:
  - `distribution_mask * class_balance_weight * progress_mask * progress_weight`

按当前默认配置可直观理解为：

- 普通 success 样本：
  - `progress_mask = 1.0`
  - `progress_weight = 1.0`
  - class weight = `success_class_weight`
- 普通 failure 样本：
  - `progress_mask = failure_progress_weight = 1.0`
  - `progress_weight = 1.0`
  - class weight = `failure_class_weight`
- intervention-boundary pseudo 样本：
  - 只有 `anchor_intervention_pseudo = true` 时才参与 progress anchor
  - 且 `progress_weight` 会额外乘 `intervention_progress_weight = 0.3`

---

## 6. 当前阶段已经得到的关键结论

### 6.1 新结构的方向是对的，但单靠结构不够

将 critic 改成 `success base - failure penalty` 之后，`V(k=1)` 的平滑性和单调性明显好于原始 concat-k 版本，说明：

- success 主干是有效的；
- `V_success` 的 inductive bias 是合理的。

但如果 failure 的 supervision 路径仍然太弱，那么：

- `P_fail` 分支很容易退化；
- `V(k=0)` 仍会贴着 `V(k=1)`；
- `critic_gap` 难以拉开。

### 6.2 failure progress supervision 不能被完全关掉

这是目前已经最明确的结论之一。

在稀疏奖励任务里，如果 failure-side `progress_mask=0`，那么：

- failure 几乎只靠 terminal difference 和 TD 自举学习；
- 这对逐帧 failure-aware value 建模远远不够。

因此：

- `V(k<1)` 必须显式看到 failure 的 progress / MC 通路；
- 否则它极易退化为 `V(k=1)` 的附庸。

### 6.3 `k=0` 需要被有意识地强化成“现实 critic 端点”

如果不对 `k=0` 额外强化，而只是让所有 `k` 平均训练，那么：

- success 数据的总量和 anchor 强度会主导训练；
- `V(k=0)` 不会自然成为“success + 全部 failure”的现实 critic。

因此：

- 固定按 `k=0` 统计 class weight 是合理的；
- 这比对每个 `k` 都单独重新 balance 更符合当前任务语义。

---

## 7. 当前仍未解决或仍需进一步讨论的问题

### 7.1 failure 的 progress target 现在虽然接回来了，但目标仍然过于粗糙

当前 natural failure 的 `segment_terminal_reward = 0`，因此对应的 `progress_return` 仍然会整体接近全 `0`。

这意味着：

- 现在恢复 failure progress supervision，主要目的是先把 `V(k=0)` 和 `V(k=1)` 拉开；
- 但 failure 轨迹内部的“真实进度形状”并没有被充分表达。

也就是说，这一版更像是：

- 一个合理的“gap 修复版”
- 而不是最终形态的 failure progress 建模方案

后续仍需要继续讨论：

- failure target 应该是全 0 吗？
- 还是应该表达某种“离成功距离 / 可恢复度 / 执行质量”？

### 7.2 `k=1` 是否应该包含 successful intervention segment

当前 `_is_success_side_segment(...)` 的实现会把：

- 原生 success segment
- 以及 reward > 0 的 intervention segment

都算入 success-side。

这样做的优点是：

- 专家恢复成功的片段也能为 success 侧提供正监督。

但问题是：

- 如果 `k=1` 的目标语义是“只看与当前 state 相似的成功自主轨迹，还差多少步到成功”
- 那么 successful intervention 是否应该属于同一个分布，需要进一步明确。

### 7.3 class balance 现在同时作用在 IQL / TD / progress 上，是否过强

当前 class balance 权重同时作用于：

- V 的 expectile / IQL
- Q 的 TD
- V 的 progress / MC anchor

这符合当前“先把 failure 影响提起来”的目标，但也可能带来一个问题：

- failure 权重过大时，是否会反过来削弱 `V(k=1)` 的稳定性？

后续可继续讨论：

- class balance 是否应该只主要作用于 `k=0` 端点？
- progress anchor 与 IQL / TD 是否应该使用完全相同的 class weight？

### 7.4 当前只验证 critic，actor 仍未进入第二阶段

目前改动主要集中在 critic：

- actor 的片段级采样权重
- intervention expert action 学习
- false-positive / false-negative 的 BC 权重
- policy original action 负样本建模

这些都还没有展开。

因此当前项目状态应明确理解为：

> 只完成了 critic 家族的第一阶段重构与问题重定位，actor 设计仍是后续阶段。

### 7.5 当前仍未系统回答模态依赖问题

此前实验中已经怀疑：

- 单模型可能更依赖本体感知信息；
- 深度信息或视觉语义未被充分利用；

但当前训练配置中：

- `dataset.include_depth = false`

因此关于：

- `V(k=0)` / `V(k=1)` 是否过度依赖 proprio
- 视觉与深度是否被 critic 真正用到

目前仍无定论。

---

## 8. 当前推荐的后续讨论重点

如果把这份总结和 `gpt_log/Plan/CPIQL.md` 一起交给 ChatGPT，建议重点追问下面这些问题：

### 8.1 如何为 failure 设计更合理的 progress target

当前最值得深入的问题是：

- natural failure 的 progress return 是否应一直为 0？
- 是否可以构造某种连续型 target，表达 failure 内部的真实进度结构？
- 这个 target 的语义应该更接近：
  - 离成功距离
  - 可恢复度
  - 执行质量
  - 风险感知后的现实期望

### 8.2 `k` 的中间区间应该如何被训练得更稳定

当前已实现：

- `k=0` 现实 critic
- `k=1` success-only critic

但中间 `k` 仍需要进一步思考：

- 是否需要 paired endpoint / consistency regularization？
- 是否需要更强的 monotonic / smooth family constraint？
- `num_k_samples=1` 与 full-grid 训练相比，会损失多少 family 结构？

### 8.3 success-side 的定义是否应进一步细分

需要继续讨论：

- 原生 success
- successful intervention
- intervention-boundary pseudo segment

它们是否应全部共享 success-side 语义，还是需要更细粒度区分？

### 8.4 后续 actor 应如何消费这个 critic family

一旦 critic 稳定下来，后续 actor 讨论重点应包括：

- 是否继续让 `Q(k=0)` 作为 action guidance
- 是否让 `Q(k=1)-Q(k=0)` 继续作为 DAC gating signal
- intervention expert segment 是否应该被 actor 高权重学习
- false positive / false negative 片段应如何设定 BC 权重

---

## 9. 建议后续向 ChatGPT 明确说明的当前状态

建议明确告诉 ChatGPT：

1. 当前分支已经不是原始 concat-k CPIQL，而是 `V_success - penalty` 结构。
2. 目前最重要的发现是：failure 的 MC / progress supervision 之前被错误屏蔽，是导致 `V(k=0)` 学不起来的核心原因之一。
3. 当前已恢复 failure progress supervision，并加入按 `k=0` 统计的固定 success/failure class balance。
4. 当前版本更像是“先修复 `critic_gap` 与 `V(k=0)` 附庸化”的中间版本，failure 内部的更细 progress 语义尚未完成。
5. actor 还没有进入主要改造阶段，现阶段讨论应仍以 critic 为主。

---

## 10. 当前关键文件

- `gpt_log/Plan/CPIQL.md`
- `agent_factory/agents/mixins/critic/CPIQL.py`
- `agent_factory/data/impl/cpiql/common.py`
- `agent_factory/modules/critics/cpiql_critic.py`
- `agent_factory/modules/encoders/state_encoder.py`

---

## 11. 一句话总结

当前 CPIQL critic 的核心进展，不是简单“换了个网络结构”，而是已经逐步确认：

> 问题的根源在于 failure-side 监督长期过弱，导致单模型在稀疏奖励下把 `V(k=0)` 学成了 `V(k=1)` 的附庸；本轮改造的核心，是把 `k=0` 明确强化成现实 critic 端点，并把 failure progress supervision 接回训练主通路。
