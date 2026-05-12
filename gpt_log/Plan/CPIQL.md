# Prompt 文档：Failure-Conditioned Progress IQL Critic 设计

## 背景与目标

当前任务是为 VLA 中的离线或半在线强化学习阶段设计一个只包含 critic 的改进方案。该方案基于现有 `Diffusion_IQL` 实现，但暂时不考虑 actor 如何利用 critic 更新策略。当前 actor 仍可以保持 diffusion behavior cloning loss，critic 的首要目标是学习一个更稳定、更具解释性的 **progress-aware value function** 与 **failure-conditioned action-value function**。

现有 IQL critic 的基本结构是双 Q 网络与一个 V 网络。Q 网络使用 target Q 取最小值构造 (\bar Q)，V 网络通过 expectile regression 拟合数据动作下的 Q，Q 网络再用 n-step reward 与 next V 构造 Bellman target。当前实现中的 V expectile loss 为：

[
\delta_t=\bar Q(o_t,a_t)-V_\psi(o_t)
]

[
L_V(\psi)
=========

\mathbb E
\left[
\left|
\tau-\mathbf 1(\delta_t<0)
\right|
\delta_t^2
\right]
]

其中 (\tau) 来自 `cfg.critic.expectile`。当前 Q loss 使用两个 Q 头的 MSE：

[
L_Q(\theta)
===========

\mathbb E
\left[
\left(Q_{\theta,1}(o_t,a_t)-y_t\right)^2
+
\left(Q_{\theta,2}(o_t,a_t)-y_t\right)^2
\right]
]

而当前实现的 Bellman target 写法中存在一个需要修正的问题：`ExpertDataset` 已经使用 n-step reward，但 `IQLCriticMixin.update_critic()` 仍额外乘单步 `cfg.env.gamma`，没有直接使用 batch 中的 `discount` 字段；严格地说，target 应使用 n-step effective discount，而不是固定单步 (\gamma)。

本方案的目标是把现有 IQL critic 扩展为：

[
Q_\theta(o_t,a_{t:t+H_p-1},k)
]

[
V_\psi(o_t,k)
]

其中 (k\in[0,1]) 是 **failure-ratio / success-bias condition**。当 (k=1) 时，critic 主要参考成功轨迹，学习更乐观的成功进度价值；当 (k=0) 时，critic 混入全部失败数据，学习更现实的失败感知价值；中间 (k) 表示不同失败数据混入比例下的条件价值估计。该 critic 不再是单一 MDP 下的唯一价值函数，而是一个 **failure-conditioned progress-aware value family**。

---

## 核心符号定义

记 (o_t) 为当前观测历史，实际由 `obs_horizon` 长度的多模态观测组成。当前项目中的状态编码器会处理多相机 RGB、可选 depth 与 proprio/state，并输出 `[B, obs_horizon, encoder_out_dim]`，Q/V 网络再把该 embedding flatten 后输入 MLP。

记：

[
a_t^c = a_{t:t+H_p-1}
]

为当前样本中的动作 chunk，其中 (H_p=\text{pred_horizon})。为了简洁，后续公式仍写作 (a_t)，但必须明确：这里的 (a_t) 实际表示动作 chunk，而不是单步动作。

记：

[
n=\text{act_horizon}
]

为环境实际推进或 next observation 偏移的步数。当前数据集样本中的 `next_observations` 来自 `start + act_horizon`，reward 是从当前步开始的 n-step 折扣奖励。

记：

[
k\in\mathcal K={0,0.2,0.4,0.6,0.8,1.0}
]

为条件变量。推荐语义为 **success-bias coefficient**，而不是简单把它解释成“失败数据比例”。训练数据分布可写为：

[
D_k = D_{\text{success}} \cup \rho(k)D_{\text{failure}}
]

其中：

[
\rho(k)=1-k
]

当 (k=1) 时，只使用成功数据或主要使用成功数据；当 (k=0) 时，使用成功数据并混入全部失败数据；当 (k=0.6) 时，混入约 (40%) 的失败数据。若失败数据存在明确的策略迭代时间顺序，优先保留较新的失败数据，因为较新的失败更接近当前 policy 的错误分布；若失败数据来自不同任务或不同采集策略，则不能简单按时间排序，需要额外记录失败类型与任务来源。

---

## Critic 网络结构设计

现有 `IQLQNet` 的输入是观测 embedding flatten 后与动作 chunk flatten 后拼接，输出两个标量 Q；`IQLVNet` 的输入是观测 embedding flatten，输出一个标量 V。 本方案在该结构上增加 (k) 条件输入，同时保证Q的多头输出设置不变（即依旧输出两个Q，取最小值）。

Q 网络应改为：

[
(Q_{\theta,1},Q_{\theta,2})
===========================

Q_\theta(e(o_t),a_t,k)
]

V 网络应改为：

[
V_\psi(o_t,k)=V_\psi(e(o_t),k)
]

其中 (e(o_t)) 是 critic encoder 输出并 flatten 后的观测 embedding。(k) 可以通过以下方式进入网络：最简单的版本是把标量 (k) 直接拼接到 flatten 后的 observation embedding 与 action chunk embedding 上；更稳的版本是对 (k) 做一层 MLP embedding，再拼接到 Q/V 输入中；如果后续希望 (k) 表达连续风险偏置，可使用 Fourier feature 或 learned embedding，但第一版不需要复杂化。

Q 网络输入形式为：

[
x_Q = \text{concat}\left(e(o_t),\text{flatten}(a_t),\phi_k(k)\right)
]

V 网络输入形式为：

[
x_V = \text{concat}\left(e(o_t),\phi_k(k)\right)
]

这里 (\phi_k(k)) 可以是 identity，也可以是小型 MLP。第一版推荐使用 MLP embedding：

[
\phi_k(k)=\text{MLP}_k(k)
]

因为这能给网络更多容量学习不同 (k) 下的价值分布，但仍保持结构简单。

---

## 动态 (\gamma) 与 progress target 设计原则

当前任务主要是稀疏奖励任务，通常只有最终状态具有显式 reward 标签。普通固定折扣 (\gamma) 会导致成功轨迹上的 return 呈指数形状：

[
G_t=\gamma^{H-t}
]

这与机器人任务中的“进度”直觉不完全一致。我们希望成功轨迹上的价值随时间推进呈现近似线性增长，即越接近成功，value 越接近 (1)；越早的状态 value 越低，但仍反映它到成功的剩余步数。

设 (K) 为任务最大允许步长，(H\le K) 为当前成功 segment 的长度，(t\in[0,H]) 为 segment 内的时间索引。定义动态折扣：

[
\gamma_t^{\text{prog}}
======================

\max\left{
\frac{K-H+t}{K-H+t+1},
\epsilon_\gamma
\right}
]

如果暂时忽略 (\epsilon_\gamma)，则从 (t) 到成功终点的 Monte Carlo return 为：

[
G_t
===

# \prod_{i=t}^{H-1}\gamma_i^{\text{prog}}

\prod_{i=t}^{H-1}
\frac{K-H+i}{K-H+i+1}
=====================

# \frac{K-H+t}{K}

1-\frac{H-t}{K}
]

因此该设计能让稀疏终点成功 reward (1) 沿轨迹反向传播时形成精确线性 progress target。它不是额外手写一个与 RL return 不一致的监督信号，而是通过 time-varying discount 在 Monte Carlo return 框架内生成进度形状。

(\epsilon_\gamma) 是数值稳定项。若 (H=K,t=0)，原始分式为 (0)，可能导致 return 完全坍为 (0) 或影响后续归一化。加入 (\epsilon_\gamma) 后严格 telescoping 会被轻微破坏，但只影响极长轨迹的早期状态。推荐 (\epsilon_\gamma\in[10^{-4},10^{-3}])，并在实现中明确命名为 `gamma_floor` 或 `epsilon_gamma`。

动态 (\gamma) 带来的额外要求是：数据集中不能只返回 n-step reward，还必须返回对应的 n-step effective discount。也就是说，对任意样本，需要返回：

[
\Gamma_{t:t+n}^{\text{prog}}
============================

\prod_{i=t}^{t+n-1}\gamma_i^{\text{prog}}
]

而不是使用固定 (\gamma^n)，更不能像当前实现那样在已经计算 n-step reward 后再额外乘一次单步 (\gamma)。当前 `IQL_summary.md` 已明确指出，现有实现存在未使用 batch `discount` 字段的问题；本方案必须在 `compute_n_step_signals()` 或新的 reward 工具函数中返回动态 effective discount，并在 `IQLCriticMixin.update_critic()` 中直接使用该字段。

---

## 奖励函数与轨迹标签设计原则

本方案默认使用稀疏显式奖励。自然成功终点给：

[
R_T=1
]

自然失败终点给：

[
R_T=0
]

若存在安全事故，例如碰撞、损坏、越界、危险接触，可以单独引入负 reward，但这会把 value scale 从 ([0,1]) 扩展到包含负数。第一版优先不引入负 reward，而是通过 `failure_type` 或额外 safety label 保留安全信息。

专家演示成功轨迹直接作为成功 segment。其终点 reward 为 (1)，中间 reward 为 (0)，使用动态 (\gamma_t^{\text{prog}}) 构造 progress return：

[
G_t^{\text{success}}
====================

\prod_{i=t}^{H-1}\gamma_i^{\text{prog}}
\approx
1-\frac{H-t}{K}
]

模型早期探索失败轨迹作为失败 segment。其终点 reward 为 (0)，中间 reward 为 (0)。它主要用于训练低 (k) critic，即 (V(o,k\approx0)) 与 (Q(o,a,k\approx0))，帮助现实 critic 学习失败区域的价值下降。

专家介入轨迹需要特殊处理。若一条轨迹由模型自主执行到专家介入点，再由专家修正，之后放权给模型并最终成功，则可以分为两个版本。严格版把专家介入点作为 **intervention truncation boundary**，不把它当作自然终点，不给显式 reward，而是在 Q target 中使用：

[
V(o_{\text{intervene}},k=0)
]

作为现实价值 bootstrap target。工程部署版可以将该复杂轨迹拆成两条独立 segment：轨迹一为模型自主执行到专家介入点，末端给一个伪奖励 (R_1\in(0,1))；轨迹二为专家修正放权后模型自主完成任务，末端给 (R_2=1)。

严格版更符合 Bellman 语义；工程版实现简单，更适合优先部署。工程版中 (R_1) 不是环境 reward，而是将介入点 bootstrap value 固化成 pseudo-terminal reward。推荐：

[
R_1
===

\text{clip}
\left(
\lambda_I \cdot \text{stopgrad}\left[
V_{\text{target}}(o_{\text{intervene}},0)
\right],
R_{\min},
R_{\max}^{I}
\right)
]

其中：

[
0<R_{\min}<R_{\max}^{I}<1
]

推荐第一版取 (R_{\min}=0.05)，(R_{\max}^{I}=0.6)，(\lambda_I\in[0.5,0.8])。这样可以鼓励模型探索到可恢复边界，但不会把“触发专家介入”错误奖励成接近成功。

若 critic 尚未具备初步判别能力，早期可以用固定伪奖励：

[
R_1=0.2
]

待 (V(o,0)) 更稳定后再切换到 target value 版本。

对于专家介入后仍失败的轨迹，需要区分自然失败与人工截断。若明确自然失败，终点 reward 为 (0)，bootstrap 关闭。若只是超时、人工停止、采集截断，并没有明确失败，则不应强行置为失败，而应作为 truncation boundary，使用：

[
V(o_{\text{end}},0)
]

作为现实 bootstrap value。

---

## 严格版 Bellman target 设计

对任意样本，critic 使用统一 target：

[
y_t(k)
======

r_t^{(n)}
+
\Gamma_{t:t+n}
\cdot V_{\text{boot}}(o_{t+n},k)
]

其中 (r_t^{(n)}) 是使用动态 (\gamma) 计算出的 n-step reward：

[
r_t^{(n)}
=========

\sum_{j=0}^{n-1}
\left(
\prod_{i=t}^{t+j-1}\gamma_i
\right)
r_{t+j}
]

(\Gamma_{t:t+n}) 是 n-step effective discount：

[
\Gamma_{t:t+n}
==============

\prod_{i=t}^{t+n-1}\gamma_i
]

(V_{\text{boot}}) 根据 boundary type 决定。如果 (o_{t+n}) 是普通非终止状态，则：

[
V_{\text{boot}}(o_{t+n},k)=V_\psi(o_{t+n},k)
]

如果 (o_{t+n}) 是自然成功或自然失败终点，则：

[
V_{\text{boot}}(o_{t+n},k)=0
]

因为终点显式 reward 已经进入 (r_t^{(n)})，自然终止后不再 bootstrap。

如果 (o_{t+n}) 是专家介入边界，则：

[
V_{\text{boot}}(o_{t+n},k)=V_\psi(o_{t+n},0)
]

这里必须使用 (k=0)，因为专家介入表示模型进入风险区域，需要用 failure-aware realistic value 评价该状态，避免用乐观 value 过度奖励模型触发专家接管。

如果 (o_{t+n}) 是普通数据截断或非风险 handover，则可以使用：

[
V_{\text{boot}}(o_{t+n},k)=V_\psi(o_{t+n},k)
]

如果是 risk truncation 或人工中止但没有明确成功失败，推荐使用：

[
V_{\text{boot}}(o_{t+n},k)=V_\psi(o_{t+n},0)
]

因此数据中必须新增或显式维护：

[
\texttt{boundary_type}
\in
{
\text{normal},
\text{success},
\text{failure},
\text{intervention},
\text{handover},
\text{timeout},
\text{risk_truncation}
}
]

并区分：

[
\texttt{terminated}
]

与：

[
\texttt{truncated}
]

专家介入不应被写成自然 `terminated=True`。它应是 `truncated=True` 且 `boundary_type="intervention"`，并在 target 中使用 (V(o,0)) bootstrap。

---

## 工程部署版 target 设计

若为了优先部署，不希望立刻修改 `boundary_type` 与复杂 bootstrap 逻辑，可以将专家介入后成功的轨迹拆成两个普通 segment。

第一个 segment 是模型自主执行但最终触发专家介入：

[
\tau_1=(o_0^M,a_0^M,\ldots,o_{\tau}^{I})
]

它被当作 pseudo-terminal trajectory，中间 reward 为 (0)，末端 reward 为：

[
R_1
===

\text{clip}
\left(
\lambda_I V_{\text{target}}(o_{\tau}^{I},0),
R_{\min},
R_{\max}^{I}
\right)
]

该 segment 的 return target 为：

[
G_t^{(1)}
=========

R_1
\prod_{i=t}^{H_1-1}\gamma_i^{(1)}
]

若不触发 (\epsilon_\gamma)，则：

[
G_t^{(1)}
=========

R_1
\cdot
\frac{K-H_1+t}{K}
]

第二个 segment 是专家修正并放权后模型自主完成任务：

[
\tau_2=(o_{\text{release}},a_{\text{release}}^M,\ldots,o_T)
]

它被当作成功 trajectory，中间 reward 为 (0)，末端 reward 为：

[
R_2=1
]

对应 return target 为：

[
G_t^{(2)}
=========

\prod_{i=t}^{H_2-1}\gamma_i^{(2)}
\approx
1-\frac{H_2-t}{K}
]

在部署版中，两个 segment 的 Q target 都可以使用普通 terminal trajectory 逻辑。区别只在 terminal reward scale：轨迹一是 (R_1<1)，轨迹二是 (R_2=1)。

为了避免模型学会“快速触发专家介入”，轨迹一应主要进入低 (k) 训练分布，或者降低其 progress anchor 权重。推荐：

[
\alpha_1 < \alpha_{\text{success}}
]

例如：

[
\alpha_1\in[0.2,0.5]\alpha_{\text{success}}
]

而轨迹二可以按成功轨迹处理，并训练多个 (k)，因为它表示专家修正后模型确实完成了任务。

---

## V 损失函数设计

新的 V 网络为：

[
V_\psi(o_t,k)
]

其主体仍然继承 IQL expectile regression。定义 target Q：

[
\bar Q(o_t,a_t,k)
=================

\min
\left(
Q_{\bar\theta,1}(o_t,a_t,k),
Q_{\bar\theta,2}(o_t,a_t,k)
\right)
]

其中 (\bar\theta) 是 target Q 网络参数。定义：

[
\delta_t(k)
===========

\bar Q(o_t,a_t,k)-V_\psi(o_t,k)
]

expectile loss 为：

[
L_V^{\text{IQL}}(\psi)
======================

\mathbb E_{(o_t,a_t,k)\sim D_k}
\left[
\rho_\tau(\delta_t(k))
\right]
]

其中：

[
\rho_\tau(u)
============

\left|
\tau-\mathbf 1(u<0)
\right|u^2
]

为了缓解 bootstrap 产生的 value drift，同时让 V 学到可解释的任务进度，需要加入 progress anchor。对于成功 segment，设其 progress return 为：

[
G_t^{\text{prog}}
=================

R_{\text{terminal}}
\prod_{i=t}^{H-1}\gamma_i^{\text{prog}}
]

其中专家演示成功或模型最终自主成功时：

[
R_{\text{terminal}}=1
]

工程部署版的介入前模型 segment 中：

[
R_{\text{terminal}}=R_1\in(0,1)
]

失败 segment 若自然失败，则：

[
R_{\text{terminal}}=0
]

但失败 segment 默认不建议使用强 progress anchor，因为失败状态不一定表示低进度，尤其 near-miss failure 可能已经非常接近成功。失败数据主要通过 Q Bellman loss 和 (k=0) 条件进入 V 的 expectile 回归。

progress anchor loss 为：

[
L_V^{\text{prog}}(\psi)
=======================

\mathbb E
\left[
m_{\text{prog}}(o_t)
\left(
V_\psi(o_t,k)-G_t^{\text{prog}}
\right)^2
\right]
]

其中 (m_{\text{prog}}) 是 mask。对专家成功轨迹、介入后成功 recovery 轨迹、部署版轨迹二，应设为 (1)。对自然失败轨迹，默认设为 (0)。对部署版轨迹一可以设为 (1)，但使用较小权重 (\alpha_1)。对严格版介入前模型段，默认不加 progress anchor，避免把专家后续成功泄漏到介入前模型动作。

完整 V loss 为：

[
L_V(\psi)
=========

L_V^{\text{IQL}}(\psi)
+
\alpha_{\text{prog}}
L_V^{\text{prog}}(\psi)
+
\lambda_{\text{mono}}L_V^{\text{mono}}(\psi)
+
\lambda_{\text{smooth}}L_V^{\text{smooth}}(\psi)
]

其中 (\alpha_{\text{prog}}) 可以根据 segment type 设置不同权重。成功专家轨迹使用 (\alpha_{\text{success}})，介入前 pseudo segment 使用 (\alpha_1)，失败轨迹默认 (\alpha=0)。

---

## Q 损失函数设计

新的 Q 网络为 twin Q：

[
Q_{\theta,1}(o_t,a_t,k),\quad Q_{\theta,2}(o_t,a_t,k)
]

严格版 Bellman target 为：

[
y_t(k)
======

r_t^{(n)}
+
\Gamma_{t:t+n}
V_{\text{boot}}(o_{t+n},k)
]

对应 Q loss 为：

[
L_Q(\theta)
===========

\mathbb E_{(o_t,a_t,k)\sim D_k}
\left[
\left(Q_{\theta,1}(o_t,a_t,k)-y_t(k)\right)^2
+
\left(Q_{\theta,2}(o_t,a_t,k)-y_t(k)\right)^2
\right]
]

这里 (y_t(k)) 需要 stop-gradient：

[
y_t(k)=\text{stopgrad}(y_t(k))
]

在实现中，target Q 用于 V expectile 的 (\bar Q)，Q Bellman target 使用 current 或 target V 需要谨慎。第一版可沿用当前 IQL 实现，使用当前 V 构造 bootstrap，但建议在 target 中 detach V 输出，防止 Q loss 反向更新 V。即：

[
V_{\text{boot}}=\text{stopgrad}\left[V_\psi(o_{t+n},k_{\text{boot}})\right]
]

其中 (k_{\text{boot}}) 由 boundary type 决定。普通 next state 使用 (k)，专家介入边界使用 (0)。

---

## k 单调性与平滑约束

由于 (k) 的目标语义是从现实失败感知估计逐步过渡到乐观成功估计，理论上应满足：

[
k_i>k_j
\Rightarrow
V(o,k_i)\ge V(o,k_j)
]

以及：

[
k_i>k_j
\Rightarrow
Q(o,a,k_i)\ge Q(o,a,k_j)
]

神经网络不会自动满足该性质。因此建议加入 monotonic regularization。对 V：

[
L_V^{\text{mono}}
=================

\mathbb E_{o,k_i>k_j}
\left[
\max
\left(
0,
V_\psi(o,k_j)-V_\psi(o,k_i)
\right)^2
\right]
]

对 Q：

[
L_Q^{\text{mono}}
=================

\mathbb E_{o,a,k_i>k_j}
\left[
\max
\left(
0,
Q_\theta(o,a,k_j)-Q_\theta(o,a,k_i)
\right)^2
\right]
]

第一版可以只对 V 加 monotonic regularization，因为 (V(o,1)-V(o,0)) 后续会作为专家介入或风险边界指标。若 Q 的训练不稳定，再扩展到 Q。

还可以加入 k-smoothness，让 (V(o,k)) 随 k 连续变化，避免不同 k 之间出现剧烈振荡：

[
L_V^{\text{smooth}}
===================

\mathbb E_{o,k_i,k_{i+1}}
\left[
\left(
V_\psi(o,k_{i+1})-V_\psi(o,k_i)
\right)^2
\right]
]

该项不应过强，否则会压制不同 k 之间的风险差异。推荐 (\lambda_{\text{smooth}}) 小于 (\lambda_{\text{mono}})。

---

## Critic gap 与专家介入指标

训练完成或训练过程中，可以定义：

[
\Delta V(o)
===========

V_\psi(o,1)-V_\psi(o,0)
]

该量不是严格 epistemic uncertainty，而是 **success-failure value discrepancy**。它表示同一状态在成功偏置 critic 与失败感知 critic 下的价值差异。

若：

[
V(o,1)\text{ 高},\quad V(o,0)\text{ 低}
]

则说明该状态在成功轨迹中像是有希望的进度状态，但在混入失败数据后现实价值下降。这通常意味着它位于成功与失败边界附近，可能是专家介入、数据采样、失败恢复训练的关键区域。

推荐专家介入触发条件不是只看 (\Delta V)，而是：

[
\Delta V(o)>\eta_\Delta
]

同时：

[
V(o,1)>\eta_{\text{hope}}
]

并且：

[
V(o,0)<\eta_{\text{risk}}
]

这样可以避免在完全无希望的状态上浪费专家介入，也避免在稳定成功状态上过度干预。

在本阶段，(\Delta V) 只作为 critic 诊断和未来数据采集指标，不设计 actor 优化。后续如果设计 actor，可以再考虑基于 (\Delta V) 调节 guidance 或 relabel 逻辑。

---

## 训练数据与模块放置建议

根据 `IQL_summary.md`，当前数据相关逻辑集中在 `agent_factory/data/impl/expert_dataset.py`、`agent_factory/data/impl/replaybuffer.py` 与 `agent_factory/data/utils.py`。其中 `ExpertDataset` 读取 H5，并使用 `compute_rl_signals()` 与 `compute_n_step_signals()` 生成 reward、value、n-step reward；ReplayBuffer 提供接近的数据契约。

本方案中，动态 (\gamma)、progress return、segment terminal reward、effective discount、boundary type 处理都应优先放在 `agent_factory/data/utils.py` 与 dataset/replaybuffer 的切片逻辑中，而不是写死在 critic 里。原因是 critic 应只消费已经构造好的训练字段：

[
\texttt{reward},\quad
\texttt{discount},\quad
\texttt{progress_return},\quad
\texttt{terminated},\quad
\texttt{truncated},\quad
\texttt{boundary_type},\quad
\texttt{k}
]

`compute_rl_signals()` 应扩展为支持 sparse success/failure reward、动态 (\gamma)、segment-level terminal reward。`compute_n_step_signals()` 应扩展为返回动态 n-step reward 与 dynamic effective discount：

[
r_t^{(n)},\quad \Gamma_{t:t+n}
]

同时需要考虑 intervention boundary。如果 n-step 跨过 intervention boundary，应截断到 intervention state，并返回 `boundary_type="intervention"`，而不是继续跨过专家介入段传播成功 reward。

`ExpertDataset.__getitem__()` 当前会返回 observations、next_observations、action、reward、terminated、cond。 本方案需要额外返回 `discount`、`progress_return`、`k`、`boundary_type`、`truncated`、`segment_type`。如果优先部署版不支持严格 boundary bootstrap，也至少需要返回 `terminal_reward` 或在预处理阶段把专家介入轨迹切成普通 segment，并把 (R_1) 写入末端 reward。

根据 `IQL_summary.md`，critic 的核心更新逻辑在 `agent_factory/agents/mixins/critic/IQL.py`，网络结构在 `agent_factory/modules/critics/iql_critic.py`。 因此 (Q(o,a,k))、(V(o,k))、V progress anchor、monotonic loss、Q target 使用 batch `discount`、intervention bootstrap 的逻辑应放在 critic mixin 与 critic module 中。

`agent_factory/modules/critics/iql_critic.py` 应修改 `IQLQNet` 与 `IQLVNet` 的输入维度，让它们接收 (k)。`agent_factory/agents/mixins/critic/IQL.py` 的 `update_critic(batch)` 应从 batch 中读取 `k`、`discount`、`progress_return`、`boundary_type`、`truncated`，并构造新的 V loss 与 Q loss。`soft_update_target()` 可以沿用现有逻辑，因为 target Q 软更新仍然适用。当前 target Q 软更新使用：

[
\bar\theta \leftarrow \alpha\theta+(1-\alpha)\bar\theta
]

其中 (\alpha=\text{cfg.soft_update_tau})。

`agent_factory/agents/impl/diffusion_iql.py` 主要是算法组装入口，当前 `_init_components()` 先构建 critic，再构建 actor，`update()` 顺序调用 `update_critic()`、`update_actor()`、`soft_update_target()`。 本阶段只需保证新的 critic config 能被注册与实例化，不需要改 actor 训练逻辑。

---

## 推荐训练流程

训练前先对原始 H5 或 replay 数据做 segment 标注。每条数据需要标记其来源：专家演示成功、模型自主失败、模型自主到专家介入、专家介入后成功、专家介入后失败、专家修正放权后模型成功。若采用严格版，需要记录 intervention start、intervention end、handover、success、failure、timeout 等边界。若采用工程部署版，则将专家介入后成功轨迹拆成两个普通 segment：模型自主到介入点的 (R_1) pseudo-terminal segment，以及专家修正放权后模型完成任务的 (R_2=1) success segment。

随后对每个 segment 计算动态 (\gamma_t^{\text{prog}})、n-step reward、effective discount、progress return。成功 segment 的 terminal reward 为 (1)，普通失败 segment 为 (0)，部署版介入前 segment 为 (R_1)。严格版介入前 segment 不写 terminal reward，而是保留 intervention boundary，让 Q target 使用 (V(o_{\text{intervene}},0)) bootstrap。

训练 batch 时，为每个样本采样或指定 (k)。专家成功数据可以用于多个 (k)，但更强调高 (k)。模型失败数据主要用于低 (k)。介入后成功 recovery 数据应进入多个 (k)，尤其应进入低 (k)，因为它能告诉 (V(o,0)) 某些风险状态可恢复，避免现实 critic 过度悲观。部署版介入前 (R_1) segment 主要进入低 (k)，或者降低其 progress anchor 权重。

每次 critic update 中，先预处理 observations 与 next_observations，经过 critic encoder 得到 embedding。接着使用 target Q 计算：

[
\bar Q(o_t,a_t,k)
=================

\min(Q_{\bar\theta,1},Q_{\bar\theta,2})
]

并计算 V expectile loss。然后根据 batch 中的 `progress_return` 与 mask 计算 progress anchor loss。随后根据 `reward`、`discount`、`terminated`、`truncated`、`boundary_type` 构造 Q target，并计算 twin Q MSE loss。最后加入 (k)-monotonic 与 smoothness regularization，更新 V optimizer 与 Q optimizer。

训练循环中仍保留 target Q soft update。是否调用 actor update 由外部流程决定，但本阶段不设计 actor 如何基于 critic 改进策略。当前 `DiffusionIQLAgent` 本来就没有完整训练循环，常规调用是外部训练循环分别调用 `update_critic()`、`update_actor()`，或调用 `update()` 组合更新。 因此可以先实现单独 critic update 与 critic-only training mode，待 critic 稳定后再接入 actor。

---

## 最终 Critic 总损失

推荐最终 critic loss 写为：

[
L_{\text{critic}}
=================

L_Q
+
\lambda_V L_V^{\text{IQL}}
+
\alpha_{\text{prog}}L_V^{\text{prog}}
+
\lambda_{\text{mono}}L_{\text{mono}}
+
\lambda_{\text{smooth}}L_{\text{smooth}}
]

其中：

[
L_Q
===

\mathbb E
\left[
\left(Q_{\theta,1}(o_t,a_t,k)-y_t(k)\right)^2
+
\left(Q_{\theta,2}(o_t,a_t,k)-y_t(k)\right)^2
\right]
]

[
y_t(k)
======

r_t^{(n)}
+
\Gamma_{t:t+n}
V_{\text{boot}}(o_{t+n},k)
]

[
L_V^{\text{IQL}}
================

\mathbb E
\left[
\rho_\tau
\left(
\bar Q(o_t,a_t,k)-V_\psi(o_t,k)
\right)
\right]
]

[
L_V^{\text{prog}}
=================

\mathbb E
\left[
m_{\text{prog}}
\left(
V_\psi(o_t,k)-G_t^{\text{prog}}
\right)^2
\right]
]

[
L_{\text{mono}}
===============

\mathbb E_{k_i>k_j}
\left[
\max
\left(
0,
V_\psi(o,k_j)-V_\psi(o,k_i)
\right)^2
\right]
]

[
L_{\text{smooth}}
=================

\mathbb E
\left[
\left(
V_\psi(o,k_{i+1})-V_\psi(o,k_i)
\right)^2
\right]
]

若后续发现 Q 的 (k) 条件也不稳定，可补充 Q monotonic loss：

[
L_Q^{\text{mono}}
=================

\mathbb E_{k_i>k_j}
\left[
\max
\left(
0,
Q_\theta(o,a,k_j)-Q_\theta(o,a,k_i)
\right)^2
\right]
]

但第一版建议先只对 V 施加 monotonic regularization，以免 Q 过度受约束。

---

## 需要优先实现的最小版本

最小可部署版本不做严格 intervention bootstrap，而是先采用工程切段策略。先改 `IQLQNet/IQLVNet`，让它们接收 (k)。然后改 dataset 预处理，让每个 segment 具备 terminal reward、progress return、dynamic discount。专家成功轨迹终点 (1)，模型失败轨迹终点 (0)，专家介入后成功轨迹拆成 (R_1) pseudo segment 与 (R_2=1) success segment。(R_1) 先用固定 (0.2)，等 critic 初步稳定后改为：

[
R_1
===

\text{clip}
\left(
\lambda_I V_{\text{target}}(o_{\text{intervene}},0),
0.05,
0.6
\right)
]

接着修改 `update_critic()`，确保 Q target 使用 batch 中的 `discount`，而不是固定 `cfg.env.gamma`。这一步是必须优先修复的，因为当前 `IQL_summary.md` 已指出现有实现没有严格使用 n-step effective discount。

最后加入 V progress anchor 与 V monotonic loss。actor 保持不动。critic 训练稳定后，再考虑 `relabel_data()` 如何使用：

[
A_k(o,a)=Q(o,a,k)-V(o,k)
]

以及如何将 critic gap：

[
\Delta V(o)=V(o,1)-V(o,0)
]

用于后续专家介入、数据采样或 actor guidance。当前阶段只保留这些量作为日志与诊断指标。

---

## 最终方案命名

该 critic 可命名为：

**Failure-Conditioned Progress IQL Critic**

更完整的表述是：

**A failure-conditioned, progress-anchored IQL critic for intervention-aware VLA reinforcement learning**

它的核心贡献是：用动态 (\gamma) 从稀疏成功 reward 中生成线性 progress return；用 (k)-conditioned critic 同时学习乐观成功价值与现实失败感知价值；用专家介入边界或其工程近似伪奖励，避免把专家后续成功错误归因给模型介入前动作；并用 (V(o,1)-V(o,0)) 作为未来专家介入和风险边界识别的 critic-side 诊断信号。
