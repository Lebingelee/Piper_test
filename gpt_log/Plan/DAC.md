# Prompt 文档：Stage-wise Risk-Scaled CPIQL-DAC Actor 优化方法

## 方法目标

当前算法库中已经实现了 **CPIQL critic**，即：

[
Q_\phi(o,a,k),\qquad V_\psi(o,k)
]

其中 (o) 表示 VLA 的观测输入，(a) 表示 action chunk，(k\in[0,1]) 表示 failure-conditioned success bias。当前阶段不再重新设计 critic，也不重新讨论动态 (\gamma)、progress return、专家介入奖励等细节。这里只使用 CPIQL 已经训练好的 critic 来优化 diffusion actor。

本方法的目标是构建一个 **CPIQL-DAC actor update**：先冻结已经训练好的 CPIQL critic，再用 (Q(o,a,k=0)) 的 action-gradient 引导 diffusion actor 的去噪训练，从而在保持 behavior cloning 稳定性的同时，让 actor 朝 failure-aware high-value action 分布偏移。

方法继承 DAC 的基本思想：把 behavior-regularized policy extraction 写成 diffusion noise regression。但和原始 DAC 不同，我们不采用 actor-critic mini-batch 交替更新，而采用更适合真实机器人部署的 **stage-wise training**：

[
\text{Train CPIQL Critic}
\Longrightarrow
\text{Freeze CPIQL Critic}
\Longrightarrow
\text{Train Q-guided Diffusion Actor}
\Longrightarrow
\text{Collect New Data}
\Longrightarrow
\text{Refresh Critic}
\Longrightarrow
\text{Refresh Actor}
]

这里的关键是：**CPIQL critic 基于 IQL 训练，训练期间不依赖 actor 采样动作**。因此 critic 可以先利用大规模离线轨迹、失败轨迹和专家介入轨迹快速训练到稳定状态。actor 只在 critic 稳定后被引入，作为一个受 frozen critic 引导的策略提取模块。

---

## 核心思想

已训练好的 CPIQL critic 提供两个重要价值估计：

[
Q_{\bar\phi}(o,a,0)
]

和：

[
Q_{\bar\phi}(o,a,1)
]

其中 (\bar\phi) 表示 frozen critic 或 EMA critic 参数。(Q(o,a,0)) 吸收全部成功与失败数据，是 **failure-aware realistic Q**；(Q(o,a,1)) 更偏向成功轨迹，是 **success-biased optimistic Q**。

actor 更新时，只使用：

[
\nabla_a Q_{\bar\phi}(o,a,0)
]

作为真正的 Q-gradient guidance。也就是说，actor 的改进方向始终来自现实失败感知 critic，而不是乐观 critic。这是为了满足真实机器人部署中的保守性要求。

同时，用：

[
Q_{\bar\phi}(o,a,1)-Q_{\bar\phi}(o,a,0)
]

衡量成功偏置 critic 与失败感知 critic 的价值分歧。若二者差异大，说明该状态-动作附近可能存在失败边界，此时 actor 应更接近 behavior cloning；若二者差异小，说明乐观 critic 和现实 critic 判断一致，此时可以允许更强的 Q-gradient guidance。

因此，本方法的策略提取目标可以抽象为：

[
\pi^*(a|o)
\propto
\pi_\beta(a|o)
\exp
\left(
\frac{Q_{\bar\phi}(o,a,0)}
{\eta_{\mathrm{eff}}(o,a)}
\right)
]

其中 (\pi_\beta(a|o)) 是 diffusion BC actor 所表示的行为分布，(\eta_{\mathrm{eff}}) 是由 CPIQL critic 风险差异动态缩放后的 behavior regularization 系数。

---

## Actor 输入与 diffusion 加噪过程

设数据集中采样得到观测与 action chunk：

[
(o,a)\sim\mathcal D
]

其中：

[
a=a_{t:t+H_p-1}\in\mathbb R^{H_p\times d_a}
]

表示 action chunk，而不是单步 action。采样扩散时间与高斯噪声：

[
t\sim p(t),\qquad \epsilon\sim\mathcal N(0,I)
]

按照 diffusion forward process 构造 noisy action：

[
x_t
===

\sqrt{\bar\alpha_t}a
+
\sqrt{1-\bar\alpha_t}\epsilon
]

diffusion actor 是一个 noise predictor：

[
\epsilon_\theta(x_t,o,t)
]

普通 behavior cloning diffusion loss 为：

[
L_{\mathrm{BC}}(\theta)
=======================

\mathbb E
\left[
\left|
\epsilon_\theta(x_t,o,t)-\epsilon
\right|_2^2
\right]
]

CPIQL-DAC 的作用是在该 loss 的 noise target 中加入基于 (Q(o,a,0)) 的 gradient guidance。

---

## Failure-aware Q-gradient

在 actor 更新阶段，CPIQL critic 冻结。使用 frozen critic 计算 noisy action 上的 Q-gradient：

[
g_0(o,x_t)
==========

\nabla_{x_t}
Q_{\bar\phi}(o,x_t,0)
]

这里 (x_t) 是 noisy action chunk，因此：

[
g_0(o,x_t)\in\mathbb R^{H_p\times d_a}
]

该梯度表示在当前 noisy action chunk 附近，如何改变 action chunk 可以提高 failure-aware realistic Q。

由于 action chunk 维度较高，必须对梯度进行归一化与裁剪。定义：

[
\widehat g_0(o,x_t)
===================

\frac{
\mathrm{clip}
\left(
g_0(o,x_t),g_{\mathrm{clip}}
\right)
}{
\mathrm{EMA}(|g_0(o,x_t)|_2)+\epsilon_g
}
]

其中 (\mathrm{clip}(\cdot,g_{\mathrm{clip}})) 表示 global norm clipping，(\mathrm{EMA}(|g_0|_2)) 是训练过程中维护的梯度范数滑动平均，(\epsilon_g) 是数值稳定项。

---

## Risk gap 与局部 (\eta) 缩放

定义 CPIQL critic 的风险分歧：

[
\Delta Q(o,a)
=============

## Q_{\bar\phi}(o,a,1)

Q_{\bar\phi}(o,a,0)
]

实际使用时，采用非负截断形式：

[
\Delta Q_+(o,a)
===============

\mathrm{stopgrad}
\left[
\mathrm{clip}
\left(
Q_{\bar\phi}(o,a_{\mathrm{ref}},1)
----------------------------------

Q_{\bar\phi}(o,a_{\mathrm{ref}},0),
0,
\Delta_{\max}
\right)
\right]
]

这里 (a_{\mathrm{ref}}) 第一版推荐使用数据动作 (a)，而不是 noisy action (x_t)。原因是数据动作处于行为数据支持内，critic 的 (Q(o,a,1)-Q(o,a,0)) 更可靠；而 (x_t) 在较高噪声下可能远离真实动作分布，直接用它计算 risk gap 容易引入 OOD 估计误差。

(\Delta Q_+) 的含义是：若成功偏置 critic 与失败感知 critic 对同一动作的估计差异较大，则说明该动作可能位于成功-失败边界附近。此时 actor 更新应更保守。

定义全局 behavior regularization 系数：

[
\eta_{\mathrm{base}}>0
]

并用 risk gap 生成局部有效系数：

[
\eta_{\mathrm{eff}}(o,a_{\mathrm{ref}})
=======================================

\mathrm{clip}
\left(
\eta_{\mathrm{base}}
\exp(\beta_\Delta\Delta Q_+(o,a_{\mathrm{ref}})),
\eta_{\min},
\eta_{\max}
\right)
]

其中 (\beta_\Delta) 控制 risk gap 对 (\eta) 的放大强度。若 (\Delta Q_+) 大，则：

[
\eta_{\mathrm{eff}}\uparrow
]

actor 更接近 behavior cloning。若 (\Delta Q_+) 小，则：

[
\eta_{\mathrm{eff}}\approx \eta_{\mathrm{base}}
]

Q-gradient 可以更明显地影响 actor。

---

## (m_Q)：Q-gradient 注入随机变量

定义：

[
m_Q\sim\mathrm{Bernoulli}(p_Q)
]

其中：

[
m_Q\in{0,1}
]

表示当前样本是否启用 Q-gradient guidance。若：

[
m_Q=1
]

则该样本可以使用 CPIQL critic 的 Q-gradient 修改 diffusion noise target。若：

[
m_Q=0
]

则该样本退化为普通 diffusion behavior cloning。

(p_Q) 是 **Q-gradient 注入比例**。真实机器人部署的第一版推荐：

[
p_Q\in[0.1,0.3]
]

这样 actor 的主体仍然是 behavior cloning，只在一部分样本中接受 Q-guided policy improvement。(m_Q) 不参与反向传播。

---

## (m_{\mathrm{safe}})：安全门控

安全门控只考虑三个部分：

[
m_{\mathrm{safe}}
=================

\mathbf 1[\Delta Q_+<\Delta_{\mathrm{hard}}]
\cdot
\mathbf 1[|g_0|*2<g*{\mathrm{raw,max}}]
\cdot
\mathbf 1[t\in\mathcal T_{\mathrm{guide}}]
]

第一项：

[
\mathbf 1[\Delta Q_+<\Delta_{\mathrm{hard}}]
]

表示当成功偏置 critic 与失败感知 critic 的分歧过大时，不启用 Q-gradient。因为这说明当前状态-动作附近可能处于失败边界，真实部署时应保持更强的 behavior cloning 约束。

第二项：

[
\mathbf 1[|g_0|*2<g*{\mathrm{raw,max}}]
]

表示当原始 Q-gradient 范数过大时，不启用 Q-gradient。异常大的梯度通常意味着 critic 局部不平滑、动作处于 OOD 区域，或者 action chunk 中某些维度被 critic 过度敏感地放大。

第三项：

[
\mathbf 1[t\in\mathcal T_{\mathrm{guide}}]
]

表示只在指定 diffusion timestep 区间启用 Q-gradient。第一版不建议在接近纯噪声的 action 上查询 critic，也不建议在最终低噪声阶段强行改变动作细节。因此 (\mathcal T_{\mathrm{guide}}) 应选择中等噪声区间。

如果使用 SNR 形式定义，可以写为：

[
\mathcal T_{\mathrm{guide}}
===========================

\left{
t:
\mathrm{SNR}*{\min}
<
\frac{\bar\alpha_t}{1-\bar\alpha_t}
<
\mathrm{SNR}*{\max}
\right}
]

当 (m_{\mathrm{safe}}=0) 时，即使 (m_Q=1)，该样本也不使用 Q-gradient guidance。

---

## Q-guided noise target

定义有效 guidance 强度：

[
\lambda_{\mathrm{eff}}
======================

m_Q
m_{\mathrm{safe}}
\frac{\lambda_0}{\eta_{\mathrm{eff}}(o,a_{\mathrm{ref}})}
]

其中 (\lambda_0) 是基础 Q-guidance 强度。由于 (\eta_{\mathrm{eff}}) 出现在分母中，risk gap 越大，(\eta_{\mathrm{eff}}) 越大，实际 Q-guidance 越弱。

定义 Q-guided noise target：

[
\epsilon_{\mathrm{target}}
==========================

## \epsilon

\lambda_{\mathrm{eff}}
\sqrt{1-\bar\alpha_t}
\widehat g_0(o,x_t)
]

展开为：

[
\epsilon_{\mathrm{target}}
==========================

## \epsilon

m_Qm_{\mathrm{safe}}
\frac{\lambda_0}{\eta_{\mathrm{eff}}(o,a_{\mathrm{ref}})}
\sqrt{1-\bar\alpha_t}
\widehat{\nabla_{x_t}Q_{\bar\phi}(o,x_t,0)}
]

这里的负号表示：actor 的 noise prediction target 被移动到能够使 denoised action 朝 (Q(o,a,0)) 上升方向变化的位置。该项保留 DAC 的 soft Q-guidance 思想，同时通过 (m_Q)、(m_{\mathrm{safe}})、(\eta_{\mathrm{eff}}) 控制更新强度。

---

## Actor loss

最终 actor loss 为：

[
L_{\mathrm{actor}}(\theta)
==========================

\mathbb E
\left[
\left|
\epsilon_\theta(x_t,o,t)
------------------------

\epsilon_{\mathrm{target}}
\right|_2^2
\right]
]

当：

[
m_Qm_{\mathrm{safe}}=0
]

时：

[
\epsilon_{\mathrm{target}}=\epsilon
]

于是：

[
L_{\mathrm{actor}}(\theta)
==========================

\mathbb E
\left[
\left|
\epsilon_\theta(x_t,o,t)-\epsilon
\right|_2^2
\right]
]

即普通 diffusion behavior cloning。

当：

[
m_Qm_{\mathrm{safe}}=1
]

时，actor 的去噪目标受到 failure-aware Q-gradient 的偏移，从而在行为分布附近向高 (Q(o,a,0)) 动作移动。

---

## (\eta_{\mathrm{base}}) 的动态更新

(\eta_{\mathrm{eff}}) 由两部分组成：

[
\eta_{\mathrm{eff}}
===================

\eta_{\mathrm{base}}
\cdot
\exp(\beta_\Delta\Delta Q_+)
]

其中只有 (\eta_{\mathrm{base}}) 参与动态更新，risk gap 指数项不参与 dual update。

定义普通 diffusion BC loss：

[
\ell_{\mathrm{BC}}
==================

\mathbb E
\left[
\left|
\epsilon_\theta(x_t,o,t)-\epsilon
\right|_2^2
\right]
]

希望它落在稳定区间：

[
b_{\min}
\le
\ell_{\mathrm{BC}}
\le
b_{\max}
]

若：

[
\ell_{\mathrm{BC}}>b_{\max}
]

说明 actor 偏离 behavior cloning 太多，应增大 (\eta_{\mathrm{base}})，削弱 Q-guidance。若：

[
\ell_{\mathrm{BC}}<b_{\min}
]

说明 actor 过于接近 BC，Q-guidance 影响不足，可以缓慢减小 (\eta_{\mathrm{base}})。

为了保证 (\eta_{\mathrm{base}}>0)，更新：

[
\xi=\log\eta_{\mathrm{base}}
]

使用：

[
\xi
\leftarrow
\xi
+
\alpha_\eta
\left[
\max(0,\ell_{\mathrm{BC}}-b_{\max})
-----------------------------------

\lambda_{\mathrm{low}}
\max(0,b_{\min}-\ell_{\mathrm{BC}})
\right]
]

然后：

[
\eta_{\mathrm{base}}=\exp(\xi)
]

其中：

[
0<\lambda_{\mathrm{low}}<1
]

真实机器人部署中应让 (\eta) 增大更快、减小更慢，因此推荐：

[
\lambda_{\mathrm{low}}\in[0.25,0.5]
]

---

## Target actor EMA

actor 训练阶段维护 EMA target actor：

[
\epsilon_{\bar\theta}
]

更新规则为：

[
\bar\theta
\leftarrow
(1-\alpha_{\mathrm{ema}})\bar\theta
+
\alpha_{\mathrm{ema}}\theta
]

部署时优先使用 EMA actor，而不是 online actor。EMA actor 能平滑 Q-guided 更新带来的震荡，降低真实机器人部署风险。

---

## 推荐算法流程

首先，加载已经训练好的 CPIQL critic：

[
Q_\phi(o,a,k),\qquad V_\psi(o,k)
]

并冻结或复制为 EMA critic：

[
Q_{\bar\phi}\leftarrow Q_\phi
]

actor 训练阶段不更新 critic。每个 actor update step 中，从数据集中采样：

[
(o,a)\sim\mathcal D
]

再采样：

[
\epsilon\sim\mathcal N(0,I),\qquad t\sim p(t)
]

构造 noisy action：

[
x_t=
\sqrt{\bar\alpha_t}a
+
\sqrt{1-\bar\alpha_t}\epsilon
]

计算 failure-aware gradient：

[
g_0=
\nabla_{x_t}
Q_{\bar\phi}(o,x_t,0)
]

计算 risk gap：

[
\Delta Q_+
==========

\mathrm{clip}
\left(
Q_{\bar\phi}(o,a,1)-Q_{\bar\phi}(o,a,0),
0,
\Delta_{\max}
\right)
]

计算有效 (\eta)：

[
\eta_{\mathrm{eff}}
===================

\mathrm{clip}
\left(
\eta_{\mathrm{base}}
\exp(\beta_\Delta\Delta Q_+),
\eta_{\min},
\eta_{\max}
\right)
]

采样 Q-gradient 注入变量：

[
m_Q\sim\mathrm{Bernoulli}(p_Q)
]

计算安全门控：

[
m_{\mathrm{safe}}
=================

\mathbf 1[\Delta Q_+<\Delta_{\mathrm{hard}}]
\cdot
\mathbf 1[|g_0|*2<g*{\mathrm{raw,max}}]
\cdot
\mathbf 1[t\in\mathcal T_{\mathrm{guide}}]
]

归一化 Q-gradient：

[
\widehat g_0
============

\frac{
\mathrm{clip}(g_0,g_{\mathrm{clip}})
}{
\mathrm{EMA}(|g_0|_2)+\epsilon_g
}
]

构造 Q-guided noise target：

[
\epsilon_{\mathrm{target}}
==========================

## \epsilon

m_Qm_{\mathrm{safe}}
\frac{\lambda_0}{\eta_{\mathrm{eff}}}
\sqrt{1-\bar\alpha_t}
\widehat g_0
]

更新 actor：

[
\theta
\leftarrow
\theta
------

\alpha_\theta
\nabla_\theta
\left|
\epsilon_\theta(x_t,o,t)
------------------------

\epsilon_{\mathrm{target}}
\right|_2^2
]

更新 (\eta_{\mathrm{base}})：

[
\log\eta_{\mathrm{base}}
\leftarrow
\log\eta_{\mathrm{base}}
+
\alpha_\eta
\left[
\max(0,\ell_{\mathrm{BC}}-b_{\max})
-----------------------------------

\lambda_{\mathrm{low}}
\max(0,b_{\min}-\ell_{\mathrm{BC}})
\right]
]

更新 EMA actor：

[
\bar\theta
\leftarrow
(1-\alpha_{\mathrm{ema}})\bar\theta
+
\alpha_{\mathrm{ema}}\theta
]

---

## 实现位置建议

如果当前算法库已经有 `DiffusionIQLAgent` 或类似结构，那么 CPIQL-DAC actor update 应该作为新的 actor mixin 或 actor update 函数加入，而不是直接重写 critic。建议命名为：

[
\texttt{CPIQLDACActorMixin}
]

或者：

[
\texttt{RiskScaledDACActorMixin}
]

该模块只需要读取 frozen CPIQL critic，不需要更新 critic 参数。核心函数可以设计为：

[
\texttt{update_actor_cpiql_dac(batch)}
]

它从 batch 中读取：

[
o,\quad a
]

并调用 frozen critic 计算：

[
Q(o,x_t,0),\quad Q(o,a,0),\quad Q(o,a,1)
]

其中 (Q(o,x_t,0)) 用于计算 action-gradient，(Q(o,a,1)-Q(o,a,0)) 用于计算 risk gap 与 (\eta_{\mathrm{eff}})。

需要注意：计算 (g_0=\nabla_{x_t}Q(o,x_t,0)) 时，必须让 (x_t) 开启 gradient：

[
x_t.\texttt{requires_grad_(True)}
]

但 critic 参数应 frozen：

[
\texttt{requires_grad=False}
]

计算出的 (g_0) 应 detach 后用于构造 (\epsilon_{\mathrm{target}})，避免 actor loss 反向更新 critic 或产生二阶梯度。

---

## 必须记录的训练日志

训练 actor 时必须记录以下量，用于判断 Q-guidance 是否稳定。首先记录普通 BC loss：

[
\ell_{\mathrm{BC}}
]

它应落在：

[
[b_{\min},b_{\max}]
]

附近。其次记录 risk gap：

[
\Delta Q_+
]

如果长期过高，说明 actor 训练样本集中大量状态处于成功-失败分歧区域，应降低 (p_Q) 或提高 (\eta_{\min})。接着记录原始梯度范数：

[
|g_0|_2
]

如果经常超过 (g_{\mathrm{raw,max}})，说明 critic 对 noisy action 的梯度不稳定，应缩小 (\mathcal T_{\mathrm{guide}}) 或进一步裁剪梯度。还应记录：

[
m_Q,\qquad m_{\mathrm{safe}},\qquad m_Qm_{\mathrm{safe}}
]

的平均值，确保 Q-guidance 实际启用比例合理。

最后记录 guided target 偏移量：

[
|\epsilon_{\mathrm{target}}-\epsilon|_2
]

它是最直接的 actor 更新强度指标。如果该值过大，说明 Q-guidance 过强。

---

## 推荐初始超参数

第一版真实部署应保守设置。可以令：

[
p_Q=0.1\sim0.3
]

[
\lambda_0=0.05\sim0.2
]

[
\eta_{\min}>0
]

并设置较小的 (\beta_\Delta)，例如：

[
\beta_\Delta\in[0.5,2.0]
]

若 CPIQL 的 (Q) 范围基本在 ([0,1])，则 (\exp(\beta_\Delta\Delta Q_+)) 不会过大；但仍必须使用 (\eta_{\max}) 限制上界。

(\Delta_{\mathrm{hard}}) 应小于或等于 (\Delta_{\max})，表示一旦成功偏置 critic 与失败感知 critic 差异超过该阈值，就完全禁用 Q-gradient。(g_{\mathrm{raw,max}}) 需要根据实际 action chunk 维度和动作归一化尺度统计得到，不建议手写固定值。

---

## 最终方法定义

本方法可以命名为：

**Stage-wise Risk-Scaled CPIQL-DAC**

其核心定义是：

[
\pi^*(a|o)
\propto
\pi_\beta(a|o)
\exp
\left(
\frac{
Q_{\bar\phi}(o,a,0)
}{
\eta_{\mathrm{base}}
\exp
\left(
\beta_\Delta
[Q_{\bar\phi}(o,a,1)-Q_{\bar\phi}(o,a,0)]_+
\right)
}
\right)
]

并通过 diffusion noise regression 近似该目标分布。实际训练公式为：

[
\epsilon_{\mathrm{target}}
==========================

## \epsilon

m_Qm_{\mathrm{safe}}
\frac{\lambda_0}{\eta_{\mathrm{eff}}}
\sqrt{1-\bar\alpha_t}
\widehat{\nabla_{x_t}Q_{\bar\phi}(o,x_t,0)}
]

[
L_{\mathrm{actor}}
==================

\mathbb E
\left[
\left|
\epsilon_\theta(x_t,o,t)-\epsilon_{\mathrm{target}}
\right|_2^2
\right]
]

该方法的关键特征是：**critic 先由 IQL 独立训练，不依赖 actor 采样；actor 更新只使用 (Q(o,a,0)) 的 failure-aware gradient；(Q(o,a,1)-Q(o,a,0)) 只用于调节保守程度；(m_Q) 控制 Q-gradient 注入比例；(m_{\mathrm{safe}}) 保证只在风险分歧小、梯度范数正常、diffusion timestep 合适时启用 Q-guidance。**
