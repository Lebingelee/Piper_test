## 1）目前存在的问题与核心矛盾

当前 CPIQL 使用同一个条件化 critic 模型，通过输入不同的 (k) 来输出不同语义的价值估计：

[
V(s,k),\quad Q(s,a,k)
]

其中 (k=1) 期望表示 **理想进度估计**，即当前状态如果沿着相似成功轨迹继续执行，距离最终成功还差多少；(k=0) 期望表示 **现实失败感知估计**，即在混入失败数据后，该状态在真实数据分布下的可达价值。

目前出现的问题是：由于 (k<1) 的训练数据中混入了失败轨迹，而失败轨迹的 MC return 通常被设为 (0)，普通 MLP 在共享参数下会把失败轨迹的 (0) target 泛化到 (k=1) 的输出上，导致：

[
V(s,k=1)\approx V(s,k=0)
]

尤其是在失败轨迹或失败边界状态上，(V(k=1)) 无法保持“理想进度估计”的语义，而是被现实失败标签压低。

这说明当前问题不是单纯的 alpha 调参问题，而是 **条件化价值函数的监督语义混叠**。如果直接用：

[
L_V = L_{\text{expectile}}+\alpha L_{\text{MC}}
]

让成功轨迹的线性 progress return 和失败轨迹的 (0) return 同时更新同一个共享 value 主干，那么 (V(k=1)) 会被失败数据污染；当 (\alpha) 小时，(V) 差异化不明显；当 (\alpha) 大时，成功轨迹的上升 target 和失败轨迹的全零 target 冲突过强，导致 value 震荡。

因此核心问题是：**失败数据的 (0) return 不应该直接更新表示理想进度的 value base；它应该主要训练现实风险惩罚项。**

---

## 2）采用梯度路由后希望达到的预期效果

我们希望将 value 结构改造成“理想进度基底 + 失败风险惩罚”的形式，而不是继续让普通 MLP 直接拟合 (V(s,k))。

推荐结构语义为：

[
V(s,k)=B(s)-(1-k)P(s)
]

或更稳定地：

[
V(s,k)=B(s)\left[1-(1-k)\sigma(p(s))\right]
]

其中：

[
B(s)=V_{\text{ideal}}(s)
]

表示 **理想进度估计**，对应 (k=1)：

[
V(s,1)=B(s)
]

而 (P(s)) 或 (\sigma(p(s))) 表示 **失败风险惩罚**，对应 (k=0)：

[
V(s,0)=B(s)-P(s)
]

或：

[
V(s,0)=B(s)(1-\sigma(p(s)))
]

采用梯度路由后，训练目标应具备以下效果。

对于成功轨迹，成功样本应该训练 (B(s)) 拟合动态 (\gamma) 产生的 progress return，使：

[
V(s_t,1)\approx G_t^{\text{prog}}
]

同时成功样本应该压低 penalty，使：

[
P(s_t)\approx0
]

或者：

[
\sigma(p(s_t))\approx0
]

因此在成功轨迹上应观察到：

[
V(s_t,1)\approx V(s_t,0)\approx G_t^{\text{prog}}
]

也就是说，**成功轨迹上的 (V(k=1)) 和 (V(k=0)) 都应随任务进度稳步上升，并且二者差距不应过大**。这表示成功轨迹处于稳定可达区域，理想估计和现实估计应基本一致。

对于失败轨迹，失败样本的 (0) return 不应直接压低 (B(s))。失败样本应主要训练 penalty，使现实价值下降：

[
V(s,0)\rightarrow0
]

但保留：

[
V(s,1)=B(s)
]

的理想进度语义。

也就是说，对于 near-miss failure 或专家介入边界这类状态，理想情况下应出现：

[
V(s,1)\text{ 较高},\quad V(s,0)\text{ 较低}
]

从而产生明显 gap：

[
\Delta V(s)=V(s,1)-V(s,0)
]

这个 gap 后续可作为失败边界、专家介入、recovery 数据采集的信号。

需要注意，并不是所有失败轨迹都应该有大 gap。early failure 或明显偏离任务流形的失败状态，可能本身就没有理想进度，此时：

[
V(s,1)\approx V(s,0)\approx0
]

是合理结果。真正应该出现明显 gap 的是 **看起来接近成功但现实执行失败的状态**。

---

## 3）使用梯度路由时的注意事项

实现梯度路由时，最重要的是确保失败样本的现实 (0) return 不会直接反向更新理想进度基底 (B(s))。

对于成功样本，可以正常训练：

[
B(s)\rightarrow G_{\text{prog}}(s)
]

并额外约束 penalty 变小：

[
P(s)\rightarrow0
]

这样成功轨迹上的 (V(k=1)) 和 (V(k=0)) 都能保持上升。

对于失败样本，应该使用 stop-gradient 切断 (B(s)) 的梯度。例如现实失败 loss 不应写成：

[
\left(B(s)-P(s)-0\right)^2
]

因为这样失败 (0) return 会同时更新 (B(s)) 和 (P(s))，继续污染 (V(k=1))。

应改成类似：

[
\left(\text{stopgrad}[B(s)]-P(s)-0\right)^2
]

或在乘法结构下使用：

[
\left(\text{stopgrad}[B(s)](1-\sigma%28p%28s%29%29)-0\right)^2
]

这样失败数据主要更新 penalty 分支，而不是更新 ideal base。

还需要注意，Q 网络最好也采用类似结构，否则 V 的 expectile loss 仍可能通过 (\bar Q(s,a,k)) 把失败数据的现实估计传回 (V(k=1))。如果暂时不改 Q，也至少需要检查 expectile loss 中失败样本是否会间接污染 (B(s))。理想情况下，Q 也应具有：

[
Q(s,a,k)=Q_B(s,a)-(1-k)P_Q(s,a)
]

并对失败样本采用类似的梯度路由。

此外，梯度路由只能防止失败 (0) return 污染 (V(k=1))，但不能凭空让失败状态获得合理的理想进度。如果失败状态与成功轨迹状态分布差异很大，那么 (B(s)) 在失败轨迹上可能仍然偏低。此时后续可以考虑为 near-miss failure 或 intervention boundary 构造 counterfactual ideal progress pseudo-label，例如通过最近成功轨迹状态匹配、专家恢复轨迹长度、或者上一轮稳定的 (V(k=1)) 生成理想进度标签。

第一版优先实现梯度路由，不要立刻加入普通 contrastive loss。直接用 contrastive loss 强行拉开 (V(k=1)) 和 (V(k=0)) 可能破坏 value 的数值语义。若后续需要拉开 gap，也应使用具有 value 解释的 risk-gap margin，只作用于 near-miss failure 或 intervention boundary，而不是所有失败状态。

最终目标是让模型学到：

[
V(s,1)=\text{ideal progress value}
]

[
V(s,0)=\text{failure-aware realistic value}
]

并让：

[
V(s,1)-V(s,0)
]

只在真正的成功-失败边界区域变大，而不是在所有失败状态或所有成功状态上无差别变大。
