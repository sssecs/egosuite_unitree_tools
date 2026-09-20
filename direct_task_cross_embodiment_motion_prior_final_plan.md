# Direct Task-Space Teleoperation + Cross-Embodiment Human Motion Prior

## 1. 项目目标

本项目目标是在 humanoid 全身遥操作中同时满足以下三点：

1. **高精度保留人类操作意图**
   - 手腕位置、手腕姿态、肩部高度、肩部朝向等控制接口尽量直接由 Ego 人体数据得到。
   - 避免 full-body retargeting 对手腕轨迹造成二次变形或精度损失。

2. **机器人全身动作尽可能像人**
   - 正常操作时能够自然站立、俯身、半蹲、深蹲、单膝跪地、跨步等。
   - 不要求机器人逐关节复制人体动作，而是希望机器人采取与人类相似的整体运动策略。

3. **允许机器人根据自身动力学主动适应**
   - 受到扰动时允许跨步、改变骨盆高度、改变支撑方式等。
   - 负载、摩擦、机器人尺寸、关节限制与人体不同，因此机器人不应被锁死在人类参考轨迹上。
   - 动作风格是 soft prior，而不是 hard reference。

核心原则：

$$
oxed{	ext{Safety / Dynamics} > 	ext{Task Accuracy} > 	ext{Human Style}}
$$

---

## 2. 为什么不以 Full-Body Retargeting 为主线

传统路线通常是：

$$
	ext{Human Motion}ightarrow	ext{Robot Retargeting}ightarrow	ext{Robot Full-body Reference}ightarrow	ext{Tracking Policy}
$$

对于本项目，这条路线存在两个主要问题。

### 2.1 Retargeting 不考虑完整机器人动力学

运动学上可行的 reference 并不一定动态稳定、满足接触约束、满足机器人力矩能力，也不一定适合负载和扰动恢复。

$$
	ext{Kinematically plausible}
eq	ext{Dynamically optimal}
$$

### 2.2 Retargeting 会损失手腕任务精度

Full-body IK / retargeting 通常需要同时平衡 pelvis、torso、shoulder、elbow、wrist、foot、knee 等多个目标，因此 wrist 只是多个优化目标之一。

$$
FK_{wrist}(q^*)
eq p_{wrist}^{human-target}
$$

而本项目的核心要求是：

> 人类手腕和肩部信息应尽量无损地进入机器人控制接口。

因此主线不采用：

$$
oxed{Humanightarrow GMRightarrow Robot\ Command}
$$

---

## 3. 最终总体架构

核心思想是把人体数据拆成两条互相独立的通道：

$$
X_t^{human}ightarrowegin{cases}c_t^{task}\ d_t^{style}\end{cases}
$$

其中：

- $c_t^{task}$：直接任务控制接口；
- $d_t^{style}$：跨形态人体动作风格描述。

机器人 policy：

$$
oxed{a_t=\pi(c_t^{task},e_t,H_t)}
$$

其中：

- $c_t^{task}$：人体 sparse command；
- $e_t$：机器人当前任务误差；
- $H_t$：机器人 proprioception/action history；
- $a_t$：29-DoF joint position action。

训练时使用 $d_t^{human}$ 与机器人自身计算得到的 $d_t^{robot}$ 构造人体动作风格约束。

```text
                  EgoSuite Human Skeleton
                           │
              ┌────────────┴─────────────┐
              │                          │
              ▼                          ▼
      Direct Task Extractor      Style Descriptor Extractor
              │                          │
              ▼                          ▼
   wrist / shoulder command      Human Style Descriptor
              │                          │
              ▼                          │
      Closed-loop Task Policy            │
              │                          │
              ▼                          │
          29-DoF Action                  │
              │                          │
              ▼                          │
             G1 ───────► Robot Style Descriptor
                                      │
                                      ▼
                               Style / Prior Reward
```

---

## 4. Task Channel：人体任务信息直接进入机器人

### 4.1 控制接口

沿用当前 sparse interface，主要包含：

- 左手腕位置；
- 左手腕姿态；
- 右手腕位置；
- 右手腕姿态；
- 双肩中点 XY；
- 左肩高度；
- 右肩高度；
- 肩部 heading；
- 左右手腕线速度。

当前核心 command 可继续保持类似：

$$
c_t^{task}\in\mathbb{R}^{24}
$$

外加 wrist velocity 等当前已有项。

### 4.2 只允许简单、可解释的几何变换

Human → Robot task command 只进行：

1. 坐标系转换；
2. 初始 SE(2) 对齐；
3. 身体比例 / reach scale；
4. 必要的 workspace clipping；
5. 必要的高度基准处理。

例如：

$$
p_r^{cmd}=T+S(p_h-p_{origin})
$$

不经过 full-body IK。

---

## 5. 当前 Sparse Policy

### 5.1 Actor 输入

当前 structured-history 设计：

#### 当前 task 信息

```text
command absolute                  24
sim task state                    24
command wrist linear velocity      6
sim wrist linear velocity          6
explicit task error               24
------------------------------------
                                  84
```

#### 25 帧 proprioception/action history

```text
base angular velocity              3
projected gravity                  3
joint position                    29
joint velocity                    29
previous action                   29
------------------------------------
                                  93 / frame
```

因此：

$$
84+25	imes93=2409
$$

Actor：

$$
2409ightarrow512ightarrow256ightarrow128ightarrow29
$$

输出 29-DoF joint position actions。

### 5.2 为什么保留 history

25 帧、50 Hz 对应约 0.5 秒。History 可以帮助 policy 隐式判断：

- 当前是否正在失稳；
- 是否受到外部扰动；
- 是否存在额外负载；
- actuator response 是否异常；
- 当前处于 squat / kneel / recovery 的哪个阶段。

因此 actor 不一定需要显式知道 payload mass、push force、contact mode 等隐变量。

---

## 6. Cross-Embodiment Style Descriptor

### 6.1 核心思想

不比较：

$$
q^{human}\leftrightarrow q^{G1}
$$

而比较：

$$
oxed{	ext{Human movement strategy}\leftrightarrow	ext{Robot movement strategy}}
$$

即：

$$
d_t^{human}\leftrightarrow d_t^{robot}
$$

这个 descriptor 应尽量：

- 与 morphology 无关；
- 与绝对身体尺寸无关；
- 与世界 XY 无关；
- 能区分 stand / bend / squat / kneel / step；
- 能表达人体整体支撑和重心策略；
- 不要求人体和 G1 具有相同关节结构。

---

## 7. Characteristic Length

定义 characteristic length：

$$
L=L_{leg}
$$

可以使用：

$$
L_{leg}=L_{thigh}+L_{shin}
$$

或者 standing pelvis height。

Human 和 G1 分别使用自己的 $L$，大量空间量都做 $x/L$ 归一化。

---

## 8. 推荐的人体 / 机器人共享 Descriptor

最终可表示为：

$$
oxed{d_t=[d_t^{posture},d_t^{support},d_t^{motion}]}
$$

维度预计约 20–40D。

### 8.1 Normalized Pelvis Height

$$
	ilde h_p=rac{h_{pelvis}}{L}
$$

或：

$$
	ilde h_p=rac{h_{pelvis}}{h_{pelvis}^{standing}}
$$

它表达的是身体整体降低了多少，而不是绝对高度。

### 8.2 Normalized Effective Leg Length / Leg Compression

定义每条腿：

$$
oxed{c_{leg}=rac{\|p_{hip}-p_{ankle}\|}{L_{thigh}+L_{shin}}}
$$

意义：

- 站直：接近 1；
- 屈膝：减小；
- 半蹲：进一步减小；
- 深蹲：更低。

这不是要求 $q_{knee}^{human}=q_{knee}^{G1}$，而是要求人和机器人具有类似的整体腿部伸展 / 压缩程度。

左右分别：

$$
c_L,\quad c_R
$$

还可以表达 kneeling 时的非对称性。

### 8.3 Torso Orientation

在 heading-local frame 中表示 torso orientation：

$$
R_{torso}^{local}=R_{heading}^{-1}R_{torso}
$$

可以使用 pitch、roll 或 6D rotation representation。

### 8.4 Foot Placement / Stance

左右脚相对 pelvis：

$$
	ilde p_f^L=rac{p_f^L-p_{pelvis}}{L},\qquad
	ilde p_f^R=rac{p_f^R-p_{pelvis}}{L}
$$

也可以使用：

$$
	ilde w_{stance}=rac{\|p_f^L-p_f^R\|_{xy}}{L}
$$

### 8.5 Support Topology

推荐：

$$
d_{support}=[c_{foot,L},c_{foot,R},c_{knee,L},c_{knee,R}]
$$

例如：

- Normal standing / squat：$[1,1,0,0]$
- Left-knee kneeling：可能出现 $[1,1,1,0]$

Contact 只作为 soft style feature，不能作为硬约束。

### 8.6 Pelvis Relative to Support

定义支撑中心：

$$
p_{support}=rac{p_f^L+p_f^R}{2}
$$

然后：

$$
oxed{d_{shift}=rac{(p_{pelvis}-p_{support})_{xy}}{L}}
$$

表达身体前后左右的重心策略。

---

## 9. Dynamic Descriptor

动作风格不应只看单帧姿态。

### 9.1 Dimensionless Pelvis Velocity

$$
	ilde v=rac{v}{\sqrt{gL}}
$$

### 9.2 Dimensionless Angular Velocity

$$
	ilde\omega=\omega\sqrt{rac{L}{g}}
$$

### 9.3 Leg Compression Rate

使用：

$$
\dot c_L,\quad \dot c_R
$$

也可以按 $\sqrt{L/g}$ 无量纲化。

---

## 10. Descriptor 示例

一个半蹲操作可能表示为：

```text
pelvis_height_norm       0.78
leg_extension_L          0.74
leg_extension_R          0.75
torso_pitch              0.18 rad

foot_contact_L           1
foot_contact_R           1
knee_contact_L           0
knee_contact_R           0

stance_width_norm        0.25
pelvis_support_shift     small
```

单膝跪地：

```text
pelvis_height_norm       low
leg_extension_L          low
leg_extension_R          medium

foot_contact_L           1
foot_contact_R           1
knee_contact_L           1
knee_contact_R           0

strong left/right asymmetry
```

这些值描述运动策略，而不是机器人 joint target。

---

## 11. 第一版 Style Reward

第一阶段建议不要立刻使用 discriminator，先直接测试 descriptor 是否有意义。

定义：

$$
r_{style}=w_h r_{pelvis}+w_c r_{leg}+w_t r_{torso}+w_s r_{stance}+w_{shift}r_{shift}+w_{contact}r_{contact}
$$

例如：

$$
r_{pelvis}=\exp\left(-rac{(	ilde h_p^r-	ilde h_p^h)^2}{\sigma_h^2}ight)
$$

腿部：

$$
r_{leg}=rac12\left[
\exp\left(-rac{(c_L^r-c_L^h)^2}{\sigma_c^2}ight)+
\exp\left(-rac{(c_R^r-c_R^h)^2}{\sigma_c^2}ight)
ight]
$$

关键原则：

$$
oxed{	ext{Style Region, not Exact Reference}}
$$

---

## 12. Reward 优先级

最终总 reward 建议：

$$
oxed{
r=r_{stability}+r_{task}+r_{physical}+g_{task}g_{recovery}\lambda_{style}r_{style}
}
$$

其中：

- $r_{task}$：手腕 / 肩部任务；
- $r_{physical}$：joint limit、collision、feet slide 等；
- $r_{stability}$：避免摔倒、保证动态可行；
- $r_{style}$：人体动作先验；
- $g_{task}$：task gate；
- $g_{recovery}$：stability / recovery gate。

---

## 13. Task Gate

人体风格不能牺牲主任务。

定义 wrist position error：

$$
e_p=\sqrt{rac{\|p_L^{cmd}-p_L\|^2+\|p_R^{cmd}-p_R\|^2}{2}}
$$

可以使用：

$$
g_{task}=\sigma\left(rac{\epsilon_p-e_p}{	au_p}ight)
$$

也可以进一步乘 orientation gate：

$$
g_{task}=g_p g_R
$$

含义：

```text
tracking error 很大
→ style 权重很低

tracking 基本完成
→ style 开始决定冗余自由度如何使用
```

即：

$$
oxed{	ext{Task first, style second}}
$$

---

## 14. Recovery Gate

Human style 是 nominal behavior，而不是硬约束。

当机器人受到扰动时：

$$
g_{recovery}ightarrow0
$$

允许 policy 暂时离开人体动作 manifold。

可以根据 torso tilt、base angular velocity、pelvis velocity、COM/support proxy、foot support state 构造：

$$
e_{stability}=w_1|	heta_{tilt}|+w_2\|\omega_{base}\|+w_3\|v_{pelvis}\|+w_4d_{support}
$$

然后：

$$
g_{recovery}=\exp(-k e_{stability})
$$

或使用 sigmoid。

---

## 15. 扰动时的期望行为

例如人体示范：

```text
semi-squat
double-foot support
```

机器人：

```text
正常状态
→ 保持类似的人体半蹲策略

受到侧向 push
→ stability error 增大
→ g_recovery 降低
→ style penalty 暂时减弱
→ 允许侧跨一步
→ 调整 pelvis / torso
→ 恢复稳定

稳定以后
→ g_recovery 恢复
→ 返回 human-like semi-squat manifold
```

因此：

$$
oxed{	ext{Human data teaches nominal motion}}
$$

$$
oxed{	ext{Physics RL learns adaptation and recovery}}
$$

---

## 16. Quiet Feet 等 Reward 必须 Stability-Aware

稳定操作时，quiet-feet reward 可以抑制无意义乱走；但受到扰动时，它会与 recovery stepping 冲突。

因此建议：

$$
r_{quiet-feet}=g_{stable}r_{quiet-feet}
$$

稳定时 $g_{stable}pprox1$，失稳时 $g_{stable}ightarrow0$，允许机器人跨步恢复。

`feet_slide` 可以继续保留，因为 stepping 与 foot sliding 不同。

---

## 17. 扰动训练

当前 Fast-V7 posture stage 暂时关闭了 push。在 style/recovery 阶段需要重新打开。

推荐随机：

### Direction

```text
front
back
left
right
diagonal
```

### Magnitude

从轻到强 curriculum。

### Timing

最好让 push 出现在：

```text
standing
reaching
half squat
deep squat
kneel transition
```

不同 phase 中。

目标是自然形成：

```text
small disturbance
→ ankle / hip response

medium disturbance
→ weight shift

large disturbance
→ recovery step

very large disturbance
→ multiple steps
```

---

## 18. EgoSuite 的使用方式

EgoSuite skeleton 同时生成：

$$
(c_t^{task},d_t^{human})
$$

不再生成 G1 full-body joint reference 作为主训练目标。

```text
EgoSuite Skeleton
       │
       ├── Wrist / Shoulder
       │       ↓
       │   Direct Task Command
       │
       └── Pelvis / Torso / Legs / Feet
               ↓
          Human Style Descriptor
```

---

## 19. Human Style 数据不必另外重新采集

如果 EgoSuite 已经有足够 full-body skeleton，那么它本身就可以作为主要 Human Style Corpus。

只在发现明显 coverage gap 时补采 targeted data，例如：

```text
rare kneeling
heavy payload
controlled external push
specific G1 workspace
special transitions
```

---

## 20. 负载

如果人体数据存在拿重物行为，可以作为人体 nominal style 的参考。

但机器人训练时仍需加入真实对应的动力学变化：

- 手中 rigid body；
- payload mass randomization；
- 或 wrist external wrench。

否则机器人只是在模仿“拿重物的外观”。

建议 critic 可以看到 true payload mass / external wrench，actor 不一定看到，让 actor 通过 history 学习 implicit payload estimation。

---

## 21. Asymmetric Actor-Critic

不需要一开始做 teacher-student，但推荐保留 asymmetric actor-critic。

### Actor

只使用部署可获得信息：

```text
direct task command
explicit error
robot FK task state

joint position
joint velocity
IMU
projected gravity
previous action
history
```

### Critic

可以额外使用：

```text
true base linear velocity
true contacts
payload mass
external push
friction
COM randomization
full simulation state
human style descriptor
```

因此：

$$
V(o^{actor},o^{privileged})
$$

但：

$$
\pi(o^{actor})
$$

保持 deployable。

---

## 22. 当前不需要 Teacher-Student

第一版建议：

$$
oxed{	ext{Direct PPO}}
$$

即直接训练：

$$
\pi(c,e,H)
$$

原因：

1. 主任务 command 已经是最终部署 interface；
2. 不想重新引入 GMR reference error；
3. RL 可以直接承担 embodiment adaptation；
4. human style 通过 descriptor/reward 提供，而不是 full-body teacher。

Teacher-student 只在后续出现以下情况时再考虑：

- privileged actor 明显比 deployable actor 强；
- direct PPO 很难发现 squat/kneel；
- MoE PPO 不稳定；
- 需要将 privileged state 蒸馏成 history-based student。

如果需要 teacher，其输入也应是：

$$
(c^{task},d^{style},s^{privileged})
$$

而不是 GMR full-body reference。

---

## 23. 第二阶段：Conditional Cross-Embodiment Motion Prior

如果直接 descriptor tracking reward 效果有限，再升级为 learned prior。

Human：

$$
d_{t-k:t}^{human}
$$

Robot：

$$
d_{t-k:t}^{robot}
$$

训练：

$$
D(d_{t-k:t},\psi(c_t))
$$

其中 $\psi(c_t)$ 表示当前 task context，例如：

```text
mean wrist height
wrist relative to shoulder
reach distance
wrist speed
shoulder height
motion direction
```

形成：

$$
oxed{p(	ext{human-like motion strategy}\mid	ext{task context})}
$$

注意 discriminator 比较的是 $d^{human}$ 与 $d^{robot}$，而不是人体关节和 G1 关节，因此不需要 GMR。

---

## 24. Prior 的时间窗口

推荐：

$$
k=6\sim10
$$

在 50 Hz 下：

```text
6 frames   ≈ 120 ms
10 frames  ≈ 200 ms
```

这样可以区分正常进入 squat 和机器人失控下坠，即使某一瞬间 pelvis height 类似。

---

## 25. 多模态问题

相同 sparse command 可能对应：

```text
bend
squat
kneel_L
kneel_R
```

因此 $p(a|c)$ 天然多模态。

如果后续需要显式保留多种模式，可以加入：

$$
z_{style}
$$

例如：

```text
auto
bend
squat
kneel_L
kneel_R
```

Policy：

$$
\pi(c,e,H,z)
$$

部署默认 $z=auto$，也可以由更高层 planner 或视觉策略选择。

---

## 26. MoE 的位置

MoE 不负责创造技能。

先让 policy 真正学会：

```text
walk
bend
squat
kneel
recovery
load adaptation
```

如果发现这些技能在单一 MLP 中发生明显参数干扰，再替换为 $\pi_{MoE}$。

例如：

```text
shared encoder
     ↓
router
     ↓
4 experts
top-2
     ↓
action
```

MoE解决：

$$
oxed{	ext{capacity + specialization + skill interference}}
$$

Style prior 负责：

$$
oxed{	ext{what motion is human-like}}
$$

---

## 27. 推荐开发阶段

### Phase 1 — 当前 V7 baseline

已经具备：

- direct sparse task command；
- fixed-world closed-loop tracking；
- explicit task error；
- 25-frame structured proprio history；
- torso upright reward 移除；
- kneeling contact 允许。

目标：

$$
	ext{精确 task tracking + posture freedom}
$$

### Phase 2 — Human / Robot Descriptor

实现同一套 $d(\cdot)$，分别作用于：

- EgoSuite human skeleton；
- G1 simulation state。

先离线验证 descriptor 是否能区分：

```text
stand
bend
half squat
deep squat
kneel_L
kneel_R
step
```

同时检查不同人体尺寸下是否稳定。

### Phase 3 — Direct Descriptor Style Reward

训练：

$$
r=r_{task}+r_{physical}+g_{task}\lambda_{style}r_{descriptor}
$$

先验证：

> 不经过 GMR 是否能让 G1 学出自然的 bend/squat/kneel。

### Phase 4 — Recovery-Aware Style

重新开启 `push_robot`，加入 $g_{recovery}$，并把 `quiet_feet`、contact prior、style reward 都改为 stability-aware。

验证：

> 半蹲/跪地状态受到扰动后，是否能主动跨步恢复，再返回 human-like posture。

### Phase 5 — Conditional Motion Prior

如果逐帧 descriptor reward 太限制策略：

$$
r_{descriptor}ightarrow r_{prior}
$$

训练 conditional cross-embodiment discriminator。

### Phase 6 — Multimodality

加入 $z_{style}$ 支持 bend / squat / kneel 等同一 task command 下的不同解决方式。

### Phase 7 — MoE

当技能数量增加并出现干扰：

$$
MLPightarrow MoE
$$

### Phase 8 — Sim2Real

加入：

```text
localization noise
delay
encoder bias
friction randomization
COM randomization
payload randomization
push disturbance
sensor noise
```

最终部署到 G1。

---

## 28. 关键 Ablation

### A. Retargeting vs Direct Task Mapping

```text
A1: GMR full-body reference tracking
A2: GMR-derived wrist command
A3: direct Human wrist/shoulder command
```

比较：

- wrist position error；
- wrist orientation error；
- long-horizon drift；
- task success。

### B. Style Mechanism

```text
B1: no style
B2: descriptor reward
B3: descriptor reward + task gate
B4: conditional motion prior
```

比较：

- task accuracy；
- human-style similarity；
- pelvis / leg / support distribution；
- 动作多样性。

### C. Recovery

```text
C1: no push
C2: push + fixed style weight
C3: push + recovery gating
```

比较：

- fall rate；
- recovery success；
- step frequency；
- wrist tracking during disturbance；
- time to return to nominal style。

### D. History

```text
D1: no history
D2: 10 frames
D3: 25 frames
```

比较：

- disturbance recovery；
- payload adaptation；
- task precision；
- inference cost。

### E. MoE

只在前面系统有效后再做：

```text
E1: MLP
E2: MoE
```

观察是否减少 squat / kneel / walk 等技能之间的 interference。

---

## 29. 核心研究假设

本项目真正需要验证的核心假设是：

$$
oxed{	ext{Humanoid teleoperation does not require full-body robot retargeting}}
$$

更具体地说：

> 人类 end-effector / shoulder task information 可以直接作为机器人 task-space command；人体 whole-body behavior 可以通过 morphology-normalized motion descriptors 或 motion prior 传递，而机器人自身的 physics-aware RL 负责把这些任务和风格约束转换成动态可行的 G1 全身动作。

因此：

$$
oxed{	ext{Human task intent}
eq	ext{Robot joint reference}}
$$

而是：

$$
oxed{	ext{Human task intent}+	ext{Human motion strategy}ightarrow	ext{Robot physics-aware solution}}
$$

---

## 30. 当前方法名称建议

可以暂时使用：

### Direct Task-Space Teleoperation with Cross-Embodiment Motion Prior

或者更短：

### Cross-Embodiment Motion Prior for Sparse Humanoid Teleoperation

核心可以概括成：

$$
oxed{	extbf{Direct Task Mapping}+	extbf{Cross-Embodiment Style Representation}+	extbf{Physics-Aware Adaptation}}
$$

---

## 31. 一句话总结

> **直接保留人类手腕和肩部的任务空间信息，不通过 full-body retargeting 生成机器人参考动作；利用人体与机器人共享的尺度归一化姿态、支撑和动态描述传递 human-like whole-body motion prior，并让 physics-aware RL 在保持任务精度的前提下，自主决定适合 G1 的俯身、下蹲、跪地、跨步以及扰动恢复行为。**
