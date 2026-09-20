# Cross-Embodiment Human Motion Descriptor Design

## 1. 目标

本 descriptor 不要求 G1 逐关节模仿人体，而是把人体动作压缩为一组跨 morphology、尺度归一化、尽量不重复 wrist/shoulder 主任务的低维特征，用于描述：

- 身体整体降低多少；
- 左右腿压缩多少；
- 躯干如何配合；
- pelvis 相对 shoulder 如何布置；
- 双脚如何形成支撑；
- 是否出现单膝跪地；
- 动作是否在合理地进入/离开这些姿态。

核心不是比较：

\[
q^{human}\leftrightarrow q^{G1}
\]

而是比较：

\[
d^{human}\leftrightarrow d^{G1}
\]

---

## 2. 统一 semantic landmarks

Human skeleton 和 G1 FK 都先映射到同一组语义关键点：

```text
pelvis
shoulder_mid
hip_L / hip_R
knee_L / knee_R
ankle_L / ankle_R
foot_center_L / foot_center_R
```

然后 Human 和 Robot 共用同一个 descriptor 计算函数。

所有水平位置关系建议使用 heading-local frame，不使用 world XY 和 absolute yaw。

---

## 3. 尺度归一化

定义 characteristic length：

\[
L=L_{leg}=L_{thigh}+L_{shin}
\]

Human 和 G1 分别使用自身尺寸。

大多数空间量用 \(x/L\) 归一化。

Pelvis height 第一版更推荐：

\[
	ilde h_p=
rac{h_{pelvis}}
{h_{pelvis}^{neutral}}
\]

它直接表达“相对正常站立时身体降低了多少”。

---

# 4. 推荐 V1：17D Descriptor

\[
oxed{
d_t=
[
h_p,
c_L,c_R,
k_L,k_R,
	heta_{torso},
r_{sp,x},r_{sp,y},
\Delta\psi,
f_{Lx},f_{Ly},
f_{Rx},f_{Ry},
C_{fL},C_{fR},
C_{kL},C_{kR}
]
}
\]

总维度：

\[
oxed{17D}
\]

---

## 4.1 Normalized Pelvis Height — 1D

\[
h_p=
rac{h_{pelvis}}
{h_{pelvis}^{neutral}}
\]

作用：区分 stand / crouch / squat / kneel。

推荐权重：高。

---

## 4.2 Normalized Effective Leg Length — 2D

左右腿分别：

\[
c_L=
rac{\|p_{hip,L}-p_{ankle,L}\|}
{L_{thigh,L}+L_{shin,L}}
\]

\[
c_R=
rac{\|p_{hip,R}-p_{ankle,R}\|}
{L_{thigh,R}+L_{shin,R}}
\]

含义：

```text
接近 1  → 腿基本伸直
减小     → 腿逐渐压缩/屈曲
```

作用：表达 stand / half squat / deep squat，以及左右腿非对称。

推荐权重：高。

---

## 4.3 Knee-to-Ground Distance — 2D

\[
k_L=
rac{d(p_{knee,L},ground)}{L}
\]

\[
k_R=
rac{d(p_{knee,R},ground)}{L}
\]

作用：区分 deep squat 和 kneel，尤其是 kneel_L / kneel_R。

推荐权重：中。

当 human knee 接近地面时，可自适应提高该项权重。

---

## 4.4 Torso Pitch — 1D

定义 torso 相对 gravity / heading-local frame 的 pitch：

\[
	heta_{torso}
\]

主要区分：

```text
bend:
  pelvis 中等
  leg compression 小
  torso pitch 大

squat:
  pelvis 低
  leg compression 大
  torso pitch 中等
```

第一版不建议加入 torso roll，因为 shoulder height 已经在主任务接口里提供大量 roll 信息。

推荐权重：中。

---

## 4.5 Shoulder-Mid Relative to Pelvis XY — 2D

\[
r_{sp}
=
R_{heading}^{T}
(
p_{shoulder-mid}-p_{pelvis}
)
\]

取水平分量并除以 \(L\)：

\[
r_{sp,x},\quad r_{sp,y}
\]

这是很重要的一组特征，因为：

- shoulder 属于 primary task；
- pelvis 属于机器人可自由选择的冗余自由度。

它直接描述：

> 人类完成同样 upper-body task 时，通常把 pelvis 放在 shoulder 的什么位置。

推荐权重：中高。

---

## 4.6 Shoulder–Pelvis Yaw Difference — 1D

\[
\Delta\psi=
wrap(
\psi_{shoulder}-\psi_{pelvis}
)
\]

作用：描述上下身扭转，用于侧向操作和自然 waist/stance coordination。

如果 EgoSuite pelvis orientation 噪声较大，可延后启用。

推荐权重：低到中。

---

## 4.7 Foot Relative to Pelvis XY — 4D

左脚：

\[
f_L=
rac{
R_{heading}^{T}(p_{foot,L}-p_{pelvis})_{xy}
}{L}
\]

右脚：

\[
f_R=
rac{
R_{heading}^{T}(p_{foot,R}-p_{pelvis})_{xy}
}{L}
\]

展开为：

\[
[
f_{Lx},f_{Ly},f_{Rx},f_{Ry}
]
\]

作用：

- stance width；
- 左右脚前后错位；
- stepping；
- 非对称站姿；
- 负载下扩大支撑；
- kneeling support geometry。

不必另外加入 stance width，因为可由 \(\|f_L-f_R\|\) 推出。

推荐权重：低到中。

---

## 4.8 Foot Pseudo-Contact — 2D

\[
C_{fL},\quad C_{fR}
\]

不建议 Human 用 binary contact。

可定义连续 pseudo-contact：

\[
C_f=
\exp
\left(
-rac{z_f^2}{\sigma_z^2}
-rac{\|v_f\|^2}{\sigma_v^2}
ight)
\]

含义：

```text
脚低 + 基本不动 → 接近 1
脚离地/快速运动 → 接近 0
```

Human 和 Robot 都应使用同一种几何公式计算 style pseudo-contact。

Robot 的 MuJoCo 精确 contact sensor 仍可用于物理 reward / safety，但不要直接混进 cross-embodiment style descriptor。

推荐权重：低。

---

## 4.9 Knee Pseudo-Contact — 2D

\[
C_{kL},\quad C_{kR}
\]

可由 knee-ground distance 与 knee velocity 构造。

主要用于区分：

```text
deep squat
kneel_L
kneel_R
```

推荐权重：低到中。

---

# 5. V1 17D 总表

| Descriptor | Dim | 主要作用 | 推荐权重 |
|---|---:|---|---|
| normalized pelvis height | 1 | 身体整体高低 | 高 |
| effective leg length L/R | 2 | 腿部压缩程度 | 高 |
| knee-ground distance L/R | 2 | squat / kneel 区分 | 中 |
| torso pitch | 1 | bend / squat 区分 | 中 |
| shoulder-mid relative pelvis XY | 2 | 上下身整体协调 | 中高 |
| shoulder–pelvis yaw difference | 1 | 上下身扭转 | 低~中 |
| foot relative pelvis XY L/R | 4 | stance / stepping | 低~中 |
| foot pseudo-contact L/R | 2 | 支撑方式 | 低 |
| knee pseudo-contact L/R | 2 | kneeling topology | 低~中 |
| **总计** | **17** |  |  |

---

# 6. Direct Reward 阶段：扩展为 20D

如果第一阶段直接使用 descriptor tracking reward，而不是 learned prior，建议额外加入：

\[
[
\dot h_p,
\dot c_L,
\dot c_R
]
\]

因此：

\[
oxed{
d_t^{reward}\in\mathbb{R}^{20}
}
\]

这三项主要用于区分：

```text
正常进入 squat
```

和：

```text
机器人失控向下坠
```

第一版不建议加入大量 pelvis XY velocity、foot velocity 等，因为恢复跨步时这些量会自然偏离 human nominal motion。

---

# 7. Temporal Prior 阶段：170D

后续如果采用 conditional motion prior / discriminator，推荐直接使用 descriptor 时间窗口：

\[
[d_{t-9},...,d_t]
\]

50 Hz 下 10 帧约等于 200 ms。

因此：

\[
oxed{
10	imes17=170D
}
\]

时序 prior 可以自行学习：

- squat transition；
- kneel transition；
- stepping；
- instability；
- recovery；
- descriptor 的一阶/二阶变化。

---

# 8. 不建议第一版加入的量

## Raw joint angles

不加入：

```text
human hip angle
human knee angle
human ankle angle
```

因为 morphology dependence 强。

## Wrist / hand

不加入，因为已经属于 primary task。

## Shoulder height / heading

不加入 style descriptor，因为已经是 task interface。

## Elbow

第一版不加入，避免重新靠近 full-body imitation。

## World XY / absolute yaw

不加入，全部使用 heading-local / relative representation。

## COM / ZMP

第一版不加入。Human skeleton 的 segment mass / COM 估计未必可靠，Human 与 G1 mass distribution 也不同。

## GRF / torque

不加入。Human 数据通常没有可靠对应量，并且高度 dynamics-specific。

---

# 9. Descriptor Reward 建议

第一版可以：

\[
r_{style}
=
w_h r_h
+
w_c r_c
+
w_k r_k
+
w_t r_t
+
w_{sp}r_{sp}
+
w_f r_f
+
w_{contact}r_{contact}
\]

不要追求 exact tracking，而是宽容的 soft tolerance。

例如 pelvis：

\[
r_h=
\exp
\left(
-rac{(h_p^r-h_p^h)^2}{\sigma_h^2}
ight)
\]

核心原则：

\[
oxed{
	ext{Style Region}

eq
	ext{Exact Reference}
}
\]

---

# 10. Reward 权重优先级

## 强

```text
normalized pelvis height
effective leg length L/R
```

## 中

```text
knee-ground distance
torso pitch
shoulder-pelvis XY
```

## 弱

```text
foot placement
pseudo-contact
shoulder-pelvis yaw
```

弱项必须允许机器人为了：

- balance；
- morphology；
- payload；
- perturbation recovery；

主动偏离 human nominal motion。

---

# 11. Recovery-Aware Gating

最终建议：

\[
r=
r_{task}
+r_{physical}
+r_{stability}
+
g_{task}
g_{recovery}
\lambda_{style}
r_{style}
\]

正常：

\[
g_{recovery}pprox1
\]

失稳：

\[
g_{recovery}ightarrow0
\]

此时允许机器人：

```text
step
change stance
raise pelvis
shift torso
```

完成恢复。

Human descriptor 表达的是 nominal behavior，不是硬 safety constraint。

---

# 12. 推荐实现顺序

## V1

先实现最核心的几类：

```text
1. pelvis height
2. effective leg length L/R
3. knee-ground distance L/R
4. torso pitch
5. shoulder-pelvis XY
6. foot relative pelvis XY
```

## V1.1

再加入：

```text
foot pseudo-contact
knee pseudo-contact
shoulder-pelvis yaw
```

形成完整 17D。

## V2

加入：

\[
\dot h_p,\dot c_L,\dot c_R
\]

形成 direct-reward 20D。

## V3

使用：

\[
10	imes17=170D
\]

训练 Conditional Cross-Embodiment Motion Prior。

---

# 13. 离线验证优先于接入 PPO

建议先对 EgoSuite skeleton 批量计算 17D descriptor，检查是否能稳定区分：

```text
stand
bend
half squat
deep squat
kneel_L
kneel_R
step
```

建议验证：

- descriptor 分布；
- PCA / UMAP；
- 简单 classifier；
- 不同采集员的一致性；
- 身高归一化前后对比；
- skeleton noise sensitivity；
- missing joint robustness。

如果 descriptor 本身不能把这些动作区分开，就不应该急着接入 PPO。

---

# 14. 推荐优先级

如果只能先实现少量项：

\[
oxed{
	ext{Pelvis Height}
>
	ext{Leg Extension}
>
	ext{Knee-Ground Distance}
>
	ext{Shoulder-Pelvis Relation}
>
	ext{Torso Pitch}
>
	ext{Foot Placement}
>
	ext{Contact}
}
\]

前五类已经足以较好地区分：

```text
stand
bend
half squat
deep squat
kneel_L
kneel_R
```

Foot / contact 主要进一步帮助：

```text
walking
stepping
kneeling support
perturbation recovery
```

---

# 15. 一句话定义

> Cross-Embodiment Descriptor 不把人体关节角映射到机器人关节角，而是用尺度归一化的 pelvis height、effective leg length、knee-ground relation、torso posture、shoulder–pelvis relation、foot placement 和 support topology 描述人体如何组织全身完成任务，再让 physics-aware RL 根据 G1 自身动力学寻找具有相似运动策略的机器人动作。
