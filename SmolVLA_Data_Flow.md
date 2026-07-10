# SmolVLA 数据流动过程

本文按 **LeRobot `SmolVLAConfig` 默认参数 + SmolVLM2-500M 配置**整理 SmolVLA 的数据流动过程。

为了让矩阵维度具体可读，本文使用一个默认推理样例：

| 参数 | 数值 |
|---|---:|
| batch size | $1$ |
| 相机 / 图像数量 | $1$ |
| 图像尺寸 | $512 \times 512$ |
| 文本长度 | padding 到 $48$ tokens |
| state 维度 | padding 到 $32$ |
| action 维度 | padding 到 $32$ |
| action chunk size | $50$ |

---

## 1. 默认参数

### SmolVLA 默认配置

| 参数 | 数值 | 含义 |
|---|---:|---|
| `n_obs_steps` | $1$ | 使用 $1$ 步历史观测 |
| `chunk_size` | $50$ | 一次输出 $50$ 步动作 |
| `n_action_steps` | $50$ | 执行 $50$ 步动作 |
| `max_state_dim` | $32$ | robot state padding 后最大维度 |
| `max_action_dim` | $32$ | action padding 后最大维度 |
| `resize_imgs_with_padding` | $(512, 512)$ | 输入图像 resize 后尺寸 |
| `tokenizer_max_length` | $48$ | 文本 token 最大长度 |
| `num_steps` | $10$ | flow matching 推理步数 |
| `num_vlm_layers` | $16$ | 使用 SmolVLM2 前 $16$ 层 |
| `expert_width_multiplier` | $0.75$ | action expert 宽度系数 |

### SmolVLM2-500M 配置

| 参数 | 数值 | 含义 |
|---|---:|---|
| SigLIP vision hidden size | $768$ | SigLIP 视觉特征维度 |
| image size | $512$ | 输入图像边长 |
| patch size | $16$ | ViT patch 边长 |
| raw patch tokens | $1024$ | 原始 patch token 数 |
| pixel shuffle factor | $4$ | token 空间压缩倍数 |
| visual tokens after reduction | $64$ | 压缩后的视觉 token 数 |
| SmolLM2 hidden size | $960$ | VLM / language hidden size |
| SmolLM2 total layers | $32$ | SmolLM2 总层数 |
| SmolVLA used layers | $16$ | SmolVLA 使用前 $16$ 层 |
| action expert hidden size | $720$ | $0.75 \times 960$ |

关键维度计算：

$$
\frac{512}{16} = 32
$$

$$
32 \times 32 = 1024
$$

$$
\frac{1024}{4^2} = 64
$$

$$
768 \times 4^2 = 12288
$$

$$
960 \times 0.75 = 720
$$

---

## 2. 总体结构

SmolVLA 的整体数据流可以概括为：

```text
图像
  ↓
SigLIP vision encoder
  ↓
visual tokens
  ↓
SmolLM2 / SmolVLM2 前 16 层
  ↓
multimodal features
  ↓
action expert
  ↓
flow matching
  ↓
continuous action chunk
```

模块关系为：

$$
\text{SmolVLA}
=
\text{SigLIP vision encoder}
+
\text{SmolLM2 decoder}
+
\text{action expert}
$$

其中：

- SigLIP 负责把图像编码成视觉 token；
- SmolLM2 负责融合文本、视觉、机器人状态；
- action expert 使用 flow matching 输出连续动作 chunk。

---

## 3. 输入矩阵

默认推理样例包含三类输入：

| 输入 | 矩阵 | 形状 |
|---|---|---|
| 图像 | $\mathbf{I}$ | $\mathbb{R}^{1 \times 3 \times 512 \times 512}$ |
| 文本 token ids | $\mathbf{x}$ | $\mathbb{N}^{1 \times 48}$ |
| robot state | $\mathbf{s}$ | $\mathbb{R}^{1 \times 32}$ |

也就是：

$$
\mathbf{I}
\in
\mathbb{R}^{1 \times 3 \times 512 \times 512}
$$

$$
\mathbf{x}
\in
\mathbb{N}^{1 \times 48}
$$

$$
\mathbf{s}
\in
\mathbb{R}^{1 \times 32}
$$

---

## 4. SigLIP 视觉编码矩阵流

### 4.1 图像切分为 patch

输入图像尺寸为 $512 \times 512$，patch size 为 $16$。

每个方向的 patch 数为：

$$
\frac{512}{16} = 32
$$

因此总 patch 数为：

$$
32 \times 32 = 1024
$$

SigLIP raw patch features 为：

$$
\mathbf{P}
\in
\mathbb{R}^{1 \times 1024 \times 768}
$$

其中：

- $1$ 是 batch size；
- $1024$ 是 raw patch token 数；
- $768$ 是 SigLIP vision hidden size。

### 4.2 Pixel Shuffle 降低视觉 token 数

SmolVLA 使用 pixel shuffle factor $4$。

token 数从 $1024$ 降到：

$$
\frac{1024}{4^2} = 64
$$

hidden 维度从 $768$ 扩展为：

$$
768 \times 4^2 = 12288
$$

因此 pixel shuffle 后视觉特征为：

$$
\mathbf{V}_{ps}
\in
\mathbb{R}^{1 \times 64 \times 12288}
$$

### 4.3 Projector 映射到 SmolLM2 hidden size

pixel shuffle 后的视觉特征需要投影到 SmolLM2 的 hidden size $960$：

$$
\mathbf{V}
=
\mathbf{V}_{ps}\mathbf{W}_{v}
+
\mathbf{b}_{v}
$$

其中：

$$
\mathbf{W}_{v}
\in
\mathbb{R}^{12288 \times 960}
$$

得到 projector 后 visual tokens：

$$
\mathbf{V}
\in
\mathbb{R}^{1 \times 64 \times 960}
$$

---

## 5. SmolLM2 文本与多模态序列

文本 instruction 被 tokenizer padding 到 $48$ 个 token：

$$
\mathbf{x}
\in
\mathbb{N}^{1 \times 48}
$$

经过 SmolLM2 token embedding：

$$
\mathbf{E}_{\text{text}}
=
\operatorname{Embed}(\mathbf{x})
$$

得到：

$$
\mathbf{E}_{\text{text}}
\in
\mathbb{R}^{1 \times 48 \times 960}
$$

其中：

- $48$ 是文本 token 长度；
- $960$ 是 SmolLM2 hidden size。

---

## 6. State Projector

robot state 被 padding 到 $32$ 维：

$$
\mathbf{s}
\in
\mathbb{R}^{1 \times 32}
$$

通过 state projector 映射成 $1$ 个 VLM token：

$$
\mathbf{E}_{\text{state}}
=
\mathbf{s}\mathbf{W}_{s}
+
\mathbf{b}_{s}
$$

其中：

$$
\mathbf{W}_{s}
\in
\mathbb{R}^{32 \times 960}
$$

得到：

$$
\mathbf{E}_{\text{state}}
\in
\mathbb{R}^{1 \times 1 \times 960}
$$

---

## 7. Action Expert 与 Flow Matching

### 7.1 构造 VLM 输入序列

SmolVLA 将文本 token、视觉 token 和 state token 拼接：

$$
\mathbf{X}_{\text{vlm}}
=
[
\mathbf{E}_{\text{text}};
\mathbf{V};
\mathbf{E}_{\text{state}}
]
$$

序列长度为：

$$
48 + 64 + 1 = 113
$$

因此：

$$
\mathbf{X}_{\text{vlm}}
\in
\mathbb{R}^{1 \times 113 \times 960}
$$

### 7.2 SmolVLM2 前 16 层输出特征

SmolVLA 使用 SmolVLM2 / SmolLM2 的前 $16$ 层作为 VLM backbone：

$$
\mathbf{F}_{\text{vlm}}
=
f_{\text{VLM}}^{(16)}(\mathbf{X}_{\text{vlm}})
$$

输出形状保持为：

$$
\mathbf{F}_{\text{vlm}}
\in
\mathbb{R}^{1 \times 113 \times 960}
$$

### 7.3 投影为 action expert memory

action expert hidden size 为：

$$
0.75 \times 960 = 720
$$

VLM 特征被投影到 action expert hidden space：

$$
\mathbf{F}_{e}
=
\mathbf{F}_{\text{vlm}}\mathbf{W}_{e}
+
\mathbf{b}_{e}
$$

其中：

$$
\mathbf{W}_{e}
\in
\mathbb{R}^{960 \times 720}
$$

得到 expert memory：

$$
\mathbf{F}_{e}
\in
\mathbb{R}^{1 \times 113 \times 720}
$$

### 7.4 Action chunk 初始化

SmolVLA 输出连续动作 chunk。

默认 action chunk size 为 $50$，action 维度为 $32$：

$$
\mathbf{a}_{t}
\in
\mathbb{R}^{1 \times 50 \times 32}
$$

训练时，$\mathbf{a}_{t}$ 是由真实动作 chunk 和噪声插值得到的中间状态。

推理时，$\mathbf{a}_{t}$ 从噪声初始化，并通过 flow matching 逐步更新。

### 7.5 Action tokens

action chunk 被投影成 action expert token：

$$
\mathbf{Q}_{a}
=
\mathbf{a}_{t}\mathbf{W}_{a}
+
\mathbf{b}_{a}
$$

其中：

$$
\mathbf{W}_{a}
\in
\mathbb{R}^{32 \times 720}
$$

得到：

$$
\mathbf{Q}_{a}
\in
\mathbb{R}^{1 \times 50 \times 720}
$$

### 7.6 Cross-Attention 融合 VLM 信息

在 action expert 中，action tokens 作为 query：

$$
\mathbf{Q}
\in
\mathbb{R}^{1 \times 50 \times 720}
$$

expert memory 作为 key / value：

$$
\mathbf{K}, \mathbf{V}
\in
\mathbb{R}^{1 \times 113 \times 720}
$$

cross-attention 的权重矩阵形状为：

$$
\mathbf{Q}\mathbf{K}^{\top}
\in
\mathbb{R}^{1 \times 50 \times 113}
$$

attention 输出为：

$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})
\in
\mathbb{R}^{1 \times 50 \times 720}
$$

### 7.7 输出 vector field

action expert 输出 flow matching 的 vector field：

$$
\mathbf{v}_{\theta}
=
F_{\theta}
(
\mathbf{a}_{t},
t,
\mathbf{F}_{e}
)
$$

形状为：

$$
\mathbf{v}_{\theta}
\in
\mathbb{R}^{1 \times 50 \times 32}
$$

### 7.8 Flow Matching 更新动作

推理时默认使用 $10$ 个 flow matching steps。

每一步根据 vector field 更新动作：

$$
\mathbf{a}_{t+\Delta t}
=
\mathbf{a}_{t}
+
\Delta t \cdot \mathbf{v}_{\theta}
$$

经过 $10$ 步更新后得到最终动作 chunk：

$$
\hat{\mathbf{a}}
\in
\mathbb{R}^{1 \times 50 \times 32}
$$

---

## 8. 总体矩阵流表

| 阶段 | 张量 | 形状 |
|---|---|---|
| 输入图像 | $\mathbf{I}$ | $\mathbb{R}^{1 \times 3 \times 512 \times 512}$ |
| 文本 token ids | $\mathbf{x}$ | $\mathbb{N}^{1 \times 48}$ |
| robot state | $\mathbf{s}$ | $\mathbb{R}^{1 \times 32}$ |
| SigLIP raw patch features | $\mathbf{P}$ | $\mathbb{R}^{1 \times 1024 \times 768}$ |
| pixel shuffle 后视觉特征 | $\mathbf{V}_{ps}$ | $\mathbb{R}^{1 \times 64 \times 12288}$ |
| projector 后 visual tokens | $\mathbf{V}$ | $\mathbb{R}^{1 \times 64 \times 960}$ |
| 文本 token embeddings | $\mathbf{E}_{\text{text}}$ | $\mathbb{R}^{1 \times 48 \times 960}$ |
| state token | $\mathbf{E}_{\text{state}}$ | $\mathbb{R}^{1 \times 1 \times 960}$ |
| VLM 输入序列 | $\mathbf{X}_{\text{vlm}}$ | $\mathbb{R}^{1 \times 113 \times 960}$ |
| VLM 输出特征 | $\mathbf{F}_{\text{vlm}}$ | $\mathbb{R}^{1 \times 113 \times 960}$ |
| expert memory | $\mathbf{F}_{e}$ | $\mathbb{R}^{1 \times 113 \times 720}$ |
| 噪声 / 中间 action chunk | $\mathbf{a}_{t}$ | $\mathbb{R}^{1 \times 50 \times 32}$ |
| action tokens | $\mathbf{Q}_{a}$ | $\mathbb{R}^{1 \times 50 \times 720}$ |
| cross-attention score | $\mathbf{Q}\mathbf{K}^{\top}$ | $\mathbb{R}^{1 \times 50 \times 113}$ |
| vector field | $\mathbf{v}_{\theta}$ | $\mathbb{R}^{1 \times 50 \times 32}$ |
| final action chunk | $\hat{\mathbf{a}}$ | $\mathbb{R}^{1 \times 50 \times 32}$ |

整体链路可以压缩表示为：

$$
\mathbf{I}
\rightarrow
\mathbf{P}
\rightarrow
\mathbf{V}_{ps}
\rightarrow
\mathbf{V}
\rightarrow
\mathbf{X}_{\text{vlm}}
\rightarrow
\mathbf{F}_{\text{vlm}}
\rightarrow
\mathbf{F}_{e}
\rightarrow
\mathbf{v}_{\theta}
\rightarrow
\hat{\mathbf{a}}
$$

对应形状为：

$$
\mathbb{R}^{1 \times 3 \times 512 \times 512}
\rightarrow
\mathbb{R}^{1 \times 1024 \times 768}
\rightarrow
\mathbb{R}^{1 \times 64 \times 12288}
\rightarrow
\mathbb{R}^{1 \times 64 \times 960}
\rightarrow
\mathbb{R}^{1 \times 113 \times 960}
\rightarrow
\mathbb{R}^{1 \times 113 \times 960}
\rightarrow
\mathbb{R}^{1 \times 113 \times 720}
\rightarrow
\mathbb{R}^{1 \times 50 \times 32}
\rightarrow
\mathbb{R}^{1 \times 50 \times 32}
$$

---

## 9. 多相机扩展说明

上文使用的是单相机默认样例。

如果实际输入有 $C$ 个相机 / 图像，则每个相机仍产生 $64$ 个 visual tokens：

$$
\mathbf{V}
\in
\mathbb{R}^{1 \times 64C \times 960}
$$

VLM 输入序列长度变为：

$$
48 + 64C + 1
$$

因此：

$$
\mathbf{X}_{\text{vlm}}
\in
\mathbb{R}^{1 \times (48 + 64C + 1) \times 960}
$$

也就是：

$$
\mathbf{X}_{\text{vlm}}
\in
\mathbb{R}^{1 \times (49 + 64C) \times 960}
$$

对应的 expert memory 为：

$$
\mathbf{F}_{e}
\in
\mathbb{R}^{1 \times (49 + 64C) \times 720}
$$

action 侧不随相机数变化，仍为：

$$
\hat{\mathbf{a}}
\in
\mathbb{R}^{1 \times 50 \times 32}
$$

---

## 10. 维度一致性检查

本文使用的关键维度满足：

$$
48 + 64 + 1 = 113
$$

$$
\frac{512}{16} = 32
$$

$$
32 \times 32 = 1024
$$

$$
\frac{1024}{4^2} = 64
$$

$$
768 \times 4^2 = 12288
$$

$$
960 \times 0.75 = 720
$$

因此，默认单相机样例的数据流为：

```text
image:        1 x 3 x 512 x 512
patches:      1 x 1024 x 768
visual:       1 x 64 x 960
text:         1 x 48 x 960
state:        1 x 1 x 960
vlm input:    1 x 113 x 960
expert memory:1 x 113 x 720
action tokens:1 x 50 x 720
output:       1 x 50 x 32
```
