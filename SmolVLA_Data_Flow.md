# SmolVLA 数据流动过程

本文按 **LeRobot `SmolVLAConfig` 默认参数 + SmolVLM2-500M 配置**整理 SmolVLA 的数据流动过程。

为了让矩阵维度具体可读，本文使用一个单相机推理样例：

| 参数 | 数值 |
|---|---:|
| batch size | $1$ |
| 相机 / 图像数量 | $1$ |
| 图像尺寸 | $512 \times 512$ |
| 文本长度 | 记为 $L$ tokens，默认 $L \le 48$ |
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
| `tokenizer_max_length` | $48$ | 文本 token 截断上限 |
| `pad_language_to` | `"longest"` | padding 到当前 batch 内最长文本，不会默认固定补到 $48$ |
| `num_steps` | $10$ | flow matching 推理步数 |
| `num_vlm_layers` | $16$ | 使用 SmolVLM2 前 $16$ 层 |
| `expert_width_multiplier` | $0.75$ | action expert 宽度系数 |

### SmolVLM2-500M 配置

| 参数                                  |     数值 | 含义                         |
| ----------------------------------- | -----: | -------------------------- |
| [CLIP 与 SigLIP](CLIP%20与%20SigLIP.md) vision hidden size |  $768$ | SigLIP 视觉特征维度              |
| image size                          |  $512$ | 输入图像边长                     |
| patch size                          |   $16$ | ViT patch 边长               |
| raw patch tokens                    | $1024$ | 原始 patch token 数           |
| pixel shuffle factor                |    $4$ | token 空间压缩倍数               |
| visual tokens after reduction       |   $64$ | 压缩后的视觉 token 数             |
| SmolLM2 hidden size                 |  $960$ | VLM / language hidden size |
| SmolLM2 total layers                |   $32$ | SmolLM2 总层数                |
| SmolVLA used layers                 |   $16$ | SmolVLA 使用前 $16$ 层         |
| action expert hidden size           |  $720$ | $0.75 \times 960$          |

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
图像 → SigLIP → visual tokens ─┐
文本 → token embedding        ├→ prefix → VLM 逐层 KV cache ─┐
state → state projector       ─┘                              │
                                                                  ├→ action expert
噪声 action + timestep → action/time MLP → suffix ───────────────┘
                                                                  ↓
                                                             vector field
                                                                  ↓
                                                      10 步 reverse-time Euler
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
- SmolLM2 前 $16$ 层处理由图像、文本和 state 组成的 prefix，并产生逐层 KV cache；
- action expert 将噪声 action 与 timestep 编码成 suffix，通过交替的 attention 读取 prefix；
- flow matching 通过 reverse-time Euler 积分生成连续动作 chunk。

---

## 3. 输入矩阵

默认推理样例包含三类输入：

| 输入 | 矩阵 | 形状 |
|---|---|---|
| 图像 | $\mathbf{I}$ | $\mathbb{R}^{1 \times 3 \times 512 \times 512}$ |
| 文本 token ids | $\mathbf{x}$ | $\mathbb{N}^{1 \times L}$，$L\le48$ |
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
\mathbb{N}^{1 \times L},
\qquad L\le48
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
$$

其中：

$$
\mathbf{W}_{v}
\in
\mathbb{R}^{12288 \times 960}
$$

当前 SmolVLM connector 使用无 bias 的线性投影。

得到 projector 后 visual tokens：

$$
\mathbf{V}
\in
\mathbb{R}^{1 \times 64 \times 960}
$$

---

## 5. SmolLM2 文本与多模态序列

文本 instruction 经 tokenizer 截断到最多 $48$ 个 token。默认
`pad_language_to="longest"`，因此序列长度是当前 batch 内的最长文本长度 $L$：

$$
\mathbf{x}
\in
\mathbb{N}^{B \times L},
\qquad L \le 48
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
\mathbb{R}^{B \times L \times 960}
$$

其中：

- $L$ 是当前 batch padding 后的文本 token 长度，且 $L \le 48$；
- $960$ 是 SmolLM2 hidden size。

只有将 `pad_language_to` 设为 `"max_length"` 时，才会固定得到
$L=48$。

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

### 7.1 构造 prefix 序列

`embed_prefix()` 的实际拼接顺序是所有图像、文本、state：

$$
\mathbf{X}_{\text{prefix}}
=
[
\mathbf{V}_1;
\ldots;
\mathbf{V}_C;
\mathbf{E}_{\text{text}};
\mathbf{E}_{\text{state}}
]
$$

默认 `add_image_special_tokens=False`，因此 prefix 长度为：

$$
P = 64C + L + 1
$$

得到：

$$
\mathbf{X}_{\text{prefix}}
\in
\mathbb{R}^{B \times P \times 960}
$$

对于单相机且显式固定 $L=48$ 的样例，$P=113$。但在当前默认
`padding="longest"` 下，$P$ 随 batch 内的实际文本长度变化。

attention block mask 使图像和文本不能关注后面的 state，而 state 可以关注
前面的图像和文本。

### 7.2 VLM prefix 与逐层 KV cache

SmolVLA 使用 SmolVLM2 / SmolLM2 前 $16$ 层处理 prefix。推理时会在每一层保存
prefix 的 key/value cache，供每个 denoising step 重复使用。

当前实现中不存在一个显式的
$\mathbf{W}_e\in\mathbb{R}^{960\times720}$ 将整个 VLM 输出投影成
`[B, P, 720]` 的 expert memory。action expert 而是逐层读取 VLM 的 KV cache，
并在 cross-attention 层中使用 expert 的 K/V projection 转换它们。

### 7.3 Action chunk 初始化

SmolVLA 输出连续动作 chunk。

默认 action chunk size 为 $50$，action 维度为 $32$：

$$
\mathbf{a}_{t}
\in
\mathbb{R}^{B \times 50 \times 32}
$$

训练时，$\mathbf{a}_{t}$ 是由真实动作 chunk 和噪声插值得到的中间状态。

推理时，$\mathbf{a}_{t}$ 从噪声初始化，并通过 flow matching 逐步更新。

### 7.4 Action 与 timestep tokens

action chunk 先被投影到 expert hidden size：

$$
\mathbf{E}_{a}
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

同时，标量 timestep $t$ 经 sine-cosine positional encoding 得到：

$$
\mathbf{E}_t
\in
\mathbb{R}^{B \times 50 \times 720}
$$

action embedding 和 timestep embedding 在 hidden 维度上拼接，再经过两层 MLP：

$$
\operatorname{Concat}_{-1}(\mathbf{E}_a,\mathbf{E}_t)
\in
\mathbb{R}^{B \times 50 \times 1440}
\rightarrow
\operatorname{Linear}
\rightarrow
\operatorname{SiLU}
\rightarrow
\operatorname{Linear}
$$

最终得到 action expert suffix：

$$
\mathbf{X}_{\text{suffix}}
\in
\mathbb{R}^{B \times 50 \times 720}
$$

### 7.5 Action expert 的交替 attention

默认 `attention_mode="cross_attn"` 且 `self_attn_every_n_layers=2`。因此
$16$ 层中交替使用两种 attention：

- 偶数层使用 action suffix self-attention，推理时 action query 同时读取
  prefix KV cache 和带 causal mask 的 action KV；
- 奇数层使用 cross-attention，action query 只读取 prefix KV cache。

对 SmolVLM2-500M，attention head 数为 $15$。若忽略 GQA 在内部对 K/V head
的扩展过程，attention score 形状为：

$$
\text{cross-attention score}
\in
\mathbb{R}^{B \times 15 \times 50 \times P}
$$

$$
\text{self-attention score}
\in
\mathbb{R}^{B \times 15 \times 50 \times (P+50)}
$$

### 7.6 输出 vector field

action expert 输出 flow matching 的 vector field：

$$
\mathbf{v}_{\theta}
=
F_{\theta}
(
\mathbf{a}_{t},
t,
\operatorname{KV}_{\text{prefix}}
)
$$

形状为：

$$
\mathbf{v}_{\theta}
\in
\mathbb{R}^{B \times 50 \times 32}
$$

### 7.7 Flow Matching 训练目标

训练时，数据 action 记为 $\mathbf{a}$，高斯噪声记为 $\boldsymbol{\epsilon}$。
当前实现使用：

$$
\mathbf{x}_t
=
t\boldsymbol{\epsilon}+(1-t)\mathbf{a}
$$

目标 vector field 为：

$$
\mathbf{u}_t
=
\boldsymbol{\epsilon}-\mathbf{a}
$$

优化目标是 $\mathbf{v}_\theta(\mathbf{x}_t,t)$ 与 $\mathbf{u}_t$ 之间的 MSE。

### 7.8 Flow Matching 推理更新

推理从 $t=1$ 的高斯噪声开始，向 $t=0$ 积分。默认使用
$10$ 个 Euler steps：

$$
\Delta t=-\frac{1}{10}
$$

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
\mathbb{R}^{B \times 50 \times 32}
$$

模型 wrapper 最后会将 $32$ 维 padding 裁剪回数据集的真实 action 维度。

---

## 8. 总体矩阵流表

| 阶段 | 张量 | 形状 |
|---|---|---|
| 输入图像 | $\mathbf{I}$ | $\mathbb{R}^{1 \times 3 \times 512 \times 512}$ |
| 文本 token ids | $\mathbf{x}$ | $\mathbb{N}^{1 \times L}$，$L\le48$ |
| robot state | $\mathbf{s}$ | $\mathbb{R}^{1 \times 32}$ |
| SigLIP raw patch features | $\mathbf{P}$ | $\mathbb{R}^{1 \times 1024 \times 768}$ |
| pixel shuffle 后视觉特征 | $\mathbf{V}_{ps}$ | $\mathbb{R}^{1 \times 64 \times 12288}$ |
| projector 后 visual tokens | $\mathbf{V}$ | $\mathbb{R}^{1 \times 64 \times 960}$ |
| 文本 token embeddings | $\mathbf{E}_{\text{text}}$ | $\mathbb{R}^{1 \times L \times 960}$ |
| state token | $\mathbf{E}_{\text{state}}$ | $\mathbb{R}^{1 \times 1 \times 960}$ |
| prefix embeddings | $\mathbf{X}_{\text{prefix}}$ | $\mathbb{R}^{1 \times P \times 960}$，$P=64+L+1$ |
| 逐层 prefix memory | $\operatorname{KV}_{\text{prefix}}$ | VLM 每层的 K/V cache |
| 噪声 / 中间 action chunk | $\mathbf{a}_{t}$ | $\mathbb{R}^{1 \times 50 \times 32}$ |
| action + timestep suffix | $\mathbf{X}_{\text{suffix}}$ | $\mathbb{R}^{1 \times 50 \times 720}$ |
| cross-attention score |  | $\mathbb{R}^{1 \times 15 \times 50 \times P}$ |
| self-attention score |  | $\mathbb{R}^{1 \times 15 \times 50 \times (P+50)}$ |
| vector field | $\mathbf{v}_{\theta}$ | $\mathbb{R}^{1 \times 50 \times 32}$ |
| final action chunk | $\hat{\mathbf{a}}$ | $\mathbb{R}^{1 \times 50 \times 32}$ |

整体链路可以压缩表示为两条输入分支：

$$
\mathbf{I}
\rightarrow
\mathbf{P}
\rightarrow
\mathbf{V}_{ps}
\rightarrow
\mathbf{V}
\rightarrow
\mathbf{X}_{\text{prefix}}
\rightarrow
\operatorname{KV}_{\text{prefix}}
$$

$$
(\mathbf{a}_t,t)
\rightarrow
\mathbf{X}_{\text{suffix}}
$$

两条分支在 action expert 中汇合：

$$
(\operatorname{KV}_{\text{prefix}},\mathbf{X}_{\text{suffix}})
\rightarrow
\mathbf{v}_{\theta}
\rightarrow
\hat{\mathbf{a}}
$$

关键形状为：

$$
\mathbb{R}^{1 \times 3 \times 512 \times 512}
\rightarrow
\mathbb{R}^{1 \times 1024 \times 768}
\rightarrow
\mathbb{R}^{1 \times 64 \times 12288}
\rightarrow
\mathbb{R}^{1 \times 64 \times 960}
\rightarrow
\mathbb{R}^{1 \times P \times 960}
\xrightarrow{\text{per-layer KV cache}}
\operatorname{KV}_{\text{prefix}}
$$

$$
\mathbb{R}^{1 \times 50 \times 32}
\xrightarrow{\text{action/time MLP}}
\mathbb{R}^{1 \times 50 \times 720}
\xrightarrow{\text{expert + output projection}}
\mathbb{R}^{1 \times 50 \times 32}
\xrightarrow{\text{Euler integration}}
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

prefix 序列长度变为：

$$
P = 64C + L + 1,
\qquad L\le48
$$

因此：

$$
\mathbf{X}_{\text{prefix}}
\in
\mathbb{R}^{1 \times (64C + L + 1) \times 960}
$$

如果显式使用 `padding="max_length"` 令 $L=48$，才可以简化为：

$$
\mathbf{X}_{\text{prefix}}
\in
\mathbb{R}^{1 \times (49 + 64C) \times 960}
$$

每层 KV cache 的序列长度也随 $P$ 变化；不存在一个独立的
`[1, P, 720]` expert-memory 张量。

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
P=64C+L+1
$$

对于单相机、显式固定 $L=48$ 的特例：

$$
P=64+48+1=113
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

因此，默认单相机数据流应表示为：

```text
image:        1 x 3 x 512 x 512
patches:      1 x 1024 x 768
visual:       1 x 64 x 960
text:         1 x L x 960, L <= 48
state:        1 x 1 x 960
prefix:       1 x P x 960, P = 64 + L + 1
prefix memory:per-layer KV cache
action suffix:1 x 50 x 720
output:       1 x 50 x 32
```

其中 action suffix 在 $16$ 层 expert 中交替执行 self-attention 和
prefix cross-attention，最后经 `action_out_proj` 生成 vector field。
