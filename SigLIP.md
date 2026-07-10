# SigLIP

## 1. 背景

图文对比学习的目标是把图片和文本映射到同一个语义向量空间中，使匹配的图文对更接近，不匹配的图文对更远。

给定一个 batch 中的 $N$ 个图文对：

$$
\{(I_1, T_1), (I_2, T_2), \dots, (I_N, T_N)\}
$$

其中：

- $I_i$ 表示第 $i$ 张图片；
- $T_i$ 表示第 $i$ 条文本；
- $(I_i, T_i)$ 是正样本对；
- $(I_i, T_j), i \ne j$ 是负样本对。

图像编码器和文本编码器分别将图片和文本映射为向量：

$$
\mathbf{v}_i = f_{\text{image}}(I_i)
$$

$$
\mathbf{u}_j = f_{\text{text}}(T_j)
$$

通常会对向量做归一化：

$$
\|\mathbf{v}_i\|_2 = 1, \quad \|\mathbf{u}_j\|_2 = 1
$$

图文相似度一般使用点积或余弦相似度：

$$
s_{ij} = \mathbf{v}_i^\top \mathbf{u}_j
$$

所有图片和文本两两计算相似度后，可以得到一个 $N \times N$ 的相似度矩阵：

$$
S =
\begin{bmatrix}
s_{11} & s_{12} & \cdots & s_{1N} \\
s_{21} & s_{22} & \cdots & s_{2N} \\
\vdots & \vdots & \ddots & \vdots \\
s_{N1} & s_{N2} & \cdots & s_{NN}
\end{bmatrix}
$$

其中，对角线元素 $s_{ii}$ 是正样本相似度，非对角线元素 $s_{ij}, i \ne j$ 是负样本相似度。

---

## 2. CLIP 与 SigLIP 的时间关系

CLIP 和 SigLIP 的时间关系是：

| 时间 | 模型 / 方法 | 关系 |
|---|---|---|
| 2021 年 | CLIP | OpenAI 提出，使用 softmax contrastive loss 做图文对比学习 |
| 2023 年 | SigLIP | Google Research 提出，在 CLIP 式双塔图文模型基础上，将 softmax loss 改为 sigmoid loss |

更具体地说：

- CLIP 更早，论文 *Learning Transferable Visual Models From Natural Language Supervision* 于 2021 年提交到 arXiv。
- SigLIP 更晚，论文 *Sigmoid Loss for Language Image Pre-Training* 于 2023 年提交到 arXiv。

因此，二者的发展关系可以概括为：

$$
\text{CLIP}
\quad \rightarrow \quad
\text{CLIP 成为图文对比学习的重要基线}
\quad \rightarrow \quad
\text{SigLIP 改进 CLIP 的训练损失}
$$

SigLIP 不是推翻 CLIP，而是在 CLIP 式图文双塔结构基础上改进训练目标：

$$
\text{SigLIP}
=
\text{CLIP 式图文双塔结构}
+
\text{pairwise sigmoid loss}
$$

---

## 3. CLIP 与 Softmax Contrastive Loss

Softmax contrastive loss 是 CLIP 使用的核心训练目标，因此理解 SigLIP 前通常需要先理解 CLIP 的训练方式。CLIP 把图文匹配问题转化为 batch 内分类问题。

对于每张图片 $I_i$，模型需要从当前 batch 的所有文本中选出正确文本 $T_i$。

对于每条文本 $T_i$，模型需要从当前 batch 的所有图片中选出正确图片 $I_i$。

因此，CLIP 的损失包含两个方向：

- Image-to-Text loss；
- Text-to-Image loss。

---

## 4. CLIP 的 Image-to-Text Loss

对第 $i$ 张图片 $I_i$，其与所有文本的相似度为：

$$
s_{i1}, s_{i2}, \dots, s_{iN}
$$

使用 softmax 得到文本 $T_j$ 与图片 $I_i$ 匹配的概率：

$$
P(T_j \mid I_i)
=
\frac{\exp(s_{ij} / \tau)}
{\sum_{k=1}^{N} \exp(s_{ik} / \tau)}
$$

其中，$\tau$ 是 temperature 参数，用于控制 softmax 分布的尖锐程度。

正确文本是 $T_i$，因此 image-to-text loss 为：

$$
\mathcal{L}_{\text{i2t}}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{\exp(s_{ii} / \tau)}
{\sum_{j=1}^{N} \exp(s_{ij} / \tau)}
$$

该损失会推动：

$$
s_{ii} \uparrow
$$

$$
s_{ij} \downarrow, \quad i \ne j
$$

也就是说，对于每张图片，正确文本的相似度应高于其他文本。

---

## 5. CLIP 的 Text-to-Image Loss

对第 $i$ 条文本 $T_i$，模型需要从所有图片中选出正确图片 $I_i$。

图片 $I_j$ 与文本 $T_i$ 匹配的概率为：

$$
P(I_j \mid T_i)
=
\frac{\exp(s_{ji} / \tau)}
{\sum_{k=1}^{N} \exp(s_{ki} / \tau)}
$$

对应的 text-to-image loss 为：

$$
\mathcal{L}_{\text{t2i}}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{\exp(s_{ii} / \tau)}
{\sum_{j=1}^{N} \exp(s_{ji} / \tau)}
$$

该损失会推动每条文本与其正确图片的相似度高于其他图片。

---

## 6. CLIP 的最终损失

CLIP 通常将两个方向的损失取平均：

$$
\mathcal{L}_{\text{CLIP}}
=
\frac{1}{2}
\left(
\mathcal{L}_{\text{i2t}}
+
\mathcal{L}_{\text{t2i}}
\right)
$$

因此，CLIP 的 softmax contrastive loss 是一个对称的对比学习损失：

$$
I \rightarrow T
$$

$$
T \rightarrow I
$$

它的本质是：

> 在 batch 内进行多分类，让每张图片选出正确文本，也让每条文本选出正确图片。

---

## 7. SigLIP 的核心思想

SigLIP 的全称是 Sigmoid Loss for Language Image Pre-Training。

它和 CLIP 的模型结构相似，也使用图像编码器和文本编码器学习共享语义空间。核心区别在于训练损失函数。

CLIP 使用 softmax contrastive loss，把图文匹配看作 batch 内的多分类问题。

SigLIP 使用 sigmoid loss，把每一个图文 pair 看作独立的二分类问题。

也就是说，SigLIP 不再问：

> 这张图片在当前 batch 的所有文本中最匹配哪一条？

而是问：

> 这张图片和这条文本是否匹配？

---

## 8. SigLIP 的 Pairwise Sigmoid Loss

SigLIP 对每一个图文 pair $(I_i, T_j)$ 分配标签：

$$
z_{ij}
=
\begin{cases}
+1, & i = j \\
-1, & i \ne j
\end{cases}
$$

其中：

- $z_{ij} = +1$ 表示正样本；
- $z_{ij} = -1$ 表示负样本。

SigLIP 首先计算图文 pair 的 logit：

$$
\ell_{ij}
=
t \cdot \mathbf{v}_i^\top \mathbf{u}_j + b
$$

其中：

- $\ell_{ij}$ 表示第 $i$ 张图片和第 $j$ 条文本的匹配 logit；
- $t$ 是可学习的 scale 参数，作用类似 CLIP 中 temperature 的倒数；
- $b$ 是可学习的 bias 参数；
- $\mathbf{v}_i^\top \mathbf{u}_j$ 是图文向量相似度。

SigLIP 的损失函数为：

$$
\mathcal{L}_{\text{SigLIP}}
=
\sum_{i=1}^{N}
\sum_{j=1}^{N}
\log
\left(
1 + \exp(-z_{ij} \ell_{ij})
\right)
$$

也可以写成 softplus 形式：

$$
\mathcal{L}_{\text{SigLIP}}
=
\sum_{i=1}^{N}
\sum_{j=1}^{N}
\operatorname{softplus}(-z_{ij} \ell_{ij})
$$

其中：

$$
\operatorname{softplus}(x)
=
\log(1 + \exp(x))
$$

还可以写成 binary logistic loss 的形式：

$$
\mathcal{L}_{\text{SigLIP}}
=
-
\sum_{i=1}^{N}
\sum_{j=1}^{N}
\log
\sigma(z_{ij} \ell_{ij})
$$

其中 sigmoid 函数为：

$$
\sigma(x)
=
\frac{1}{1 + \exp(-x)}
$$

---

## 9. SigLIP 损失的直观理解

对于正样本 $(I_i, T_i)$：

$$
z_{ii} = +1
$$

损失项为：

$$
\log(1 + \exp(-\ell_{ii}))
$$

为了降低损失，模型需要让：

$$
\ell_{ii} \uparrow
$$

即提高正确图文对的匹配分数。

对于负样本 $(I_i, T_j), i \ne j$：

$$
z_{ij} = -1
$$

损失项为：

$$
\log(1 + \exp(\ell_{ij}))
$$

为了降低损失，模型需要让：

$$
\ell_{ij} \downarrow
$$

即降低错误图文对的匹配分数。

所以 SigLIP 的训练目标可以概括为：

$$
\ell_{ii} \text{ 越大越好}
$$

$$
\ell_{ij} \text{ 越小越好}, \quad i \ne j
$$

---

## 10. SigLIP 的梯度直觉

单个 pair 的损失为：

$$
\mathcal{L}_{ij}
=
\operatorname{softplus}(-z_{ij}\ell_{ij})
$$

对 logit $\ell_{ij}$ 求导：

$$
\frac{\partial \mathcal{L}_{ij}}{\partial \ell_{ij}}
=
-z_{ij} \cdot \sigma(-z_{ij}\ell_{ij})
$$

对于正样本：

$$
z_{ij} = +1
$$

$$
\frac{\partial \mathcal{L}_{ij}}{\partial \ell_{ij}}
=
-\sigma(-\ell_{ij})
$$

当 $\ell_{ij}$ 不够大时，梯度会推动 $\ell_{ij}$ 增大。

对于负样本：

$$
z_{ij} = -1
$$

$$
\frac{\partial \mathcal{L}_{ij}}{\partial \ell_{ij}}
=
\sigma(\ell_{ij})
$$

当 $\ell_{ij}$ 过大时，梯度会推动 $\ell_{ij}$ 减小。

---

## 11. Bias 参数的作用

在一个 batch 中：

- 正样本数量为 $N$；
- 负样本数量为 $N(N - 1)$。

负样本数量远多于正样本数量。

如果没有 bias 参数，模型容易受到大量负样本主导，使整体 logit 倾向于偏小。SigLIP 引入可学习 bias $b$，用于调整二分类决策阈值，使正负样本比例不平衡时训练更加稳定。

SigLIP 的 logit 为：

$$
\ell_{ij}
=
t \cdot \mathbf{v}_i^\top \mathbf{u}_j + b
$$

其中 $b$ 的作用是整体平移 logit 分布。

---

## 12. CLIP 与 SigLIP 的区别

| 对比项 | CLIP | SigLIP |
|---|---|---|
| 训练视角 | batch 内多分类 | pairwise 二分类 |
| 目标 | 从候选集合中选正确匹配 | 判断每个图文 pair 是否匹配 |
| 正样本 | 对角线 $s_{ii}$ | 对角线 $(I_i, T_i)$ |
| 负样本 | softmax denominator 中的其他样本 | 每个非匹配 pair 独立作为负样本 |
| 损失形式 | cross entropy over softmax | binary logistic loss over sigmoid |
| 是否依赖全局归一化 | 是 | 否 |
| batch size 依赖 | 较强 | 相对较弱 |
| 分布式训练通信 | 通常需要全局 batch logits 或 embedding | 更容易解耦 |
| 推理方式 | 使用图文相似度排序 | 使用图文相似度或 logit 排序 |

---

## 13. 本质区别

CLIP 的 softmax contrastive loss 关心的是相对排序：

$$
P(T_i \mid I_i)
=
\frac{\exp(s_{ii} / \tau)}
{\sum_{j=1}^{N}\exp(s_{ij} / \tau)}
$$

它要求正确文本在当前 batch 的所有文本中概率最高。

SigLIP 的 sigmoid loss 关心的是独立匹配判断：

$$
\sigma(z_{ij}\ell_{ij})
$$

它要求每一个图文 pair 都被独立判断为匹配或不匹配。

因此，两者可以简洁概括为：

$$
\text{CLIP}:
\quad
\text{多分类，选正确项}
$$

$$
\text{SigLIP}:
\quad
\text{二分类，判断每对是否匹配}
$$

---

## 14. 推理阶段

训练完成后，SigLIP 和 CLIP 的使用方式基本一致。

### 零样本分类

给定一张图片 $I$ 和一组类别文本 prompt：

$$
\{T_1, T_2, \dots, T_C\}
$$

分别计算：

$$
s_j = \mathbf{v}^\top \mathbf{u}_j
$$

选择相似度最高的类别：

$$
\hat{y}
=
\arg\max_j s_j
$$

### 图文检索

给定文本查询 $T$，计算其与所有图片的相似度：

$$
s_i = \mathbf{v}_i^\top \mathbf{u}
$$

然后按照 $s_i$ 从高到低排序，即可得到最相关的图片。

---

## 15. 总结

CLIP 和 SigLIP 都用于学习图文共享语义空间，但它们的训练目标不同。

CLIP 使用 softmax contrastive loss，将图文匹配建模为 batch 内分类问题：

$$
\text{在 } N \text{ 个候选中选出正确匹配}
$$

SigLIP 将图文匹配建模为 pairwise 二分类问题：

$$
\text{判断每个图文 pair 是否匹配}
$$

SigLIP 的关键优势在于不依赖 softmax 的全局归一化，训练时更容易扩展到大规模分布式场景，并且对 batch size 的依赖相对更弱。

核心公式可以记为：

$$
\mathcal{L}_{\text{CLIP}}
=
\frac{1}{2}
\left(
\mathcal{L}_{\text{i2t}}
+
\mathcal{L}_{\text{t2i}}
\right)
$$

$$
\mathcal{L}_{\text{SigLIP}}
=
\sum_{i=1}^{N}
\sum_{j=1}^{N}
\operatorname{softplus}(-z_{ij}\ell_{ij})
$$

其中：

$$
z_{ij}
=
\begin{cases}
+1, & i = j \\
-1, & i \ne j
\end{cases}
$$

$$
\ell_{ij}
=
t \cdot \mathbf{v}_i^\top \mathbf{u}_j + b
$$
