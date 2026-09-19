# lecture 13：生成模型（一）

## 生成模型

判别模型学习 $p(y|x)$，即给定输入预测标签。生成模型学习数据分布 $p(x)$，条件生成模型学习 $p(x|y)$。

学习数据分布后，可以生成新样本、检测异常数据，也可以学习没有标签的特征。

### 生成模型分类

| 类型 | 是否能计算 $p(x)$ | 例子 |
| --- | --- | --- |
| 显式密度 | 可以精确或近似计算 | 自回归模型、VAE |
| 隐式密度 | 不能直接计算，但可以采样 | GAN、扩散模型 |

## 最大似然估计

给定训练数据 $x^{(1)},\cdots,x^{(N)}$，选择使训练数据概率最大的参数：
$$
\theta^*=\arg\max_\theta\prod_{i=1}^{N}p_\theta(x^{(i)})
$$

通常转化为最大化对数似然：
$$
\theta^*=\arg\max_\theta\sum_{i=1}^{N}\log p_\theta(x^{(i)})
$$

训练时也可以最小化负对数似然。

## 自回归模型

将数据按顺序拆分，利用概率链式法则：
$$
p(x)=\prod_{i=1}^{D}p(x_i|x_1,\cdots,x_{i-1})
$$

模型预测下一个元素的条件分布。使用掩码Transformer时，训练可以并行计算各位置的损失；生成时仍要从前到后逐个采样。

对于图像，可以按从左到右、从上到下的顺序预测像素，例如PixelRNN和PixelCNN。也可以先将图像压缩为离散token，再使用Transformer生成。

优点：能够精确计算似然，训练稳定。

缺点：生成速度慢，而且图像的扫描顺序是人为规定的。

## 潜变量模型

引入低维潜变量 $z$ 表示数据中的隐藏因素：
$$
p_\theta(x)=\int p_\theta(x|z)p(z)\,dz
$$

先从简单先验 $p(z)$ 中采样，再根据 $p_\theta(x|z)$ 生成数据。但这个积分通常难以直接计算。

## 变分自编码器（VAE）

VAE包含两个网络：

- 编码器 $q_\phi(z|x)$：输入数据 $x$，输出潜变量的分布。
- 解码器 $p_\theta(x|z)$：输入潜变量 $z$，输出数据的分布。

编码器通常输出均值和标准差：
$$
q_\phi(z|x)=\mathcal{N}(\mu(x),\operatorname{diag}(\sigma^2(x)))
$$

### ELBO

由于 $p_\theta(x)$ 难以直接计算，VAE最大化其下界：
$$
\log p_\theta(x)\geq
\mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(x|z)]
-D_{KL}(q_\phi(z|x)\|p(z))
$$

- 重建项：要求解码结果接近输入。
- KL项：要求编码器输出接近先验 $p(z)=\mathcal{N}(0,I)$。

两项存在竞争：重建项希望不同输入容易区分，KL项希望潜空间接近统一的高斯分布。

### 重参数化

直接采样无法反向传播，因此将随机性移到独立变量 $\epsilon$：
$$
\epsilon\sim\mathcal{N}(0,I),\qquad z=\mu+\sigma\odot\epsilon
$$

这样梯度可以传到 $\mu$ 和 $\sigma$。

### 训练与生成

训练：
$$
x\rightarrow(\mu,\sigma)\rightarrow z\rightarrow\hat{x}
$$

生成时不需要编码器：
$$
z\sim\mathcal{N}(0,I)\rightarrow Decoder\rightarrow x
$$

VAE训练稳定，潜空间连续且容易插值，但生成结果往往比较模糊。
