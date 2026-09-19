# CS231n Assignment 2

本目录包含 CS231n Assignment 2 的 NumPy 与 PyTorch 实现。
作业重点从全连接网络扩展到归一化、正则化、卷积网络和循环神经网络。

## 学习目标

完成本作业后，应能够：

- 理解 Batch Normalization 和 Layer Normalization 的计算过程。
- 推导归一化层的反向传播，并实现简化梯度公式。
- 理解 Dropout 在训练和测试阶段的不同行为。
- 手动实现卷积层和最大池化层的前向、反向传播。
- 使用快速卷积层训练三层卷积神经网络。
- 使用 PyTorch 构建和训练自定义卷积网络。
- 理解普通 RNN 与 LSTM 的状态更新过程。
- 使用 RNN 根据图像特征逐词生成描述。
- 使用数值梯度检查和固定值测试验证实现。

## Notebook 导览

### `BatchNormalization.ipynb`

研究深层网络中的归一化方法：

- 实现 Batch Normalization 前向传播。
- 区分训练模式和测试模式。
- 更新并使用滑动均值与滑动方差。
- 实现普通版本和简化版本的反向传播。
- 将 Batch Normalization 接入全连接网络。
- 实现并验证 Layer Normalization。

### `Dropout.ipynb`

实现倒置 Dropout，并观察其正则化效果：

- 在训练阶段随机屏蔽神经元。
- 使用保留概率对输出进行缩放。
- 在测试阶段直接传递输入。
- 实现 Dropout 的反向传播。
- 使用随机种子完成确定性的梯度检查。

### `ConvolutionalNetworks.ipynb`

从底层实现卷积神经网络：

- 实现朴素卷积前向传播。
- 计算 padding、stride 和输出空间尺寸。
- 实现卷积层对输入、卷积核和偏置的梯度。
- 实现最大池化的前向与反向传播。
- 编译并测试 Cython 快速层。
- 构建 `conv-relu-pool-affine-relu-affine` 网络。

### `PyTorch.ipynb`

使用 PyTorch 重新实现图像分类模型：

- 熟悉 PyTorch Tensor 和自动微分。
- 使用函数式 API 编写三层卷积网络。
- 使用 `nn.Module` 管理网络参数。
- 编写训练、验证和准确率计算循环。
- 尝试 BatchNorm、Dropout 和更深的 CNN。

### `RNN_Captioning_pytorch.ipynb`

使用循环神经网络完成图像描述：

- 加载预处理后的 MS COCO 图像特征和描述。
- 实现普通 RNN 的单步和序列前向传播。
- 实现词嵌入层与时间仿射层。
- 使用带 mask 的时间 Softmax 损失。
- 实现 `CaptioningRNN.loss` 的训练流程。
- 在小数据集上过拟合 RNN 描述模型。
- 实现测试阶段的逐词采样。

## 核心代码

- `cs231n/layers.py`：归一化、Dropout、卷积和池化等基础层。
- `cs231n/layer_utils.py`：常用层组合的便捷接口。
- `cs231n/fast_layers.py`：卷积与池化的快速实现。
- `cs231n/im2col.py`：卷积展开所需的 Python 工具。
- `cs231n/im2col_cython.pyx`：快速 `im2col` 的 Cython 实现。
- `cs231n/classifiers/fc_net.py`：支持归一化和 Dropout 的全连接网络。
- `cs231n/classifiers/cnn.py`：三层卷积神经网络。
- `cs231n/classifiers/rnn_pytorch.py`：RNN/LSTM 图像描述模型。
- `cs231n/rnn_layers_pytorch.py`：RNN、LSTM、词嵌入和时间层。
- `cs231n/captioning_solver_pytorch.py`：图像描述模型训练器。
- `cs231n/coco_utils.py`：COCO 特征、描述和词汇表加载。
- `cs231n/image_utils.py`：从 URL 读取和展示图像。
- `cs231n/solver.py`：NumPy 模型的训练和验证流程。
- `cs231n/optim.py`：参数更新算法。
- `cs231n/gradient_check.py`：数值梯度检查工具。

## 推荐完成顺序

1. `BatchNormalization.ipynb`
2. `Dropout.ipynb`
3. `ConvolutionalNetworks.ipynb`
4. `PyTorch.ipynb`
5. `RNN_Captioning_pytorch.ipynb`

前三部分延续 Assignment 1 的模块化网络设计；PyTorch 部分引入自动微分；
RNN 图像描述部分相对独立，但需要理解仿射层、Softmax 和序列张量形状。

## 环境要求

- Windows、Linux 或 macOS。
- Python 3.10 及以上版本；本项目使用 Python 3.12 验证。
- NumPy、SciPy、Matplotlib、Jupyter、ImageIO 和 h5py。
- PyTorch，用于 CNN 与 RNN Notebook。
- Cython 和可用的 C/C++ 编译工具，用于快速卷积扩展。

## 本地运行

从项目根目录启动 Jupyter：

```powershell
.\.venv312\Scripts\Activate.ps1
jupyter notebook assignment\assignment2
```

进入 Notebook 后，先运行 setup 单元格。项目已配置 `%autoreload 2` 时，修改
Python 文件后通常无需重新启动；若扩展模块或导入状态异常，再重启 Kernel。

## CIFAR-10 数据集

CIFAR-10 应位于：

```text
cs231n/datasets/cifar-10-batches-py/
```

它用于归一化、Dropout、卷积网络和 PyTorch 分类实验。

## COCO 图像描述数据

预处理后的数据应位于：

```text
cs231n/datasets/coco_captioning/
```

主要文件包括：

- `coco2014_captions.h5`：编码后的训练和验证描述。
- `coco2014_vocab.json`：单词与整数索引之间的映射。
- `train2014_vgg16_fc7_pca.h5`：训练图像的 PCA 特征。
- `val2014_vgg16_fc7_pca.h5`：验证图像的 PCA 特征。
- `train2014_urls.txt` 和 `val2014_urls.txt`：原始图片 URL。

下载脚本为 `cs231n/datasets/get_coco_dataset.sh`。数据包不包含完整 COCO 图片，
Notebook 只在可视化时根据 URL 临时获取少量图片。

## 实现与检查建议

- 始终记录张量形状，例如 `(N, T, D)` 或 `(N, C, H, W)`。
- 归一化时明确均值和方差沿哪个轴计算，并正确使用 `keepdims`。
- 卷积输出尺寸必须能由 stride、padding 和卷积核大小整除得到。
- 最大池化反向传播只把梯度传给窗口中的最大值位置。
- PyTorch 新建张量时优先使用 `x.new_zeros` 或指定 `dtype` 与 `device`。
- 交叉熵标签必须使用 `torch.long`。
- RNN 训练使用真实的前一个单词，采样时使用模型刚预测的单词。
- 每完成一个函数，立即运行对应固定值测试和数值梯度检查。

## 目录说明

Notebook 用于实验流程、结果展示和问题回答，`cs231n` 包保存模型、网络层和工具。
建议只修改标记为 `TODO` 的区域，并保持函数签名不变。先通过局部梯度检查，
再进行耗时更长的模型训练，可以更快定位形状、精度和梯度方面的问题。
