# CS231n Assignment 1

本目录包含 CS231n Assignment 1 的 NumPy 实现与实验 Notebook。
作业从基础分类器出发，逐步过渡到全连接神经网络、优化算法和图像特征分类。

## 学习目标

完成本作业后，应能够：

- 理解图像分类任务的数据表示和基本评估方法。
- 实现 K 近邻分类器并使用交叉验证选择超参数。
- 推导并实现 Softmax 分类器的损失与梯度。
- 比较循环实现与向量化实现的效率差异。
- 实现仿射层、ReLU 和 Softmax 损失等基础组件。
- 构建两层网络和任意深度的全连接网络。
- 使用 SGD、Momentum、RMSProp 和 Adam 优化模型。
- 利用验证集选择模型，避免使用测试集调参。
- 使用 HOG 和颜色直方图等特征改善分类效果。

## Notebook 导览

### `knn.ipynb`

实现 K 近邻分类器，并完成以下实验：

- 加载和观察 CIFAR-10 数据。
- 使用不同循环数量计算样本间距离。
- 理解 L1 距离和 L2 距离的性质。
- 使用最近邻标签完成分类。
- 通过 K 折交叉验证选择合适的 `k`。
- 比较不同距离实现的运行时间。

### `softmax.ipynb`

实现线性 Softmax 分类器，主要内容包括：

- Softmax 损失函数的朴素循环版本。
- 损失函数的向量化版本。
- 解析梯度与数值梯度检查。
- L2 正则化及其对梯度的影响。
- 学习率和正则化强度搜索。
- 训练集、验证集和测试集准确率比较。

### `two_layer_net.ipynb`

从基础层开始搭建两层神经网络：

- 实现仿射层的前向与反向传播。
- 实现 ReLU 的前向与反向传播。
- 实现向量化 Softmax 损失。
- 组合 `affine-relu-affine-softmax` 计算图。
- 使用 `Solver` 训练两层网络。
- 观察损失曲线和准确率曲线。

### `FullyConnectedNets.ipynb`

将两层网络推广为任意深度的全连接网络：

- 使用模块化层构建深层网络。
- 检查所有权重和偏置的梯度。
- 实现 SGD with Momentum。
- 实现 RMSProp 和 Adam。
- 比较不同更新规则的收敛行为。
- 研究网络深度、隐藏维度和正则化的影响。

### `features.ipynb`

研究手工图像特征对分类任务的作用：

- 提取方向梯度直方图（HOG）。
- 提取 HSV 颜色直方图。
- 拼接多种特征并进行标准化。
- 使用线性分类器处理提取后的特征。
- 使用两层网络完成特征分类。
- 对比原始像素与手工特征的表现。

## 核心代码

### 分类器

- `cs231n/classifiers/k_nearest_neighbor.py`：KNN 距离计算与预测。
- `cs231n/classifiers/linear_classifier.py`：线性分类器训练框架。
- `cs231n/classifiers/softmax.py`：Softmax 损失的两种实现。
- `cs231n/classifiers/fc_net.py`：两层网络与深层全连接网络。

### 网络组件

- `cs231n/layers.py`：仿射层、ReLU 和 Softmax 等基础层。
- `cs231n/layer_utils.py`：常用层组合的便捷封装。
- `cs231n/gradient_check.py`：数值梯度检查工具。

### 训练工具

- `cs231n/solver.py`：mini-batch 训练和验证流程。
- `cs231n/optim.py`：SGD、Momentum、RMSProp 和 Adam。
- `cs231n/data_utils.py`：CIFAR-10 数据加载与预处理。
- `cs231n/features.py`：HOG 和颜色特征提取。

## 推荐完成顺序

1. `knn.ipynb`
2. `softmax.ipynb`
3. `two_layer_net.ipynb`
4. `FullyConnectedNets.ipynb`
5. `features.ipynb`

前两个 Notebook 介绍分类和梯度基础；后续内容会依赖基础层、优化器和
`Solver`。按顺序完成可以减少接口与张量形状方面的问题。

## 环境要求

- Windows、Linux 或 macOS。
- Python 3.10 及以上版本；本项目使用 Python 3.12 验证。
- NumPy、SciPy、Matplotlib、Jupyter 和 ImageIO。
- 本作业只使用 NumPy，不要求 GPU 或 PyTorch。

安装本地依赖：

```powershell
python -m pip install -r requirements-local.txt
```

## 本地运行

在 `assignment1` 目录中创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-local.txt
jupyter notebook
```

如果已经在项目根目录创建虚拟环境，可以直接选择对应的 Jupyter Kernel，
不需要为每个 assignment 重复创建环境。

## CIFAR-10 数据集

数据应位于：

```text
cs231n/datasets/cifar-10-batches-py/
```

目录中应包含 `data_batch_1` 到 `data_batch_5`、`test_batch` 和 `batches.meta`。
数据通常会被划分为训练集、验证集、测试集和少量开发集。验证集用于调参，
测试集只用于最终评估，避免测试信息泄漏到模型选择过程。

## 实现与检查建议

- 每完成一个函数，先运行相邻的数值梯度检查单元格。
- 梯度误差明显偏大时，优先检查正则化系数、平均因子和矩阵转置。
- 向量化实现应避免针对样本或类别的 Python 循环。
- 所有偏置通常不参与 L2 正则化。
- 修改 `.py` 文件后，可使用 `%autoreload 2` 自动重载。
- 若 Notebook 状态混乱，可重启 Kernel，再从上到下重新运行。

## 目录说明

Notebook 负责实验、可视化和结果分析，`cs231n` 包保存可复用的模型与层实现。
建议只在标记为 `TODO` 的区域内编写作业代码，并保留给定函数的输入输出接口，
以便继续使用 Notebook 中提供的测试、梯度检查和训练工具。
