# CS231n 学习笔记与课程作业

本仓库用于记录 Stanford CS231n（Deep Learning for Computer Vision）的学习过程，包含中文课程笔记、三个阶段的课程作业、实验 Notebook 及配套代码。

## 项目结构

```text
.
├── note/                  # Lecture 2–15 中文笔记
├── assignment/
│   ├── assignment1/       # 传统分类器与全连接网络
│   ├── assignment2/       # CNN、归一化、RNN 与 PyTorch
│   ├── assignment3/       # Transformer、自监督学习与生成模型
│   └── LOCAL_STRUCTURE.md # 本地目录补充说明
└── README.md
```

## 课程笔记

- 图像分类、线性分类器、正则化与优化
- 神经网络、反向传播、卷积神经网络与经典 CNN 架构
- 循环神经网络、注意力机制与 Transformer
- 目标检测、图像分割、视频理解与分布式训练
- 自监督学习、生成模型以及视觉与语言

## 课程作业

- **Assignment 1**：KNN、Softmax、两层网络、全连接网络与图像特征。
- **Assignment 2**：Batch Normalization、Dropout、卷积网络、PyTorch 与 RNN 图像描述。
- **Assignment 3**：Transformer 图像描述、SimCLR、DDPM、CLIP 与 DINO。

## 本地运行

建议使用 Python 3.10 及以上版本，并在虚拟环境中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install jupyter
jupyter notebook
```

各作业的详细依赖、运行方式和完成顺序请查看对应目录中的 `README.md` 或 `requirements*.txt`。

## 说明

本仓库仅用于课程学习与个人记录。课程内容及原始作业版权归 Stanford University 和 CS231n 课程团队所有。
