# 本地作业目录说明

## 为什么 Assignment 1 和 Assignment 2 都有 `cs231n`

它们是两套相互独立的作业代码包，不是重复文件。运行某个作业时，应当从对应的作业目录启动 Notebook：

- Assignment 1：`assignment/assignment1`
- Assignment 2：`assignment/assignment2`

这样 Notebook 中的 `import cs231n` 会导入当前作业自己的代码，避免 A1 和 A2 相互覆盖。

## 已做的兼容调整

A2 的 `cs231n/layers.py` 明确要求从 A1 复用以下五个基础函数，因此这里只迁移了它们：

- `affine_forward`
- `affine_backward`
- `relu_forward`
- `relu_backward`
- `softmax_loss`

A2 其余 TODO（批归一化、Dropout、卷积、RNN 等）均保持原样，仍需按 A2 的 Notebook 要求完成。

## CIFAR-10 数据

A2 的 `cs231n/datasets/cifar-10-batches-py` 是一个 Windows Junction，指向 A1 中已经下载的数据集。这样两份作业共用一份数据，不会额外占用约 170 MB 空间。

注意：使用 A2 时不要删除或移动 A1 的 `cifar-10-batches-py`。如果以后把整个项目复制到别处，Windows Junction 可能需要重新创建。

## 提交作业

提交时仍按每个 Assignment 自带的提交脚本或课程说明操作。不要把 CIFAR-10 数据目录打包进提交文件。
