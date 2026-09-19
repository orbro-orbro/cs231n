# CS231n Assignment 包结构整理设计

## 目标

保持 Assignment 1 与 Assignment 2 各自独立、可提交的 `cs231n` 包，同时把课程明确要求从 Assignment 1 继承的实现迁移到 Assignment 2，并让两个作业在本地共用同一份 CIFAR-10 数据。

## 目录结构

```text
assignment/
├── LOCAL_STRUCTURE.md
├── assignment1/
│   └── cs231n/
│       ├── datasets/
│       │   └── cifar-10-batches-py/
│       └── ...A1 独立源码
└── assignment2/
    └── cs231n/
        ├── datasets/
        │   └── cifar-10-batches-py -> A1 数据集目录（Windows Junction）
        └── ...A2 独立源码
```

两个 `cs231n` 包不合并。每个 Notebook 从所属 Assignment 目录运行，因此 `import cs231n` 始终解析到同目录下的包。

## 代码迁移范围

只把以下五个已由用户在 A1 完成、且 A2 明确标注为“Copy over your solution from Assignment 1”的函数体迁移到 `assignment2/cs231n/layers.py`：

- `affine_forward`
- `affine_backward`
- `relu_forward`
- `relu_backward`
- `softmax_loss`

不整体覆盖 `layers.py`，不复制 A1 的 `fc_net.py`，不填写 A2 的归一化、Dropout、卷积、RNN 或其他新 TODO。A1 文件保持不变。

## 数据与生成文件

`assignment2/cs231n/datasets/cifar-10-batches-py` 创建为 Windows 目录联接，目标是 `assignment1/cs231n/datasets/cifar-10-batches-py`。联接只服务于本地运行；提交脚本不应包含数据集。

删除两个 `cs231n` 目录内生成的 `__pycache__` 目录和 `.pyc` 文件。源码、Notebook、模型存档和数据集不删除。

## 本地说明

创建 `assignment/LOCAL_STRUCTURE.md`，说明两个包为何必须独立、从哪个目录打开各 Assignment Notebook、数据联接的用途，以及 Assignment 2 只继承上述五个函数。

## 验证

- 确认 A1 的五个源函数在操作前后内容哈希不变。
- 确认 A2 的五个目标函数与 A1 对应函数产生相同数值结果。
- 分别从 `assignment1` 和 `assignment2` 启动 Python，确认 `cs231n.__file__` 指向各自目录。
- 确认 A2 能通过联接发现 CIFAR-10 的 `data_batch_1` 和 `test_batch`。
- 确认 A2 其余 TODO 数量和文本未被迁移操作意外修改。
- 确认所有 `__pycache__` 已清理。

## 恢复策略

修改 A2 前，在工作区内创建仅包含被修改文件的临时备份；验证成功后删除临时备份。数据目录联接可直接删除而不会删除 A1 的真实 CIFAR-10 数据。
