# CS231n Assignment 包结构整理实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 保持 A1/A2 独立可运行与可提交，仅把课程明确要求继承的五个 A1 函数迁入 A2，并在本地安全共享 CIFAR-10 数据。

**Architecture:** `assignment1/cs231n` 与 `assignment2/cs231n` 继续作为两个互不混用的 Python 包。A2 只接收五个指定函数体；数据通过 Windows Junction 指向 A1 的真实数据目录，生成缓存被清理。

**Tech Stack:** Python 3.12、NumPy 1.26.4、PowerShell、Windows Junction、pytest 风格的一次性断言脚本

**Spec:** `docs/superpowers/specs/2026-09-05-assignment-package-structure-design.md`

## Global Constraints

- 不移动、覆盖或删除 `assignment1/cs231n` 中的源码与数据。
- 不合并两个 `cs231n` 包，不修改 Notebook 的 `from cs231n...` 导入。
- A2 只继承 `affine_forward`、`affine_backward`、`relu_forward`、`relu_backward`、`softmax_loss`。
- 不填写 A2 的归一化、Dropout、卷积、RNN 或其他新作业内容。
- 当前目录不是 Git 仓库；用 SHA-256 和临时备份提供回滚能力。

---

### Task 1: 建立安全快照与迁移测试

**Files:**
- Create: `.structure-backup/assignment2-layers.py`
- Create: `.structure-backup/verify_assignment_structure.py`
- Read: `assignment/assignment1/cs231n/layers.py`
- Read: `assignment/assignment2/cs231n/layers.py`

**Interfaces:**
- Consumes: A1/A2 两份 `layers.py` 与 A1 CIFAR-10 数据目录
- Produces: A2 原文件备份、A1 源文件哈希、迁移前必然失败的五函数对比测试

- [ ] **Step 1: 保存 A2 文件备份与 A1 SHA-256**

创建 `.structure-backup`，复制 A2 的 `layers.py` 为 `assignment2-layers.py`，并把 A1 `layers.py` 的 SHA-256 写入 `a1-layers.sha256`。备份只包含本次会修改的源码。

- [ ] **Step 2: 创建一次性验证脚本**

脚本通过 `importlib.util.spec_from_file_location` 分别加载两份 `layers.py`，用固定数组验证以下等价关系：

```python
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


a1 = load_module("a1_layers", ROOT / "assignment/assignment1/cs231n/layers.py")
a2 = load_module("a2_layers", ROOT / "assignment/assignment2/cs231n/layers.py")

x = np.linspace(-0.4, 0.6, 24).reshape(2, 3, 4)
w = np.linspace(-0.2, 0.3, 60).reshape(12, 5)
b = np.linspace(-0.1, 0.2, 5)
dout = np.linspace(-0.3, 0.4, 10).reshape(2, 5)

for left, right in zip(a1.affine_forward(x, w, b), a2.affine_forward(x, w, b)):
    if isinstance(left, tuple):
        continue
    np.testing.assert_allclose(left, right)

a1_cache = a1.affine_forward(x, w, b)[1]
a2_cache = a2.affine_forward(x, w, b)[1]
for left, right in zip(a1.affine_backward(dout, a1_cache), a2.affine_backward(dout, a2_cache)):
    np.testing.assert_allclose(left, right)

relu_x = np.linspace(-1, 1, 12).reshape(3, 4)
np.testing.assert_allclose(a1.relu_forward(relu_x)[0], a2.relu_forward(relu_x)[0])
relu_dout = np.ones_like(relu_x)
np.testing.assert_allclose(
    a1.relu_backward(relu_dout, relu_x),
    a2.relu_backward(relu_dout, relu_x),
)

scores = np.array([[0.2, -0.4, 1.1], [1.2, 0.1, -0.2]])
labels = np.array([2, 0])
a1_loss, a1_dx = a1.softmax_loss(scores, labels)
a2_loss, a2_dx = a2.softmax_loss(scores, labels)
np.testing.assert_allclose(a1_loss, a2_loss)
np.testing.assert_allclose(a1_dx, a2_dx)
print("FIVE_FUNCTIONS_MATCH")
```

- [ ] **Step 3: 运行迁移前测试并确认失败**

Run: `.venv312\Scripts\python.exe .structure-backup\verify_assignment_structure.py`

Expected: FAIL，因为 A2 五个函数当前保留空实现。这证明测试能发现尚未迁移的状态。

---

### Task 2: 精确迁移五个函数

**Files:**
- Modify: `assignment/assignment2/cs231n/layers.py`
- Read: `assignment/assignment1/cs231n/layers.py`
- Test: `.structure-backup/verify_assignment_structure.py`

**Interfaces:**
- Consumes: A1 五个函数的完整定义
- Produces: A2 中签名不变、数值行为与 A1 一致的五个函数

- [ ] **Step 1: 替换五个完整函数定义**

使用 Python AST 取得两份源码中函数的 `lineno`/`end_lineno`，把 A1 五个完整函数定义逐个替换进 A2；按 A2 中的原始位置从后向前替换，避免行号漂移。允许替换的函数名集合严格固定为：

```python
ALLOWED = {
    "affine_forward",
    "affine_backward",
    "relu_forward",
    "relu_backward",
    "softmax_loss",
}
```

迁移脚本必须断言 A1 与 A2 都恰好找到五个名称，且除这五段以外的 A2 前后源码片段保持原样。

- [ ] **Step 2: 运行五函数等价测试**

Run: `.venv312\Scripts\python.exe .structure-backup\verify_assignment_structure.py`

Expected: 输出 `FIVE_FUNCTIONS_MATCH`，退出码为 0。

- [ ] **Step 3: 检查 A1 未发生变化**

重新计算 `assignment/assignment1/cs231n/layers.py` 的 SHA-256，与 `.structure-backup/a1-layers.sha256` 比较，必须完全一致。

- [ ] **Step 4: 检查 A2 新作业范围未被填写**

确认 A2 中 `batchnorm_forward` 及其后所有新函数的源码，与 `.structure-backup/assignment2-layers.py` 中对应函数逐字节相同。

---

### Task 3: 共享数据并说明本地结构

**Files:**
- Create: `assignment/assignment2/cs231n/datasets/cifar-10-batches-py`（Windows Junction）
- Create: `assignment/LOCAL_STRUCTURE.md`

**Interfaces:**
- Consumes: A1 的真实 CIFAR-10 数据目录
- Produces: A2 可读取的数据路径、用户可读的目录说明

- [ ] **Step 1: 验证联接两端路径**

源必须解析为工作区内的 `assignment/assignment1/cs231n/datasets/cifar-10-batches-py`，目标必须是工作区内尚不存在的 `assignment/assignment2/cs231n/datasets/cifar-10-batches-py`。如果目标已经存在则停止，不覆盖。

- [ ] **Step 2: 创建 Windows Junction**

使用 PowerShell `New-Item -ItemType Junction` 创建目录联接；不移动 A1 数据。验证从 A2 路径可读取 `data_batch_1`、`data_batch_5`、`test_batch` 和 `batches.meta`。

- [ ] **Step 3: 添加本地结构说明**

`assignment/LOCAL_STRUCTURE.md` 必须说明：两个包需要独立；运行 Notebook 时工作目录应是所属 Assignment；A2 数据目录是指向 A1 的本地联接；提交时不包含数据；A2 只继承五个指定函数。

---

### Task 4: 清理与完整验收

**Files:**
- Delete: `assignment/assignment1/cs231n/**/__pycache__`
- Delete: `assignment/assignment2/cs231n/**/__pycache__`
- Delete after success: `.structure-backup/`

**Interfaces:**
- Consumes: 完成后的两个 Assignment 包
- Produces: 无缓存污染、两个包可独立导入、可正常继续作业的目录

- [ ] **Step 1: 验证独立导入**

分别以 `assignment1`、`assignment2` 为工作目录执行：

```python
import cs231n
print(cs231n.__file__)
```

输出必须分别位于对应 Assignment 目录。

- [ ] **Step 2: 验证 A1 核心函数**

运行 A1 五函数的固定数值测试，确认迁移过程没有修改 A1。运行 `knn.ipynb` 不作为本次验收条件，因为它依赖用户当前作业进度与执行状态。

- [ ] **Step 3: 验证 A2 五函数与数据读取**

重新运行 `verify_assignment_structure.py`，并调用 A2 `load_CIFAR10` 读取联接目录；确认训练集为 50,000 张、测试集为 10,000 张。

- [ ] **Step 4: 安全清理生成缓存**

先把每个待删目录解析为绝对路径并确认位于两个 `cs231n` 目录内，再用 PowerShell `Remove-Item -LiteralPath ... -Recurse -Force` 删除 `__pycache__`。不删除任何 `.py`、`.ipynb`、`.npy` 或 CIFAR 数据文件。

- [ ] **Step 5: 删除临时备份**

仅在所有验证均通过后删除 `.structure-backup`。如果任一验证失败，用其中的 `assignment2-layers.py` 恢复 A2，然后保留备份并报告失败证据。
