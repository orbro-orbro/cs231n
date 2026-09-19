# lecture 7：循环神经网络

序列建模任务：生成图片，识别视频等

RNN拥有隐藏状态h，会随着$x_t$改变
$$
h_t=f_W(h_{t-1},x_t)\\
h_t:new\ state\\ h_{t-1}:old\ state
\\x_t:input\ vector \ at\ some \ time\ step
\\f_W:some\ function\ with\ parameters\ W
$$
h决定生成y
$$
y_t=f_{W_{hy}}(h_t)
$$
<img src="https://pic2.zhimg.com/v2-629abbab0d5cc871db396f17e9c58631_r.jpg" alt="全面理解RNN及其不同架构 - 知乎" style="zoom:15%;" />

$h_0$需要我们初始化

一般来说基础公式
$$
h_t=tanh(W_{hh}h_{t-1}+W_{xh}x_t)\\
y_t=W_{hy}h_t
$$
**example**：

输入X ：0、1组成的序列，要求输出Y：检测当前是否有两个1同时出现

我们令：   $h_t=ReLU(W_{hh}h_{t-1}+W_{xh}x_t)$    我们理想中h是这么携带信息的：$h_t=(current,previous,1)$   最后一个取1是为了数学的推导  $y_t=ReLU(W_{hy}h_t)$

```python
w_xh=np.array([[1],[0],[0]])
w_hh=np,array([[0,0,0],
			   [1,0,0],
			   [0,0,1]])
w_yh=np.array([1,1,-1])
x_seq=[0,1,0,1,1,1,0,1,1]
h_t_prev=np.array([[0],[0],[1]])

for t,x in enumerate(x_seq):
	h_t=relu(w_hh@h_t_prev+w_xh@x)
	y_t=relu(w_yh@h_t)
	h_t_prev=h_t
```

## 时间维度上的反向传播

### 基本思路：BPTT

将RNN按时间展开，前向传播保存每个 $h_t$，反向传播则从 $T$ 到1。设
$$
a_t=W_{hh}h_{t-1}+W_{xh}x_t,\quad h_t=\tanh(a_t),\quad L=\sum_tL_t
$$
则隐藏状态的梯度既来自当前输出，也来自下一时间步：
$$
\delta_t=\frac{\partial L}{\partial a_t}
=\left(W_{hy}^T\frac{\partial L_t}{\partial y_t}+W_{hh}^T\delta_{t+1}\right)\odot(1-h_t^2)
$$
由于各时间步共享参数，所以梯度需要累加：
$$
\frac{\partial L}{\partial W_{hh}}=\sum_t\delta_th_{t-1}^T,\qquad
\frac{\partial L}{\partial W_{xh}}=\sum_t\delta_tx_t^T
$$

### 时间窗口：截断BPTT

完整BPTT需要保存整个序列，序列过长时开销很大。可以设置窗口长度 $K$，第 $t$ 步只向前回传到 $t-K+1$，窗口边界处停止梯度。

这样能降低显存和计算量，但窗口外的信息无法通过梯度学习，$K$ 太小会影响长期依赖。

### 分块处理

将长序列切成长度为 $K$ 的若干块。上一块的最终隐藏状态传给下一块以保留上下文，但在块边界使用 `detach` 断开计算图：

```python
h = h0
for x_block, y_block in blocks:
    h = h.detach()
    y_pred, h = model(x_block, h)
    loss = criterion(y_pred, y_block)
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

分块是截断BPTT的常见实现；梯度裁剪可以缓解长序列中的梯度爆炸。

为了解决梯度消失和信息丢失的问题：我们可以引入门控，LSTM模型
