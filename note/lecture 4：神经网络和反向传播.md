# lecture 4：神经网络和反向传播

## 神经网络

全连接神经网络：
$$
f=W_2max(0,W_1,x)
$$
这就是一个双层神经网络，中间引入了一个非线性函数——使得模型拜托了线性的约束

如果没有max函数（激活函数），**我们可以通过数学证明多层的线性变换等价于单次的线性变化**

激活函数:$ReLU $    $Sigmoid$     $tanh$     等。通常来说Relu效果不错

激活函数是为了引入非线性的表达能力

更多的神经元会带来更强大的表达能力，但很容易导致过拟合----通过正则化或者超参数的选择来优化
$$
L(W)=\frac{1}{N}\sum_{i=1}^NL_i(f(x_i,W),y_i)+\lambda R(W)
$$
当$\lambda$ 比较大的时候损失函数会过多的关注正则项，而忽略实际误差

## 计算图

把神经网络中的而所有操作搭建起来

<img src="https://img2024.cnblogs.com/blog/1078349/202411/1078349-20241122144508657-318432892.png" alt="image" style="zoom:50%;" />

上述图只是举例子，不代表神经网络的计算图

## 反向传播

###### 从计算图的末端往回推，本质是一个递归过程

#### 矩阵乘法的反向传播

设 $Y=AB$，上游梯度为 $G=\frac{\partial L}{\partial Y}$，则
$$
\frac{\partial L}{\partial A}=GB^T,\qquad
\frac{\partial L}{\partial B}=A^TG
$$

例如：
$$
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\quad
B=\begin{bmatrix}1&0&1&0\\0&1&0&1\\1&1&1&1\end{bmatrix}
$$
则 $Y=AB=\begin{bmatrix}4&5&4&5\\10&11&10&11\end{bmatrix}$。若 $G$ 为全 $1$ 矩阵，则
$$
\frac{\partial L}{\partial A}=\begin{bmatrix}2&2&4\\2&2&4\end{bmatrix},\qquad
\frac{\partial L}{\partial B}=\begin{bmatrix}5&5&5&5\\7&7&7&7\\9&9&9&9\end{bmatrix}
$$









# 注：我对矩阵求导的运算法则不熟悉！！！
