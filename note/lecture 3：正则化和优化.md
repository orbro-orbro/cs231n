# lecture 3:正则化和优化

## 正则化项：

$$
L(W)=\frac{1}{N}\sum_{i=1}^NL_i(f(x_i,W),y_i)+\lambda R(W)
$$

最后这个$**\lambda R(W)**$就是正则化项，  作用是：**防止过拟合**

其中的$\lambda$是超参数
$$
L1\ regularization:R(W)=\sum_k\sum_l|W_{k,l}|\\
L2\ regularization:R(W)=\sum_k\sum_lW_{k,l}^2
$$
L1正则化：会把极小值压得更小，会让权重矩阵出现更多的0或者接近0的值

L2正则化则不会，**原因是平方项使得小值更小**，相对而言缩小正则化项的收益小
$$
L_i=\sum_{j \neq y_i}max(0,s_j-s_{y_i}+\lambda)
$$


上述定义为铰链损失

## 优化：

梯度下降：

```py
while True:
	weights_grad=evaluate_gradient(loss_fun,data,weights)
	weights += -step_size*weights_grad
```

随机梯度下降（sgd）：每次只看一个子集

```python
while True：
	data_batch=sample_training_data(data,256)
	weights_grad=evaluate_gradient(loss_fun,data_batch,weights)
	weights += -step_size*weights_grad
```

局部极小值、鞍点可能会影响sgd的优化

sgd+momentum：加上动量项，当参数优化到上述困境，动量项可以利用惯性将其带出去,同时可以防止过度振荡

```python
vx=0
while True:
	dx=compute_gradient(x)
	vx=rho*vx+dx
	x -= learning_rate*vx
```

但是收敛的更慢

RMSProp:对于不同的梯度使用不同的步长，梯度越小步长越大————少往陡峭处走，多往平坦处走

```
grad_squared=0 
while True:
	dx=compute_gradient(x)
	grad_squared=decay_rate*grad_squared+(1-decay_rate)*dx*dx
	x -= learning_rate*dx/np.sqrt(grad_squared+1e-7)
```

Adam（最常用）：

```python
first_moment=0
second_moment=0
for t in range(1,num_itetations):
	dx=compute_gradient(x)
	first_moment=beta1*first_moment+(1-beta1)*dx
	second_moment=beta2*second_moment+(1-beta2)*dx*dx
    first_unbias=first_moment/(1-bata1**t)
    second_unbias=second_moment/(1-beta2**t)
	x -= learning_rate*first_unbias/(np.sqrt(second_unbias)+1e-7)
```

学习率也可以在训练中挑战：每跑够一定的epoch，学习率/10