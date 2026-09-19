# lecture 6：CNN经典架构

## CNN架构

### Layers in CNNs

###### 图像->卷积->池化->全连接层->scores

归一化层：先标准化输入，再通过可学习参数 $\gamma$、$\beta$ 进行缩放和平移：
$$
y=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

BatchNorm：训练时按批次、按通道统计均值和方差；测试时使用训练阶段的移动平均值。

LayerNorm：对每个样本的特征单独计算均值和方差，训练和测试方式相同。

Dropout层：训练时以概率 $p$ 丢弃神经元，减少神经元间的依赖。常用 inverted dropout 将保留的激活除以 $1-p$，因此测试时无需缩放。

### Activation Functions

卷积核也是线性运算。需要我们引入激活函数

sigmoid函数因为在极大值或极小值会出现梯度消失的现象

现在我们常用Relu、Gelu

### CNN Architectures

3\*3的卷积核常用：3个3\*3的卷积层等价于7\*7的卷积层，而且参数更少

越深的神经网络越难训练学习，为了解决这个问题：我们使用残差链接

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260905085527191.png" alt="image-20260905085527191" style="zoom: 25%;" />

这样设计可以更快捷的学习到恒等映射、并且更方便传递梯度

### Weight Initialization

权重不能全部初始化为0，否则同层神经元会始终学习到相同特征。偏置通常可以初始化为0。

Xavier初始化适合tanh等激活函数：$W\sim N(0,\frac{2}{fan_{in}+fan_{out}})$；ReLU通常使用He初始化：$W\sim N(0,\frac{2}{fan_{in}})$。

```python
W = np.random.randn(D_in, D_out) * np.sqrt(2 / D_in)  # He初始化
```

## 训练模型

### Data Preprocessing

常见做法是先将像素缩放到 $[0,1]$，再按通道标准化。均值和标准差只能从训练集计算，验证集和测试集必须复用，避免数据泄漏。

```python
mean = train_images.mean(axis=(0, 1, 2), keepdims=True)
std = train_images.std(axis=(0, 1, 2), keepdims=True)
train_images = (train_images - mean) / (std + 1e-8)
val_images = (val_images - mean) / (std + 1e-8)
```



### Data augmentation

我们会将训练图像随机翻转、缩放、裁剪或遮挡，以增加数据多样性。验证和测试阶段通常不使用随机增强。

### Transfer Learning

我们发现相同的标签图片在最后的分类器前的向量在空间中往往很接近

我们可以使用在ImageNet上训练好的模型，替换最后的分类器并冻结特征提取层。分类器训练稳定后，可用较小学习率解冻部分网络进行微调。

### Hyperparameter Selection

- [ ] check initial loss
- [ ] overfit  a small sample
- [ ] find LR that makes loss go down
- [ ] coarse grid of hyperparams ,train for ~1-5epochs
- [ ] refine grid ,train longer
- [ ] look at loss and curves

超参数应根据验证集选择，测试集只用于最终评估。
