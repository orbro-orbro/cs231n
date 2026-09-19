# lecture 12：自监督学习

###### 人工标注对质量要求高，成本高

dataset（无标签）通过encoder获得learned representation，再经过decoder、classifier转换到输出。这些输出往往只与图片本身的性质有关比如：					这种任务叫做pretext（要求通用性）

- 将图片旋转 \(0^\circ、90^\circ、180^\circ、270^\circ\)，让模型预测旋转角度。
- 遮住图片的一部分，让模型恢复被遮挡的内容。
- 判断两张经过不同增强的图片是否来自同一张原图。

接下来保留encoder，后续接入任务所需要的组件，使用有标签的数据集来进行正常的训练

MAE：随机掩码，遮蔽75%的图块，将剩下的token输入encoder，再放入decoder来补全图像，和初始的图片进行对比。



我们也会引入负样本的概念。初始图像为一只猫x，经过变换得到x+图像，data中其他差别大的图片x-（负样本）
$$
score(f(x),f(x^+))>>score(f(x),f(x^-))\\
Loss=-E_x[log\frac{e^{s(f(x),f(x^+))}}{e^{s(f(x),f(x^+))+\sum_{j=1}^{N-1}e^{s(f(x),f(x_j^-))}}}]
$$
这个loss类似于多分类的交叉熵