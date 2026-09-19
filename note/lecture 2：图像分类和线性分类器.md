# lecture2：图像分类和线性分类器

#### **core task:       image classifiction**

###### 图像分类：给一张图片分配标签

图像=数据张量（高度，宽度，通道）

挑战：1、角度      2、光线     3、背景     4、遮挡     5、形变     6、内差异     7、上下文

## 1、data-driven approaches

机器学习识别图像的基本流程：

​	1、收集图像标签的数据集（dataset）

​	2、使用机器学习算法构建训练一个classifier

​	3、在新图像上评估这个模型

## 2、linear classification

参数化方法：找到权重、偏置来将输入映射成类别

输入图像先转换成向量，通过W ，b转换为类别分数

找到高维空间的超平面来将散乱在空间中的点进行划分，但是具相当大的局限

损失函数（目标函数），用来衡量不满意度
$$
Loss=\frac{1}{N}\sum_iL_i(f(x_i,W),y_i)
$$
softmax分类器：

将类别分数转换为概率（分数没有上下界，差值较大）
$$
P(Y=k|X=x_i)=\frac{e^{s_k}}{\sum_je^{s_j}}
$$
在多分类问题中loss函数(最大似然估计):$L_i=-logP(Y=y_i|X=x_i)$

## 3、k-nearest neighbor（极其粗糙）

###### 通过最近的k个样本的类别来决定test的归属

`def train(images,labels):`

`return model`        `训练`

`def predict(model,test_images):`

`return test_labels`	预测



假设：训练集中有五个图，test图像为一只猫，我们计算test图像和训练集中各个图像的distance（多种），代表相似程度


$$
d_1=\sum_p|I_1^p-I_2^p|
\\
d_2=\sqrt{\sum_p(I_1^p-I_2^p)^2}
$$




 d1:像素差值的绝对值之和，曼哈顿距离——对特征值很敏感

d2：差值的平方和的开方

`for i inn xrange(X.shape[0]):`

`		distance=np.sum(np.abs(self.Xtr-X[i,:]),axis=1)`		这里图是已经做过flatten处理X.shape=(批量大小，像素)

​												self.Xtr是多张训练图，每张图都要减去第i张测试图，再分别求和

`		min_index=np.argmin(distance)`	选出distance最小的小标

`		Ypred[i]=self.ytr[min_index]`	

`		return Ypred`

超参数：k，距离函数

设置超参数：选在验证集中表现最好的那一组超参数，k折交叉验证