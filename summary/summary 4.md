# 百度飞桨领航团零基础图像分类速成营 课程总结4
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/11939)	`https://aistudio.baidu.com/aistudio/course/introduce/11939`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   
[推荐学习网站](https://www.runoob.com/python3/python3-tutorial.html)	`https://www.runoob.com/python3/python3-tutorial.html`  

****
## 目录
* [前置知识](#前置知识)
* [基础理论](#基础理论)
* [建模实战](#建模实战)
* [作业](#作业)
## 参考资料
* [关于LeNet的前世今生](https://www.jiqizhixin.com/graph/technologies/6c9baf12-1a32-4c53-8217-8c9f69bd011b)

* [常用网络结构](https://www.jiqizhixin.com/articles/2020-05-06-16)

* [GoogleNet解析](https://www.itread01.com/content/1544969366.html)

* [你必须要知道CNN模型：ResNet](https://zhuanlan.zhihu.com/p/31852747)

* [Alexnet论文](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

* [GoogleNet论文](https://arxiv.org/pdf/1409.4842v1.pdf)

* [Resnet论文](https://arxiv.org/pdf/1512.03385.pdf)


# 课节4：卷积神经网络基础：蝴蝶图像识别分类
## 前置知识
### 矩阵
1. 矩阵加法运算 
	I2 = I1 + b
    
	1.1 如果I1和b维度一致时，结果为两个矩阵对应项相加。

	1.2 如果I1和b维度不一致，但是满足一定条件，亦可进行加法计算。

	假设I1的矩阵形状为shape_I1=（h,w,c）,那么在b的矩阵形状shape_b为shape_I1的某个切片相等时，I1和b矩阵可以进行“广播”相加。

2. 矩阵和数乘法运算
	
    I3 = a * I1
    
    I1各个元素分别乘以a
    
3. 矩阵乘法运算
	
	I2 = I1 * A
    
    1. 二维矩阵
	I1的r行向量与A的c列向量的点积作为I3的（r,c）元素
    
    2. 多维矩阵
    所得结果的后两维的shape与二维矩阵的计算方式一致。不同的是高维度（3位以上）的尺寸大小的计算方式-取较大的尺寸。作为多维矩阵，数据还是保持在后两维中，高维度只是数据的排列方式的定义。本质上，高维矩阵的乘法还是二维矩阵之间的乘法，再加上排列方法的保持，当高维尺寸不同时，只要可以“广播”，就依然可以计算。
    
    譬如：
    2.1 I1的shape为 (c,n,s),A的shape为（c，s，m）时，I2的shape为（c,n,m).
    结果shape的计算过程：I1的后两维shape为（n,s), A的后两维shape为（s,m),类似于二维矩阵乘法所得结果的后两维shape为（n,m), 高维度的尺寸取c。
    
    2.2 I1的shape为 (n,c,h,s),A的shape为（1,1，s，w）时，I2的shape为（n,c,h,w).
    结果shape的计算过程：I1的后两维shape为（h,s), A的后两维shape为（s,m),类似于二维矩阵乘法所得结果的后两维shape为（h,w), 高维度的尺寸取（n,c）和（1,1）中尺寸较大值（n,c）。
```python
I1 = [[j*4+i for i in range(4)] for j in range(4)]
I1 = np.array(I1,dtype='float32')
A = np.random.uniform(0,1,(4,4)) # numpy.random.uniform(low,high,size) 随机数

## 二维矩阵乘法 1
I2 = np.matmul(I1,A)
## 二维矩阵乘法 2
I3 = np.zeros_like(I2)
for i in range(4):
    for j in range(4):
        I3[i,j] = np.dot(I1[i,:],A[:,j])

value = {}
value['I1'] = I1
value['A'] = A
value['I2'] = I2
value['I3'] = I3
print("I1={I1},\nA={A},\nI2={I2}\n".format(**value))
print("I1={I1},\nA={A},\nI3={I2}\n".format(**value))
print("I2 ==I3: {}\n".format(np.allclose(I2,I3)))
```
```python
## 高维矩阵乘法
## (3,100,9) , (3,9,99)
I1 = np.random.uniform(0,1,(3,100,9))
A = np.random.uniform(0,1,(3,9,99))
I2 = np.matmul(I1,A)
value = {}
value['I1'] = I1
value['A'] = A
value['I2'] = I2
#print("I1={I1},\na={a},\nI2={I2}\n".format(**value))
print("I1 shape={},A shape={},I2 shape={}".format(I1.shape,A.shape,I2.shape))

```
```python
## 高维矩阵乘法
## (100,3,100,9) , (1,3,9,99)
I1 = np.random.uniform(0,1,(100,3,100,9))
A = np.random.uniform(0,1,(1,3,9,99))
I2 = np.matmul(I1,A)
value = {}
value['I1'] = I1
value['A'] = A
value['I2'] = I2
#print("I1={I1},\na={a},\nI2={I2}\n".format(**value))
print("I1 shape={},A shape={},I2 shape={}".format(I1.shape,A.shape,I2.shape))
```
### 算子
传统计算机视觉利用算子，以及算子集对进行图像处理，从而获得特征图像，CNN中的卷积核即是与之相对应的概念。不同是，前者需要算法开发者手动设计，而后者通过训练数据的驱动来自动选择，而且通常情况下CNN网络中的卷积核数量远超过手动设计的。

Sobel算子
![](https://ai-studio-static-online.cdn.bcebos.com/5dcab1a62a764dc783cef4a8da3706d6ec59c432868d4a45b8d9b272b8ce2791)


Laplace算子
![](https://ai-studio-static-online.cdn.bcebos.com/c6bef46b534e4af5bde59f6ed07726d1810e87e3defa44a9a20c7c59f526bb95)

### 卷积
该过程可理解为算子（核）窗口在图像上沿指定维度滑动，同时将矢量点积值作为对应输出矩阵元素的值。

###批归一化
批归一化（Batch Normalization）

### 感受野
感受野（Receptive Field）的定义是卷积神经网络每一层输出的特征图（feature map）上每个像素点在原始图像上映射的区域大小，这里的原始图像是指网络的输入图像，是经过预处理（如resize，warp，crop）后的图像。

作者：有一个错别字
链接：https://www.jianshu.com/p/9997c6f5c01e
来源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

## 作业
作业说明：

1. 能够跑通项目得到结果，得到底分60分；
2. 通过修改模型、数据增强等方法，使得模型在测试数据集上的准确度达到85%及以上的，在底分上再得到加分30分；
3. 通过修改模型、数据增强等方法，使得模型在测试数据集上的准确度达到90%及以上的，直接得到满分100分。

