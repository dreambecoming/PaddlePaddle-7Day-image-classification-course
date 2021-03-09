# 百度飞桨领航团零基础图像分类速成营 课程总结5
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



# 课节4：卷积神经网络基础：蝴蝶图像识别分类
## 前置知识

图像分类比赛的一般解题流程：

  1. 数据EDA （Pandas、Matplotlib）
  2. 数据预处理 （OpenCV、PIL、Pandas、Numpy、Scikit-Learn）
  3. 根据赛题任务定义好读取方法，即Dataset和Dataloader（PaddlePaddle2.0）
  4. 选择一个图像分类模型进行训练 （PaddlePaddle2.0）
  5. 对测试集进行测试并提交结果（PaddlePaddle2.0、Pandas）

图像分类竞赛常见难点：

  1. 类别不均衡
  2. One-Shot和Few-Shot分类
  3. 细粒度分类
一、EDA（Exploratory Data Analysis）与数据预处理
* EDA
&emsp;&emsp;探索性数据分析（Exploratory Data Analysis，简称EDA），是指对已有的数据（原始数据）进行分析探索，通过作图、制表、方程拟合、计算特征量等手段探索数据的结构和规律的一种数据分析方法。一般来说，我们最初接触到数据的时候往往是毫无头绪的，不知道如何下手，这时候探索性数据分析就非常有效。

&emsp;&emsp;对于图像分类任务，我们通常首先应该统计出每个类别的数量，查看训练集的数据分布情况。通过数据分布情况分析赛题，形成解题思路。

* 数据预处理

Compose实现将用于数据集预处理的接口以列表的方式进行组合。
```python
# 定义数据预处理
data_transforms = T.Compose([
    T.Resize(size=(32, 32)),
    T.Transpose(),    # HWC -> CHW
    T.Normalize(
        mean=[0, 0, 0],        # 归一化
        std=[255, 255, 255],
        to_rgb=True)    
])
```
图像标准化与归一化

&emsp;&emsp;最常见的对图像预处理方法有两种，一种叫做图像标准化处理，另外一种方法叫做归一化处理。数据的标准化是指将数据按照比例缩放,使之落入一个特定的区间。
将数据通过去均值，实现中心化。处理后的数据呈正态分布,即均值为零。数据归一化是数据标准化的一种典型做法,即将数据统一映射到[0,1]区间上。

作用:
1. 有利于初始化的进行
2. 避免给梯度数值的更新带来数值问题
3. 有利于学习率数值的调整
4. 加快寻找最优解速度

<center><p>
  
   标准化

  ![](https://ai-studio-static-online.cdn.bcebos.com/0c290c7b103c41b09f118f977e7e1fe26871a6416e314dc6ae1e534548701114)
 
  
   归一化

  ![](https://ai-studio-static-online.cdn.bcebos.com/b6133a31796a4ccc964e62b4acb1b5b907d9c5d86e244acd9495126b4849ced1)
  
  </p></center>
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

```python

```
```python

``````python

```
