# 百度飞桨领航团零基础图像分类速成营 课程总结2
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/11939)	`https://aistudio.baidu.com/aistudio/course/introduce/11939`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   
[推荐学习网站](https://www.runoob.com/python3/python3-tutorial.html)	`https://www.runoob.com/python3/python3-tutorial.html`  

****
## 目录
* [图像处理的概念与基本操作](#图像处理的概念与基本操作)
* [OpenCV库进阶操作](#OpenCV库进阶操作)
* [图像分类任务概念导入](#图像分类任务概念导入)
* [PaddleClas数据增强代码解析](#PaddleClas数据增强代码解析)
* [作业](#作业)


# 课节2：图像处理入门基础（一）

## 图像处理的概念与基本操作
```python
 # 引入依赖包
%matplotlib inline
import numpy as np
import cv2
import matplotlib.pyplot as plt
import paddle
from PIL import Image
```
```python
# 引入依赖包
%matplotlib inline
import numpy as np
import cv2
import matplotlib.pyplot as plt
import paddle
from PIL import Image
```

```python
# 加载一张手写数字的灰度图片
# 从Paddle2.0内置数据集中加载手写数字数据集，本文第3章会进一步说明
from paddle.vision.datasets import MNIST
# 选择测试集
mnist = MNIST(mode='test')
# 遍历手写数字的测试集
for i in range(len(mnist)):
    # 取出第一张图片
    if i == 0:
        sample = mnist[i]
        # 打印第一张图片的形状和标签
        print('图片的形状',sample[0].size, '图片的标签',sample[1])
        print('MNIST数据集第一条：',mnist[0])
        print('MNIST数据集第一条第一项：',sample[0],'\n','MNIST数据集第一条第二项：',sample[1])

```
输出：
```python
图片的形状 (28, 28) 图片的标签 [7]
MNIST数据集第一条： (<PIL.Image.Image image mode=L size=28x28 at 0x7F1676CBBA50>, array([7]))
MNIST数据集第一条第一项： <PIL.Image.Image image mode=L size=28x28 at 0x7F1676C6C890> 
 MNIST数据集第一条第二项： [7]
```
```python
# 查看测试集第一个数字
plt.imshow(mnist[0][0])
print('手写数字是：', mnist[0][1])
```
灰度值与光学三原色（RGB）：红、绿、蓝(靛蓝)。光学三原色混合后，组成像素点的显示颜色，三原色同时相加为白色，白色属于无色系（黑白灰）中的一种。
```python
img = Image.open('lena.jpg')

# 使用PIL分离颜色通道
r,g,b = img.split()

# 将图片转为矩阵表示
np.array(img)

# 获取第一个通道转的灰度图
img.getchannel(0)

# 获取第二个通道转的灰度图
img.getchannel(1)

# 获取第三个通道转的灰度图
img.getchannel(2)

# 将矩阵保存成文本，数字格式为整数
np.savetxt('lena-r.txt', r, fmt='%4d')
np.savetxt('lena-g.txt', g, fmt='%4d')
np.savetxt('lena-b.txt', b, fmt='%4d')

```

## OpenCV库进阶操作

## 图像分类任务概念导入

## PaddleClas数据增强代码解析

## 作业
用课程所学内容实现各类图像增广

    常用图像增广方法主要有：左右翻转(上下翻转对于许多目标并不常用)，随机裁剪，变换颜色(亮度，对比度，饱和度和色调)等等，我们拟用opencv-python实现部分数据增强方法。
    结构如下：
           ```python
           class FunctionClass:
               def __init__(self, parameter):
                   self.parameter=parameter

               def __call__(self, img):       
           ```
要求
1.补全代码
2.验证增强效果
3.可自选实现其他增强效果
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

filename = '1.png'
## [Load an image from a file]
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
```
```python
 print(img.shape)
```
1. 图片缩放
 ```python
class Resize:
    def __init__(self, size):
        self.size=size

    def __call__(self, img):

        # 此处插入代码
        res = cv2.resize(img, self.size)
        return(res)


resize=Resize( (600, 600))
img2=resize(img)
plt.imshow(img2)
 ```
  
2. 图片翻转
```python
 class Flip:
    def __init__(self, mode):
        self.mode=mode

    def __call__(self, img):

        # 此处插入代码
        dst = cv2.flip(img, self.mode)
        return(dst)



flip=Flip(mode=0)
img2=flip(img)
plt.imshow(img2)
```
3. 图片旋转 
  
```python
class Rotate:
    def __init__(self, degree,size):
        self.degree=degree
        self.size=size

    def __call__(self, img):

        # 此处插入代码
        rows,cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), self.degree, self.size)
        dst = cv2.warpAffine(img, M, (cols, rows))
        return(dst)

rotate=Rotate( 45, 0.7)
img2=rotate(img)
plt.imshow(img2)
```
4. 图片亮度调节
```python
class Brightness:
    def __init__(self,brightness_factor):
        self.brightness_factor=brightness_factor

    def __call__(self, img):

        # 此处插入代码
        dst =  np.uint8(np.clip((img +self.brightness_factor * 100), 0, 255))
        return(dst)

brightness=Brightness(0.6)
img2=brightness(img)
plt.imshow(img2)
```
5. 图片随机擦除
```python
import random
import math

class RandomErasing(object):
    def __init__(self, EPSILON=0.5, sl=0.02, sh=0.4, r1=0.3,
                 mean=[0., 0., 0.]):
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            area = img.shape[0] * img.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            
            # 此处插入代码
            if w < img.shape[0] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[1] - h)
                y1 = random.randint(0, img.shape[0] - w)
                if img.shape[2] == 3:
                    img[ x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    img[ x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    img[ x1:x1 + h, y1:y1 + w, 2] = self.mean[2]
                else:
                    img[x1:x1 + h, y1:y1 + w,0] = self.mean[0]
                    return img
            return img


erase = RandomErasing()
img2=erase(img)
plt.imshow(img2)    
```

