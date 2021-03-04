# 百度飞桨领航团零基础图像分类速成营 课程总结2
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/11939)	`https://aistudio.baidu.com/aistudio/course/introduce/11939`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   
[推荐学习网站](https://www.runoob.com/python3/python3-tutorial.html)	`https://www.runoob.com/python3/python3-tutorial.html`  

****
## 目录
* [图像处理的概念与基本操作](#图像处理的概念与基本操作)
* [OpenCV库进阶操作](#opencv库进阶操作)
* [图像分类任务概念导入](#图像分类任务概念导入)
* [PaddleClas数据增强代码解析](#paddleclas数据增强代码解析)
* [作业](#作业)
## 参考资料
* [面向初学者的OpenCV-Python教程](http://codec.wang/#/opencv/)
* [OpenCV学习—OpenCV图像处理基本操作](https://www.bilibili.com/video/BV1VC4y1h7wq?p=2)
* [OpenCV 4计算机视觉项目实战（原书第2版）](https://github.com/PacktPublishing/Learn-OpenCV-4-By-Building-Projects-Second-Edition)

# 课节2：图像处理基本概念

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

# PIL库的crop函数做简单的图片裁剪
# 使用Image中的open(file)方法可返回一个打开的图片，使用crop([x1,y1,x2,y2])可进行裁剪
r.crop((100,100,128,128))

# 将裁剪后图片的矩阵保存成文本，数字格式为整数
np.savetxt('lena-r-crop.txt', r.crop((100,100,128,128)), fmt='%4d')
```
分辨率=画面水平方向的像素值 * 画面垂直方向的像素值

使用OpenCV加载并保存图片

    加载图片，显示图片，保存图片  
    OpenCV函数：cv2.imread(), cv2.imshow(), cv2.imwrite()

 大部分人可能都知道电脑上的彩色图是以RGB(红-绿-蓝，Red-Green-Blue)颜色模式显示的，但OpenCV中彩色图是以B-G-R通道顺序存储的，灰度图只有一个通道。

 OpenCV默认使用BGR格式，而RGB和BGR的颜色转换不同，即使转换为灰度也是如此。一些开发人员认为R+G+B/3对于灰度是正确的，但最佳灰度值称为亮度（luminosity），并且具有公式：0.21R+0.72G+0.07*B

 图像坐标的起始点是在左上角，所以行对应的是y，列对应的是x。

 * 加载图片
 使用cv2.imread()来读入一张图片：

  * 参数1：图片的文件名

    * 如果图片放在当前文件夹下，直接写文件名就行了，如'lena.jpg'
    * 否则需要给出绝对路径，如'D:\OpenCVSamples\lena.jpg'
  
  * 参数2：读入方式，省略即采用默认值

    * cv2.IMREAD_COLOR：彩色图，默认值(1)
    * cv2.IMREAD_GRAYSCALE：灰度图(0)
    * cv2.IMREAD_UNCHANGED：包含透明通道的彩色图(-1)
```python
%matplotlib inline
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 加载彩色图
img = cv2.imread('lena.jpg', 1)
# 将彩色图的BGR通道顺序转成RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 显示图片
plt.imshow(img)

# 打印图片的形状
print(img.shape)
# 形状中包括行数、列数和通道数
height, width, channels = img.shape
# img是灰度图的话：height, width = img.shape
```
```python
# 将彩色图的BGR通道直接转为灰度图
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img,'gray')

cv2.imwrite('lena-grey.jpg',img)
```
```python
# 加载四通道图片
img = cv2.imread('cat.png',-1)
# 将彩色图的BGR通道顺序转成RGB，注意，在这一步直接丢掉了alpha通道
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
img.shape
```
## OpenCV库进阶操作
### ROI
Region of Interest，感兴趣区域。
### 通道分割与合并
彩色图的BGR三个通道是可以分开单独访问的，也可以将单独的三个通道合并成一副图像。分别使用cv2.split()和cv2.merge()
```python
import math
import random
import numpy as np
%matplotlib inline
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('lena.jpg')
# 通道分割
b, g, r = cv2.split(img)

RGB_Image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12,12))
#显示各通道信息
plt.subplot(141)
plt.imshow(RGB_Image,'gray')
plt.title('RGB_Image')
plt.subplot(142)
plt.imshow(r,'gray')
plt.title('R_Channel')
plt.subplot(143)
plt.imshow(g,'gray')
plt.title('G_Channel')
plt.subplot(144)
plt.imshow(b,'gray')
plt.title('B_Channel')
```
### 颜色空间转换

 最常用的颜色空间转换如下：
  * RGB或BGR到灰度（COLOR_RGB2GRAY，COLOR_BGR2GRAY）
  * RGB或BGR到YcrCb（或YCC）（COLOR_RGB2YCrCb，COLOR_BGR2YCrCb）
  * RGB或BGR到HSV（COLOR_RGB2HSV，COLOR_BGR2HSV）
  * RGB或BGR到Luv（COLOR_RGB2Luv，COLOR_BGR2Luv）
  * 灰度到RGB或BGR（COLOR_GRAY2RGB，COLOR_GRAY2BGR）

 颜色转换其实是数学运算，如灰度化最常用的是：gray=R*0.299+G*0.587+B*0.114。

特定颜色物体追踪

  HSV是一个常用于颜色识别的模型，相比BGR更易区分颜色，转换模式用COLOR_BGR2HSV表示。

  OpenCV中色调H范围为[0,179]，饱和度S是[0,255]，明度V是[0,255]。虽然H的理论数值是0°~360°，但8位图像像素点的最大值是255，所以OpenCV中除以了2，某些软件可能使用不同的尺度表示，所以同其他软件混用时，记得归一化。
  
  示例1：  
 一个使用HSV来只显示视频中蓝色物体的例子，步骤如下：

     1. 捕获视频中的一帧
     2. 从BGR转换到HSV
     3. 提取蓝色范围的物体
     4. 只显示蓝色物体

```python
# 加载一张有天空的图片
sky = cv2.imread('sky.jpg')

# 蓝色的范围，不同光照条件下不一样，可灵活调整
lower_blue = np.array([15, 60, 60])
upper_blue = np.array([130, 255, 255])

# 从BGR转换到HSV
hsv = cv2.cvtColor(sky, cv2.COLOR_BGR2HSV)
# inRange()：介于lower/upper之间的为白色，其余黑色
mask = cv2.inRange(sky, lower_blue, upper_blue)
# 只保留原图中的蓝色部分
# bitwise_and(src1, src2, dst=None, mask=None) 
res = cv2.bitwise_and(sky, sky, mask=mask)

# 保存颜色分割结果
cv2.imwrite('res.jpg', res)

res = cv2.imread('res.jpg')
res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
plt.imshow(res)
```
蓝色的HSV值的上下限lower和upper范围，参考标准蓝色BGR值的转换值 [[[120 255 255]]]：
```python
# uint8类型取值范围：0到255
blue = np.uint8([[[255, 0, 0]]])
hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
print(hsv_blue)
```
### 阈值分割

    使用固定阈值、自适应阈值和Otsu阈值法"二值化"图像  
    OpenCV函数：cv2.threshold(), cv2.adaptiveThreshold()

 * 固定阈值分割
 
 固定阈值分割很直接，一句话说就是像素点值大于阈值变成一类值，小于阈值变成另一类值。

 cv2.threshold()用来实现阈值分割，ret是return value缩写，代表当前的阈值。函数有4个参数：

   * 参数1：要处理的原图，一般是灰度图
   * 参数2：设定的阈值
   * 参数3：最大阈值，一般为255
   * 参数4：阈值的方式，主要有5种，详情：ThresholdTypes
     * 0: THRESH_BINARY  当前点值大于阈值时，取Maxval,也就是第四个参数，否则设置为0
     * 1: THRESH_BINARY_INV 当前点值大于阈值时，设置为0，否则设置为Maxval
     * 2: THRESH_TRUNC 当前点值大于阈值时，设置为阈值，否则不改变
     * 3: THRESH_TOZERO 当前点值大于阈值时，不改变，否则设置为0
     * 4:THRESH_TOZERO_INV  当前点值大于阈值时，设置为0，否则不改变

 ```python
 import cv2
 import matplotlib.pyplot as plt
 # 灰度图读入
 img = cv2.imread('lena.jpg', 0)
 # 颜色通道转换
 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 # 阈值分割
 ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

 # 应用5种不同的阈值方法
 # THRESH_BINARY  当前点值大于阈值时，取Maxval,否则设置为0
 ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
 # THRESH_BINARY_INV 当前点值大于阈值时，设置为0，否则设置为Maxval
 ret, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
 # THRESH_TRUNC 当前点值大于阈值时，设置为阈值，否则不改变
 ret, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
 # THRESH_TOZERO 当前点值大于阈值时，不改变，否则设置为0
 ret, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
 # THRESH_TOZERO_INV  当前点值大于阈值时，设置为0，否则不改变
 ret, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

 titles = ['Original', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
 images = [img, th1, th2, th3, th4, th5]

 plt.figure(figsize=(12,12))
 for i in range(6):
     plt.subplot(2, 3, i + 1)
     plt.imshow(images[i], 'gray')
     plt.title(titles[i], fontsize=8)
     plt.xticks([]), plt.yticks([])
 ```
 * 自适应阈值

 看得出来固定阈值是在整幅图片上应用一个阈值进行分割，它并不适用于明暗分布不均的图片。 cv2.adaptiveThreshold()自适应阈值会每次取图片的一小部分计算阈值，这样图片不同区域的阈值就不尽相同。它有5个参数，其实很好理解，先看下效果：

  * 参数1：要处理的原图
  * 参数2：最大阈值，一般为255
  * 参数3：小区域阈值的计算方式
    * ADAPTIVE_THRESH_MEAN_C：小区域内取均值
    * ADAPTIVE_THRESH_GAUSSIAN_C：小区域内加权求和，权重是个高斯核
  * 参数4：阈值方式（跟前面讲的那5种相同）
  * 参数5：小区域的面积，如11就是11*11的小块
  * 参数6：最终阈值等于小区域计算出的阈值再减去此值

 ```python
 # 自适应阈值对比固定阈值
 img = cv2.imread('lena.jpg', 0)

 # 固定阈值
 ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
 # 自适应阈值, ADAPTIVE_THRESH_MEAN_C：小区域内取均值
 th2 = cv2.adaptiveThreshold(
     img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
 # 自适应阈值, ADAPTIVE_THRESH_GAUSSIAN_C：小区域内加权求和，权重是个高斯核
 th3 = cv2.adaptiveThreshold(
     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)

 titles = ['Original', 'Global(v = 127)', 'Adaptive Mean', 'Adaptive Gaussian']
 images = [img, th1, th2, th3]
 plt.figure(figsize=(12,12))
 for i in range(4):
     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
     plt.title(titles[i], fontsize=8)
     plt.xticks([]), plt.yticks([])
 ```
 * Otsu阈值法就提供了一种自动高效的二值化方法

### 图像几何变换

     实现旋转、平移和缩放图片
     OpenCV函数：cv2.resize(), cv2.flip(), cv2.warpAffine()
     
  缩放图片

  缩放就是调整图片的大小，使用cv2.resize()函数实现缩放。可以按照比例缩放，也可以按照指定的大小缩放： 我们也可以指定缩放方法interpolation，更专业点叫插值方法，默认是INTER_LINEAR

  缩放过程中有五种插值方式：

   * cv2.INTER_NEAREST 最近邻插值
   * cv2.INTER_LINEAR 线性插值
   * cv2.INTER_AREA 基于局部像素的重采样，区域插值
   * cv2.INTER_CUBIC 基于邻域4x4像素的三次插值
   * cv2.INTER_LANCZOS4 基于8x8像素邻域的Lanczos插值
```python
img = cv2.imread('cat.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 按照指定的宽度、高度缩放图片
res = cv2.resize(img, (400, 500))
# 按照比例缩放，如x,y轴均放大一倍
res2 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

plt.subplot(121)
plt.imshow(res)
plt.title('res')
plt.subplot(122)
plt.imshow(res2)
plt.title('res2')
```
翻转图片

镜像翻转图片，可以用`cv2.flip()`函数：  
其中，参数2 = 0：垂直翻转(沿x轴)，参数2 > 0: 水平翻转(沿y轴)，参数2 < 0: 水平垂直翻转。
```python
dst = cv2.flip(img, 1)
plt.imshow(dst)
```
平移图片

要平移图片，我们需要定义下面这样一个矩阵，tx,ty是向x和y方向平移的距离：

$$
 M = \left[
 \begin{matrix}
   1 & 0 & t_x \newline
   0 & 1 & t_y 
  \end{matrix}
  \right] 
$$

平移是用仿射变换函数 cv2.warpAffine() 实现的：
```python
# 平移图片
import numpy as np
# 获得图片的高、宽
rows, cols = img.shape[:2]

# 定义平移矩阵，需要是numpy的float32类型
# x轴平移200，y轴平移500
M = np.float32([[1, 0, 100], [0, 1, 500]])
# 用仿射变换实现平移
dst = cv2.warpAffine(img, M, (cols, rows))
plt.imshow(dst)
```
### 绘图功能
    绘制各种几何形状、添加文字
    OpenCV函数：cv2.line(), cv2.circle(), cv2.rectangle(), cv2.ellipse(), cv2.putText()

绘制形状的函数有一些共同的参数，提前在此说明一下：

* img：要绘制形状的图片
* color：绘制的颜色
   * 彩色图就传入BGR的一组值，如蓝色就是(255,0,0)
   * 灰度图，传入一个灰度值就行
* thickness：线宽，默认为1；对于矩形/圆之类的封闭形状而言，传入-1表示填充形状
* lineType：线的类型。默认情况下，它是8连接的。cv2.LINE_AA 是适合曲线的抗锯齿线。

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
