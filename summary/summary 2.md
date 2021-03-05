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
    OpenCV函数：cv2.line(), cv2.polylines()，cv2.circle(), cv2.rectangle(), cv2.ellipse(), cv2.putText()，cv2.polylines()

绘制形状的函数有一些共同的参数，提前在此说明一下：

* img：要绘制形状的图片
* color：绘制的颜色
   * 彩色图就传入BGR的一组值，如蓝色就是(255,0,0)
   * 灰度图，传入一个灰度值就行
* thickness：线宽，默认为1；对于矩形/圆之类的封闭形状而言，传入-1表示填充形状
* lineType：线的类型。默认情况下，它是8连接的。cv2.LINE_AA 是适合曲线的抗锯齿线。

画线
```python
img = cv2.imread('lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 画一条线宽为5的红色直线，参数2：起点，参数3：终点
cv2.line(img, (0, 0), (800, 512), (255, 0, 0), 5)
plt.imshow(img)
```
画矩形
```python
img = cv2.imread('lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 画一个矩形，左上角坐标(40, 40)，右下角坐标(80, 80)，框颜色为绿色
img1 = cv2.rectangle(img, (40, 40), (80, 80), (0, 255, 0),1) 

# 画一个矩形，左上角坐标(40, 40)，右下角坐标(80, 80)，框颜色为绿色，填充这个矩形,CV_FILLED
img2 = cv2.rectangle(img, (40, 40), (80, 80), (0, 255, 0),-1) 

plt.subplot(121)
plt.imshow(img1)
plt.title('img1')
plt.subplot(122)
plt.imshow(img2)
plt.title('img2')
```
添加文字

使用cv2.putText()添加文字，它的参数也比较多，同样请对照后面的代码理解这几个参数：
* 参数2：要添加的文本
* 参数3：文字的起始坐标（左下角为起点）
* 参数4：字体
* 参数5：文字大小（缩放比例）

```python
# 添加文字，加载字体
font = cv2.FONT_HERSHEY_SIMPLEX
# 添加文字hello
img = cv2.putText(img, 'hello', (10, 200), font,
            4, (255, 255, 255), 2, lineType=cv2.LINE_AA)
plt.imshow(img)
```
引入中文
```python
# 参考资料 https://blog.csdn.net/qq_41895190/article/details/90301459
# 引入PIL的相关包
from PIL import Image, ImageFont,ImageDraw
from numpy import unicode

def paint_chinese_opencv(im,chinese,pos,color):
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    # 加载中文字体
    font = ImageFont.truetype('NotoSansCJKsc-Medium.otf',25)
    # 设置颜色
    fillColor = color
    # 定义左上角坐标
    position = pos
    # 判断是否中文字符
    if not isinstance(chinese,unicode):
        # 解析中文字符
        chinese = chinese.decode('utf-8')
    # 画图
    draw = ImageDraw.Draw(img_PIL)
    # 画文字
    draw.text(position,chinese,font=font,fill=fillColor)
    # 颜色通道转换
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    return img
```
添加中文
```python
plt.imshow(paint_chinese_opencv(img,'中文',(100,100),(255,255,0)))
```
### 图像间数学运算
    图片间的数学运算，如相加、按位运算等
    OpenCV函数：cv2.add(), cv2.addWeighted(), cv2.bitwise_and()

图片相加

要叠加两张图片，可以用cv2.add()函数，相加两幅图片的形状（高度/宽度/通道数）必须相同。numpy中可以直接用res = img + img1相加，但这两者的结果并不相同：

```python
x = np.uint8([250])
y = np.uint8([10])
print(cv2.add(x, y))  # 250+10 = 260 => 255
print(x + y)  # 250+10 = 260 % 256 = 4
```
图像混合

图像混合cv2.addWeighted()也是一种图片相加的操作，只不过两幅图片的权重不一样，γ相当于一个修正值：

dst=α×img1+β×img2+γdst = \alpha\times img1+\beta\times img2 + \gamma
dst=α×img1+β×img2+γ
```python
img1 = cv2.imread('lena.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('cat.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = cv2.resize(img2, (350, 350))
# 两张图片相加
res = cv2.add(img1, img2)
# 两张图片加权相加
res2 = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)

plt.subplot(131)
plt.imshow(res)
plt.title('cv2.add')
plt.subplot(132)
plt.imshow(img1+img2)
plt.title('img1+img2')
plt.subplot(133)
plt.imshow(res2)
plt.title('cv2.addWeighted')
```
按位操作

    按位操作包括按位与/或/非/异或操作
    cv2.bitwise_and(), cv2.bitwise_or(), cv2.bitwise_not(), cv2.bitwise_xor()

如果将两幅图片直接相加会改变图片的颜色，如果用图像混合，则会改变图片的透明度，所以我们需要用按位操作。掩膜（mask）的概念：掩膜是用一副二值化图片对另外一幅图片进行局部的遮挡。掩膜图像白色区域是对需要处理图像像素的保留，黑色区域是对需要处理图像像素的剔除，其余按位操作原理类似只是效果不同而已。
```python
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('lena.jpg')
img2 = cv2.imread('logo.jpg')
img2 = cv2.resize(img2, (350, 350)) # logo 图片比 lena 还大，转换成一样大

rows, cols = img2.shape[:2]
roi = img1[:rows, :cols]

# 创建掩膜
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# cv2.threshold(src, thresh, maxval, type[, dst]) → retval, dst 返回阈值和图像。cv2.THRESH_BINARY 大于阈值10 取 255，否则0
ret, mask = cv2.threshold(img2gray, 240, 255, cv2.THRESH_BINARY) # mask 背景依然是白色，彩色logo是黑色
mask_inv = cv2.bitwise_not(mask) # mask 背景黑色，彩色logo  白色

# 保留除logo外的背景
img2_bg = cv2.bitwise_and(roi, roi, mask=mask)
img2_fg = cv2.bitwise_and(img2, img2, mask= mask_inv)
dst = cv2.add(img2_bg, img2_fg)  # 进行融合
img1[:rows, :cols] = dst  # 融合后放在原图上

plt.subplot(151);plt.imshow(mask,'gray');plt.title('mask')
plt.subplot(152);plt.imshow(mask_inv,'gray');plt.title('imask_inv')
plt.subplot(153);plt.imshow(img2_bg);plt.title('img2_bg')
plt.subplot(154);plt.imshow(img2_fg);plt.title('img2_fg')
plt.subplot(155);plt.imshow(dst);plt.title('dst')
```
### 平滑图像
    模糊/平滑图片来消除图片噪声
    OpenCV函数：cv2.blur(), cv2.GaussianBlur(), cv2.medianBlur(), cv2.bilateralFilter()
滤波与模糊

关于滤波和模糊：
* 它们都属于卷积，不同滤波方法之间只是卷积核不同（对线性滤波而言）
* 低通滤波器是模糊，高通滤波器是锐化
低通滤波器就是允许低频信号通过，在图像中边缘和噪点都相当于高频部分，所以低通滤波器用于去除噪点、平滑和模糊图像。高通滤波器则反之，用来增强图像边缘，进行锐化处理。

    常见噪声有椒盐噪声和高斯噪声，椒盐噪声可以理解为斑点，随机出现在图像中的黑点或白点；高斯噪声可以理解为拍摄图片时由于光照等原因造成的噪声。

均值滤波

均值滤波是一种最简单的滤波处理，它取的是卷积核区域内元素的均值，用`cv2.blur()`实现，如3×3的卷积核：

$$
 kernel = \frac{1}{9}\left[
 \begin{matrix}
   1 & 1 & 1 \newline
   1 & 1 & 1 \newline
   1 & 1 & 1
  \end{matrix}
  \right]
$$
```python
img = cv2.imread('lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
blur = cv2.blur(img, (9, 9))  # 均值模糊
plt.imshow(blur)
```
方框滤波

方框滤波跟均值滤波很像，如3×3的滤波核如下：

$$
k = a\left[
 \begin{matrix}
   1 & 1 & 1 \newline
   1 & 1 & 1 \newline
   1 & 1 & 1
  \end{matrix}
  \right]
$$

用 cv2.boxFilter() 函数实现，当可选参数normalize为True的时候，方框滤波就是均值滤波，上式中的a就等于1/9；normalize为False的时候，a=1，相当于求区域内的像素和。
```python
# 前面的均值滤波也可以用方框滤波实现：normalize=True
blur = cv2.boxFilter(img, -1, (9, 9), normalize=True)
plt.imshow(blur)
```
高斯滤波

前面两种滤波方式，卷积核内的每个值都一样，也就是说图像区域中每个像素的权重也就一样。高斯滤波的卷积核权重并不相同：中间像素点权重最高，越远离中心的像素权重越小。

显然这种处理元素间权值的方式更加合理一些。图像是2维的，所以我们需要使用[2维的高斯函数](https://en.wikipedia.org/wiki/Gaussian_filter)，比如OpenCV中默认的3×3的高斯卷积核：

$$
k = \left[
 \begin{matrix}
   0.0625 & 0.125 & 0.0625 \newline
   0.125 & 0.25 & 0.125 \newline
   0.0625 & 0.125 & 0.0625
  \end{matrix}
  \right]
$$
OpenCV中对应函数为`cv2.GaussianBlur(src,ksize,sigmaX)`:
参数3 σx值越大，模糊效果越明显。高斯滤波相比均值滤波效率要慢，但可以有效消除高斯噪声，能保留更多的图像细节，所以经常被称为最有用的滤波器。均值滤波与高斯滤波的对比结果如下（均值滤波丢失的细节更多）
```python
# 均值滤波vs高斯滤波
gaussian = cv2.GaussianBlur(img, (9, 9), 1)  # 高斯滤波
plt.imshow(gaussian)
```
中值滤波

中值又叫中位数，是所有数排序后取中间的值。中值滤波就是用区域内的中值来代替本像素值，所以那种孤立的斑点，如0或255很容易消除掉，适用于去除椒盐噪声和斑点噪声。中值是一种非线性操作，效率相比前面几种线性滤波要慢。
```python
median = cv2.medianBlur(img, 9)  # 中值滤波
plt.imshow(median)
```
双边滤波

模糊操作基本都会损失掉图像细节信息，尤其前面介绍的线性滤波器，图像的边缘信息很难保留下来。然而，边缘（edge）信息是图像中很重要的一个特征，所以这才有了双边滤波。用cv2.bilateralFilter()函数实现：可以看到，双边滤波明显保留了更多边缘信息。
```python
blur = cv2.bilateralFilter(img, 9, 75, 75)  # 双边滤波
plt.imshow(blur)
```
图像锐化
```python
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
dst = cv2.filter2D(img, -1, kernel=kernel)
plt.imshow(dst)
```
边缘检测
[Canny J . A Computational Approach To Edge Detection[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1986, PAMI-8(6):679-698.](https://www.computer.org/cms/Computer.org/Transactions%20Home%20Pages/TPAMI/PDFs/top_ten_6.pdf)

     OpenCV函数：cv2.Canny()
     
Canny边缘检测方法常被誉为边缘检测的最优方法：

cv2.Canny()进行边缘检测，参数2、3表示最低、高阈值。

Canny边缘提取的具体步骤如下：

1. 使用5×5高斯滤波消除噪声：

边缘检测本身属于锐化操作，对噪点比较敏感，所以需要进行平滑处理。
$$
K=\frac{1}{256}\left[
 \begin{matrix}
   1 & 4 & 6 & 4 & 1 \newline
   4 & 16 & 24 & 16 & 4  \newline
   6 & 24 & 36 & 24 & 6  \newline
   4 & 16 & 24 & 16 & 4  \newline
   1 & 4 & 6 & 4 & 1
  \end{matrix}
  \right]
$$
2. 计算图像梯度的方向：

首先使用Sobel算子计算两个方向上的梯度$ G_x $和$ G_y $，然后算出梯度的方向：
$$
\theta=\arctan(\frac{G_y}{G_x})
$$
保留这四个方向的梯度：0°/45°/90°/135°，有什么用呢？我们接着看。

3. 取局部极大值：

梯度其实已经表示了轮廓，但为了进一步筛选，可以在上面的四个角度方向上再取局部极大值

4. 滞后阈值：

经过前面三步，就只剩下0和可能的边缘梯度值了，为了最终确定下来，需要设定高低阈值：
- 像素点的值大于最高阈值，那肯定是边缘
- 同理像素值小于最低阈值，那肯定不是边缘
- 像素值介于两者之间，如果与高于最高阈值的点连接，也算边缘，所以上图中C算，B不算

Canny推荐的高低阈值比在2:1到3:1之间。

```python
# 先阈值分割后检测
img = cv2.imread('lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
_, thresh = cv2.threshold(img, 124, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(thresh, 30, 70)
plt.imshow(edges)
```
### 腐蚀与膨胀
    OpenCV函数：cv2.erode(), cv2.dilate(), cv2.morphologyEx()

形态学操作其实就是改变物体的形状，比如腐蚀就是"变瘦"，膨胀就是"变胖"。

#### 腐蚀

腐蚀的效果是把图片"变瘦"，其原理是在原图的小区域内取局部最小值。因为是二值化图，只有0和255，所以小区域内有一个是0该像素点就为0。

这样原图中边缘地方就会变成0，达到了瘦身目的

OpenCV中用 cv2.erode() 函数进行腐蚀，只需要指定核的大小就行：
```python
img = cv2.imread('lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel)  # 腐蚀
plt.imshow(erosion)
```
这个核也叫结构元素，因为形态学操作其实也是应用卷积来实现的。结构元素可以是矩形/椭圆/十字形，可以用 cv2.getStructuringElement() 来生成不同形状的结构元素，比如：

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆结构
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形结构
```
#### 膨胀

膨胀与腐蚀相反，取的是局部最大值，效果是把图片"变胖"：
```python
dilation = cv2.dilate(img, kernel)  # 膨胀
plt.imshow(dilation)
```
开/闭运算 
先腐蚀后膨胀叫开运算（因为先腐蚀会分开物体，这样容易记住），其作用是：分离物体，消除小区域。  
闭运算则相反：先膨胀后腐蚀（先膨胀会使白色的部分扩张，以至于消除/"闭合"物体里面的小黑洞，所以叫闭运算）。  
这类形态学操作用cv2.morphologyEx()函数实现
```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算

plt.subplot(121);plt.imshow(closing);plt.title('closing')
plt.subplot(122);plt.imshow(opening);plt.title('opening')
```
### 使用OpenCV摄像头与加载视频
    OpenCV函数：cv2.VideoCapture(), cv2.VideoWriter()
```python
import IPython
# 加载视频
IPython.display.Video('demo_video.mp4')
```
#### 打开摄像头

要使用摄像头，需要使用`cv2.VideoCapture(0)`创建VideoCapture对象，参数0指的是摄像头的编号，如果你电脑上有两个摄像头的话，访问第2个摄像头就可以传入1，依此类推。

``` python
# 打开摄像头并灰度化显示
import cv2

capture = cv2.VideoCapture(0)

while(True):
    # 获取一帧
    ret, frame = capture.read()
    # 将这帧转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break
```

`capture.read()`函数返回的第1个参数ret(return value缩写)是一个布尔值，表示当前这一帧是否获取正确。`cv2.cvtColor()`用来转换颜色，这里将彩色图转成灰度图。

另外，通过`cap.get(propId)`可以获取摄像头的一些属性，比如捕获的分辨率，亮度和对比度等。propId是从0~18的数字，代表不同的属性，完整的属性列表可以参考：[VideoCaptureProperties](https://docs.opencv.org/4.0.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d)。也可以使用`cap.set(propId,value)`来修改属性值。比如说，我们在while之前添加下面的代码：

``` python
# 获取捕获的分辨率
# propId可以直接写数字，也可以用OpenCV的符号表示
width, height = capture.get(3), capture.get(4)
print(width, height)

# 以原分辨率的一倍来捕获
capture.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height * 2)
```

> 经验之谈：某些摄像头设定分辨率等参数时会无效，因为它有固定的分辨率大小支持，一般可在摄像头的资料页中找到。

#### 播放本地视频

跟打开摄像头一样，如果把摄像头的编号换成视频的路径就可以播放本地视频了。回想一下`cv2.waitKey()`，它的参数表示暂停时间，所以这个值越大，视频播放速度越慢，反之，播放速度越快，通常设置为25或30。

```python
# 播放本地视频
capture = cv2.VideoCapture('demo_video.mp4')

while(capture.isOpened()):
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitKey(30) == ord('q'):
        break
```

#### 录制视频

之前我们保存图片用的是`cv2.imwrite()`，要保存视频，我们需要创建一个`VideoWriter`的对象，需要给它传入四个参数：

- 输出的文件名，如'output.avi'
- 编码方式[FourCC](https://baike.baidu.com/item/fourcc/6168470?fr=aladdin)码
- 帧率[FPS](https://baike.baidu.com/item/FPS/3227416)
- 要保存的分辨率大小

FourCC是用来指定视频编码方式的四字节码，所有的编码可参考[Video Codecs](http://www.fourcc.org/codecs.php)。如MJPG编码可以这样写： `cv2.VideoWriter_fourcc(*'MJPG')`或`cv2.VideoWriter_fourcc('M','J','P','G')`

```python
capture = cv2.VideoCapture(0)

# 定义编码方式并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
outfile = cv2.VideoWriter('output.avi', fourcc, 25., (640, 480))

while(capture.isOpened()):
    ret, frame = capture.read()

    if ret:
        outfile.write(frame)  # 写入文件
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
```
## 图像分类任务概念导入
### 计算机视觉的子任务:
* Image Classification： 图像分类，用于识别图像中物体的类别（如：bottle、cup、cube）。
* Object Localization： 目标检测，用于检测图像中每个物体的类别，并准确标出它们的位置。
* Semantic Segmentation： 图像语义分割，用于标出图像中每个像素点所属的类别，属于同一类别的像素点用一个颜色标识。
* Instance Segmentation： 实例分割，值得注意的是，目标检测任务只需要标注出物体位置，实例分割任务不仅要标注出物体位置，还需要标注出物体的外形轮廓。
### 图像分类问题的经典数据集:
MNIST手写数字识别  
MNIST是一个手写体数字的图片数据集，该数据集来由美国国家标准与技术研究所（National Institute of Standards and Technology (NIST)）发起整理，一共统计了来自250个不同的人手写数字图片，其中50%是高中生，50%来自人口普查局的工作人员。该数据集的收集目的是希望通过算法，实现对手写数字的识别。[数据集链接](http://yann.lecun.com/exdb/mnist/)

Cifar数据集
* CIFAR-10
CIFAR-10数据集由10个类的60000个32x32彩色图像组成，每个类有6000个图像。有50000个训练图像和10000个测试图像。    
数据集分为五个训练批次和一个测试批次，每个批次有10000个图像。测试批次包含来自每个类别的恰好1000个随机选择的图像。训练批次以随机顺序包含剩余图像，但一些训练批次可能包含来自一个类别的图像比另一个更多。总体来说，五个训练集之和包含来自每个类的正好5000张图像。    
[CIFAR-10 python版本](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)   
* CIFAR-100
CIFAR-100数据集就像CIFAR-10，除了它有100个类，每个类包含600个图像。，每类各有500个训练图像和100个测试图像。CIFAR-100中的100个类被分成20个超类。每个图像都带有一个“精细”标签（它所属的类）和一个“粗糙”标签（它所属的超类）    
[CIFAR-100 python版本](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)    

ImageNet数据集
* ImageNet数据集是一个计算机视觉数据集，是由斯坦福大学的李飞飞教授带领创建。该数据集包合 14,197,122张图片和21,841个Synset索引。Synset是WordNet层次结构中的一个节点，它又是一组同义词集合。ImageNet数据集一直是评估图像分类算法性能的基准。  
* ImageNet 数据集是为了促进计算机图像识别技术的发展而设立的一个大型图像数据集。2016 年ImageNet 数据集中已经超过干万张图片，每一张图片都被手工标定好类别。ImageNet 数据集中的图片涵盖了大部分生活中会看到的图片类别。ImageNet最初是拥有超过100万张图像的数据集。如图下图所示，它包含了各种各样的图像，并且每张图像都被关联了标签（类别名）。每年都会举办使用这个巨大数据集的ILSVRC图像识别大赛。
[http://image-net.org/download-imageurls](http://image-net.org/download-imageurls)

## PaddleClas数据增强代码解析
```python
# 创建一副图片
img = cv2.imread('lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

```python
class RandFlipImage(object):
    """ random flip image 随机翻转图片
        flip_code:
            1: Flipped Horizontally 水平翻转
            0: Flipped Vertically 上下翻转
            -1: Flipped Horizontally & Vertically 水平、上下翻转
    """

    def __init__(self, flip_code=1):
        # 设置一个翻转参数，1、0或-1
        assert flip_code in [-1, 0, 1
                             ], "flip_code should be a value in [-1, 0, 1]"
        self.flip_code = flip_code

    def __call__(self, img):
        # 随机生成0或1（即是否翻转）
        if random.randint(0, 1) == 1:
            return cv2.flip(img, self.flip_code)
        else:
            return img

# 初始化实例，默认随机水平翻转
flip = RandFlipImage()
plt.imshow(flip(img))
```
```python
class RandCropImage(object):
    """ random crop image """
    """ 随机裁剪图片 """

    def __init__(self, size, scale=None, ratio=None, interpolation=-1):

        self.interpolation = interpolation if interpolation >= 0 else None
        if type(size) is int:
            self.size = (size, size)  # (h, w)
        else:
            self.size = size

        self.scale = [0.08, 1.0] if scale is None else scale
        self.ratio = [3. / 4., 4. / 3.] if ratio is None else ratio

    def __call__(self, img):
        size = self.size
        scale = self.scale
        ratio = self.ratio

        aspect_ratio = math.sqrt(random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio

        img_h, img_w = img.shape[:2]

        bound = min((float(img_w) / img_h) / (w**2),
                    (float(img_h) / img_w) / (h**2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = img_w * img_h * random.uniform(scale_min, scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)

        i = random.randint(0, img_w - w)
        j = random.randint(0, img_h - h)

        img = img[j:j + h, i:i + w, :]
        if self.interpolation is None:
            return cv2.resize(img, size)
        else:
            return cv2.resize(img, size, interpolation=self.interpolation)
            
img = cv2.imread('lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
crop = RandCropImage(350)
plt.imshow(crop(img))
```
```python

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
        
img = cv2.imread('lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
erase = RandomErasing()
plt.imshow(erase(img))
```

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

filename = 'lena.jpg'
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

img = cv2.imread('lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
erase = RandomErasing()
img2=erase(img)
plt.imshow(img2)    
```
