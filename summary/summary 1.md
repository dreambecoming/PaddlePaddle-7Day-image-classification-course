# 百度飞桨领航团零基础图像分类速成营 课程总结1（课前预习）
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/11939)	`https://aistudio.baidu.com/aistudio/course/introduce/11939`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   

****
## 目录
* [python 的__call__()方法](#python-的__call__方法)
* [图像基础知识](#图像基础知识)
* [图片格式](#图片格式)
* [常用库](#常用库)
* [常见操作](#常见操作)
* [预习作业](#预习作业)


# 课节1：课前资料

## python 的__call__()方法


外加总结：

调用父类(超类)的方法：

1. 明确指定：
```python
class  C(P):
     def __init__(self):
             P.__init__(self)
             print 'calling Cs construtor'
 ```
2. 使用super()方法 ：
```python
class  C(P):
    def __init__(self):
            super(C,self).__init__()
            print 'calling Cs construtor'
```
* 单继承结果一样，多继承super更好。
* 对于定义的类，在Python中会创建一个MRO(Method Resolution Order)列表，它代表了类继承的顺序。MRO的查找顺序是按广度优先
* super().method()是调用父类中的方法，这个搜索顺序当然是按照MRO从前向后开始进行的。

     super([type][, object-or-type])
     
     根据官方文档，super函数返回一个委托类type的父类或者兄弟类方法调用的代理对象。super函数用来调用已经再子类中重写过的父类方法。

     常见的是直接调用super(),这其实是super(type, obj)的简写方式，将当前的类传入type参数，同时将实例对象传入type-or-object参数，这两个实参必须确保isinstance(obj, type)为True。

     使用该方法调用的super函数返回的代理类是obj所属类的MRO中，排在type之后的下一个父类。
     
示例1：
```
继承结构：
 Base
  /  \
 /    \
A      B
 \    /
  \  /
   C
```

* 不使用super:
```python 
class Base(object):
    def __init__(self):
        print ("enter Base")
        print ("leave Base")

class A(Base):
    def __init__(self):
        print ("enter A")
        Base().__init__()
        print ("leave A")

class B(Base):
    def __init__(self):
        print ("enter B")
        Base().__init__()
        print ("leave B")

class C(A, B):
    def __init__(self):
        print ("enter C")
        A().__init__()
        B().__init__()
        print ("leave C")
        
print (C.mro())
C()
```
输出：
```python
[<class '__main__.C'>, <class '__main__.A'>, <class '__main__.B'>, <class '__main__.Base'>, <class 'object'>]
enter C
enter A
enter Base
leave Base
enter Base
leave Base
leave A
enter A
enter Base
leave Base
enter Base
leave Base
leave A
enter B
enter Base
leave Base
enter Base
leave Base
leave B
enter B
enter Base
leave Base
enter Base
leave Base
leave B
leave C
<__main__.C at 0x7f734302db10>
```

* 使用super:
```python
class Base(object):
    def __init__(self):
        print ("enter Base")
        print ("leave Base")

class A(Base):
    def __init__(self):
        print ("enter A")
        super(A, self).__init__()
        print ("leave A")

class B(Base):
    def __init__(self):
        print ("enter B")
        super(B, self).__init__()
        print ("leave B")

class C(A, B):
    def __init__(self):
        print ("enter C")
        super(C, self).__init__()
        print ("leave C")
        
print (C.mro())
C()
```
输出：

```python
[<class '__main__.C'>, <class '__main__.A'>, <class '__main__.B'>, <class '__main__.Base'>, <class 'object'>]
enter C
enter A
enter B
enter Base
leave Base
leave B
leave A
leave C
<__main__.C at 0x7f734304e090>
```


* isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。

    * isinstance() 与 type() 区别：

        type() 不会认为子类是一种父类类型，不考虑继承关系。  

        isinstance() 会认为子类是一种父类类型，考虑继承关系。
   * isinstance(object, classinfo)  
     object -- 实例对象。  
     classinfo -- 可以是直接或间接类名、基本类型或者由它们组成的元组。其中基本类型：int，float，bool，complex，str(字符串)，list，dict(字典)，set，tuple。
     ```python
     class A:
         pass

     class B(A):
         pass

     isinstance(A(), A)    # returns True
     type(A()) == A        # returns True
     isinstance(B(), A)    # returns True
     type(B()) == A        # returns False
     ```
     ```python
     print('A\t:',A)
     print('A()\t:',A())
     print('type(A())\t:',type(A()))
     print('type(A())== A\t:', type(A()) == A  )
     print('判断A()实例对象，A类名 的类型是否相同：',isinstance(A(), A))
     ```
     输出：
     ```python
     A	: <class '__main__.A'>
     A()	: <__main__.A object at 0x7f7342ff3350>
     type(A())	: <class '__main__.A'>
     type(A())== A	: True
     判断A()实例对象，A类名 的类型是否相同： True
     ```
     
        
   



## 图像基础知识
* 在计算机中, 图像是由一个个像素点组成，像素点就是颜色点，而颜色最简单的方式就是用 RGB 或 RGBA 表示。如果有A通道就表明这个图像可以有透明效果。  
R,G,B 每个分量一般是用一个字节(8位)来表示，所以RGB图中每个像素大小就是3 `*` 8=24位图, 而RGBA图中每个像素大小是4 `*` 8=32位。
* 图像是二维数据，数据在内存中只能一维存储，二维转一维有不同的对应方式。比较常见的只有两种方式: 按像素“行排列” 从上往下 或者 从下往上。
* 一般只会有RGB,BGR, RGBA, RGBA, BGRA这几种排列据。 绝大多数图形库或环境是BGR/BGRA排列，cocoa中的NSImage或UIImage是RGBA排列。

像素32位对齐  
在x86体系下，cpu一次处理32整数倍的数据会更快，图像处理中经常会按行为单位来处理像素。24位图，宽度不是4的倍数时，其行字节数将不是32整数倍。这时可以采取在行尾添加冗余数据的方式，使其行字节数为32的倍数。 比如，如果图像宽为5像素，不做32位对齐的话，其行位数为24`*`5=120，120不是32的倍数。是32整数倍并且刚好比120大的数是128，也就只需要在其行尾添加1字节(8位)的冗余数据即可。(一个以空间换时间的例子) 有个公式可以轻松计算出32位对齐后每行应该占的字节数
* 1. 调用函数计算：`byteNum = ceil(bm.bmWidth / 32) * 4`。图片实际存储宽度 除以32 再向上取整，32位相当于4字节，所以 乘以4 得出行字节数。
* 2. 使用位运算：`byteNum = ((bm.bmWidth  + 31) & ~31) >> 3`。
```python
#图像宽为5像素，不做32位对齐的话，其行位数为24*5=120，120不是32的倍数。是32整数倍并且刚好比120大的数是128
print('二进制31\t',bin(31))
print('二进制~31\t',bin(~31))
print('二进制32\t',bin(32))

print('\n')
print('二进制120+31=151\t',bin(151))
print('二进制~31\t',bin(~31))
print('十进制(151 & ~31):\t{a}\n二进制(151 & ~31):\t{b} '.format(a=(151 & ~31),b=bin(151 & ~31)))
```
```python
二进制31	 0b11111
二进制~31	 -0b100000
二进制32	 0b100000


二进制120+31=151	 0b10010111
二进制~31	 -0b100000
十进制(151 & ~31):	128
二进制(151 & ~31):	0b10000000 
```
图片实际存储宽度 加了31 与  取反后的31 按位与之后，后面5个位都置为0，这相当于向下取32的最大倍数；>>3右位移3位，即除以8，得出行字节数。
## 图片格式

## 常用库

## 常见操作


## 预习作业

飞桨安装文档：https://paddlepaddle.org.cn/install/quick

1.本地安装PaddlePaddle，截图并上传

     提示：使用 python 进入python解释器，输入import paddle ，再输入 paddle.utils.run_check()。
     
   ![](https://ai-studio-static-online.cdn.bcebos.com/ab991a76629f42eba4ed7235dbf60e9d1219104cdf2d433ead9d742654399b5b)


2.本地安装open-cv-python，截图上传

     终端下输入： pip show opencv-python
     
   ![](https://ai-studio-static-online.cdn.bcebos.com/e8d9c1486b4945dab96135794119b7394d5786203d584377b421015c9e1a7029)
