# 百度飞桨领航团零基础图像分类速成营 课程总结1（课前预习）
![paddlepaddle](https://paddlepaddle-org-cn.cdn.bcebos.com/paddle-site-front/favicon-128.png  "百度logo")

[课程链接](https://aistudio.baidu.com/aistudio/course/introduce/11939)	`https://aistudio.baidu.com/aistudio/course/introduce/11939`  
[飞桨官网](https://www.paddlepaddle.org.cn/)	`https://www.paddlepaddle.org.cn/`   

****
## 目录
* [python 的__call__()方法](#python 的__call__()方法)
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
