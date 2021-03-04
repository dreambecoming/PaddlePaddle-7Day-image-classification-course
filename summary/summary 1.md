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


对于定义的类，在Python中会创建一个MRO(Method Resolution Order)列表，它代表了类继承的顺序。查看MRO列表：

print C.mro()

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
 
c=C()
```
* 能够在多继承中按照广度优先的继承树搜索关系链来有序地调用与父类相同名称的方法，且在每个子类拥有super的都会执行父类方法

* isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。

    * isinstance() 与 type() 区别：

        type() 不会认为子类是一种父类类型，不考虑继承关系。  

        isinstance() 会认为子类是一种父类类型，考虑继承关系。
   * isinstance(object, classinfo)
     object -- 实例对象。  
     classinfo -- 可以是直接或间接类名、基本类型或者由它们组成的元组。
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
