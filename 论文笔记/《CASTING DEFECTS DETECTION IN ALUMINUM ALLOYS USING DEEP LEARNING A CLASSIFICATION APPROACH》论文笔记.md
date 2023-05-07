# 《CASTING DEFECTS DETECTION IN ALUMINUM ALLOYS USING DEEP LEARNING: A CLASSIFICATION APPROACH》论文笔记

## 基于深度学习的的铝合金铸造缺陷检测：一种分类方法

铝合金材料与铁基材料相比较，他们具有<font color='red'>良好的力学性能</font>，且<font color='red'>质量密度和耐腐蚀性较低</font>。而铸件缺陷会对其最终的力学性能起着重要的作用。

常用的人工智能算法基本可以分为<font color='orange'>浅层机器学习（ML）</font>和<font color='orange'>深度学习（DL）</font>。与标准的ML方法相比，DL方法往往具有<font color='red'>更大的计算量</font>，但在许多类型的任务上，它很有可能获得<font color='red'>更高的精确度</font>。对于大多数包括图像操作的DL问题，CNN方法应该是首选。

在铸造和焊接时，<font color='red'>铸件的质量控制是决定铸件是否能够承受使用寿命的基础</font>。

CNN模型表示如下图所示：

![image-20230214151657852](F:\Learning_notes\论文笔记\image-20230214151657852.png)

该网络由3个二维卷积层块组成。每个卷积滤波器后都使用了RELU函数激活、maxpooling层和批量归一化。maxpooling过滤器的大小为（2,2），池化层的步长为2。对于第一，二，三层的卷积块，滤波器的大小分别为$(15,15)*32,(10,10)*32,(5,5)*32$。