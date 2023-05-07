# **《Research on Automatic Recognition of Casting Defects Based on Deep Learning》论文笔记**

## 基于深度学习的铸件缺陷自动识别研究

针对<font color='red'>人工检测速度慢，效率低，鲁棒性差</font>等问题，提出了一种YOLOv3_134模型，该模型<font color='red'>提高了缺陷的检测能力，特别是微小缺陷的检测能力</font>。改论文使用<font color='orange'>DR图像</font>对缺陷进行检测。DR图像是X光成像。

铸件缺陷检测的重要性：

><font color='red'>气孔，裂纹，夹渣等缺陷影响铸件的质量</font>
>
><font color='red'>降低铸件的性能</font>
>
><font color='red'>甚至会造成严重的交通事故和国家财产损失</font>

使用人工检测的缺陷：

> <font color='red'>人工成本高，效率低</font>
>
> 由于检测者的个人能力和主观性，导致<font color='red'>准确性低</font>
>
> <font color='red'>检测任务量大，检测强度大</font>

基于深度学习的缺陷检测方法的优点：

> <font color='red'>较低的人工劳动量和较高的检测效率</font>
>
> 测试标准统一客观，<font color='red'>稳定性高</font>
>
> 面对大量的检测任务时，<font color='red'>较少的误检和漏检</font>

<font color='orange'>传统的缺陷检测</font>方法<font color='red'>只能识别缺陷的大致位置和大小信息，缺乏缺陷的分类信息</font>。

>  Liu et al.[1] established deep belief networks and trained it according to the source domain sample feature, but it needs to use transfer learning to extract feature information. Yu et al.[2]  proposed a deep convolution neural network-based method to detect the casting defect in X-ray images, which is not effective in detecting small defects of casting.
>
> To further improve the accuracy of the defect detection, Ferguson et al[3]. showed the influence of defect detection of casting used varied feature extractor, such as VGG-16 and ResNet-101. Xu et al[3]. used feature cascade in VGG-16 to achieve better performance in defect detection. However, the methods above are not real-time, and cannot meet the balance between the speed and accuracy of defect detection.

翻译：

> 刘[1]等人建立了深度信念网络，并根据源域样本特征对其进行训练，但需要使用迁移学习来提取特征。于[2]等人提出了一种基于深度卷积网络的X射线特向铸件缺陷检测方法，该方法对铸件微小缺陷检测效果不佳。
>
> 为了进一步提高缺陷检测的准确性，Ferguson等人[3]研究了VGG-16和Resnet-101等多种特征提取器对铸件缺陷检测的影响。徐等人[14]在VGG-16中使用特征级联来实现更好地缺陷检测性能。但是上述的方法都不具有实时性，不能满足缺陷检测速度和精度之间的平衡。

文献出处：

> [1] R. Liu, M. Yao, and X. Wang, ‘‘Defects detection based on deep learning and transfer learning,’’ Metall. Mining Ind., vol. 7, pp. 312–321, Jul. 2015.
>
> [2] Y. Yu, L. Du, C. Zeng, and J. Zhang, ‘‘Automatic localization method of small casting defect based on deep learning feature,’’ Chin. J. Sci. Instrum., vol. 37, no. 6, pp. 1364–1370, Jun. 2016.
>
> [3] M. Ferguson, R. Ak, Y.-T.-T. Lee, and K. H. Law, ‘‘Automatic localization of casting defects with convolutional neural networks,’’ in Proc. IEEE Int. Conf. Big Data (Big Data), Dec. 2017, pp. 1726–1735.
>
> [4] X. Xu, Y. Lei, and F. Yang, ‘‘Railway subgrade defect automatic recognition method based on improved faster R-CNN,’’ Sci. Program., vol. 2018, pp. 1–12, Jun. 2018.

在目标检测网络模型中，特征提取网络的设计有着关键作用。YOLOv3采用<font color='orange'>DarkNet-53</font>架构提取特征，整个网络采用完全卷积的方式涉及。YOLOv3的每个卷积组件由<font color='orange'>卷积层（Conv），批归一化层（BN）和激活层（Leaky ReLu）</font>组成。

并将<font color='orange'>残差分量Resn</font>引入YOLOv3中，<font color='red'>有利于解决梯度消失和爆炸问题，增强网络的学习能力</font>。

YOLOv3网络结果图如下所示：

![image-20221201110948353](F:\Learning_notes\论文笔记\image-20221201110948353.png)

原YOLOv3每个检测模块在预测之前会让特征通过一个$3*3$和一个$1*1$卷积，前者是用来提取图像特征，增加网络通道数，后者是改变网络通道数的大小。

<font color='red'>本文改进点一：使用双密度卷积层代替原先的单密度卷积层进行检测</font>。

> 将检测模块改进为一个$3*3$卷积层和两个串联$3*3$卷积层的并联双密度检测层模块。
>
> 在$3*3$卷积层较小的感受野中显示了更详细的信息。
>
> 两个序列的$3*3$卷积层具有较大的感受野和较多的语义信息，可以有效地检测出较大的缺陷。
>
> ![image-20221201154012171](F:\Learning_notes\论文笔记\image-20221201154012171.png)
>
> 因此YOLOv3网络在提取特征时得到了两种不同的感受野。<font color='red'>网络模块利用了更多的细节信息和语义信息，提高了检测的能力</font>。

<font color='red'>本文改进点二：添加一个检测头</font>

<font color='red'>本文改进点三：改进锚选框初始化聚类算法</font>

论文中使用<font color='orange'>引导滤波</font>对图像进行预处理，<font color='red'>增强了缺陷区域的轮廓和细节，抑制了背景区域灰度值的增加，有利于铸件缺陷的检测</font>。

<font color='orange'>损失函数曲线</font>反映了网络模型参数的更新速度，网络模型是否收敛，训练是否完成。损失函数下降速度越快，网络模型在不断学习更新过程中的速度就越快。当损失函数值变化曲线趋于稳定时，模型训练即将完成，这意味着网络模型训练接近收敛。

<font color='pink'>改进后的模型mAP达到了88.02%,参数量高达2284MB</font>。

## 该论文相较于v5缺点：

> * <font color='red'>参数量过大，不易部署</font>
> * <font color='red'>精度有待进一步提高</font>

