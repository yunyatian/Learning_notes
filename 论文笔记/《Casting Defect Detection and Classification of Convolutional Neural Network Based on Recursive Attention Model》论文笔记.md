# 《Casting Defect Detection and Classification of Convolutional Neural Network Based on Recursive Attention Model》论文笔记

## 基于递归注意模型的卷积神经网络铸造缺陷检测和分类

铸件表面缺陷分类是工业缺陷检测的关键环节。然而，在传统行业中，这一环节往往是手工完成的。人工铸件容易出现<font color='orange'>气孔、粘砂、夹砂、砂眼，胀砂、冷保温、浇筑不足等</font>问题。为了取代人工操作，人们希望机器能利用计算机视觉技术自动检测铸件的表面的缺陷。

由于<font color='red'>铸件表面缺陷的图像受光线和材料变化的影响，且铸件内部缺陷的外观差异较大，不同类别的缺陷具有相似的方面</font>，因此利用计算机视觉技术对缺陷进行分类仍然是一个巨大的挑战。

本文对铸件的检测主要采用递归注意模型下的<font color='orange'>SA层模块</font>，识别采用卷积神经网络的方法。底面检测完成后，通过检测系统对合格铸件进行翻转和再次检测。

<font color='red'>两阶段分类算法检测精度更高，单阶段回归算法检测速度更快。</font>

YOLOV3借鉴<font color='orange'>残差网络结构</font>，<font color='red'>形成更深的网络层次，在保持单阶段回归算法速度优势的前提下，提高了检测精度和小目标的检测能力</font>。

本文主要采用ROI的注意提取模型，通过SA层识别模块在一定范围内提高缺陷检测的准确性。

DBL是YOLOV3的基本组件。它由<font color='orange'>卷积层（Conv），批处理归一化层（BN）和激活层（LeakRelu）</font>组成。

Resn_unit利用resnet的思想，在<font color='red'>两个DBL之后执行一个剩余层跳转连接</font>。

递归关注模块（SA）是一个<font color='orange'>多尺度上下文级联模块</font>，包括<font color='orange'>顶部模块</font>和<font color='orange'>多级池模块</font>。

顶部模块直接对输入特征进行1*1卷积处理，得到空间全局特征。

多级池模块包含3个下采样，分别获得1/2输入图像，1/4输入图像和1/8输入图像的三种不同分辨率特征，并获得不同尺度的上下文信息，直至输入恢复特征尺寸。最后与顶层模块结合，作为结构的最终输出。

通过多次下采样，得到不同尺寸铸件的特征，并利用卷积层进行特征选择和提取。大区域可用于获取高层上下文信息，小区域借用于获取低层上下文信息。该层次组合能够<font color='red'>保持不同尺寸的层次依赖性，适应不同尺度的特征映射，avoid other the misclassification of symbiotic and mixed castings in land（避免了。。。。。。误分类）</font>。

Channel shuffle是shuffleNet的核心，其目的是解决群卷积干扰群间信息交换，导致性能不佳的问题，当每组信道数为3时，信道变换不能完全避免组间的信息丢失。第2卷积的每组只接收第1卷积层每组的1个信道，导致每组中的其他两个信道被忽略。

<font color='orange'>MixConv</font>用不同大小的卷积核代替深度卷积，大卷积核可以在一定范围内提高模型的精度，多卷积核可以提高模型在不同分辨率下的适应度。

为了解决<font color='orange'>组间信息丢失</font>的问题，本论文<font color='red'>借鉴MENet中合并和进化的思想。利用<font color='orange'>窄幅特征映射</font>对组间信道信息进行融合编码，并将匹配变换和原始网络相结合，获得更明显的特征</font>。