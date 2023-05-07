# 《Underwater target detection based on improved YOLOv7》论文笔记

## 基于改进的YOLOV7的水下目标检测

本文代码地址：[https://github.com/NZWANG/YOLOV7-AC](https://github.com/NZWANG/YOLOV7-AC)

目标检测是计算机视觉的一个核心分支，它包含了<font color='orange'>目标分类</font>和<font color='orange'>定位</font>等基本任务。

现有的目标检测方法大致可以分为两类：<font color='orange'>传统的目标检测算法</font>和<font color='orange'>基于深度学习的目标检测算法</font>。

传统的目标检测算法通常分为三个阶段：<font color='orange'>区域选择</font>，<font color='orange'>特征提取</font>和<font color='orange'>特征分类</font>。

区域选择的目标是对目标进行局部定位，因为目标在图像中的位置和长宽比可能会发生变化。这一阶段通常通过<font color='Sky Blue'>使用滑动窗口策略$^{[1]}$遍历整个图像</font>来完成，其中考虑了不同的比例和宽高比。随后<font color='Sky Blue'>利用特征提取算法HOG（Histogram of Oriented Gradients）$^{[2]}$和SIFT（Scale Invariant feature Transform）$^{[3]}$提取相关特征</font>。最后<font color='Sky Blue'>利用支撑向量机（SVM，Support Vector Machines）$^{[4]}$和Ada-Boost（Adaptive Boosting）$^{[5]}$等分类器对提取的特征进行分类</font>。

> * [1] Samantaray.; Sahil.; Rushikesh Deotale.; Chiranji Lal Chowdhary. Lane detection using sliding
>   window for intelligent ground vehicle challenge. Innovative Data Communication Technologies and
>   Application: Proceedings ofICIDCA 2020. Springer Singapore, 2021. pp. 871-881.
> * [2] Bakheet.; Samy.; Ayoub Al-Hamadi. A framework for instantaneous driver drowsiness detection
>   based on improved HOG features and naïve Bayesian classification. Brain Sciences 11.2, 2021. pp.
>   240.
> * [3] Bellavia.; Fabio. SIFT matching by context exposed. IEEE Transactions on Pattern Analysis and
>   Machine Intelligence, 2022. pp. 2445 - 2457.
> * [4] Koklu, Murat, et al. A CNN-SVM study based on selected deep features for grapevine leaves
>   classification. Measurement 188, 2022. pp. 110425.
> * [5] Sevinç.; Ender. An empowered AdaBoost algorithm implementation: A COVID-19 dataset study.
>   Computers & Industrial Engineering 165, 2022. pp. 107912.

传统目标检测算法存在<font color='pink'>两个主要缺陷</font>：

> * <font color='red'>采用滑动窗口的区域选择缺乏特异性，导致时间复杂度高和窗口冗余</font>
> * <font color='red'>手工设计的特征对姿态变化的鲁棒性不强</font>。

卷积神经网络（CNN）已经证明了其在目标检测任务中提取和建模特征的卓越能力。

大量研究证明，基于深度学习的方法优于依赖手工设计特征的传统方法。

<font color='pink'>基于深度学习的目标检测算法</font>主要有两大类：<font color='orange'>基于区域提议的算法（两阶段目标检测算法）</font>和<font color='orange'>基于回归的算法（一阶段目标检测算法）</font>

<font color='orange'>基于区域提议的算法（两阶段目标检测算法）：</font>该算法是基于粗定位和细分类的原理，首先识别出包含目标的候选区域，然后进行分类。

<font color='orange'>基于回归的算法（一阶段目标检测算法）：</font>该算法直接通过CNN提取特征，预测目标分类和定位。

以上两种算法的优缺点：

> * <font color='orange'>基于区域的算法</font>具有<font color='red'>较高的精度</font>，但是他们往往<font color='red'>速度较慢，不适合实时应用</font>。
> * <font color='orange'>基于回归的算法</font>，由于直接预测分类和定位，所以该算法拥有<font color='red'>更快的检测速度</font>

本文<font color='pink'>创新性</font>如下：

> * 为了提取更多的信息特征，提出了<font color='orange'>整合全局注意力机制（GAM）$^{[6]}$</font>的方法。这种机制<font color='red'>有效的捕捉了特征的通道和空间信息，增加了跨纬度交互的重要性</font>。
> * 为了进一步提高网络的性能，引入了<font color='orange'>ACmix</font>$^{[7]}$（一种融合了自注意和卷积优点的混合模型）
> * <font color='orange'>ResNet-ACmix模块</font>的设计目的在于<font color='red'>增强骨干网络的特征提取能力，通过捕获更多的信息特征，加速网络的收敛</font>。
> * YOLOv7网络中的E-ELAN模块通过整合Skip Connection和模块之间的$1*1$卷积结构进行优化，并使用ACmix模块替换$3*3$卷积层。这<font color='red'>提高了特征提取能力，并提高了推理过程中的速度</font>。

<font color='pink'>上述论文出处</font>：

> * [6] Liu, Yichao.; Zongru, Shao.; Nico Hoffmann. Global attention mechanism: Retain information to
>   enhance channel-spatial interactions. arXiv preprint arXiv:2112.05561, 2021.
> * [7] Pan, X.; Ge, C.; Lu, R.; Song, S.; Chen, G.; Huang, Z.; Huang, G. On the integration of self-attention
>   and convolution. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022. pp. 815-825.

### YOLOv7

YOLOv7模型整合了<font color='orange'>E-ELAN（Extended efficient layer aggregation networks，扩展有效层聚合网络）</font>，<font color='orange'>基于级联的模型</font>和<font color='orange'>模型重参数化</font>等策略，<font color='red'>实现了检测效率和精度之间的良好平衡</font>。

YOLOv7由4个不同的模块组成：<font color='orange'>输入模块，骨干网，头部网络和预测网络</font>

YOLOv7原始网络模型结构如下图所示：

![image-20230218140001623](F:\Learning_notes\论文笔记\image-20230218140001623.png)

<font color='orange'>输入模块</font>：YOLOv7模型的预处理阶段采用拼接和混合数据增强技术，利用YOLOv5建立的自适应锚选框计算方法，保证输入的彩色图像均匀缩放到$640*640$大小，从而满足骨干网络输入大小的要求。

<font color='orange'>骨干网络</font>：YOLOv7骨干网络主要由<font color='orange'>CBS、E-ELAN和MP1</font>构成。<font color='sky blue'>CBS模块由卷积，批量归一化和SiLU激活函数</font>构成。<font color='sky blue'>E-ELAN模块保持原有ELAN设计结构，通过引导不同特征组计算块学习更多样化的特征，提高网络的学习能力，保持原有的梯度路径</font>。<font color='sky blue'>MP1由CBS和MaxPool组成，分为上下两个分支</font>。上分支使用MaxPool将图像的长度和宽度减半，使用具有128个输出通道的CBS将图像半径减半。下分支通过核为$1*1$，stride为$2*2$的CBS将图像通道减半，核为$3*3$，stride为$2*2$的CBS将图像长度和宽度减半，最后通过拼接（cat）操作将两个分支提取的特征进行融合。<font color='red'>MaxPool提取小局部区域的最大值信息，CBS提取小局部区域的所有值信息，从而提高了网络的特征提取能力</font>。

<font color='orange'>头部网络</font>：YOLOv7的头部网络采用<font color='orange'>特征金字塔网络结构（FPN）</font>，采用<font color='orange'>PANet</font>设计。该网络包括<font color='sky blue'>多个卷积、批处理归一化和SiLU激活函数（CBS）块，并引入空间金字塔池和卷积空间金字塔池（Sppcspc）结构、扩展有效层聚合网络（E-ELAN）和MaxPool-2(MP2)</font>。<font color='orange'>Sppcspc结构</font>通过在空间金字塔池（Spatial Pyramid Pooling，SPP）结构中引入卷积空间金字塔（Convolutional Spatial Pyramid，CSP）结构来改善网络的感知场，同时利用较大的残差边缘来辅助优化和特征提取。<font color='orange'>ELAN-H层</font>时基于E-ELAN的多个特征层的融合，进一步增强了特征提取。<font color='orange'>MP2块</font>具有与MP1块相似的结构，只是对输出通道的数量略有修改。

<font color='orange'>预测网络</font>：YOLOv7的预测网络<font color='sky blue'>采用Rep结构对头部网络的输出特征进行图像通道数的调整，然后应用$1*1$卷积对置信度、类别、锚选框进行预测</font>。Rep结构受RepVGG的启发，引入了一种特殊的残差设计，以帮助训练过程。在实际预测中，<font color='red'>这种独特的残差结构可以简化为一个简单的卷积，从而在不牺牲其预测性能的情况下降低网络复杂度</font>。

### GAM

<font color='orange'>注意力机制</font>是一种通过在神经网络中<font color='red'>为输入的不同部分分配不同的权重来改进在复杂背景下的特征提取的方法</font>。这种方法使模型能够<font color='red'>关注相关信息而忽略无关信息，从而提高网络模型性能</font>。注意力机制有<font color='orange'>像素注意力机制（pixel attention）</font>，<font color='orange'>通道注意力机制（channel attention）</font>和<font color='orange'>多阶注意力机制（multi-order attention）</font>。

<font color='red'><font color='orange'>GAM</font>可以通过减少信息分散和放大全局交互表示来提高深度神将网络的性能</font>。该模块结构如下图所示：

![image-20230218170253329](F:\Learning_notes\论文笔记\image-20230218170253329.png)

GAM包括<font color='orange'>通道注意子模块</font>和<font color='orange'>空间注意子模块</font>。

<font color='orange'>通道注意子模块</font>被<font color='sky blue'>设计为三维变换，使其能够保留输入的三维信息。然后是两层的多层感知（multilayer perception，MLP），其作用是放大通道空间中的维度间依赖性，从而使网络能够聚焦于图像中更有意义和前景的区域</font>。如下图所示：

![image-20230218191651428](F:\Learning_notes\论文笔记\image-20230218191651428.png)

<font color='orange'>空间注意子模块</font>包含了<font color='sky blue'>两个卷积层，以有效的整合空间信息，使网络能够集中在图像中上下文重要的区域</font>，如下图所示：

![image-20230218192238384](F:\Learning_notes\论文笔记\image-20230218192238384.png)

### ACmix

<font color='orange'>自注意力机制</font>和<font color='orange'>卷积</font>都严重依赖于$1*1$卷积运算，而<font color='orange'>ACmix</font>很好的<font color='red'>结合了自注意力机制和卷积，并且拥有很小的计算开销</font>。输出结果为：$F_{out}=\alpha*F_{attention}+\beta*F_{conv}$。结构如下图所示：

![image-20230218204543038](F:\Learning_notes\论文笔记\image-20230218204543038.png)

### ResNet-ACmix

在YOLOv7的骨干网络中<font color='red'>引入ResNet-ACmix模块，有效的保持了提取特征信息的一致性</font>。该模块基于ResNet的结构，<font color='sky blue'>使用ACmix模块取代原$3*3$卷积，可以自适应聚焦不同的区域，捕捉更多信息的特征</font>。如下图所示，<font color='sky blue'>将输入分为主输入和剩余输入，避免了信息的丢失，同时减少了参数数量和计算需求</font>。<font color='red'>ResNet-ACmix模块可以使网络获得更深的深度，而不会遇到梯度消失，学习结果对网络权值的波动更敏感</font>。

![image-20230218210414429](F:\Learning_notes\论文笔记\image-20230218210414429.png)

### AC-E-ELAN

与传统的ELAN网络不同，<font color='orange'>E-ELAN网络</font>采用了一种expand，shuffle和merge cardinallity方法，这种方法<font color='red'>能够在不破坏原始梯度流的情况下不断增强网络的学习能力，从而提高参数利用率和计算效率</font>。将<font color='orange'>RepVGG架构中的残差结构（即$1*1$卷积分制和跳跃连接分支）纳入其中</font>。该结构集成了ACmix模块，由$3*3$卷积块组成。这种组合使网络<font color='red'>既能从多分支模型的训练中获得丰富的特征，又能从单路径模型中获得快速、内存高效的推理</font>。

![image-20230218210827669](F:\Learning_notes\论文笔记\image-20230218210827669.png)

<font color='pink'>改进后的YOLOv7网络结构</font>：

![image-20230218205548216](F:\Learning_notes\论文笔记\image-20230218205548216.png)

引入<font color='orange'>AC-E-ELAN结构</font>，<font color='red'>增强了模型关注输入图像样本有价值的内容和位置的能力，丰富了网络提取的特征，减少了模型的推理时间</font>。

<font color='orange'>ResNet-ACmix</font><font color='red'>有效地保留了骨干网络中收集到的特征，提取小目标和复杂背景目标的特征信息，同时加快网络收敛速度，提高检测精度</font>。

加入<font color='orange'>GAM注意力机制</font>，<font color='red'>增强了网络有效提取深度和重要特征的能力</font>。

### 缺陷

<font color='red'>YOLOv7-AC模型在高度复杂的水下环境中仍然存在误检和漏检的情况</font>。















