# 《Image Classification Method Based on Multi-Agent Reinforcement Learning for Defects Detection for Casting》论文笔记

## 基于多智能体强化学习的铸件缺陷图像分类方法

<font color='green'>铸件缺陷检测是机械行业的一个重要问题。高质量的铸件对汽车、工程设备等产品具有重要意义。及时发现和处理这些缺陷铸件是十分重要的，否则会损害下游工业产品的质量</font>。

现阶段铸件缺陷的<font color='pink'>一个问题</font>：

> 在过去的十年中，为了提高分类精度，<font color='red'>模型被设计的越来越复杂，这导致参数数量越来越大，计算负担呈指数级增加</font>，特别是对于大图像。铸件的图像尺寸很大，但是通常缺陷仅位于少量像素中，这意味着图像中的大量像素对于缺陷的检测任务是无用的，但是他们仍然消耗相同的计算资源。