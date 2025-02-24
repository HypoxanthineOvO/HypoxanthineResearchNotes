# 简介
这篇论文提出了一种多尺度的基础块（MSRB）和层次特征融合（HFFS）的方法来进行超分，取得了SOTA的效果。多尺度的思想借鉴与Inception的思想，使用 3x3 与 5x5 的卷积来获得不同尺度的信息；HFFS的思想则与 [DRCN](https://zhuanlan.zhihu.com/p/76868378)很是相似，[DRCN](https://zhuanlan.zhihu.com/p/76868378) 中间的递归子模型展开之后就是线性堆叠的模型层，这里也是将中间各个 Block 的输出进行concat之后，然后再来进行重建。

这里特征堆叠的思路与 [DRCN](https://zhuanlan.zhihu.com/p/76868378) 差异的地方在于：
1. [DRCN](https://zhuanlan.zhihu.com/p/76868378) 中递归子模型中的权重是共享的，而这里的 Block 之间的权重是相互独立不共享的；
2. [DRCN](https://zhuanlan.zhihu.com/p/76868378) 中是递归子模型的输出都重建然后再加权平均得到最终的结果，这里是 Block 的输出concat起来然后再来进行重建；
3. [DRCN](https://zhuanlan.zhihu.com/p/76868378) 是先上采样再超分，这里是上采样模块在模型中。
# 主要创新点 
 1. 提出了一种MSRB的多尺度特征提取块，可以自适应的学习图像特征，还可以进行特征融合，这是第一次将多尺度应用于残差结构中。  
 2. 在不需要很深的网络模型的情况下，MSRN的效果就已经超过其他SOTA的模型，并且可以直接推广到其他的一些重建任务中。  
 3. 提出了一种简单的HFFS的特征融合方法，其可以简单地推广到任意的上采样尺度中。

# Motivation：当前模型的缺点 

> 1. 较难复现。缺乏原文的超参配置，一些训练技巧的差异就会导致无法达到原文的结果。  
> 2. 模型特征利用率不够。  
> 3. 灵活性不够。将低分辨率输入插值到高分辨率的预处理操作会使得模型的耗时变得更长，不同的上采样倍数均需要设计模型。

文中对于缺点2中的问题解决方案是很明显，HFFS的特征融合利用；对于缺点3的上采样问题也很针对，使用同一组架构来进行不同倍数的超分。然而缺点1的问题是通用的，文中也没有进行很多的超参设置对比实验。

# 模型结构
![[2-03-MSRN-011.png]]

整个模型结构主要包括两个部分，一个是特征提取，一个是重建。在特征提取网络中，通过堆叠N个**MSRB**来获得足够的非线性，并且将这些 Block 的输出concat起来，也就是文中所说的**HFFS**的层次融合方法。由于这些concat之后的特征维度可能很巨大，因此使用了一个Bottleneck Layer来进行降维，再来作为重建网络的输入。在重建网络中，第一个卷积层用来匹配不同上采样倍数时所需要的通道倍数，不同的上采样倍数均使用[ESPCN](https://zhuanlan.zhihu.com/p/76338220)中的[Pixel Shuffle](https://zhida.zhihu.com/search?content_id=105212630&content_type=Article&match_order=1&q=Pixel+Shuffle&zhida_source=entity)的整数倍上采样方法。

在 [DRCN](https://zhuanlan.zhihu.com/p/76868378) 中进行了特征是否融合的对比实验，而这里并没有进行是否采用层次特征融合的对比实验。直觉层面上来讲，HFFS是会加速训练，而且会带来效果的提升。但是这里缺乏对比实验，不好确定HFFS的贡献度如何。

![[2-03-MSRN-012.png]]

每一个MSRB中，都有着 3x3 和 5x5 来进行多尺度的信息融合。

![[2-03-MSRN-013.png]]

这种思想在[ResNet](https://zhida.zhihu.com/search?content_id=105212630&content_type=Article&match_order=1&q=ResNet&zhida_source=entity)、[DenseNet](https://zhida.zhihu.com/search?content_id=105212630&content_type=Article&match_order=1&q=DenseNet&zhida_source=entity)与Inception的结构设计中就已经有所体现。 3x3 不知道怎么选不要紧，小孩子才做选择，大人全都要。直接 3x3 与 5x5 一起使用。

同时，这里还是用了**local residual learning**的思想，在每一个MSRB的子模块中，均使用了残差连接。**局部残差的思想是DRRN中提出来的，这里并没有引用DRRN的论文，而且还没有进行是否使用局部残差的对比实验。**

![[2-03-MSRN-014.png]]

这里对于不同的上采样倍数，都是直接使用Pixel Shuffle来实现。不同的上采样倍数只需要相应地设置上采样之前的那层卷积的通道数即可。

![[2-03-MSRN-015.png]]

相比于一些其他的上采样的方法，这种直接设置上采样倍数比较灵活、简单。

这篇论文也是在YCbCr中的Y通道上来进行的超分。

![[2-03-MSRN-017.png]]

从2、3、4、8的超分倍数来看，其与 [EDSR](https://zhida.zhihu.com/search?content_id=105212630&content_type=Article&match_order=1&q=EDSR&zhida_source=entity) 的效果相当，没有明显的优势。

![[2-03-MSRN-018.png]]

因此文中也放出了与EDSR的细节对比，相比而言比EDSR是要轻量很多。

![[2-03-MSRN-019.png]]

文中也对不同的MSRB的数量进行了对比实验，可以看到，随着MSRB的数量的提升，最终的不同倍数上的超分效果都会提升。感觉这里也完全可以像EDSR一样，将 Block 数量设置成32，然后看看效果会不会更好。

![[2-03-MSRN-020.png]]

在基准 Block 上，这里与常用的ResNet & DenseNet  Block 进行了对比，可以看到不同的超分倍数下的结果都要更好，而且优势较为明显。

![[2-03-MSRN-021.png]]

同时，模型的特征图利用率上也进行了对比，可以看到MSRB的特征图的特征利用率会更高一些。

![[2-03-MSRN-022.png]]

最骚的是这里居然没有与EDSR进行对比，可能与EDSR的效果相当。这里上面一张图中的2倍倍数下模型的结构均出现了条纹的异变；下面图中MSRN的线复原得更好，[SRCNN](https://zhuanlan.zhihu.com/p/76520991) 的结果中出现很多细节变异。

![[2-03-MSRN-023.png]]
最后还强行在高斯降噪与去雾上做了点实验，表明其通用性。但是并没有与其他的降噪和去雾的算法进行效果对比，感觉这里并不能说明什么。

# 结论

这篇论文提出了MSRB的多尺度特征提取和HFFS的层次特征融合的方法，取得了SOTA的效果。对于MSRB的有效性问题，与ResNet & DenseNet  Block 进行了对比，最终的效果和特征利用率上均更好；还进行了不同数量的MSRB对比实验。

> [!notes] 妈的好无聊啊纯纯炼丹吧