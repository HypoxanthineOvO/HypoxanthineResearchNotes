# 1. 概述
这篇文章是 SRResnet 的升级版——EDSR，其对网络结构进行了优化 (去除了 BN 层)，省下来的空间可以用于提升模型的 size 来增强表现力。作者提出了一种基于 EDSR 且适用于多缩放尺度的超分结构——MDSR。EDSR 在 2017 年赢得了 NTIRE 2017 超分辨率挑战赛的冠军。

# 2. 主要贡献
1. **作者推出了一种加强版本的基于 Resnet 块的超分方法，它实际上是在 SRResnet 上的改进，去除了其中没必要的 BN 部分**，从而在节省下来的空间下提升模型的 size 来增强表现力，它就是 EDSR，其取得了当时 SOTA 的水平。
	- **原因：BN 使得特征范围受到限制**。BN通过对特征进行归一化，强制将特征值限制在一个固定的范围内（均值为0，方差为1）。然而，在超分辨率任务中，特征的动态范围非常重要，尤其是高频细节的恢复需要网络能够灵活地处理不同尺度的特征。
2. 此外，作者在文中还介绍了一种基于 EDSR 的多缩放尺度融合在一起的新结构——MDSR。
3. EDSR、MDSR 在 2017 年分别赢得了 NTIRE 2017 超分辨率挑战赛的冠军和亚军。
4. 此外，作者通过实验证明使用 L 1 Loss 比 L 2 Loss 具有更好的收敛特性。

## 3. 引言
近几年来，深度学习在 SR 领域展现了较好的图像高分辨率重建表现，但是网络的结构上仍然存在着一些待优化的地方：
1. 深受神经网络的影响，SR 网络在超参数 (Hyper-parameters)、网络结构 (Architecture) 十分敏感。
2. 之前的算法 (除了 VDSR) 总是为特定 up-scale-factor 而设计的 SR 网络，即 scale-specific，将不同缩放尺度看成是互相独立的问题，因此我们需要一种网络来处理不同缩放尺度的 SR 问题，比如×2,3,4，这比训练 3 个不同缩放尺度的网络节省更多的资源消耗。

针对第一个网络结构问题，作者在 SRResNet 的基础上，对其网络中多余的 BN 层进行删除，从而节约了 BN 本身带来的存储消耗以及计算资源的消耗，简化了网络结构。此外，选择一个合适的 loss function，作者经过实验证明 $L1\text{ Loss}$ 比 $L2\text{ Loss}$ 具有更好的收敛特性。

## 4. 提出的方法
本节将正式开始介绍一种增强版本的 SRResNet——EDSR (一种 single-scale 网络)，它通过移除了适合分类这种高级计算机视觉任务而不适合 SR 这种低级计算机视觉任务的 BN 层来减少计算资源损耗。除此之外，本节还会介绍一种集合了多尺度于一个网络中的 multi-scale 超分网络——MDSR。

## 4.1 残差块
![[2-02-EDSR&MDSR-001.png|496x406]]
移除 BN 有以下三个好处：
1. 这样模型会更加轻量。BN 层所消耗的存储空间等同于上一层 CNN 层所消耗的，作者指出相比于 SRResNet，EDSR 去掉 BN 层之后节约了 40%的存储空间。
2. 在 BN 腾出来的空间下插入更多的类似于残差块等 CNN-based 子网络来增加模型的表现力。
3. BN 层天然会拉伸图像本身的色彩、对比度，这样反倒会使得输出图像会变坏，实验也证明去掉 BN 层反倒可以增加模型的表现力。

## 4.2 单尺度模型
EDSR 是 SRResNet 的增强版本，是一种基于残差块的网络结构。
![[2-02-EDSR&MDSR-002.png]]
EDSR 的结构：最上面一排是网络结构，可以大致分为**低层特征信息提取、高层特征信息提取、反卷积 (上采样) 层、重建层**，基本和 SRResNet、SRDCNN 类似的。下面第二层分别表示残差块的构造以及反卷积层 (分别是×2、×3、×4) 的构造。
Note：
1. 连接 ① 是将不同 level 的特征信息进行合并；连接 ② 是 ResNet 块内部的残差连接。
2. 在 EDSR 的 baseline 中，是没有 residual scaling 的，因为只是用到了 64 层 feature map，相对通道数较低，几乎没有不稳定现象。但是在最后实验的 EDSR 中，作者是设置了 residual scaling 中的缩减系数为 $0.1$，且 $B=32$, $F=256$。

增加模型表现力最直接的方式就是增加模型的参数 (复杂度)，一般可以通过增加模型层数 (即网络深度) 以及滤波器个数 (即网络宽度或者说通道数)。对于存储资源的消耗大约是 $O(BF)$，增加的参数大约是 $O (BF^2)$，因此增加滤波器个数才能在有限存储空间下最大化参数个数。
![[2-02-EDSR&MDSR-003.png|286x444]]
在 Inception-ResNet 这篇文章以及本文中都指出，过大的滤波器个数 (feature map 个数，或者说通道数) 会导致网络不稳定，最佳的解决办法不是降低学习率或者增加 BN 层，而是通过在残差块最后一层卷积后加上 Residual scaling 层。
### 4.3 多尺度模型

![[2-02-EDSR&MDSR-004.png]]
上图蓝色线表示的用训练好的 up-scale-factor=2 的 EDSR 网络作为训练时候的初始化参数，结果来看收敛速度以及表现力的提升都是有目共睹的，一定程度上说明了不同缩放尺度之间是存在某种内在联系的。
因此作者设计了一种在单一网络中实现多尺度融合的 SR 网络——MDSR，其结构如下：
![[2-02-EDSR&MDSR-005.png|552x467]]
如上图所示是 MDSR 的网络结构，每个预处理模块由 2 个 5×5 的卷积层组成，针对每一种 up-scale-factor 设置不同的残差快；中间是一个共享残差网络；最后是针对不同缩放倍数设计的上采样网络。
![[2-02-EDSR&MDSR-006.png|403x325]]
Note：
1. 总体来说，MDSR 是基于 EDSR 的结构。
2. 预处理阶段的残差块中的卷积采用较大的卷积核来增大初始阶段的感受野。
3. 作者统计了一笔数据，训练 3 个单独的 EDSR-baseline 来实现不同放大倍数的 SR 需要消耗 1.5 M∗3=4.5 M 的参数量；而训练一个 MDSR 的 baseline 需要 3.2 M 的参数量，而 MDSR 在后续实验中表现也还不错，因此 MDSR 是一种资源消耗相对少且有一定表现力的 SR 网络。

下表是 SRResNet、EDSR、MDSR 资源占用统计：
![[2-02-EDSR&MDSR-007.png]]
# 5. 实验

## 5.1 数据集
需要介绍一下新的数据集 DIV 2 K，这是包含了 2 K 高分辨率图像的数据集：训练集 800 张、验证集 100 张、测试集 100 张。其余的标准 benchmark：Set 5、Set 14、B 100、Urban 100。

## 5.2 训练细节
1. 输入是数据集中的 patch 部分，RGB 格式的大小 48×48。
2. 通过水平翻转和 90°旋转来做数据增强。
3. Adam 做优化。
4. Mini-batch=16。
5. 学习率从 10−4 开始，每过 2×105 个 epoches，就减半一次。
6. 对于 EDSR 中的×3、4 网络训练的初始化参数，是采用训练完毕的×2 EDSR 网络的参数。而×2 的 EDSR 是从头开始训练的。
7. EDSR 和 MDSR 都采用 L 1 Loss，作者通过大量实验表明 L 1 比 L 2 有更好的收敛特性。

## 5.3 几何自集成
几何自集成的方法用于在测试的时候，将每一张输入图像经过 8 种不同 (其中一种是原图) 的变换方式进行转换：然后将 8 个结果通过网络输出成，然后将每一个值经过转置处理：最后在此基础上进行平均处理：最后拿着最终的结果去计算 PSNR/SSIM，即图表中的 EDSR+、MDSR+，从实验结果来看，self-ensemble 确实可以提升表现力。

## 5.4 在 DIV 2K 数据集上的评估
在 DIV 2K 验证集 (测试集不公开) 中实验结果如下：
![[2-02-EDSR&MDSR-010.png]]
Note：
1. 从结果来看，L 1 比 L 2-Loss 更能对表现力进行提升。
2. Geometric Self-ensemble 确实可以提升表现力。
3. EDSR 在 DIV 2 K 上获取最佳的表现，其次 MDSR 也表现尚可。

## 5.5 基准测试结果
作者对多种 SR 算法在 Benchmark 上的表现进行统计，结果如下：
![[2-02-EDSR&MDSR-008.png]]
![[2-02-EDSR&MDSR-009.png]]

Note：
1. 总体来看，EDSR 和 MDSR 是包揽了最佳和次佳的表现结果。
---
### **EDSR（Enhanced Deep Super-Resolution Network）和 MDSR（Multi-scale Deep Super-Resolution Network）的网络结构细节及相关原理**

EDSR 和 MDSR 是用于单图像超分辨率（Single Image Super-Resolution, SISR）任务的深度卷积神经网络，分别针对单尺度和多尺度超分辨率任务进行了优化。以下是它们的网络结构细节和相关原理。

---

### **1. EDSR（单尺度超分辨率网络）**

#### **网络结构**
EDSR 的核心是一个**深度残差网络**，主要由以下几个部分组成：
1. **浅层特征提取**：
   - 使用一个卷积层从输入的低分辨率图像中提取浅层特征。
   - 卷积层的滤波器数量为 256，核大小为 3×3。

2. **残差块（Residual Blocks）**：
   - EDSR 包含 32 个残差块，每个残差块由两个卷积层组成，卷积层的滤波器数量为 256，核大小为 3×3。
   - 每个残差块内部使用**跳跃连接（Skip Connection）**，将输入直接添加到输出中，以学习残差信息。
   - 在残差块的最后一层卷积后引入**残差缩放（Residual Scaling）**，通过一个缩放因子（如 0.1）稳定训练过程。

3. **全局跳跃连接（Global Skip Connection）**：
   - 将浅层特征提取的输出直接添加到残差块的最终输出中，以保留低频信息。

4. **上采样模块（Upsampling Module）**：
   - 使用**亚像素卷积层（Sub-pixel Convolution Layer）**实现上采样。
   - 亚像素卷积层通过重排特征图的方式将低分辨率特征图转换为高分辨率图像。

5. **重建层（Reconstruction Layer）**：
   - 使用一个卷积层将上采样后的特征图转换为最终的高分辨率图像。
   - 卷积层的滤波器数量为 3（对应 RGB 通道），核大小为 3×3。

#### **原理**
- **残差学习**：EDSR 通过残差块学习低分辨率图像和高分辨率图像之间的残差（差值），而不是直接学习高分辨率图像。这种方式使得网络更容易训练，且能够更好地恢复高频细节。
- **移除批归一化层（BN）**：BN 在超分辨率任务中会限制特征的灵活性，移除 BN 后，网络能够更好地捕捉图像中的细节信息。
- **残差缩放**：通过引入残差缩放，稳定了深层网络的训练过程，特别是在使用大量滤波器时。
- **后上采样策略**：EDSR 采用后上采样策略，即在网络的最后一层进行上采样，减少了计算复杂度，且能够更好地保留图像细节。

---

### **2. MDSR（多尺度超分辨率网络）**

#### **网络结构**
MDSR 是 EDSR 的扩展版本，旨在处理多个尺度的超分辨率任务。其核心思想是**共享大部分参数**，同时为每个尺度引入特定的处理模块。MDSR 的结构如下：
1. **共享主干网络（Shared Main Branch）**：
   - MDSR 包含一个共享的主干网络，由 80 个残差块组成，每个残差块的结构与 EDSR 相同。
   - 主干网络用于提取多尺度共享的特征。

2. **尺度特定预处理模块（Scale-specific Pre-processing Modules）**：
   - 每个尺度（如×2、×3、×4）都有一个特定的预处理模块，用于减少输入图像的尺度差异。
   - 每个预处理模块由两个残差块组成，卷积层的核大小为 5×5，以覆盖更大的感受野。

3. **尺度特定上采样模块（Scale-specific Upsampling Modules）**：
   - 每个尺度都有一个特定的上采样模块，用于将低分辨率特征图转换为高分辨率图像。
   - 上采样模块的结构与 EDSR 相同，使用亚像素卷积层实现上采样。

4. **重建层（Reconstruction Layer）**：
   - 使用一个卷积层将上采样后的特征图转换为最终的高分辨率图像。

#### **原理**
- **参数共享**：MDSR 通过共享主干网络的大部分参数，显著减少了模型的计算复杂度和参数量。
- **尺度特定处理**：每个尺度都有特定的预处理模块和上采样模块，以处理不同尺度的超分辨率任务。
- **多尺度训练**：在训练过程中，MDSR 随机选择一个尺度进行训练，只更新与该尺度相关的模块，从而实现了多尺度联合训练。

---

### **3. 训练策略**

#### **EDSR 的训练策略**
1. **损失函数**：
   - 使用 L 1 损失函数代替传统的 L 2 损失函数，实验表明 L 1 损失能够带来更好的收敛性和性能。
 
2. **预训练策略**：
   - 先训练低尺度模型（如×2），然后将其作为高尺度模型（如×3、×4）的初始化，加速训练并提升最终性能。

3. **几何自集成（Geometric Self-ensemble）**：
   - 在测试阶段，通过对输入图像进行几何变换（如翻转和旋转）生成多个增强输入，然后将这些输入的超分辨率结果进行平均，进一步提升模型性能。

#### **MDSR 的训练策略**
1. **多尺度训练**：
   - 在训练过程中，随机选择一个尺度进行训练，只更新与该尺度相关的模块。
 
2. **参数共享**：
   - 通过共享主干网络的大部分参数，减少了模型的计算复杂度和参数量。

---

### **4. 性能对比**

| 特性                | EDSR                          | MDSR                          |
|---------------------|-------------------------------|-------------------------------|
| 网络深度             | 32 个残差块（约 64 层）           | 80 个残差块（约 160 层）         |
| 网络宽度             | 256 个滤波器                    | 64 个滤波器                    |
| 残差学习             | 残差块 + 残差缩放              | 残差块 + 残差缩放              |
| 批归一化层           | 明确移除                       | 明确移除                       |
| 上采样策略           | 后上采样（亚像素卷积）          | 后上采样（亚像素卷积）          |
| 多尺度处理           | 单尺度                         | 多尺度系统（×2、×3、×4）       |
| 训练策略             | L 1 损失函数 + 预训练 + 自集成    | 多尺度训练 + 参数共享          |
| 性能                | 最先进（NTIRE 2017 冠军）        | 最先进（NTIRE 2017 亚军）        |

---

### **总结**
- **EDSR**通过更深的网络、更宽的滤波器、优化的残差学习、移除 BN、后上采样策略以及几何自集成，显著提升了单尺度超分辨率任务的性能。
- **MDSR**通过共享主干网络、引入尺度特定处理模块以及多尺度训练策略，实现了在单个模型中处理多个尺度的超分辨率任务，显著减少了计算复杂度和参数量。

EDSR 和 MDSR 在多个基准数据集上均取得了当时最先进的性能，并在 NTIRE 2017 超分辨率挑战赛中分别获得了第一名和第二名。