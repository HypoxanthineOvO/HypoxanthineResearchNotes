```table-of-contents
title: 111
style: nestedList # TOC style (nestedList|nestedOrderedList|inlineFirstLevel)
minLevel: 0 # Include headings from the specified level
maxLevel: 1 # Include headings up to the specified level
includeLinks: true # Make headings clickable
hideWhenEmpty: false # Hide TOC if no headings are found
debugInConsole: false # Print debug info in Obsidian console
```
# 1.1: SRCNN
## 概述

### 1. **SRCNN 与传统稀疏编码方法的主要区别是什么？**
   - **端到端学习 vs 分步优化**：
     - **传统稀疏编码**：需要多步骤流程（块提取、字典学习、稀疏编码、重建），每一步独立优化，==缺乏全局一致性==。
     - **SRCNN**：通过卷积神经网络直接学习低分辨率到高分辨率的端到端映射，所有步骤（块提取、非线性映射、重建）联合优化，减少人工干预。
   - **隐式特征表示**：
     - 传统方法依赖显式设计的字典或流形，而 SRCNN 通过隐式卷积层自动学习特征表示。
   - **计算效率**：
     - SRCNN 无需在线迭代优化（如稀疏编码中的迭代求解），前向传播速度快，适合实时应用。

### 2. **SRCNN 的网络结构及每层作用**
   - **三层卷积结构**：
     1. **块提取与表示层（Patch Extraction）**：
        - 输入：低分辨率图像（经双三次插值上采样）。
        - 操作：使用大尺寸卷积核（如 9×9）提取局部块，映射为高维特征。
        - 输出：64 个特征图，捕捉局部结构信息。
     2. **非线性映射层（Non-linear Mapping）**：
        - 操作：1×1 卷积（或更大尺寸如 5×5）对特征进行非线性变换。
        - 输出：32 个特征图，将低分辨率特征映射到高分辨率空间。
     3. **重建层（Reconstruction）**：
        - 操作：5×5 卷积聚合高分辨率特征，生成最终输出。
        - 输出：单通道（Y）或三通道（RGB）高分辨率图像。
  - 可以理解为，本质上就是一组调整了不同配置的卷积在嗯卷。

### 3. **SRCNN 的实验优势**
   - **定量指标**：
     - **PSNR/SSIM**：在 Set 5、Set 14、BSD 200 等数据集上显著优于 SC、ANR、A+等方法（如 Set 5 上 PSNR 提升 0.15-0.3 dB）。
     - **速度**：纯 C++实现下，处理一幅图像仅需 0.05-0.6 秒（取决于网络规模），远快于传统优化方法。
   - **定性结果**：
     - 生成边缘更锐利、细节更丰富的图像，避免伪影（如图 14-16 中的蝴蝶、斑马纹理）。
   - **灵活性**：
     - 可通过调整滤波器数量（如 64→128）、尺寸（如 9→11）平衡性能与速度。

### 4. **颜色图像处理策略**
   - **传统方法**：仅处理亮度通道（YCbCr 中的 Y），色度通道（Cb/Cr）通过双三次插值上采样。
   - **SRCNN 改进**：
     - **多通道输入**：直接处理 RGB 三通道，利用通道间相关性提升重建质量。
     - **实验结果**：RGB 训练相比单通道 Y，PSNR 提升 0.07 dB，色度通道质量更高（如 Cb/Cr 的 PSNR 提升 1-2 dB）。

### 5. **训练策略与参数选择**
   - **损失函数**：均方误差（MSE），因其与 PSNR 直接相关。
   - **学习率**：
     - 前两层：$10^{-4}$，最后一层：$10^{-5}$（防止梯度爆炸）。
   - **数据准备**：
     - 从 ImageNet 裁剪 33×33 子图，模拟低分辨率图像（高斯模糊+下采样+双三次上采样）。
   - **初始化**：滤波器权重从高斯分布（均值 0，标准差 0.001）随机初始化。

---
### 6. **深层网络的局限性**
   - **训练难度**：无池化/全连接层，对初始化敏感，易陷入局部最优。
   - **参数爆炸**：增加层数（如 9-1-1-5）导致参数量倍增，但性能提升有限（如四层仅提升 0.1 dB）。
   - **梯度问题**：深层网络梯度传播不稳定，需精细调整学习率。
---
### 7. **评估指标**
   - **PSNR/SSIM**：衡量像素级精度和结构相似性。
   - **感知指标**：
     - **IFC**：基于自然场景统计的信息保真度。
     - **NQM/WPSNR**：加权信噪比，考虑视觉敏感度。
     - **MSSSIM**：多尺度结构相似性，更贴近人眼感知。
   - **速度指标**：单图处理时间（CPU/GPU）。

## 简单评价
 SRCNN 也太朴素了，就是设计了一个 CNN 对图像进行优化，从现在的角度来看，当时应该是不能确定这种方法 Work，并且缺乏框架（必须手动实现全流程），但实际上 Idea 难度并不大


---
# 1.2. ESPCN

## **1. 相关工作（Related Work）**
- **传统方法**：基于插值、边缘统计、稀疏编码等方法，计算复杂且性能有限。
- **深度学习方法**：
  - SRCNN：首次将 CNN 用于 SR，但需先上采样 LR 图像，计算量大。
  - TNRD：可训练非线性扩散模型，性能优秀但速度慢。
- **关键缺陷**：现有方法在 HR 空间操作，导致高计算成本。

---
## 2. **Method 章节详解**
### **2.1. 整体思路**
传统超分辨率方法（如 SRCNN）通常先将低分辨率（LR）图像通过双三次插值上采样到高分辨率（HR）空间，再通过卷积网络增强细节。这种方法存在两个问题：
1. **计算复杂度高**：所有卷积操作在 HR 空间进行，计算量与图像分辨率平方成正比。
2. **信息冗余**：固定插值（如双三次）未引入新信息，仅依赖后续卷积修复误差。

**ESPCN 的核心改进**：
- **特征提取在 LR 空间**：所有卷积层直接在 LR 图像上操作，降低计算量。
- **末端子像素卷积上采样**：仅最后一层通过可学习的子像素卷积将 LR 特征图转换为 HR 图像。

### **2.2. 网络结构设计**
ESPCN 由三部分组成（如图 1 所示）：
1. **LR 特征提取层**（2 层卷积）：
   - **第一层**：5×5 卷积，输出 64 通道特征图。
   - **第二层**：3×3 卷积，输出 32 通道特征图。
   - **激活函数**：前两层使用 ReLU，最后一层无激活函数。
   - **作用**：提取 LR 图像的多尺度特征（边缘、纹理等）。

2. **子像素卷积层**（Sub-pixel convolution layer）（核心创新）：
   - **输入**：LR 特征图（尺寸为 $H \times W \times C$，如 32 通道）。
   - **操作**：
     - 通过 1 层 3×3 卷积，将通道数扩展为 $C \times r^2$（$r$ 为上采样因子，如$r=3$ 时通道数变为 32×9=288）。
     - **周期洗牌（Periodic Shuffling, PS）**：将通道维度转换为空间维度，生成 HR 图像（尺寸$rH \times rW \times C$）。
![[1-CNN-Based-Method-001.png]]
### **2.3. 子像素卷积的数学实现**
- **公式定义**（公式 3-4）：
  $$
  I^{SR} = f_L (I^{LR}) = \text{PS}(W_L * f_{L-1}(I^{LR}) + b_L)
  $$ 
  - $W_L$：卷积核，尺寸为 $n_{L-1} \times r^2 C \times k_L \times k_L$（$k_L=3$）。
  - $\text{PS}$：周期洗牌操作，将 $H \times W \times r^2 C$ 转换为 $rH \times rW \times C$。

- **周期洗牌的具体步骤**：
  假设输入特征图尺寸为 $H \times W \times r^2 C$，输出为 $rH \times rW \times C$：
  1. **通道分组**：将 $r^2 C$ 通道分为 $C$ 组，每组 $r^2$  通道。
  2. **空间重排**：每组中的 $r^2$ 通道按位置排列为 $r \times r$  的网格，填充到 HR 空间对应位置。
  *示例*：当 $r=2$ 时，通道 0-3 分别对应 HR 图像的 $(0,0)$、$(0,1)$、$(1,0)$、$(1,1)$ 位置。

### **2.4. 计算复杂度分析**
ESPCN 的复杂度优势来自两方面：
3. **LR 空间操作**：
   - 假设放大因子为$r$，传统方法在 HR 空间卷积的计算量为 $O ((rH \cdot rW) \cdot k^2)$，而 ESPCN 在 LR 空间的计算量为 $O ((H \cdot W) \cdot k^2)$，减少$r^2$ 倍。
4. **子像素卷积的高效性**：
   - 反卷积层需在 HR 空间计算，而子像素卷积的 PS 操作仅为内存重排（零计算成本）。

### **2.5. 训练细节**
- **输入预处理**：
  - 从 HR 图像裁剪 17×17 像素块，下采样为 LR 块（尺寸$\lfloor 17/r \rfloor \times \lfloor 17/r \rfloor$）。
  - 数据增强：随机旋转、翻转。
- **损失函数**：像素级均方误差（MSE）。
- **优化器**：随机梯度下降（SGD），初始学习率 0.01，逐步衰减至 0.0001。
- **激活函数**：实验发现**tanh**比 ReLU 更优（可能因 ReLU 的稀疏性不利于密集像素预测）。
### **2.6. 与传统方法的对比**

| **方法**    | **上采样位置** | **计算复杂度**    | **可学习上采样**   |
| --------- | --------- | ------------ | ------------ |
| SRCNN     | 输入阶段      | 高（HR 空间）     | 否（双三次插值）     |
| 反卷积网络     | 中间层       | 中            | 是（反卷积层）      |
| **ESPCN** | **最后一层**  | **低（LR 空间）** | **是（子像素卷积）** |

> [!notes] 感觉这个论文的核心思路是，尽量把提取特征放在 LR 图片上（节约不必要的计算量），然后设计了一个网络代替普通的插值上采样；这个“周期性洗牌”实际上就是先在通道层面生成更多数据，然后把数据排布到更大的长宽上（相当于：LR 的一个像素生成 $r^2$ 个通道，然后排成 $r\times r$ 个像素）。

# 1.3. RCAN
# 该篇文章的笔记

## 1. 该篇文章所研究的任务介绍
- **研究任务**：单图像超分辨率（Single Image Super-Resolution, SISR），即从低分辨率（LR）图像重建高分辨率（HR）图像。应用领域包括安全监控、医学影像和物体识别。
- **现有方法的局限性**：
  - 深层 CNN 网络（如 EDSR、MDSR）存在训练困难问题。
  - 现有方法对通道特征（channel-wise features）的等权重处理限制了模型的判别能力，无法区分低频和高频信息。

## 2. 该篇文章的研究动机
- **现有方法的主要问题**：
  1. **深层网络训练困难**：当网络深度超过 400 层时，直接堆叠残差块（Residual Block）会导致性能下降。
  2. **通道特征处理不灵活**：现有方法（如 EDSR）对所有通道特征采用统一处理，无法自适应地关注高频信息。
- **研究动机**：
  - **低频信息冗余**：LR 输入和中间特征包含大量低频信息，直接传递这些信息会浪费计算资源。
  - **高频信息重建需求**：图像 SR 的核心是恢复高频细节（如边缘、纹理），需要更高效的通道特征建模机制。

## 3. 该篇文章所提出的主要方法
### 3.1 整体架构
提出的 **Residual Channel Attention Network (RCAN)** 包含四个部分：
1. **浅层特征提取**：单层卷积提取初始特征 $F_0 = H_{SF}(I_{LR})$。
2. **残差中的残差结构（RIR）**：由多个残差组（Residual Group, RG）构成，每个 RG 包含多个残差通道注意力块（RCAB）。
3. **上采样模块**：使用 ESPCNN 方法进行分辨率提升。
4. **重建模块**：单层卷积生成最终 HR 图像。

### 3.2 核心创新点
- **残差中的残差结构（RIR）**：
  - **长跳跃连接（LSC）**：跨多个 RG 的残差连接，公式为 $F_{DF} = F_0 + W_{LSC}F_G$。
  - **短跳跃连接（SSC）**：单个 RG 内部的残差连接，公式为 $F_g = F_{g-1} + W_gF_{g,B}$。
  - **作用**：通过多级跳跃连接绕过低频信息，使网络专注于学习高频残差。

- **通道注意力机制（CA）**：
  - **全局平均池化**：生成通道统计量 $z_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W x_c(i,j)$。
  - **门控机制**：通过全连接层和 Sigmoid 函数生成通道权重 $s = f(W_U \delta(W_D z))$。
  - **特征重标定**：对通道特征进行自适应缩放 $\hat{x}_c = s_c \cdot x_c$。

### 3.3 残差通道注意力块（RCAB）
- 结构：两个卷积层 + 通道注意力模块（见图 4）。
- 公式：$F_{g,b} = F_{g,b-1} + R_{g,b}(X_{g,b}) \cdot X_{g,b}$，其中 $X_{g,b} = W_{2,g,b} \delta(W_{1,g,b} F_{g,b-1})$。

## 4. 该篇文章的实验效果
### 4.1 定量结果
- **BI 退化模型**（Set 5 数据集）：

| 方法          | ×2 PSNR   | ×4 PSNR   | ×8 PSNR   |
| ----------- | --------- | --------- | --------- |
| EDSR [10]   | 38.11     | 32.46     | 26.96     |
| RCAN (ours) | **38.27** | **32.63** | **27.31** |

- **BD 退化模型**（Urban 100 数据集）：

| 方法     | ×3 PSNR | 
|------------|---------|
| RDN [17] | 28.46 | 
| RCAN (ours)| **28.81** |

### 4.2 定性结果
- **视觉效果**：RCAN 在恢复高频细节（如网格、文字边缘）上显著优于 EDSR 和 MemNet（见图 5 和图 6）。
- **计算效率**：RCAN 参数量（16 M）少于 EDSR（43 M），但 PSNR 提升 0.16 dB（Set 5 ×4）。

### 4.3 消融实验
- **RIR 和 CA 的贡献**：

| 配置                  | Set 5 ×2 PSNR |
| ------------------- | ------------- |
| Baseline (无 RIR/CA) | 37.45         |
| + RIR               | 37.87 (+0.42) |
| + RIR + CA          | **37.90**     |

### 4.4 下游任务提升
- **物体识别**：使用 RCAN 超分辨率后的图像，ResNet-50 的 Top-1 错误率从 0.449（ENet-E）降低到 **0.393**（见表 4）。


# 1.4. HAN

# 1.5. MSFIN


