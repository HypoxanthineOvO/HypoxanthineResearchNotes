# 1. 论文阅读参考资料
[SuperResolution 入门综述](https://zhuanlan.zhihu.com/p/558813267)
# 2. Before Deep Learning？
## 2.1. 基于插值的方法
基于插值的方法是基于一定的数学策略，从相关点计算出待恢复目标点的像素值，具有**低复杂度**和**高效率**的特点。但结果**图像的边缘效应明显**，**插值过程中没有产生新的信息**，无法恢复图像的细节。

### 2.1.1. 最近邻插值
最近邻插值是一种简单且常用的图像插值方法。其基本思想是将待插值点的像素值直接赋值为距离其最近的已知像素点的值。具体步骤如下：
1. **确定目标像素位置**：找到高分辨率图像中待求像素的位置。
2. **寻找最近邻像素**（插值）：在低分辨率图像中，找到距离目标像素位置最近的已知像素点。
3. **赋值**：将最近邻像素点的像素值赋值给目标像素位置。

这种方法计算简单，速度快，但容易产生锯齿状边缘和块状效应，图像质量较低。

### 2.1.2. 双线性插值
双线性插值方法通过在水平和垂直方向上对邻近像素值进行线性插值来的值。与更简单的最近邻插值相比，双线性插值能够产生更平滑的图像，并减少了锯齿状效应。但它仍然会导致某些细节缺失和模糊，特别是当进行较大倍数的图像放大时。具体步骤如下：
1. **确定目标像素位置**：根据高分辨率图像中待求像素的位置，找到最近的四个邻近像素（通常为 $(x_1, y_1)$，$(x_2, y_1)$，$(x_1, y_2)$，$(x_2, y_2)$）。
2. **水平插值**：通过水平方向上的线性插值，计算出位于目标位置上下两个邻近纵坐标 $(y_1, y_2)$ 之间的像素值。
    - 在水平方向上，对于目标位置的 $x$ 坐标处，利用邻近纵坐标处的像素值进行线性插值计算。
    - 计算公式：$I_h = I(x_1) \cdot \frac{x_2 - x}{x_2 - x_1} + I(x_2) \cdot \frac{x - x_1}{x_2 - x_1}$
3. **垂直插值**：通过垂直方向上的线性插值，使用步骤2中的结果计算出目标位置的像素值。
    - 在垂直方向上，对于目标位置的 $y$ 坐标处，利用上下两个水平插值结果进行线性插值计算。
    - 计算公式：$I_v = I_h \cdot \frac{y_2 - y}{y_2 - y_1} + I(y_2) \cdot \frac{y - y_1}{y_2 - y_1}$
4. **赋值**：通过双线性插值的计算结果，获得目标位置的像素值。

### 2.1.3. 双三线性插值
双三次插值方法通过在水平和垂直方向上应用三次样条插值来估计目标位置的值。相较于双线性插值，双三次插值能够提供更平滑的图像插值结果，并且能够更好地保留图像的细节和纹理。然而，双三次插值需要操作更多的邻近像素，计算复杂度也较高。具体步骤如下：

1. **确定目标像素位置**：根据高分辨率图像中待求像素的位置，找到最近的16个邻近像素（通常为 $(x_1, y_1)$，$(x_2, y_1)$，$(x_3, y_1)$，…，$(x_4, y_4)$）。

2. **水平插值**：通过水平方向上的三次插值，计算位于目标位置上下两个邻近纵坐标 $(y_1, y_2)$ 之间的像素值。
    - 在水平方向上，对于目标位置的 $x$ 坐标处，利用16个邻近像素的值进行三次样条插值计算。
    - 计算公式采用三次多项式函数：$I_h = a_0 + a_1x + a_2x^2 + a_3x^3$，其中 $x = \frac{\text{target}_x - x_1}{x_2 - x_1}$

3. **垂直插值**：通过垂直方向上的三次插值，使用步骤2中的结果计算出目标位置的像素值。
    - 在垂直方向上，对于目标位置的 $y$ 坐标处，利用上下两个水平插值结果进行三次样条插值计算。
    - 计算公式采用三次多项式函数：$I_v = a_0 + a_1y + a_2y^2 + a_3y^3$，其中 $y = \frac{\text{target}_y - y_1}{y_2 - y_1}$

4. **得到目标像素值**：通过双三次插值的计算结果，获得目标位置的像素值。

双三次插值方法通过在水平和垂直方向上应用三次样条插值来估计目标位置的值。相较于双线性插值，双三次插值能够提供更平滑的图像插值结果，并且能够更好地保留图像的细节和纹理。然而，双三次插值需要操作更多的邻近像素，计算复杂度也较高。因此，在实际应用中根据实际需求和计算资源来选择合适的插值方法。


## 2.2. 基于重构的方法
基于重构的方法对成像过程进行建模，整合来自同一场景的不同信息，获得高质量的重构结果。通常，这些方法以时间差异换取空间分辨率的提高，这通常需要预先注册和大量的计算。
### 2.2.1. 联合最大后验概率（Joint MAP）
通过**融合多幅低分辨率图像的信息**，结合先验知识（如点扩散函数，PSF）来重建高分辨率图像。这种方法能够获得较高的重建质量，但计算量大且需要精确的图像配准


### 2.2.2. 稀疏回归和自然图像先验（Sparse Regression and Natural Image Prior）

#### 2.2.2.1. 稀疏回归（Sparse Regression）
**基本概念**

稀疏回归是一种基于稀疏表示的技术，其核心思想是假设图像patch可以在一个过完备字典（overcomplete dictionary）中通过稀疏线性组合来表示。在超分辨率任务中，稀疏回归的目标是从低分辨率图像patch中学习稀疏系数，然后利用这些系数在高分辨率字典中重建高分辨率patch。

**稀疏回归的步骤**

1. **字典学习**：
   - 首先，从训练数据中学习一对低分辨率和高分辨率字典。低分辨率字典用于编码低分辨率patch，而高分辨率字典用于重建高分辨率patch。
   - 常用的字典学习方法包括K-SVD[1]和在线字典学习[2]。
2. **稀疏编码**：
   - 对于输入的低分辨率patch，通过稀疏编码算法（如OMP[3]或Lasso[4]）找到其在低分辨率字典中的稀疏表示。
   - 稀疏编码的目标是最小化以下目标函数：
     $$ \min_{\alpha} \|y - D_l \alpha\|_2^2 + \lambda \|\alpha\|_1 $$
     其中，$y$ 是低分辨率patch，$D_l$ 是低分辨率字典，$\alpha$ 是稀疏系数，$\lambda$ 是正则化参数。
3. **高分辨率重建**：
   - 使用稀疏系数 $\alpha$ 和高分辨率字典 $D_h$ 重建高分辨率patch：
     $$ x = D_h \alpha $$

**稀疏回归的局限性**
- **计算复杂度**：稀疏编码过程通常需要迭代优化，计算成本较高。
- **字典依赖性**：重建质量高度依赖于字典的质量，而字典学习本身是一个复杂的优化问题。

#### 2.2.2.2. 自然图像先验（Natural Image Prior）

**基本概念**

自然图像先验是指利用自然图像的统计特性来约束超分辨率问题的解空间。由于自然图像具有特定的统计规律（如平滑性、边缘连续性等），这些先验信息可以帮助恢复更真实的高分辨率图像。

**常见的自然图像先验**

1. **平滑性先验**：
   - 假设图像在局部区域内是平滑的，因此可以通过正则化项（如Total Variation[5]）来抑制噪声和不连续性。
2. **边缘先验**：
   - 自然图像通常包含清晰的边缘结构，因此可以通过边缘检测或梯度约束来增强重建图像的边缘锐度。
3. **自相似性先验**：
   - 自然图像在不同尺度上具有自相似性，即图像中的局部结构在不同分辨率下重复出现。基于自相似性的方法可以利用这一特性从低分辨率图像中提取高分辨率信息。
4. **稀疏性先验**：
   - 自然图像的梯度或小波系数通常具有稀疏性，因此可以通过稀疏正则化（如 $L_1$ 范数）来约束解空间。

**自然图像先验的优势**
- **物理意义**：自然图像先验基于对图像统计特性的理解，能够提供更符合人类视觉感知的重建结果。
- **鲁棒性**：先验信息可以帮助算法在缺乏足够训练数据的情况下仍然表现良好。

**自然图像先验的局限性**
- **通用性不足**：不同的图像可能具有不同的统计特性，单一的先验可能无法适用于所有场景。
- **计算复杂度**：某些先验（如自相似性）需要在图像中搜索相似patch，计算成本较高。


#### 稀疏回归与自然图像先验的结合
在实际应用中，稀疏回归和自然图像先验常常结合使用。例如：
- 在基于稀疏编码的超分辨率方法中，稀疏回归用于学习patch的表示，而自然图像先验（如平滑性或边缘连续性）用于约束重建过程。
- 一些方法（如Yang等人[7]的工作）将稀疏回归与自相似性先验结合，通过利用图像的多尺度信息来提高重建质量。

### 2.2.3. 核回归（Kernel Regression）

**核回归（Kernel Regression）**是一种非参数回归方法，广泛应用于图像超分辨率（Super-Resolution, SR）任务中。它通过利用局部像素之间的关系来估计高分辨率图像中的像素值。以下是对核回归在图像超分辨率中的详细介绍：

#### 核回归的基本概念

核回归是一种基于核函数的非参数回归方法，用于估计目标变量与输入变量之间的关系。在图像超分辨率中，核回归的目标是从低分辨率图像中估计高分辨率图像的像素值。

核函数 $K(\cdot)$ 是一个非负、对称且积分为1的函数，用于衡量输入变量之间的相似性。常见的核函数包括：
- **高斯核**：
  $$ K(u) = \exp\left(-\frac{u^2}{2h^2}\right) $$
  其中，$h$ 是带宽参数，控制核函数的平滑程度。
- **Epanechnikov核**：
  $$ K(u) = \begin{cases} 
  \frac{3}{4}(1 - u^2) & \text{if } |u| \leq 1 \\
  0 & \text{otherwise}
  \end{cases} $$

#### 核回归在图像超分辨率中的应用

在图像超分辨率中，核回归假设高分辨率图像的像素值可以通过低分辨率图像中邻近像素的加权平均来估计。权重由核函数计算，反映了邻近像素与目标像素之间的相似性。

给定低分辨率图像 $Y$，核回归估计高分辨率图像 $X$ 中的像素值 $x_i$ 为：
$$ x_i = \frac{\sum_{j \in \mathcal{N}(i)} K\left(\frac{\|y_j - y_i\|}{h}\right) y_j}{\sum_{j \in \mathcal{N}(i)} K\left(\frac{\|y_j - y_i\|}{h}\right)} $$
其中：
- $y_i$ 是低分辨率图像中与 $x_i$ 对应的像素值。
- $\mathcal{N}(i)$ 是 $y_i$ 的邻域像素集合。
- $K(\cdot)$ 是核函数，$h$ 是带宽参数。

#### 核回归的步骤
1. **邻域选择**：对于每个高分辨率像素 $x_i$，选择低分辨率图像中对应的邻域像素 $\mathcal{N}(i)$。
2. **权重计算**：使用核函数计算邻域像素的权重。
3. **像素估计**：通过加权平均估计高分辨率像素值。

#### 核回归的优势与局限性

**优势**

- **非参数性**：核回归不需要假设数据的分布形式，具有较强的灵活性。
- **局部适应性**：核回归能够根据局部像素关系自适应地估计高分辨率像素值。
- **简单易实现**：核回归的实现相对简单，计算效率较高。

**局限性**

- **带宽选择**：核回归的性能高度依赖于带宽参数 $h$ 的选择，不合适的带宽可能导致过平滑或欠平滑。
- **计算复杂度**：对于大规模图像，核回归的计算成本较高。
- **边缘保持**：核回归在处理图像边缘时可能产生模糊效应。

#### 核回归与其他超分辨率方法的比较

| 特性                | 核回归                          | 稀疏回归                          | 深度学习                          |
|---------------------|---------------------------------|-----------------------------------|-----------------------------------|
| **方法类型**        | 非参数方法                      | 参数方法                          | 数据驱动方法                      |
| **计算复杂度**      | 中等                            | 高                                | 高                                |
| **边缘保持**        | 一般                            | 较好                              | 优秀                              |
| **适用场景**        | 小规模图像                      | 中大规模图像                      | 大规模图像                        |


## 2.3. 基于浅层学习的方法
### 2.3.1. 近邻嵌入（Neighbor Embedding, NE）

#### 2.3.1.1 基本概念
近邻嵌入是一种基于流形学习的超分辨率方法，假设低分辨率和高分辨率图像patch在流形空间中具有相似的局部几何结构。通过利用低分辨率patch的邻域信息，NE可以估计对应的高分辨率patch。

#### 2.3.1.2 方法步骤
1. **构建邻域**：
   - 对于输入的低分辨率patch，在训练集中找到其 $k$ 个最近邻的低分辨率patch。
2. **计算权重**：
   - 通过最小化重构误差计算低分辨率patch与其邻域patch之间的权重：
     $$ \min_w \|y - \sum_{i=1}^k w_i y_i\|_2^2 \quad \text{s.t.} \quad \sum_{i=1}^k w_i = 1 $$
3. **重建高分辨率patch**：
   - 使用相同的权重组合对应的高分辨率patch：
     $$ x = \sum_{i=1}^k w_i x_i $$

#### 2.3.1.3 优势与局限性
- **优势**：
  - 能够捕捉局部几何结构，适用于流形假设成立的数据。
  - 实现简单，计算效率较高。
- **局限性**：
  - 对邻域选择敏感，不合适的 $k$ 值可能导致重建质量下降。
  - 在处理复杂纹理和边缘时可能表现不佳。


### 2.3.2. 基于 SVM 的方法（SVM-Based Methods）
#### 2.3.2.1 基本概念
基于 SVM 的方法是一种基于机器学习的超分辨率技术，利用支持向量机（SVM）学习低分辨率和高分辨率图像patch之间的映射关系。SVM通过最大化分类间隔来学习最优的回归函数。

#### 2.3.2.2 方法步骤
1. **特征提取**：
   - 从低分辨率图像patch中提取特征（如梯度、纹理等）。
2. **训练SVM模型**：
   - 使用提取的特征和对应的高分辨率patch训练SVM回归模型：
     $$ \min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*) $$
     其中，$\xi_i$ 和 $\xi_i^*$ 是松弛变量，$C$ 是正则化参数。
3. **预测高分辨率patch**：
   - 使用训练好的SVM模型预测高分辨率patch：
     $$ x = w^T \phi(y) + b $$
     其中，$\phi(\cdot)$ 是核函数。

#### 2.3.2.3 优势与局限性
- **优势**：
  - SVM具有较强的泛化能力，能够处理高维特征。
  - 通过核函数可以捕捉非线性关系。
- **局限性**：
  - 训练SVM模型的计算复杂度较高，尤其是在大规模数据集上。
  - 对特征选择敏感，不合适的特征可能导致性能下降。

### 2.3.3. 三种方法的比较

| 特性                | 近邻嵌入（NE）                  | 稀疏编码（SC）                  | 基于 SVM 的方法                  |
|---------------------|--------------------------------|--------------------------------|----------------------------------|
| **核心思想**        | 利用局部几何结构                | 稀疏表示                        | 学习回归映射                      |
| **计算复杂度**      | 低                             | 高                             | 高                               |
| **表示能力**        | 一般                           | 强                             | 较强                             |
| **适用场景**        | 流形假设成立的数据              | 多种图像处理任务                | 高维特征数据                      |
| **局限性**          | 对邻域选择敏感                  | 计算复杂度高                    | 对特征选择敏感                    |

### 2.3.4. 总结
- **近邻嵌入（NE）** 通过利用局部几何结构实现超分辨率，适用于流形假设成立的数据，但对邻域选择敏感。
- **稀疏编码（SC）** 通过稀疏表示和字典学习实现超分辨率，具有较强的表示能力，但计算复杂度较高。
- **基于 SVM 的方法** 通过学习回归映射实现超分辨率，具有较强的泛化能力，但对特征选择敏感且计算复杂度较高。


# 论文阅读笔记
## 经典论文阅读笔记
[[1-CNN-Based-Method]]
[[2-ResNet-Based-Method]]
[[3-GAN-Based-Method]]

## 前沿工作阅读
[[前沿工作整理]]
