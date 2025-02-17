[https://robustnerf.github.io/public/](https://robustnerf.github.io/public/)

# Abstract

神经辐射场（NeRF）擅长在给定静态场景的多视图、校准图像时合成新的视图。当场景中包含干扰物时，这些干扰物在图像捕捉过程中并不持久（移动物体、光照变化、阴影），伪影就会以视线相关的效果或 "漂浮物 "出现。为了应对干扰因素，我们主张对NeRF的训练进行一种稳健的估计，将训练数据中的干扰因素建模为一个优化问题的离群值。我们的方法成功地从场景中消除了异常值，并在合成和真实世界的场景中改进了我们的基线。我们的技术很容易被纳入现代的NeRF框架，只有很少的超参数。它不需要对干扰物的类型有先验的了解，而是专注于优化问题，而不是预处理或对瞬态物体进行建模。更多结果请见我们的网页 `https://robustnerf.github.io/public`

# Introduction

仅从二维图像了解静态三维场景的结构是计算机视觉的一个基本问题[44]。它可以在AR/VR中找到应用，用于绘制虚拟环境[6, 36, 61]，在自主机器人中用于行动规划[1]，以及在摄影测量中用于创建现实世界物体的数字副本[34]。

神经场[55]最近通过在神经网络的权重中存储三维表征[39]，彻底改变了这项经典任务。这些表征通过反向传播的图像差异进行优化。当场域存储与视线相关的辐射度并采用体积渲染时[21]，我们可以以照片般的精确度捕捉三维场景，我们将生成的表示称为神经辐射场，或NeRF[25]。

训练NeRF模型通常需要大量的图像集合，并配备精确的相机校准，这通常可以通过结构-运动来恢复[37]。在其简单性的背后，NeRF隐藏着几个假设。由于模型通常在训练中最小化 RGB 颜色空间中的误差，图像在光度上的一致性是最重要的——从同一有利位置拍摄的两张照片在噪声方面应该是相同的。除非采用一种明确说明的方法[35]，否则应该手动保持相机的焦点、曝光、白平衡和ISO固定。

然而，正确地配置相机并不是捕捉高质量的NeRFs的全部要求——避免Distractors也很重要：任何在整个捕捉过程中不持久的东西。Distractors有多种形式，从操作者探索场景时投下的硬阴影，到宠物或孩子在摄像机视野内随意走动。Distractors的去除很繁琐，因为这需要逐个像素进行标记。它们的检测也很繁琐，因为典型的NeRF场景是从数百张输入图像中训练出来的，而Distractors的类型并不是事先就知道的。如果忽略Distractors，重建场景的质量就会大打折扣；见[[RobustNeRF]]。

![[02-论文笔记/NeuralRendering（神经渲染）/RobustNeRF/images/Untitled.png|Untitled.png]]

图1. NeRF假设一个场景的观察图像具有光度一致性。违反这一假设，就像最上面一行的图像一样，会产生以 "漂浮物 "形式出现的内容不一致的重建场景（用省略号强调）。我们介绍一种简单的技术，通过自动忽略分心物而产生干净的重建，无需明确的监督。

在一个典型的捕捉过程中，人们没有能力从同一有利位置捕捉同一场景的多张图像，这使得干扰因素在数学上的建模具有挑战性。更具体地说，虽然视线依赖效应是赋予NeRF逼真外观的原因，但该模型如何区分干扰物和视线依赖效应？  
尽管有这些挑战，研究界已经设计了几种方法来克服这个问题。  

- 如果已知干扰物属于一个特定的类别（例如，人），我们可以用预先训练好的语义分割模型来移除它们[35, 43]——这个过程并不能推广到 "意外 "的干扰物，例如阴影。
- 人们可以将干扰物建模为每个图像的瞬时现象，并控制瞬时/持久建模的平衡[23] 。——然而，很难调整控制这一帕累托最优目标的损失。
- 人们可以对数据进行时间建模（即高帧率视频），并将场景分解为静态和动态（即干扰物）部分[53] 。——但这显然只适用于视频而不是照片采集的捕捉。

相反，我们通过在NeRF优化中把Distractor建模为Outliers来处理Distractor的问题。

我们通过 robust estimation 的角度来分析上述技术，使我们能够理解它们的行为，并设计出一种不仅实现起来更简单而且更有效的方法（见图1）。因此，我们得到了一个直接实现的方法，需要最小甚至没有超参数调整，并实现了最先进的性能。我们评估了我们的方法。

- 从数量上看，用合成的、但又是照片般逼真的数据进行重建。
- 在公开可用的数据集上进行定性评估（通常对以前的方法进行微调，以便有效地工作）；
- 在一个新的自然和合成场景集合上，包括那些由机器人自主获得的场景，使我们能够证明以前的方法对超参数调整的敏感性。

  

# Some Formula

$$\mathcal L_{\text{rgb}}^{\mathbf r,i}(\mathbf \theta)=\Vert \mathbf C(\mathbf r;\mathbf\theta)-\mathbf C_i(\mathbf r)\Vert_2^2$$

$$\left.\begin{aligned}&\mathbf c(\mathbf x,\mathbf d)\\&\sigma(\mathbf x)\end{aligned}\right\}f(\mathbf x,\mathbf d;\theta)$$

# Methods

![[02-论文笔记/NeuralRendering（神经渲染）/RobustNeRF/images/Untitled 1.png|Untitled 1.png]]

图2. 模糊性——一个简单的二维场景，一个静态物体（蓝色）被三台摄像机捕获。在第一次和第三次拍摄时，由于分心物在视野范围内，所以场景不是照片一致的。场景中不一致的部分最终会被编码为与视线相关的效果--即使我们假设地面真实的几何形状。

经典的 [[RobustNeRF]] 对于捕捉光度一致的场景是有效的，这导致了我们现在习惯于在最近的研究中看到的照片般真实的新视角合成。然而，"当场景中的一些元素在整个捕捉过程中不具有持续性时，会发生什么？" 这类场景的简单例子包括：一个物体只出现在部分观察到的图像中，或者在所有观察到的图像中可能没有保持在相同的位置。例如，[[RobustNeRF]]描述了一个二维场景，包括一个持久的物体（卡车），以及几个瞬时的物体（如人和狗）。当来自三个摄像机的蓝色光线与卡车相交时，来自摄像机1和3的绿色和橙色光线与瞬时物体相交。对于视频捕捉和时空NeRF模型来说，持久的物体包括场景的 "静态 "部分，而其余部分则被称为 "动态"。

## 1. Sensitivity to Outliers

对于Lambertian场景，photo-consistent structure是独立于视图的，因为场景辐射度只取决于入射光线[16]。对于这样的场景，通过最小化 [[RobustNeRF]] 来训练的与视线相关的 [[RobustNeRF]] 承认局部优化，其中瞬时物体由与视线相关的项来解释。[[RobustNeRF]]描述了这一点，输出的颜色对应于outliers的memorized color——即view-dependent radiance。这样的模型利用了模型的视线依赖能力来过度拟合观测结果，有效地记忆了瞬态物体。人们可以改变模型以消除对 $\mathbf d$ 的依赖，但L2损失仍然是有问题的，因为最小二乘法（LS）估计器对outliers或heavy-tailed noise distribution很敏感。

在更自然的条件下，放弃Lambertian假设，问题变得更加复杂，因为非兰伯斯反射现象和离群值都可以解释为视线依赖的辐射度。虽然我们希望模型能够捕捉到与照片一致的视线相关的辐射度，但离群值和其他瞬态现象最好被忽略。而在这种情况下，用L2损失进行优化会产生明显的重建误差；见图1。像这样的问题在NeRF模型拟合中普遍存在，特别是在具有复杂反射率、非刚性或独立移动物体的非控制环境中。

## 2. Robustness to outliers

### **Robustness via semantic segmentation**

在NeRF模型优化过程中，减少outlier contamination（污染）的一种方法是依靠一个 oracle $\mathbf S$，指定图像 $i$ 中的一个给定像素 $\mathbf r$ 是否是outlier，应该从经验损失中排除，将 [[RobustNeRF]] 替换为：

$$\mathcal L_{\text{oracle}}^{\mathbf r,i}(\bm{\theta})=\mathbf S_i(\mathbf r)\cdot \Vert \mathbf C(\mathbf r;\bm\theta)-\mathbf C_i(\mathbf r)\Vert_2^2$$

在实际操作中，一个预先训练好的（语义）分割网络 $\mathbf S$ 可能当做 oracle 使用， $\mathbf S_i=\mathcal S(\mathbf C_i)$ 。例如，Nerf-in-the-wild[23]采用了一个语义分割器来移除被人占据的像素，因为他们在照片旅游的背景下代表了离群值。Urban Radiance Fields[35]将天空像素分割出来，而LOL-NeRF[33]忽略了不属于人脸的像素。这种方法的明显问题是需要一个 oracle 来检测任意干扰者的离群值。

### **Robust estimators**

另一种降低对异常值敏感度的方法是用 Robust 的损失函数代替传统的L2-Loss（3）（例如，[2，41]），这样在优化过程中可以降低光度不一致的观测值的权重。给定一个robust kernel $\kappa(\cdot)$ ，我们将训练损失改写为：

$$\mathcal L_{\text{robust}}^{\mathbf r,i}(\bm{\theta})=\mathbf \kappa\left(\Vert \mathbf C(\mathbf r;\bm\theta)-\mathbf C_i(\mathbf r)\Vert_2\right)$$

其中 $\kappa(\cdot)$ 为正值且单调增。例如，MipNeRF[3]采用了L1损失 $κ(\epsilon)=|\epsilon|$ ，它在NeRF训练期间对异常值提供了某种程度的鲁棒性。鉴于我们的分析，一个有效的问题是我们是否可以直接采用robust kernel来处理我们的问题，如果可以，考虑到大量的robust kernel[2]，哪一个是首选的核。

![[02-论文笔记/NeuralRendering（神经渲染）/RobustNeRF/images/Untitled 2.png|Untitled 2.png]]

图3. **直方图**——当残差分布与估计器所暗示的分布一致时，鲁棒估计器表现良好（例如，L2的高斯，L1的拉普拉斯）。这里我们可以看到残差的真实分布（左下角），它与任何简单的参数分布都很难匹配。

![[02-论文笔记/NeuralRendering（神经渲染）/RobustNeRF/images/Untitled 3.png|Untitled 3.png]]

图4. **Kernel**（左上）来自[2]的robust kernels，包括L2（α=2）、Charbonnier（α=1）和GemanMcClure（α=2）。(右上)在训练中期，对于分散注意力的人和精细的细节来说，残差的大小是相似的，具有大残差的像素被学习得更慢，因为重新下降的核的梯度变平了。(右下)在降低大残差权重方面过于激进的核子会同时去除异常值和高频细节。(左下）不那么积极的核子不能有效地去除异常值。

不幸的是，正如上文所讨论的，outlier和non-Lambertian effects都可以被建模为view dependent effects（见[[RobustNeRF]]）。因此，只是简单地应用稳健的估计器，就很难将信号和噪音分开。[[RobustNeRF]]显示了一些例子，在这些例子中，outliers被移除，但细粒度的纹理和视线依赖的细节也被丢失，或者相反，细粒度的细节被保留，但异常值在重建的场景中造成了伪影。我们还可以观察到这些情况的混合体，即细节没有被很好地捕获，离群值也没有被完全去除。我们发现，这种行为在许多不同的稳健估计器和参数设置中都会持续发生。

训练时也可能出现问题。robust estimator 的梯度与模型参数的关系可以用链式法则表示为：

$$\frac{\partial\kappa(\epsilon(\bm\theta))}{\partial\bm\theta}\bigg|_{\bm\theta^{(t)}}=\frac{\partial\kappa(\epsilon)}{\partial\epsilon}\bigg|_{\epsilon(\bm\theta^{(t)})}\cdot\frac{\partial\epsilon(\bm\theta)}{\partial\bm\theta}\bigg|_{\bm\theta^{(t)}}$$

> $\epsilon$ 是指内部 颜色空间 的Loss

第二个factor是经典的NeRF梯度。第一个factor是在当前误差残差（ $\epsilon (\theta ^{(t)})$）上evaluate的kernel gradient。在训练过程中，大的残差可能来自尚未学习的高频细节，也可能来自outliers（见[[RobustNeRF]]（底部））。这就解释了为什么以 [[RobustNeRF]] 形式实现的稳健优化不应该被期望从outliers中解开high-frequency details。此外，当采用 roubust kernel 时，如Redescending估计器，这也解释了视觉保真度的损失。也就是说，因为（大）残差的梯度被内核的（小）梯度降低了权重，减慢了对这些细粒度细节的学习（见图4（顶部））。

## **Robustness via Trimmed Least Squares**

在下文中，我们主张采用一种带有剪枝最小二乘法（LS）损失的迭代再加权最小二乘法（IRLS）来进行NeRF模型拟合。

### **Iteratively Reweighted least Squares**

IRLS是一种广泛使用的稳健估计方法，包括解决一连串的加权LS问题，这些问题的权重被调整以减少异常值的影响。为此，在迭代 $t$ 中，我们可以将损失写成：

$$\begin{aligned}\mathcal L_{\text{robust}}^{\mathbf r,i}(\bm{\theta}^{(t)})&=\omega(\bm\epsilon^{(t-1)}(\mathbf r))\cdot\left(\Vert \mathbf C(\mathbf r;\bm\theta)-\mathbf C_i(\mathbf r)\Vert^2_2\right)\\\bm\epsilon^{(t-1)}(\mathbf r)&=\Vert\mathbf C(\mathbf r;\bm\theta^{(t-1)})-\mathbf C_i(\mathbf r)\Vert_2\end{aligned}$$

对于由 $ω(\epsilon )= \epsilon^{-1} \cdot ∂κ(\epsilon)/∂\epsilon$ 给出的权重函数，我们可以证明，在合适的条件下，迭代收敛到(5)的局部最小值（见[41，第3节]）。

这个框架接纳了一个广泛的损失系列，包括heavy-tailed noise过程的最大似然估计器。图4中的例子包括Charbonnier损失（平滑的L1），以及更aggressive的redescending估计器，如Lorentzian或Geman-McClure[2]。(4)中的目标也可以被看作是一个加权的LS目标，其二进制权重由一个 oracle 提供。而且，正如下面详细讨论的那样，我们也可以通过IRLS和加权LS的视角来看待最近的几种方法，如NeRFW[23]和D2NeRF[53]。

尽管如此，为NeRF优化选择一个合适的权重函数 $ω(\epsilon)$ 并不困难，这在很大程度上是由于view-dependent radiance phenomena 和 outliers 之间的内在模糊性。人们可能会尝试通过学习神经权重函数来解决这个问题[40]，尽管生成足够多的注释训练数据可能是令人望而却步的。相反，下面采取的方法是利用outlier structure中的inductive bias，结合robust、trimmed LS estunatir的简单性。

### **Trimmed Robust Kernels**

我们的目标是开发一个用于迭代加权LS优化的权重函数，它是简单的，并能捕捉到对NeRF优化有用的inductive biases。为了简单起见，我们选择了一个具有直观参数的二元权重函数，通过模型拟合自然适应，这样就可以快速学习到 not outliers 的细粒度图像细节。捕捉典型outliers的结构性也很重要，这与大多数 robust estimator 公式中典型的i.i.d.假设相反。为此，权重函数应该捕捉到outliers过程的空间平滑性，认识到物体通常具有continuous local support，因此outliers有望占据图像的大面积和连接区域（例如，要从照片-旅游数据集中分割出的人物轮廓）。  
令人惊讶的是，一个相对简单的权重函数体现了这些特性，并在实践中表现得非常好。该权重函数是基于所谓的trimmed estimators，这些估计器用于修剪最小二乘法，如修剪ICP[7]中使用的。我们首先对残差进行分类，并假设低于某个百分位数的残差是inliers。为了方便起见，我们选取50%的百分位数（即中位数），定义为  

$$\tilde{\omega}(\mathbf r)=\epsilon(\mathbf r)\leq\mathcal T_\epsilon,\quad \mathcal T_\epsilon = \text{Median}_{\mathbf{r}}\{\epsilon(\mathbf r)\}$$

为了捕捉 outliers 的spatial smoothness，我们用 $3×3$ 的 box-kernel $\mathcal B_{3×3}$ 进一步在空间上扩散inlier/outlier labels $ω$。形式上，我们定义

$$\mathcal W(\mathbf r)=(\tilde{\omega}(\mathbf r)\circledast \mathcal B_{3\times3})\geq T_{\circledast},\quad T_{\circledast}\geq 0.5$$

这倾向于去除被归类为异常值的高频细节，允许它们在优化期间被NeRF模型捕获（见[[RobustNeRF]]）。

![[02-论文笔记/NeuralRendering（神经渲染）/RobustNeRF/images/Untitled 4.png]]

图5. **Algorithm ——** 我们在两个例子上可视化了我们通过残差计算的权重函数。(上图）从训练视角呈现的（训练中期）NeRF的残差，（下图）包含小空间范围的残差（点、线）和大空间范围的残差（方块）的玩具残差图像。注意到幅度大但空间范围小的残差（盒子的纹理，点，线）被包括在优化中，而空间范围大的较弱的残差被排除在外。请注意，虽然我们对斑块进行操作，但为了便于可视化，我们将整个图像上的权重函数可视化。

虽然修剪后的权重函数（9）提高了模型拟合的稳健性，但它也会在训练的早期错误地分类细粒度的纹理细节，因为NeRF模型首先捕捉的是粗粒度的结构。这些局部的纹理元素可能会出现，但只有在很长的训练时间后才会出现。我们发现，对空间连贯性更强的归纳偏向可以使细粒度的细节更快地被学习。为此，我们在 $16×16$ 的邻域上对异常值的检测进行了汇总；也就是说，我们根据 $\mathcal W$ 在patch的 $16×16$ 邻域中的行为，将整个 $8×8$ 的patch标记为inliers or outliers。形式上，将 $\mathbf r$ 周围像素的 $N×N$ 邻域表示为 $\mathcal R_N(\mathbf r)$ ，我们定义为

$$\omega(\mathcal R_8(\mathbf r))=\mathbb E_{\mathbf S\sim\mathcal R_{16}(\mathbf r)}[\mathcal W(\mathbf S)]\geq \mathcal T_{\mathcal R},\quad \mathcal T_{\mathcal R}=0.6$$

![[02-论文笔记/NeuralRendering（神经渲染）/RobustNeRF/images/Untitled 5.png]]

图6. **残差 ——** 对于上行所示的数据集，我们可视化了RobustNeRF训练残差的动态，它显示了随着时间的推移，估计的分散注意力的权重如何从随机（ $(t/T)=0.5%$ ）到识别分散注意力的像素（ $(t/T)=100%$ ）而无需任何明确的监督。

请注意，这个 robust 的权重函数在优化过程中会发生变化，正如人们对IRLS所期望的那样，权重是上一次迭代中残差的函数。也就是说，在训练过程中，将像素标记为 inliers/outliers 的做法会发生变化，并在训练收敛时围绕类似于oracle 提供的掩码进行解决（见图6）。