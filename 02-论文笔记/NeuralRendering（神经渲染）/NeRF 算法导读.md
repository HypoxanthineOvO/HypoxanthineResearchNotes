
![|700x487](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=OGZmYzBhYzA4Yzg5Y2FjNWE0OGIxNzM0NGQzN2UzNjNfcE9yMEg4Z0pkcmYxVUpKQ1VqWDN5YjRsUTRHSGJDZVdfVG9rZW46TnZQa2J1WlMwb0JsMXR4MTBicGN1R2VlbnNnXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

[NeRF](https://so.csdn.net/so/search?q=NeRF&spm=1001.2101.3001.7020)是2020年ECCV论文，任务是**新视角的合成**和**三维重建**（关于新视角合成，可以参考链接[新视角合成-任务定义](https://zhuanlan.zhihu.com/p/486710656?utm_psn=1806068236170690560)），是借助[深度学习](https://so.csdn.net/so/search?q=%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0&spm=1001.2101.3001.7020) 技术的计算机图形学任务，实现了摄像机级别的逼真的新视图合成。仅仅2年时间，相关work和论文就已经大量涌现。
NeRF 最大的贡献是，启发了使用神经网络来编码场景的方式。传统方法主要使用数学方法去解析场景信息，具有比较强的局限性。而 NeRF 所提出的“位置编码（Positional Encoding）”，它把原本难以学习的信息暴露出来，使其变得容易被神经网络所学习，达到了非常惊人的“照片级合成”的效果。
当然，其局限性也非常明显，过大的神经网络和过多的采样点数量（参见后文“采样”概念）导致其运算速度非常感人，即使在最好的 GPU 上也需要很长时间。后续改进工作主要考虑改进质量（让图像更真实）和算法使用范围（让算法不再局限于重建单个静态物体），或是优化速度（实时）。
- **[论文原文](https://arxiv.org/abs/2003.08934)**
- **[TensorFlow 代码](https://github.com/bmild/nerf)**
- **[PyToch 代码](https://github.com/yenchenlin/nerf-pytorch)**
- **[官方数据](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)**
# **一、技术原理**

> 以下为原理简述，更多物理公式推导请看最后一章

1. ## **概览**
    

NeRF可以简要概括为用一个 MLP（多层感知机，其结构全部为**全连接层**加上激活层，不包括任何卷积操作）去隐式地学习一个静态3D场景，实现复杂场景的任意新视角合成（渲染）。

为了训练网络，针对一个**静态场景**，需要提供：

- 包含**大量相机参数（此处的参数被称为内参）已知的图片**的训练集
    
- **图片对应的相机位置和朝向**（此处的参数被称为外参）。
    

其训练过程和推理过程分别为：

- **训练过程**中，使用多视角的数据进行训练，空间中目标位置具有更高的密度和更准确的颜色，促使神经网络预测一个连续性更好的场景模型。
    
- **推理过程**中，以**任意的相机位置+朝向**作为**输入**，经过训练好的神经网络进行体绘制（Volume Rendering），即可以从渲染出**图片**结果了。
    

![](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=M2FiOWZhYzUzNGU5NmU3NzRlMGFmMjk5NDBmYjdhMzRfb3gybld2QzVZVlhIeUtpc2tnTHdPZlU4bVE0Z29iNmhfVG9rZW46T3NmaWJOTGpjb0lMTEN4YnNMdWNVeVMzbnRjXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

2. ## **基于神经辐射场（Neural Radiance Field）的体素渲染算法**
    

NeRF 函数是将一个连续的场景表示为一个输入为 5D 向量的函数，包括：

- 一个空间点的 3D 坐标位置 $\mathbf x=(x,y,z)$
    
- 其所对应的视角方向 $(θ,ϕ)$
    

> 位置 $\mathbf $，指的是由各个相机原点出发的，经过对应图像中每一像素引起的射线，所经过的采样点位置，方向指该射线的方向

输出为视角相关的该 3D 点的颜色 $\mathbf c=(r,g,b$，和对应位置（体素）的密度 $σ$。

> 体素密度 $$ 和方向相关的颜色值 $\mathbf c=(r,g,b$，共同决定了该位置在后续渲染时所提供的对渲染结果的数值

实践中，用 3D 笛卡尔单位向量 $\mathbf $ 来表示方向，因此这个神经网络可以写作：

$F_Θ:(\mathbf x,\mathbf d)→(\mathbf c,σ)$。

![|700x266](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=NjIzNWNmOTRiYzU2OGE5ZTc2OTRlZTQyNjNjMTc0ZjRfaWREd1p1dllHak16UkozZDNSaktyTXFqQlppRXNtMkdfVG9rZW46WVJpMWJtZDhUb203anZ4WGFBU2NST2M5bkZjXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

在具体的实现中，$\mathbf $ 首先输入到MLP网络中，并输出 $$ 和一个256维的中间特征，中间特征和 $\mathbf $ 再一起输入到额外的全连接层（128维）中预测颜色，如下图所示。

![](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=ZjM1ZmRlM2IxMTlmZGVhMjRlZWJlMGNjNjJhNTU4YTVfTGlLOThWaVhESU1lWERPZ3QwRXEyM0hDM0l6MFBRTmFfVG9rZW46SFpPdmJhWElwb2FGTkh4d0JuYWNtSm9kbkliXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

因此，**体素密度只和空间位置有关**，**而颜色则与空间位置以及观察的视角有关**。基于 view dependent 的颜色预测，能够**得到不同视角下不同的光照效果**。输入数据 $\mathbf $ 和 $\mathbf $ 都先经过了位置信息编码（Position Encoding），即$γ(∙$。

值得注意的是，上述网络中的权重/参数，在一个场景中，为所有的像素射线共享。

1. ## **体素渲染算法**
    

### **传统体渲染方法**

**体素密度** $σ(\mathbf x)$可以被理解为，一条穿过空间的射线，在 $\mathbf $ 处被一个无穷小的粒子终止的概率，这个概率是可微分的，可以将其近似理解为**该位置点的不****透明度**。

相机沿着特定方向进行观测，其观测射线上的点是连续的，则该相机成像平面上对应的像素颜色，可以理解为由对应射线经过的点的颜色积分得到。

将一条射线的原点标记为 $\mathbf $ ，射线方向（即相机视角）标记为 $\mathbf $，则可将射线表示为 $\mathbf r(t)=\mathbf o+t\mathbf $， $$ 的近端和远端边界分别为 $t_$ 和 $t_$。

![](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=MTdjZWE3ZTc2ZWExYjJlNTBlNWY0NGZiZjAxOWI4MjlfUUQ2aEtRNFNoVzZTN2ZRYUhCckJRRDFvMGFKR3AyNTJfVG9rZW46WTl6RWJDN3FKb1pMUFd4RjZWUGNUeks2blNkXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

可将这条射线的颜色，用积分的方式表示为：

$C(r)=\int_{t_n}^{t_f}T(t)\cdot \sigma(\mathbf{r}(t))\cdot \mathbf c(\mathbf{r}(t), \mathbf{d})\text{d}t$

其中，$T(t$表示的是射线从$t_$到$$这一段的累计透明度，即该射线从 $t_$ 到 $$ 都没有因击中任何粒子而被停下的概率，具体写作：

$T(t)=\exp\left(-\int_{t_n}^t \sigma(\mathbf r(s))\text{d} s\right)$

在连续的辐射场中，针对任意视角进行渲染，就需要对穿过目标虚拟相机的每个像素的射线，求取上述颜色积分，从而得到每个像素的颜色，渲染出该视角下的成像图片。

### **分段近似渲染方法**

用 NeRF 难以估计射线上的连续点，这就对其进行分段近似。作者提出了一种**分层抽样****（Stratified Sampling）的方法**：首先将射线需要积分的区域 $[t_n,t_f$ 均匀分为 $$ 份，再在每个小区域进行均匀随机采样。则以上预测颜色$C(\mathbf{r}$ 的积分，可以简化为求和的形式：

$\hat{C}(\mathbf{r})=\sum_{i=1}^N T_i\cdot(1-\exp(-\sigma_i\cdot\delta_i))\cdot\mathbf{c}_i$

其中，$δ(i$ 为两个近邻采样点之间的距离，此处 $T(t$ 改写作：

$T_i=\exp\left(-\sum_{j=1}^{i-1}\sigma_j\delta_j\right)$

这种**从所有****采样点****的** $(\mathbf c_i,σ_i$ **集合求和得到射线渲染颜色的方法也是可微分**的，并且可以简化为传统的透明度混合（Alpha Blending）算法，其中 Alpha 值 $α(i)=1−\exp(−σ(i)δ(i)$。

2. ## **位置信息编码（Positional encoding）**
    

由于神经网络难以学习到高频信息，直接将**位置**和**视角**作为网络的输入，渲染效果分辨率低；使用位置信息编码的方式**将输入先映射到高频**可以有效地解决这个问题。

$$ 用于将输入映射到高维空间中，论文中使用的是正余弦周期函数的形式：

$\gamma(p)=\big(\sin(2^0\cdot\pi\cdot p),\cos(2^0\cdot \pi\cdot p),\cdots,\sin(2^{L-1}\cdot\pi\cdot p),\cos(2^{L-1}\cdot \pi\cdot p)\big)$

位置和视角先进行归一化，到 $[-1,1$ 之间。对 3D 位置， $γ(\mathbf x)$的默认设置为 $L=1$；视角信息 $γ(\mathbf d$ 设置 L=4。

3. ## **多层级体素采样**
    

NeRF的渲染策略是对相机出发的每条射线都进行N个采样，将颜色加权的求和，得到该射线颜色。**由于大量的对渲染没有贡献的空的或被遮挡的区域仍要进行采样计算，这样占用了过多的计算量**。

作者设计了一种“coarse to fine”的多层级体素采样方法，同时优化coarse和fine两个网络：首先，使用分层采样的方法**均匀采集较为稀疏的****Nc****个点**，在这些采样点上计算coarse网络的渲染结果，改写前述的离散求和函数：

![](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTVhZjZjMmFlMmEzZTc4YWFhNjhiYjZkOGEwMjEzZWVfaHJiTnp3Rk91cUZCcHJoekNnRVVFOVFJb1pib3BoRUJfVG9rZW46SU0zaGJ3dWFSb1dkQWJ4bWpiOWM1M05DblhiXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

$w(i)=T(i)\cdot\bigg(1-\exp\big(−σ(i)δ(i)\big)\bigg)$，对 $w(i$ 进行归一化：

![](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=ZThlZWE0Y2U2YTVkZjk4NjRiMDZmMGY5YmJmMTM4YTdfaWx1bUxabW1rOTdWVFV5a2FVbHlqZE52M2RnWFY3SjdfVG9rZW46WGppR2I4czd6b1BBY2l4N0FMc2NlQWt5bnZjXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

归一化后的 $w(i)$ 可以看作是沿着射线方向的概率密度函数，如下左图所示。通过这个概率密度函数，我们可以粗略地得到射线方向上物体的分布情况：

![](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=MzJkZmY2NTBmMGQ0OGNkMzE2ZDhlYmVkMjIxYjMyODRfblhyNzJzblZqcVVVcmJvR0ltdjFBaEswaVhEZUpUN0xfVG9rZW46SEZiVmJmRXg4b3dER2J4WWJlcmNXWUpabnpiXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

随后，基于粗采样得到的概率密度函数，使用逆变换采样（inverse transform sampling）方法，再采样出Nf个密集点，如上右图。**这个方法可以从包含更多可见内容的区域中得到更多的****采样点**，然后在 $N_c+N_$ 的采样点集合上，计算 refine 网络的渲染结果。

针对不同的场景，需要进行独立训练一个 NeRF 。**训练损失**直接定义为：渲染结果的颜色，与图像真实像素值的 L2损失。**同时优化coarse和fine网络**。

# 二**、几何学原理**

神经辐射场采用简单的体渲染作为一种方法，通过利用可见性的概率概念来使得通过射线-三角形交叉点变得可微分。这是通过假设场景由一团发光粒子组成的来实现的，这些粒子的密度在空间中发生变化 （在基于物理的渲染的术语中，这将被描述为具有吸收和发射但没有散射的体积。在下文中，为了说明简单，并且不失一般性，我们**假设发射的光不作为观察方向的函数而改变**。

## **透射率**

设密度场为 $σ(x$，其中 $\mathbf x∈\mathbb R^$ 表示射线撞击粒子的微分似然度 （即在行进无穷小的距离时撞击粒子的概率）。我们重新参数化沿给定射线 $\mathbf r=(\mathbf o,\mathbf d$ 的密度作为标量函数 $σ(t$，因为沿射线的任何点 $\mathbf x$ 都可以写成$\mathbf r(t)=\mathbf o+t\mathbf $。密度与透射率函数 $T (t$ 密切相关，它表示光线在区间 $[0, t$ 上传播而没有击中任何粒子的概率。那么当走差分步 $d$ 时 没有撞击粒子的概率 $T (t+dt$ 等于 $T (t$，即射线到达 $$ 的可能性，乘以 $(1 − dt · σ(t)$，在该步骤中没有击中任何东西的概率：

![](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=YTJjOTM4ZmYyNjUyNjU4ZThiOWJhMDU4YTg1NjBlODRfNkU4dXZVWjRJbHpkT2dBemhsQjh3aVAyNTdKbmRzZm1fVG9rZW46RkxyQWIzaklXb3l0RVV4ZEdNM2MwSGpyblBjXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

这是一个经典的微分方程，可以如下求解:

![](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTkyNzQ3ODJkZjhjNWQ5NDZjZWZjOWNhMjY2ZTZlY2FfMWhSSWVDRzBzTklqcXR6bVpidGp6UHl0OVg4VnpRbDRfVG9rZW46UDFabWJDQTEyb3E0T254dmhOU2NxSFVxbk5iXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

其中我们将 $T (a → b$ 定义为光线从距离 $$到 $$而没有碰到粒子的概率，这与前面的符号 $T (t) = T (0 → t$。

## **概率解释**

请注意，我们也可以将函数 $1 − T (t$（通常称为“不透明度”）解释为累积分布函数（CDF），表示射线确实在某个时间之前到达距离 $$ 并击中粒子的概率。那么 $T (t) · σ(t$ 是相应的概率密度函数 (PDF)，给出了射线在距离$$ 处正好停止的可能性。

## **体渲染**

我们现在可以计算当光线从 $t=$ 传播到 $$ 时体积中的粒子发出的光的预期值，合成在背景颜色之上。由于在 $$ 处停止的概率密度为 $T (t) · σ(t$，因此预期颜色为

![|700x63](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=NGI1YzFkMmE5Y2I0ZjY5OWJmOGFmZmYzNzVhN2MwOGVfVGZEdVpmMWxJWFZPNmxuOWNRbG1LS0VVaE9tV3o4ZHRfVG9rZW46SkZ5Q2JaMDJLb1BZSjh4c09WbmNUWDEwbndjXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

其中 $\mathbf c_{bg$ 是根据残差透射率 $\mathcal T (D)$ 与前景场景合成的背景色。不失一般性，我们在下文中省略了背景术语。

## **同质媒介**

我们可以通过积分计算一些具有恒定颜色 $\mathbf c_$ 和密度 $σ_$ 在射线段 $[a, b$ 上的均匀体积介质的颜色：

![](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=YjgxYzc5YTAxYmFmM2Y3ZTZlNmU4ODA2MjUyMThlZDFfZGJ2STNKZ2p2TXkyeE84czZGNGdacW5xdzQ2Rk5RcUtfVG9rZW46Uk9HQWIwQ0prb1h0eE94Rm9NSGNDZVRBbkFjXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

## **透射率****是乘法**

请注意，透射率分解如下：

![](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=YzBhMzVjMDliNzMwNjVlNGE0YzVhMTA3YjFiZjA3MjZfVjlYbkFEdmJ5TVRwQjdyMlhjY2Zkb3k1RUtKUVVqV3hfVG9rZW46QjZGRWJYWlBZb0hFbDN4MzVaN2M4TWNVbkRjXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

这也来自 $ 的概率解释，因为射线没有击中 $[a, c$ 内的任何粒子的概率是它没有击中任何粒子的两个独立事件的概率的乘积在 $[a, b$ 或 $[b, c$ 内。

## 分段常数数据的透射率

给定一组区间

![](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=ZGVlMjA0MGI5NzQwMWEzNzEyNDFjMGQ2YzFkNmM3NjBfRE15UXlCdlY2WVJQWUxaVU11dmZzekRuWUkyOGtyYUlfVG9rZW46RDA0OGIxRmZjb2phbnh4TURvZWNwQ0c4bmxjXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

在第 $$ 段内具有恒定密度 $σ_n$，并且 $t_1=0$ 和 $δ_n=t_{n+1}−t_{n}$，透射率等于

![](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=YTkzNzc5Y2E4NmMyNDI5MGU2NzAyOWVkZTlmODA5MTdfVVpyWGU0M3k2Vm0yOHl0T3RoNHI0WlJkSmNicVdvaDBfVG9rZW46SXdvY2JDRlQ5b2pQZm14eXVTSWNmUkRObmdlXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

## **分段常数数据的体积渲染**

结合以上，我们可以通过具有分段常数颜色和密度的介质来评估体绘制积分：

![](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTI1OTZmNDFjZDk2MjQ3NDRmZDQyZDM0ZWY5NDE5NjJfaFdQS1dUdDF6QVNSTTVuN0h0VUFvR0JxcDRNVVRDWWlfVG9rZW46VEVUQ2JBb0dnb1NGaUZ4Wm5ab2N6bURHbkRiXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

这导致来自 NeRF [3, Eq.3] 的体绘制方程：

![](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=MWNjN2Y2MGFiODUzYTFkNTIxZmIwOTZjNDkwZTNlZThfdndlTHZINmg0N2xKRTRlb1lMVFUybUUxYzFCak9Jd3hfVG9rZW46VE9EMGI5bmJ6b2pyRXl4dlJQQWM0TVMxbnZjXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

![](https://kcnnokebdj6q.feishu.cn/space/api/box/stream/download/asynccode/?code=MWQ3NjA3MGYyNmFlMTQ0MTg1NzgwYzVkMDdhNmYzYWFfYmZqOFRhREU1WXpScXM2V200aFlOQ2FLdkpYUVZHV2dfVG9rZW46TUJJV2JLS095b0xZa2l4NE82NmM4dDRTbmplXzE3Mzk3NjIzMjM6MTczOTc2NTkyM19WNA)

结合恒定的每间隔密度，该恒等式产生与 (24) 相同的表达式.