# 2.24（周一）
## 1. 论文写作
### 1.1. QYGG 的写作部分

### 1.2. 我的论文写作
需要写投影部分的优化。
- 和 Deepseek 合作先进行了一次写作，大概确定了写作框架
- 先简单介绍现有方法的优化是怎么做的（也可能提到前面）
- 然后讲投影操作和光栅化操作，可以多写点公式吓唬人

### 1.3. 我的论文绘图


## 2. 论文阅读
今天读基于 GAN 的 SRGAN，看看效果如何：
- 感觉用 GAN 的框架还是比较 Make Sense 的（这个思路确实不错，分析也比较合理）
- 大体就是用两个网络做生成和判别；目前都是两个基于残差块的卷积网络
- GAN 的核心应该是损失函数的设计，要平衡两个网络的结果。现在的损失基本上是一部分质量损失一部分感知损失
- VGG 损失是用一个训练好的语义判别网络来做损失，相当于 VGG 某一层的输出和标准值的 MSE



# 2.25（周二）

# 2.26（周三）

# 2.27（周四）