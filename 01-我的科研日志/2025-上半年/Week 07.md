# 3.31（周一）
## 1. 论文绘图
- 第一张图：第二代技术的三个主要管线
- 第二张图：具有代表性的神经渲染技术对比
- 第三张图：Time Breakdown（学习 NeuRex）

# 4.1（周二）
## 1. 论文绘图
- 第四张图：算子的运算密度
- 第五张图：Hash 的仿真结果
## 2. 论文数据整理
- 干脆把每个代码都跑一遍，直接整理统计结果
- Dirty Work 交给 lcf 和 dqh 好了，让他俩和 qygg 竞争三四五作
## 近期工作计划整理
近期两大主要工作：TCSVT 论文和光栅器。
- TCSVT：分为几个主要部分：后端部分仿真、论文写作、补充数据和画图（Dirty Work）
- 光栅器：需要完善代码（修 Bug + 实现 Indirect VPLs）；需要若干人手去和浙大对接
	- 和浙大那边就说我们在跑 NVDLA 之类的网络加速器（所以最近没啥动静），现在重新捡了光栅器（可能需要恶补一下知识）
	- 需要做的事情：
		1. 确保 Shadow Map 没有问题
		2. 支持 .mtl 等文件格式（这样好看效果）
		3. 实现 Indirect VPLs 采样功能
因此最近（2 周内）主要分工如下：
- HYX：主要部分的写作和关键图的绘制，四处帮忙
- ZYZ：跑好后端数据，写实验部分
- CQY：读几篇论文，扒拉出数据来补充 Introduction 里的数据；也可以帮忙跑代码
- LCF：跑代码整理数据为主；完成之后需要帮忙画一部分图；跟着 DQH 跟进光栅器
- DQH：统筹 LCF 的跑代码工作和 CQY 的读数据工作；改光栅器；画图

# 4.2（周三）
## 1. Group Meeting
准备用实际案例来讲 Nsight Compute & Nsight System
——感觉没时间丢给下周吧
这周讲几个 AI 加速器的工作，带大家梳理一下
## 2. NaiLong Marp
核心包括：主题色，奶龙角标和背景图片

# 4.3（周四）
## 1. Group Meeting
讲了三篇论文
## 2. 准备后面找娄老师 Meeting
一个是问问和创业比赛相关的，然后帮 zf 问问
# 4.4（周五）
## 1. 毕业论文正稿
今天得把背景部分先写一写，先介绍 Background。
梳理一下逻辑：
- 第一章：绪论
	- 研究背景：主要讲的是三维重建技术的重要含义
	- 研究现状：
		- 软件：主要介绍一下传统重建技术和几代 NeRF 技术；
		- 硬件：主要介绍 GPU 和若干 NeRF 加速器即可
	- 论文主要研究内容：就是 TCSVT 的 Intro
	- 论文结构安排
- 第二章：背景介绍（应该是介绍 NeRF 和 NGP）
- 第三章：Ray Casting 系统设计
- 第四章：哈希设计
- 总结与展望就随便了

## 2. TCSVT 论文
主要梳理了第三节
明天根据原内容把第四节重新写一遍，然后梳理出来所有要跑的数据
# 4.5（周六）
## 1. 论文绘图
先把噪声的图做好。

## 2. 毕业论文写作
思考一下第二章怎么写——先把介绍糊上去，然后把第一章写完
等论文正稿写完之后再写具体的模块部分

# 4.6（周日）
## 1. TCSVT 论文推进
1. 跟进一下 DQHGG 那边的数据进度，Push 他们跑跑代码
2. 可能需要考虑一下 Density Grid 的展示方式（图片选择和配色）
3. 完善投影方法的图上的标记内容
4. 和 QYGG 交代哈希的图

## 2. 毕业论文写作
主要是背景部分的写作，把几代之前的 NeRF 的故事写一写

## 3. 创新创业大赛
### 计划
1. 看文件，了解比赛内容和细节
2. 打磨我们的故事，开始写故事文档
### 内容整理
- 赛道：信息学科专场
- 需求：主要是需要一份专利，关于我们的核心技术。
	- 我们的核心技术肯定要和光栅相关，所以最好在截止日期之前把混合光栅的 Demo 跑出来（不知道够不够）
## 4. 和娄老师 Meeting
首先：Meeting 之后主要整理出三件要做的事情：
1. 推进 TCSVT 论文（这个已经比较乐观了）
2. 和浙大的 Meeting：下周必须把 Meeting 办起来了
3. 创业：要和 DQH 一起把问题想明白。
### 4.1. 和浙大的 Meeting
下周直接问问他们之前的网络有没有优化，然后说我们想要了解超分的进展情况。
如果问我们就说我们在做光栅和看网络加速器（不问就不提）。
时间放在周四 OR 周五。
### 4.2. 创业
要思考的问题是：我们做的产品是什么？软件？硬件？
要论证的三个问题：
1. 我们的产品有广阔的市场（产品有价值）
2. 我们有能力把产品做出来
3. 我们有能力超过竞争对手
目前的产品只能说核心在光栅器，比较想做的方向还是协处理器（但是需要 Demo 来证明）。
感觉可以考虑用软件先搭建一版 Demo 来跑实时的混合渲染。