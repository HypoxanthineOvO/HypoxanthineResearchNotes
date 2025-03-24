# 3.24（周一）
今天要 Push 娄老师。
## 1. TCSVT 的论文写作（Meeting）
1. Review 的时候如果可以的话，Cue 一下 Gaussian
2. Profiling 阶段：尽量避免直接提及 Instant NGP，而是先提及 Neural Rendering（分析出其通用问题），再用 Instant NGP 作为经典的代表（工程上比较完善，效果比较好）
	- 可以考虑一下找更多基于 Hash 的方法
	- 得先分析为什么 Hash 类的方法是 Neural 方法里做的 TradeOff 做的最好的
		- 三平面的计算量（？）
		- 树状结构的存储量很大
	- 要强调我们有泛化性，选 NGP 是因为它是以后的 Baseline
3. Review 的逻辑：NeRF-NGP-Gaussian。NeRF 之所以效率不够高的原因，但 Ray 的处理还是慢慢的（不像 Gaussian 的投影效率高）。我们要解决的是 NeRF 的效率问题。我们找到了两个核心的问题（找点和计算的访存），分别解决了它。