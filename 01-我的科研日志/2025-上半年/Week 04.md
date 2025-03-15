# 3.10（周一）
## 1. 混合渲染器
先思考一下混合渲染器的设计：
1. Step 1：确定相机空间（CModel 和光栅器是否是一组相机空间？需要做系统验证）
2. Step 2：封装 PCAccNR 的接口，把计算部分单独提取出来
3. Step 3：修改 PCAccNR 的代码，实现生成深度图和采样点（最近点和最远点）
4. Step 4：写好投影算法的流程细节，推敲逻辑
5. Step 5：设计好 Deferred Shading 的 Entry 内容，开始实现混合投影，先输出深度图
6. Step 6：实现混合着色功能

超凡说短期没事，先把两个代码给他看看。
## 2. 浮点数转换器
今天要整理浮点数相互转换的笔记（飞书？）
超凡说可以让王珺和 zyk 一起做这个项目，感觉是合理的：正好期中可以复习 CA
OK，交代好了
# 3.11（周二）
## 1. 混合渲染器
准备今天进行 Meeting，分配后续工作。

## 2. Neural SDF 相关
主要看 `instant-nsr-pl` 的 Model 里的 236 行附近：
这是 NeRF 的 Forward：
```
def forward_(self, rays):
	n_rays = rays.shape[0]
	rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

	def sigma_fn(t_starts, t_ends, ray_indices):
		ray_indices = ray_indices.long()
		t_origins = rays_o[ray_indices]
		t_dirs = rays_d[ray_indices]
		positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
		density, _ = self.geometry(positions)
		return density[...,None]

	def rgb_sigma_fn(t_starts, t_ends, ray_indices):
		ray_indices = ray_indices.long()
		t_origins = rays_o[ray_indices]
		t_dirs = rays_d[ray_indices]
		positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
		density, feature = self.geometry(positions)
		rgb = self.texture(feature, t_dirs)
		return rgb, density[...,None]

	with torch.no_grad():
		ray_indices, t_starts, t_ends = ray_marching(
			rays_o, rays_d,
			scene_aabb=None if self.config.learned_background else self.scene_aabb,
			grid=self.occupancy_grid if self.config.grid_prune else None,
			sigma_fn=sigma_fn,
			near_plane=self.near_plane, far_plane=self.far_plane,
			render_step_size=self.render_step_size,
			stratified=self.randomized,
			cone_angle=self.cone_angle,
			alpha_thre=0.0
		)  
	ray_indices = ray_indices.long()
	t_origins = rays_o[ray_indices]
	t_dirs = rays_d[ray_indices]
	midpoints = (t_starts + t_ends) / 2.
	positions = t_origins + t_dirs * midpoints  
	intervals = t_ends - t_starts
	density, feature = self.geometry(positions)
	rgb = self.texture(feature, t_dirs)

	weights = render_weight_from_density(t_starts, t_ends, density[...,None], ray_indices=ray_indices, n_rays=n_rays)
	opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
	depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
	comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
	comp_rgb = comp_rgb + self.background_color * (1.0 - opacity)      
....
```

这是 NeuS 的：
```
def forward_(self, rays):
    n_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

    with torch.no_grad():
        ray_indices, t_starts, t_ends = ray_marching(
            rays_o, rays_d,
            scene_aabb=self.scene_aabb,
            grid=self.occupancy_grid if self.config.grid_prune else None,
            alpha_fn=None,
            near_plane=None, far_plane=None,
            render_step_size=self.render_step_size,
            stratified=self.randomized,
            cone_angle=0.0,
            alpha_thre=0.0
        )
    
    ray_indices = ray_indices.long()
    t_origins = rays_o[ray_indices]
    t_dirs = rays_d[ray_indices]
    midpoints = (t_starts + t_ends) / 2.
    positions = t_origins + t_dirs * midpoints
    dists = t_ends - t_starts

    if self.config.geometry.grad_type == 'finite_difference':
        sdf, sdf_grad, feature, sdf_laplace = self.geometry(positions, with_grad=True, with_feature=True, with_laplace=True)
    else:
        sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True)
    normal = F.normalize(sdf_grad, p=2, dim=-1)
    alpha = self.get_alpha(sdf, normal, t_dirs, dists)[...,None]
    rgb = self.texture(feature, t_dirs, normal)

    weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
    opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
    depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
    comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)

    comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
    comp_normal = F.normalize(comp_normal, p=2, dim=-1)
```

根据我的 NGP_Trainer 结构，应该也是主要改这一段即可。
SDF 模型特征：
1. 输出 SDF，SDF Grad（Normal）和 Feature
2. Feature 经过 Texture 得到 RGB
接下来计划读 NeuS 的论文来写代码。
## 3. ASIC_Sim 浮点数开发
和 wj & zyk Meeting，交代任务：
1. 先期的两个工作：修改定点数赋值部分的截断舍入，实现浮点数之间的截断和舍入
2. 要求先看一遍之前的文档，可以适当对之前的文档内容进行补充测试用例。
3. 基本上就是各种构造函数的实现。
值得注意的是应该强调一下之前有很多问题，建议边写边改，适当把重复功能封装成函数，在文档里写清楚。读文档和代码的过程就可以整理笔记放在旁边。
两个人建议一起写一起测。
# 3.12（周三）
## 1. 混合渲染器
- 今天建了仓库和文档，准备先自己把 NGP 的功能拼起来。
- 要求大家都先在自己的 Branch 上开发，然后再 Merge 到 Main 里。
- 计划是：先写好 CPU 上的版本，然后如果有必要再单独开发一个异构计算的版本。
# 2. 整体计划
和 szhgg 聊了聊，大概明确了后面的计划：
1. 把混合渲染器写出来，然后 Demo 到一个能用的程度
2. 可以尝试搞一篇论文
3. 接下来，搞一波数据采集和训练优化，把 RGBD 相机和图片分割等功能结合起来，做出一套好的数据采集管线，和我们的设备结合起来
4. 这个阶段可以开始拉投资准备搞公司了
5. 这个阶段之后可以考虑继续做加速器，可以把浙大的项目结合进来（相当于单独写一个网络加速器的版本）
6. 接下来再怎么走就看具体的项目优化了。
# 3.13（周四）
## 1. 这周雍治哥要去听讲座，正好大家都比较忙，Reading Group 暂停一次
## 2. Mix Render
### 2.1. 相机空间验证
【看来搞不出来】
### 2.2. 代码合并测试
出现了一些逆天的报错。感觉没懂是什么原因，难道要单步断点调试吗（）
我操我真是无敌了，先把原本的 VecXf 改成 Vec 3f，里面又重新用 VecXf 接受，巧妙的制造了一个能跑的代码错误，真是傻逼到家了我吃
妈的这也行我真是无敌了，愺

### 2.3. 和 szhgg 的沟通
- 很多动态的 NGP 工作实际上是编码好 NGP 的 Weight，然后按次序做渲染，几乎不用修改 NGP 的结构
- 可以研究一手，如果好的话可以把这个功能 Merge 进去

## 3. NeuS 论文阅读
【看来没时间了】

## 4. 和岱悟的 Meeting
### 4.1. Background
- 岱悟现在对接了一个多人实景 VR 的项目，现在是用两台 NVIDIA AGX，但功耗比较大
- 希望我们可以推出一版低功耗的 Gaussian 芯片
- 估计主要是伟航他们沟通，我们一起协作一下
### 4.2. Meeting 内容
输入数据
- 组里的 Gaussian 刚刚完成软件的 Demo，还没有搞硬件 Demo。
- 伟航他们改了一些 Gaussian 的训练过程（Sort-Free Gaussian）。
- 提供的数据有输入和位姿等等。
他们现在做的事情：
[2024-2025年LBE VR大空间市场观察（综合篇）](https://zhuanlan.zhihu.com/p/20191380099)
- 已经做了环境场景的三维扫描，可以转换成任意格式
- VR LBE（大空间多人互动技术）：一个 3D 场景，大约 300 平米，墙上有花纹（用于 SLAM 定位），若干用户戴着 VR 在场景里走，场景里会有交互——类似一个 3D 电影院
- 目前上海已经有几百家类似的公司在做
- 希望能把 Gaussian 放在 VR 上——需要低功耗，45 分钟的 VR 播放不用换电池
- 背包式的运算主机有三万左右，并且重量达到 10 kg 以上（笔记本+电池，非常暴力）。
- 串流？——国内合法的 WIFI 只能支持 8 个视频通道的播放
- 三种场景：
	- 空中航拍的俯视场景：范围大，但不需要很精细
	- CITY Work：远景和近景的混合
	- 纯室内的场景
- 可以接受一个场景一个 GB 级别的内容
- LOD：这个可以引到我们的 idea 上

- 提供了一个 Relighting 的 Idea：用网络做打光计算，但是只能处理直接光照？

建议：尽量提高兼容性，无论模型是什么都能做 Baking（是不是可以开发一个从现成模型到我们模型的蒸馏转换器——做一波虚拟观测）

# 3.14（周五）
## 1. NeuS 论文阅读
简单阅读了一下，好多公式 QAQ
感觉要把式子推明白，然后照着开始改（）
还好这个结构应该和我的 Trainer 差不多？试试看吧
## 2. MixRender
### 2.1. 相机空间验证
Pushing——

### 2.2. NGP 代码封装
简单测了测应该是没问题了，之前的 Bug 也太蠢了（把 VecXf 改错成 Vec 3f 之后又用另一个 VecXf 接收了导致莫名访问非法内存）


# 3.15（周六）
## 1. 尝试 Obsidian + Zotero

## 2. 整理沈哲灏的论文
### 2.1. HiFi4G（主要是压缩方案）
HiFi4G 的压缩方案是其实现高效存储和传输的关键部分。该方案通过残差补偿、量化和熵编码三个步骤，显著减少了 4D 高斯数据的存储需求，同时保持了高质量的渲染效果。以下是对该压缩方案的详细介绍：
#### 2.1.1. 残差补偿（Residual Compensation）
在压缩过程中，HiFi4G 采用了残差补偿策略，以减少数据的动态范围，从而提高压缩效率。具体步骤如下：
- **保留关键帧属性**：对于每个关键帧，保留其完整的高斯属性（包括位置、旋转、球谐系数、不透明度和缩放系数）。
- **计算残差**：对于非关键帧，计算其高斯属性与关键帧之间的残差。对于外观属性（球谐系数、不透明度和缩放系数），由于时间正则化的作用，这些属性在帧间的变化较小，因此可以直接通过减法计算残差。对于位置和旋转属性，由于帧间可能存在较大的运动，HiFi4G 使用运动补偿技术来计算残差。具体来说，首先将关键帧的高斯属性通过 ED 图进行插值，得到当前帧的估计值，然后计算当前帧的实际属性与估计值之间的残差。
通过残差补偿，HiFi4G 显著减少了非关键帧属性的动态范围，使得后续的量化步骤更加高效。
#### 2.1.2. 量化（Quantization）
量化是将连续的高斯属性值转换为离散值的过程，以减少数据的存储需求。HiFi4G 的量化策略如下：
- **确定量化位数**：根据属性的动态范围和精度需求，HiFi4G 为不同的属性分配不同的量化位数。例如，对于关键帧的外观属性，使用 9 位量化；对于非关键帧的运动属性，使用 11 位量化；对于非关键帧的外观属性，使用 7 位量化。
- **缩放和舍入**：在量化过程中，首先将属性值缩放到指定的量化范围内，然后进行舍入操作，将其转换为离散的整数值。
通过量化，HiFi4G 进一步减少了数据的存储需求，同时保持了较高的渲染质量。
#### 2.1.3. 熵编码（Entropy Encoding）
熵编码是利用数据的统计特性进行压缩的过程。HiFi4G 采用了 Ranged Arithmetic Numerical System (RANS) 算法进行熵编码，具体步骤如下：
- **构建频率分布**：首先，计算每个量化后的属性值的频率分布。由于残差补偿和量化的作用，这些属性值的分布通常集中在零附近，具有较高的偏斜性。
- **RANS 编码**：利用 RANS 算法，HiFi4G 对量化后的属性值进行编码。RANS 算法通过将每个属性值与编码器的当前状态进行处理，生成一个整数流，表示压缩后的数据序列。
通过熵编码，HiFi4G 进一步压缩了数据，显著减少了存储需求。
#### 2.1.4. 压缩效果
HiFi4G 的压缩方案实现了约 25 倍的压缩率，每帧的存储需求降低至不到 2MB。具体效果如下：
- **存储需求**：在压缩前，每帧的 4D 高斯数据需要约 48.24MB 的存储空间。通过高比特量化（0 位用于运动，9 位用于外观），存储需求降低至 7.41MB；通过低比特量化（11 位用于运动，7 位用于外观），存储需求进一步降低至 3.67MB。最终，结合残差补偿和低比特量化，HiFi4G 的每帧存储需求降低至不到 2MB。
- **渲染质量**：尽管压缩显著减少了存储需求，HiFi4G 仍能保持高质量的渲染效果。通过残差补偿和自适应量化，HiFi4G 在压缩后几乎没有损失渲染质量。
#### 2.1.5. 总结
HiFi4G 的压缩方案通过残差补偿、量化和熵编码三个步骤，显著减少了 4D 高斯数据的存储需求，同时保持了高质量的渲染效果。该方案使得 HiFi4G 能够在各种平台上（如 VR/AR 头显）实现沉浸式的高保真人类表演渲染。
### 2.2. Instant NVR（主要是前端的输入处理）
#### 2.2.1. 该篇文章所研究的任务介绍
本文提出了一种名为 **Instant-NVR** 的神经方法，用于从单目 RGBD 相机中实现即时的体积人体-物体跟踪和渲染。该方法通过多线程的跟踪-渲染机制，将传统的非刚性跟踪与最新的即时辐射场技术相结合。具体任务包括：
- **人体-物体交互的 4D 建模**：通过单目 RGBD 输入，实时生成人体和物体的辐射场，实现高质量的自由视角合成。
- **即时渲染**：通过高效的在线重建和渲染策略，能够在复杂的交互场景中实现实时的高质量渲染。
#### 2.2.2. 该篇文章的研究动机
**<font color='red'><b>所遇到的现有方法的主要问题</b></font>**：
- 传统的多视角系统需要密集的摄像头或复杂的校准，不适合日常使用。
- 现有的单目方法虽然更加实用，但生成的几何分辨率有限，无法生成逼真的外观结果。
- 现有的动态 NeRF 方法依赖于多视角输入，并且需要耗时的逐场景训练，无法满足即时应用的需求。
**<font color='blue'><b>我们提出的方法概述</b></font>**：
- 提出了一种基于单目 RGBD 相机的即时神经体积渲染系统，能够实时生成人体和物体的辐射场。
- 通过多线程的跟踪-渲染机制，结合传统的非刚性跟踪与即时辐射场技术，实现了高质量的自由视角合成。
#### 2.2.3. 该篇文章所提出的主要方法
本文的主要方法包括以下几个关键模块：
1. **跟踪前端**：
   - 使用现成的即时分割技术区分输入 RGBD 流中的人体和物体。
   - 采用嵌入式变形图（ED）和 SMPL 模型来跟踪人体的非刚性运动，使用 ICP 算法跟踪物体的刚性运动。
2. **渲染后端**：
   - 采用分离的即时神经表示，分别重建动态人体和静态物体的辐射场。
   - 引入混合变形模块，利用非刚性运动先验来优化初始变形。
   - 通过基于关键帧的训练过程，逐步优化辐射场，实现即时渲染。
3. **在线关键帧选择与渲染感知的优化策略**：
   - 提出了一种在线关键帧选择方案，考虑捕获视图和人体姿态的多样性，避免捕获区域分布不均。
   - 通过渲染感知的优化策略，进一步改善渲染外观细节。
#### 2.2.4. 该篇文章的实验效果
本文通过大量实验验证了 **Instant-NVR** 的有效性和效率，具体实验结果如下：
1. **与现有方法的对比**：
   - 与基于融合的方法（如 RobustFusion 和 NeuralHOFusion）和基于 NeRF 的方法（如 NeuralBody 和 HumanNerf）相比，**Instant-NVR** 在渲染质量和效率上均表现出色。
   - 定量结果表明，**Instant-NVR** 在 PSNR 和 SSIM 指标上均优于对比方法，且渲染时间显著缩短。
2. **在线人体和物体渲染的评估**：
   - 在线关键帧选择策略显著提高了渲染质量，避免了固定时间间隔选择关键帧导致的噪声问题。
   - 通过渲染感知的优化策略，进一步提升了渲染的逼真度。
3. **运行时间评估**：
   - 跟踪前端的刚性跟踪和非刚性跟踪分别耗时 40ms 和 62ms，变形网络耗时 5ms。
   - 渲染后端的训练时间通过快速搜索策略从 205.53ms 缩短到 17.95ms，渲染过程耗时 15.38ms。
4. **混合变形模块的评估**：
   - 混合变形模块通过显式变形块和隐式变形网络的结合，显著提高了渲染结果的准确性和纹理质量。
#### 2.2.5. 结论
本文提出了一种基于单目 RGBD 相机的即时神经体积渲染系统 **Instant-NVR**，通过结合传统的非刚性跟踪与即时辐射场技术，实现了高质量的自由视角合成。实验结果表明，该方法在复杂的人体-物体交互场景中能够实时生成逼真的渲染结果，具有广泛的应用前景。

## 3. 娄老师给的初创公司（Bolt Graphics 做的 Zeus GPU）
这家公司主要聚焦在高性能计算和实时光线追踪：
### 3.1. 架构设计 ：
>  Zeus GPU的设计理念围绕“数据本地化优先”，通过**RISC-V标量核心+向量加速单元+专用协处理器**的三级异构架构，实现了计算资源的高效调度。
>  其核心创新包括基于RVA23扩展和RVV 1.0向量指令集的RISC-V指令集深度定制，结合模块化设计降低了开发成本，并兼容开源生态如LLVM编译器。
>  在硬件层面，Zeus配备128MB-512MB片上缓存，搭配LPDDR5X（带宽273GB/s）与DDR5（最高1.45TB/s）的混合内存方案，显著提升了高带宽场景下的吞吐量。
>  此外，其可扩展集群架构支持通过400GbE/800GbE网络或PCIe 5.0构建分布式计算集群，单节点内存容量最高达2.3TB，突破了传统GPU显存容量限制。
### 3.2. 实时光线追踪
>  在实时光线追踪领域，Zeus通过**硬件级光线追踪流水线**与**Glowstick渲染引擎**的协同优化，实现了行业首个“所见即所得”的实时路径追踪工作流。
>  在4K/30fps实时光追场景下，Zeus 2c仅需28块GPU即可完成传统方案（如RTX 5090）需280块GPU的任务，能效比提升7倍。
>  Glowstick支持统计采样、渐进式渲染和无偏蒙特卡洛积分，结合OpenUSD、MaterialX等开放标准，实现了物理级真实感渲染，并通过嵌入Blender、Houdini等DCC软件，解决了传统渲染器预览与最终结果不一致的痛点。
### 3.3. 高性能计算
>  在高性能计算领域，Zeus GPU通过双精度浮点（FP64）原生支持与大规模并行优化，在电磁仿真、蛋白质折叠等对精度和内存敏感的场景中展现出颠覆性优势。
>  例如，在电磁仿真中，Zeus 4c对38亿网格单元的处理速度是NVIDIA B200的40倍，能效比高达300倍。
>  其内存扩展性尤为突出，通过DDR5内存池扩展技术，单集群可支持180TB内存，使完整光子芯片或飞行器雷达的全尺寸仿真成为可能。
### 启发和思考
- 关于“兼容性”：我们自己定义的管线迟早要融入到现有的软件生态里，因此也不能完全闭门造车，至少要学会使用现有的生态；
	- 可以考虑和第三方合作开发和现有软件生态的兼容体系
	- 用光栅的问题就是怎么打出效果牌的问题，毕竟只用光栅
- 关于“总体架构”：后续如果不是和格兰菲合作的话确实要思考一下主题的控制系统是不是要一个 RISCV IP，感觉把这些东西用起来并不是简单的事情
- 总体而言，还是要进一步思考我们能做的事情，我觉得核心应该是尽量把前沿的一些技术融合成一个独具特色和优点的 Demo

# 3.16（周日）
