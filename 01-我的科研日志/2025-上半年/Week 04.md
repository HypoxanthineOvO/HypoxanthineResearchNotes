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

## 2. MixRender
### 2.1. 相机空间验证

### 2.2. NGP 代码封装

## 2.3. 整理沈哲灏的论文
# 3.15（周六）
## 1. 尝试 Obsidian + Zotero

# 3.16（周日）
