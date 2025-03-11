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
# 3.13（周四）
# 3.14（周五）
# 3.15（周六）
# 3.16（周日）
