# 我草

NGP 的 样例图片并非直接读取的 `r_0.png`
或者说，读取之后要把第四个通道乘到前面三个通道里，这NM是个半透明图片
我的评价是：6

> [!important] 重要：精度问题

Python 和 CUDA 的精度有所区别（体现在小数点后四位，可能是浮点数表示的原因）

因此，**CUDA 训练 Python 推理 会导致 PSNR 有 2dB 左右的下降**

- 考虑一下，似乎有可能是 float 16 的原因，不太确定，再看看

# 找到问题了

是相机方案的问题

钧然的代码里的方案是正确的，两个方法在 direction 上会有误差

然后对像素位置做一个调整即可

# Introduction

Python + tiny-cuda-nn 实现 Instant NGP 的推理部分

# Modules 实现细节

## 超参数部分

原版 NGP 还有 SPP 参数，值得注意

## Camera 部分

Camera 部分主要参考的是 `nerf-pytorch` 。但是，NGP 在细节上做了很多修改。

其修改核心有两个部分：

1. 坐标轴的轮换。
    
    ```Python
    self.rays_o = self.rays_o[..., [1,2,0]]
    self.rays_d = self.rays_d[..., [1,2,0]]
    # Normalize rays_d
    self.rays_d = self.rays_d / np.linalg.norm(self.rays_d, axis=-1, keepdims=True)
    ```
    
2. 在生成采样点的时候的 scale。
    - 对于 Density Grid，其 Scale 为 `position[0] * scales[scene] * 2 + 0.5)`
    - 对于 Hash Grid，其 Scale 为 `position * scales[scene] + 0.5`
    - Scale 数值基本为 `0.33` ，但 比较奇怪的 Drums 是 `0.3`

  

## Ray Marching

[[Ray Marching 部分]]

[[NGP 附录 C]]

## Encoder and Networks

- 参数的加载：
    - 注意 Create Hash Grid 的时候要指定 `"per_level_scale": 1.38191288`
        - 默认值为 2
        - 这个值对应的是 Finest Resolution 1024
    - NGP 中 `params_binary` 的存储顺序：
        - `hash_network` - `rgb_network` - `hash_grid`

## Rendering

核心是渲染公式所对应的代码。

```Python
Color, Opacity <- (0,0,0), 0
t <- Init t value
while(t valid)
	alpha_raw, rgb_raw <- Computation
	T = 1 - Opacity
	alpha = 1 - exp(-exp(alpha_raw) * STEP_SIZE)
	weight = T * alpha
	rgb = sigmoid(rgb_raw) * weight
	Color += rgb
	t <- next_t
```

### 朴素版本的主要部分代码

```Python
for i in trange(camera.resolution[0]):
        for j in range(0, camera.resolution[1], BATCH_SIZE):
            ray_o = torch.from_numpy(camera.rays_o[i, j: j + BATCH_SIZE]).to(DEVICE)
            ray_d = torch.from_numpy(camera.rays_d[i, j: j + BATCH_SIZE]).to(DEVICE)

            t = 0.05
            # Skip the empty ray
            if isinstance(t, str):
                continue
            color = torch.zeros(3, dtype = torch.float32, device = DEVICE)
            opacity = torch.zeros(1, dtype = torch.float32, device = DEVICE)
            while (t <= 6.):
                position = ray_o + t * ray_d
                if(grid.intersect(position[0] * scales[scene] * 2 + 0.5)):
                    # Case of we need run
                    pos_hash = position * scales[scene] + 0.5
                    hash_feature = hashenc(pos_hash)
                    sh_feature = shenc((ray_d + 1)/2)
                    feature = torch.concat([hash_feature, sh_feature], dim = -1)

                    alpha_raw = hash_feature[:, 0]
                    rgb_raw = rgb_net(feature)
                    T = 1 - opacity
                    alpha = 1 - torch.exp(-torch.exp(alpha_raw) * STEP_LENGTH)
                    weight = T * alpha
                    rgb = torch.sigmoid(rgb_raw) * weight
                    opacity += weight
                    color += rgb[0]
                    
                t += STEP_LENGTH
            camera.image[i, j: j + BATCH_SIZE] = color.cpu().detach().numpy()
```

### `grid.intersect` 的优化

```Python
def intersect(self, points):
        idxs = torch.sum(
            torch.floor(
                (points - self.aabb[0]) / (self.aabb[1] - self.aabb[0]) * 128) 
                * 
                torch.tensor([128 * 128, 128, 1], device = points.device
            ),dim = -1, dtype = torch.int32)
        
        # Noticed that: a point out of aabb may map to a index in [0, 128**3)
        # So we must check by this
        masks_raw = ((points >= self.aabb[0]) & (points <= self.aabb[1]))
        masks = torch.all(masks_raw, dim = 1).type(torch.int32)
        valid_idxs = idxs * masks
        return self.grid[valid_idxs]
```

### 多像素并行的实现

主要是展开了图片的存储，以及修改了 mask 的实现

```Python
pixels = camera.resolution[0] * camera.resolution[1]
    for pixel_index in trange(0, pixels, BATCH_SIZE):
        BATCH = min(BATCH_SIZE, pixels - pixel_index)
        ray_o = torch.from_numpy(camera.rays_o[pixel_index: pixel_index + BATCH]).to(DEVICE)
        ray_d = torch.from_numpy(camera.rays_d[pixel_index: pixel_index + BATCH]).to(DEVICE)

        """
        Naive Ray Marching
        """ 
        t = NEAR_DISTANCE
        color = torch.zeros([BATCH, 3], dtype = torch.float32, device = DEVICE)
        opacity = torch.zeros([BATCH, 1], dtype = torch.float32, device = DEVICE)
        while (t <= FAR_DISTANCE):
            position = ray_o + t * ray_d
            \#if(grid.intersect(position[0] * 2 + 0.5)):
            masks = grid.intersect(position * 2 + 0.5).reshape((-1, 1))
            # Case of we need run
            pos_hash = position + 0.5
            hash_feature = hashenc(pos_hash)
            sh_feature = shenc((ray_d + 1)/2)
            feature = torch.concat([hash_feature, sh_feature], dim = -1)

            alpha_raw = hash_feature[:, 0:1]
            rgb_raw = rgb_net(feature)
            T = 1 - opacity
            alpha = 1 - torch.exp(-torch.exp(alpha_raw) * STEP_LENGTH)
            
            weight = T * alpha * masks
            rgb = torch.sigmoid(rgb_raw) * weight
            
            opacity += weight
            color += rgb
                
            t += STEP_LENGTH

        camera.image[pixel_index: pixel_index + BATCH_SIZE] = color.cpu().detach().numpy()
```

这个版本比单光线的并行快了一倍多点，但影响不大，都是一个量级的

### 单光线上并行渲染的代码

这个代码的核心是**把对 `opacity` 的累加变成用 `torch.cumprod` 的并行操作**

我自己的轮子 `cumprod_exclusive` 需要的输入是**一维向量**，但给进去的 `alpha` 一般是 `(counts, 1)` 形状的向量，从而导致累加出来的结果是错误的。

因此在这里重新实现了对于 NGP 的 `cumprod`

```Python
def cumprod_exclusive_ngp(tensor: torch.Tensor) -> torch.Tensor:
    ### Support for my implementation
    in_shape = (tensor.shape[0],)
    out_shape = (tensor.shape[0], 1)
    tensor = tensor.reshape(in_shape)
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod.reshape(out_shape)
```

并行部分的结构就是：

```Python
ts = torch.reshape(torch.linspace(NEAR_DISTANCE, FAR_DISTANCE, NERF_STEPS, device = DEVICE), (-1, 1))
        pts = ray_o + ts * ray_d
        occupancy = grid.intersect(pts * 2 + 0.5)
        if(torch.sum(occupancy) == 0):
            continue
        color = torch.zeros([BATCH, 3], dtype = torch.float32, device = DEVICE)
        opacity = torch.zeros([BATCH, 1], dtype = torch.float32, device = DEVICE)
        pts_truth = pts[torch.where(occupancy)]

        hash_features = hashenc(pts_truth + 0.5)
        sh_features = torch.tile(shenc((ray_d+1) / 2), (hash_features.shape[0], 1))
        features = torch.concat([hash_features, sh_features], dim = -1)

        alphas_raw = hash_features[..., 0:1]
        rgbs_raw = rgb_net(features)
        alphas = (1. - torch.exp(-torch.exp(alphas_raw) * STEP_LENGTH))
        
        # Cumprod_exclusive need a 1D array input!
        weights = alphas * cumprod_exclusive_ngp((1. - alphas)) 
        colors = weights * torch.sigmoid(rgbs_raw)
        para_color = torch.sum(colors, dim = -2)
        camera.image[pixel_index] = para_color.cpu().detach().numpy()
```

# 工作进度与计划

## 7.10

- 了解 TCNN 的调用方式

## 7.15

- 实现了最简单的 Pipeline
- 调用参数数量和 NGP 不同

## 7.17

- 找到了 TCNN 参数量的问题

## 7.18

- 正确 Load 了 TCNN 的参数
- 相机位姿不对

## 7.19

- 调整好了相机位姿的问题，现在和 Occupancy Grid 的相交图像基本正确
- 但是 采样点的位置不变

## 7.20

- 开始做采样点位置的测试
    
    [[ray test]]
    
    - 确定 Load 进来的 Hash Table 和 Network 都是没有问题的，输入正确则输出正确
    - 但是意识到：
        - Camera 生成的采样点也得调整
        - 得完整的写一版 Ray Marching
        - 所有的 输出 都要过 Sigmoid 或者 ReLU，Rendering 公式可能也要调整
- Camera 的光线生成问题基本上也解决了，现在对应点的误差小于 `5e-4`

## 7.21

- 开始实现 Ray Marching
- 折戟沉沙.jpg
- 不过传入的 SH 方向好像确实需要变成 `(ray_d+1)/2`

## 7.22

- 探究 Ray Marching 的 奥秘
- 感觉不是这个的问题

## 7.23

- 核心问题是 两个位置似乎不对应
- 得把传入哈希表的scale一下 才能对应
- 已经能跑出看起来正确的图了

## 7.24

- 跑出来了！
- 整理代码并开源

## 7.31

- 想到可以多像素并行实现

## 8.1

- 实现了多像素并行
- 40s 一张 800x800 的图
- 但是质量好像低点，不知道是不是没有超采样？

## 8.3

- 思考一下，在做新实验的时候还是得逐光线并行
- 研究一下为什么逐光线并行有问题

## 8.4

- 找到逐光线并行的错误了！
- 完成了逐光线并行

## 8.5

- 写了一版 Naive 的新方法，但是质量有问题

## 8.7

- 新方法为什么会有问题呢？按理来说应该只是把采样范围变大了啊
- 原来是范围设定有问题，并且步长要匹配

## 8.8

- 试着用超采样处理复现方法 PSNR 低了两个 dB 的问题
- 调好了新方法，并且发现自己之前的想法太麻烦了，优化