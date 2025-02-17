# 函数细节

## Part 01: `get_init_t_value`

目的：给定 `ray_o, ray_d` 和 `aabb` ，获取这跟光线和 `aabb` 相交的 `t` 值

- 本质上是与 aabb 的相交判断

## Part 02: `get_next_voxel`

给定一组 `ray_o, t, ray_d` ，获得一个 `dt` ，对应 `ray_o + (t+dt)*ray_d` 为下一个有效点

- 首先，获取这个点所对应的 index
    - `point_idx = np.floor((point - self.aabb[0]) / 2 * 128)`
    - `scale` 是 `aabb` 的大小，在这个项目里应该为 2
- 然后，判断在三个方向上走到“下一个格子”所需要的距离
    - 下一个格子对应的位置是？——idx 在三个位置上 +1 所需要的距离
    - 得到新的 idx 之后，一样先算出格子左下角的位置，然后通过 方向的正负决定要不要+1
    - 对应的方向除一下即可
- 最终的 `t` 就是这三个距离的最小值

## Part 03: `steping`
循环调用，直至离开 density grid

```Python
def get_init_t_value(aabb, ray_o, ray_d):
    """
    Given a ray with ray_o, ray_d and an axis-aligned bounding box
    return the initial t value.
    AABB is a 2x3 array, each row is the min and max of the AABB
    """
    ray_o, ray_d = torch.reshape(ray_o, (3,)), torch.reshape(ray_d, (3,))
    # Noticed that for each axis, tmin and tmax is not fixed order
    ts = (aabb - ray_o )/ ray_d
    
    tmins, tmaxs = torch.min(ts, dim = 0), torch.max(ts, dim = 0)
    t_enter = torch.max(tmins.values)
    t_exit = torch.min(tmaxs.values)
    
    if (t_enter < t_exit) and (t_exit >= 0):
        return t_enter + 1e-4
    return "No Intersection"
def get_next_voxel(position, direction):
    index = (position + 0.5) * 64
    next_index = torch.floor(index + 0.5 + 0.5 * torch.sign(direction))
    
    if (next_index <= 0).any() or (next_index >= 128).any():
        return "Termination"
    delta_distance = next_index - index
    
    dts = delta_distance / 64. / direction
    dt = torch.min(dts + 1e-5)
    if dt <= 0:
        return 0.0
    return dt
```