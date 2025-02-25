# Introduction
Nsight Compute 是 NVIDIA 的一款对 GPU 程序进行 Profiling 的工具，主要用于对 CUDA Kernel（Kernel Function，核函数，即运行在 GPU 上的 CUDA 函数）的性能分析，可以输出运行时间、计算量、访存量、Cache 命中率等信息。

# Usage
我们以一个常见的例子来展现 Nsight Compute 的使用：
```bash
{ncu_path} --csv --page=details --details-all --export {report_path} --force-overwrite --kernel-name {kernel} --set full --launch-skip 16 --launch-count 16 --kill on ./instant-ngp --snapshot={snapshot_path}/{scene}.msgpack --width=800 --height=800
```
Profiling 相关设置：
- `--kernel-name {KERNEL-NAME}` ：要 Profiling 的 Kernel Name
- `--launch-skip {LAUNCH-SKIP}`：跳过前多少次 Kernel Launch
- `--launch-count {LAUNCH-COUNT}`：记录多少次 Launch
输出相关设置：
- `--csv --page=details --details-all`：用 csv 的格式输出数据
- `--export {REPORT-PATH} --force-overwrite` ：输出 Report
- `--set full` ：输出所有数据

# Nsight Compute Metrics 简介
## I. 频率与吞吐量
1. **DRAM** **Frequency**
    - **定义**: DRAM 内存控制器的工作频率（如：显存频率，单位 MHz）。
    - **意义**: 反映显存带宽潜力的核心指标。
2. **SM** **Frequency**
    - **定义**: Streaming Multiprocessor (SM) 核心的运行频率（单位 MHz）。
    - **意义**: 直接影响计算吞吐量的核心参数。
3. **Memory** **Throughput**
    - **定义**: 全局内存（DRAM）的读写吞吐量（单位 GB/s）。
    - **意义**: 衡量显存带宽的实际利用率。
4. **DRAM** Throughput
    - **定义**: 显存控制器的数据传输速率（可能是 Memory Throughput 的子集）。
5. **L1/TEX Cache** **Throughput**
    - **定义**: L1 缓存和纹理缓存的综合吞吐量。
    - **备注**: TEX 指纹理单元（Texture Unit）。
6. **L2 Cache** **Throughput**
    - **定义**: L2 缓存的读写吞吐量。

## II. 周期与时间
1. **Elapsed Cycles**
    - **定义**: GPU 执行某段代码的总时钟周期数。
    - **意义**: 直接反映时间成本。
2. **Duration**
    - **定义**: 分析的持续时间（单位 ms/μs）。
    - **备注**: 由 `--duration` 参数设置或自动统计。
3. **SM Active Cycles**
    - **定义**: SM 核心处于活跃状态（未停滞）的周期数。
4. **Total [Module] Elapsed Cycles**
    - 各模块（如 DRAM、L1、L2、SM）运行总周期数，衡量模块负载。

## III. 缓存效率

1. **L1/TEX Hit Rate**
    - **定义**: L1/TEX 缓存的命中率（单位 %）。
    - **公式**: `(缓存命中次数 / 总访问次数) × 100%`
2. **L2 Hit Rate**
    - **定义**: L2 缓存的命中率（单位 %）。
3. **L2 Compression Success Rate**
    - **定义**: L2 缓存数据压缩的成功率（单位 %）。
    1. **应用**: Ampere/Ada 架构引入的显存压缩技术。
4. **L2 Compression Ratio**
    - **定义**: L2 缓存的压缩比（如 2:1 表示数据体积减半）。

## IV. 计算效率
1. **Compute (SM)Throughput**
    - **定义**: SM 中算术逻辑单元（ALU）的指令吞吐量（单位 OP/s）。
2. **Executed** **Ipc** **Active**
    - **定义**: SM 在活跃周期内每周期执行的指令数（Instructions Per Cycle）。
3. **Issue Slots Busy**
    - **意义**: 衡量指令级并行（ILP）效率。

## V. 调度与 Warp 效率
1. **Active Warps Per Scheduler**
    - **定义**: 每个调度器（Scheduler）当前活跃的 Warp 数量。
    - **备注**: CUDA 每个 SM 有 4 个调度器。
2. **Eligible Warps Per Scheduler**
    - **定义**: 可被调度器选择的就绪 Warp 数量。
3. **Warp Cycles Per Issued Instruction**
    - **定义**: 每个指令发射所需的平均 Warp 周期数。
4. **Avg. Active Threads Per Warp**
    - **定义**: 每个 Warp 中活跃线程的平均数量。
    - **意义**: 反映 Warp 执行效率（避免分支发散）。
 

## VI.内存访问瓶颈
1. **Mem** **Busy**
    - **定义**: 内存总线的繁忙周期占比（单位 %）。
    - **意义**: 高值表示显存带宽是瓶颈。
2. **Mem** **Pipes** **Busy**
    - **定义**: 内存管线（Load/Store Unit）的利用率。
    - **范围**: Fermi/Ampere 架构有独立 LOAD/STORE 管线。
3. **Max** **Bandwidth**
    - **定义**: 显存的理论峰值带宽（单位 GB/s）。
        

## VII. 线程资源与寄存器

1. **Registers Per Thread**
    - **定义**: 每个线程占用的寄存器数量。
2. **Block Limit [SM/Registers/Shared** **Mem****]**
    - **定义**: 每个 SM 支持的最大 Block 数量，受寄存器、共享内存等限制。
3. **Shared Memory** **Configuration Size**
    - **定义**: 每个 Block 分配的静态共享内存量（单位 KB）。


## VIII. 并行度与占用率
1. **Theoretical Occupancy**
    - **定义**: 理论最大占用率（每个 SM 活跃 Warp 数 / 最大支持数）。
    - **公式**: 由资源（线程/寄存器/共享内存）限制计算。
2. **Achieved Occupancy**
    - **定义**: 实际运行的平均占用率。
3. **Waves Per** **SM**
    - **定义**: 每个 SM 中同时调度的 Wavefront（类似 Warp 的概念）。


## IX. 分支与指令效率
1. **Branch Efficiency**
    - **定义**: 有效分支比例 = 非分支发散的分支次数 / 总分支次数。
2. **Branch Instructions Ratio**
    - **定义**: 分支指令在总指令中的占比。
3. **Avg. Divergent Branches**
    - **定义**: 每条指令平均导致 Warp 分支发散的次数。

## X. 采样与调试
1. **Maximum Buffer Size**
    - **定义**: 性能数据缓冲区的容量（单位 MB）。
2. **Dropped Samples**
    - **定义**: 因缓冲区不足或性能过载导致的数据丢失次数。