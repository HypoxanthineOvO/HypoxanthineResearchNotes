Nsight System 是 NVIDIA 的一款 Profiling 工具，主要用于对 GPU 的程序总体进行性能分析（e.g. 不同函数的执行时长等）。

我们以一个常见的例子展示 Nsight System 的使用：

```
{nsys_path} profile --trace=cuda --delay=8 --duration=8 --output={report_file_path} {RUN PROGRAM }
```
- `nsys profile` ：进行 Profiling
    - `--trace=cuda` ：只检查 CUDA Kernels
    - `--delay=m --duration=n` ：延迟 m 秒后开始 Profiling，持续 n 秒
    - `--output`: 输出路径

```
{nsys_path} stats --report cuda_gpu_kern_sum --force-export=true --format csv {REPORT-NAME} --output ./{CSV-NAME}
```

- `nsys stats` ：将 Report 的数据整理输出
    - `--report cuda_gpu_kern_sum` ：整理 CUDA Kernels 的数据
    - `--force-export=true --format csv` ：强制输出，输出格式