# Lab 01 · 测出性能基线

阅读：[CS336 L2：Resource accounting](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_02.json)、[GPU slides](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf)。

## 任务与验收

1. 复用第三课模型，手算参数、梯度、优化器状态和 attention score 的数量及字节数。
2. 固定配置，区分数据加载、forward、backward、optimizer 的耗时。加速设备计时前后同步，先 warmup，再测多次。
3. 改变 context length 至少三档，画耗时与内存趋势；结合复杂度解释。
4. 分开报告 CPU/RAM、CUDA 显存、MPS 可获得的统计，未能测量的项标记不可用。
5. 本目录 `notes.md` 保存配置、计时边界、median 与离散程度、profile 结果和首要瓶颈假设。

通过条件：另一进程能按记录重跑；知道最大的时间/内存来源，并能解释测量误差。状态见 [schedule](../schedule.md)；当前为实验规格。
