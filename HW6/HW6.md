# 异构计算 作业六

> 通过主机端 API 函数 ，实现事件命令同步。基本要求：
>
> 1. 参考资料 [1] 中提供了样例代码，编写 OpenCL 程序；
> 2. 说明样例程序中使用到的事件命令同步方式；
> 3. 完成程序调试，并分析程序输出结果。

我对参考资料的代码进行了部分修改。在[附录](#附录)中详述。

样例程序使用 OpenCL 的事件（Event）进行同步。具体而言：

1. 所有的 `clEnqueueWriteBuffer()` 都是非阻塞写入；对 `src2MemObj` 写入时，令其等待 `src1MemObj` 的写入事件。
2. 在启动第一个 Kernel 前，令其等待前述两个 `clEnqueueWriteBuffer()`；
3. 令第二个 Kernel 等待第一个 Kernel；
4. 最后的 `clEnqueueReadBuffer()` 是阻塞读取；且它要等待第二个 Kernel。

这样做的好处是主机端不必阻塞；在设备端进行操作时，主机端可以执行其他的事。

## 附录

我对参考资料的代码进行了以下修改：

1. 对 `cl_context` 等类型实现 RAII 方式的资源管理；可以类比智能指针和手动 `malloc`/`free`。
2. `clCreateCommandQueue()` 被标记为弃用；使用 `clCreateCommandQueueWithProperties()` 取代之。
