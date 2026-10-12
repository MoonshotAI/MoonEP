# npu_patch（MoonEP Ascend NPU 后端）

`npu_patch` 是 MoonEP 仓库内为 **Ascend（NPU）** 提供的后端模块，由
[`ascend-moonep`](../../../ascend-moonep) 迁移而来，保持 Python `Buffer` API
（shmem 对称内存，`dispatch` / `prefetch_weight` / `combine` / `reduce_grad`）
与调用面兼容。C++ runtime 负责 CANN SHMEM 生命周期、对称内存与 runtime
workspace 管理；AscendC kernel 经预编译的 `ascend_moonep_kernels`（native
CMake）暴露 C ABI launcher，再由 PyTorch C++ extension `npu_patch._C` 调用。

当前构建目标仅面向 **Ascend950 / A5**：UDRMA/UDMA（URMA）通信依赖 A5 的
对称内存 RMA 能力（SIMT），故不支持 A2/A3 等其它芯片。

## 目录结构

```text
npu_patch/
├── 3rdparty/
│   ├── shmem/                    # CANN SHMEM submodule
│   └── catlass/                  # CATLASS submodule
├── __init__.py                   # 导出 Buffer / MoonEPCommPlan / ShmemRuntime
├── api.py                        # MoonEP Buffer API（shmem 对称内存）
├── buffer_c.py                   # 对称分配/包装 + 框架侧 VMM 助手（create_vmm_physical 等）
├── planning.py                   # MoonEPCommPlan / 计划输出分配
├── shmem_runtime.py              # ShmemRuntime（shmem Python API 生命周期）
├── kernel_op.py / ops.py         # KernelOp 基类与各算子对象（AscendC 迁移接缝）
├── launchers.py                  # moonep launch_* 兼容 shim（launch_planning / launch_dispatch/…）
├── _common.py                    # 2MiB 对齐 / sym put-get / 跨 rank 屏障
├── _device.py                    # 把 ASCEND_RT_VISIBLE_DEVICES 映射成可见索引 0..N-1
├── native/                       # CMakeLists、AscendC/Bisheng kernel、C++ host csrc
│   ├── csrc/ops/                 # 每个 Python 侧算子的参数校验、输出分配、tiling 构造
│   ├── kernels/                  # AscendC kernel launcher 与 kernel（含 *_udma 变体）
│   └── include/ascend_moonep/kernels/
├── scripts/
│   ├── build_base.sh             # 唯一构建实现（CANN 发现 / shmem / kernel 库 / pip install）
│   ├── build_a5.sh               # Ascend950/A5 薄封装
│   └── run_test.sh               # 测试驱动（含 BUILD=1 重建）
├── benchmarks/                   # MoonEP benchmark 集（--engine {ubmem,urma}）
└── lib/                          # 随包分发的 kernel / shmem .so（构建产物）
```

## 获取代码

```bash
git clone --recursive <repository-url>
cd MoonEP
```

已有仓库时初始化 submodule：

```bash
git submodule update --init --recursive
```

当前 submodule：

```text
npu_patch/3rdparty/shmem    https://gitcode.com/cann/shmem.git
npu_patch/3rdparty/catlass  https://gitcode.com/cann/catlass.git
```

## 构建

先加载 CANN 环境（未设置 `ASCEND_HOME_PATH` 时构建脚本会自动发现）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

一键构建（Ascend950 / A5）：

```bash
bash npu_patch/scripts/build_a5.sh [CANN_SET_ENV]
```

或显式指定 SoC：

```bash
bash npu_patch/scripts/build_base.sh -soc_type Ascend950 [CANN_SET_ENV]
```

`build_base.sh` 是构建逻辑的**唯一实现**，依次：

1. （如配置CANN_SET_ENV）`source` CANN `set_env.sh`；
2. `git submodule update --init --recursive`（cann/shmem、catlass）；
3. 为当前 SoC 构建 cann/shmem 到 `npu_patch/3rdparty/shmem/install`；
4. `cmake -S npu_patch/native -B <build dir>`，用 Bisheng 编译 kernel，
   生成 `ascend_moonep_kernels`；
5. 将 kernel 与 shmem `.so` 拷入 `npu_patch/lib/`；
6. `pip install -e .`，编译并安装 `npu_patch._C` PyTorch extension。

> kernel `.so` 与 host 扩展 `npu_patch._C` 是**链接**关系，必须整套跑完，
> 不能只跑其中一步。可用环境变量 `FORCE_SHMEM=1`（重建 shmem）、
> `SHMEM_HOME_PATH`（复用已构建 install）、`BUILD_DIR`（指定 cmake 构建目录）
> 调整。

## 运行测试

统一入口是 `npu_patch/scripts/run_test.sh`（仓库根目录执行即可），按引擎拆分
**ubmem（MTE/AscendC，默认）** 与 **urma（URMA/UDMA，仅 A5/Ascend950）**：

```bash
# ubmem / MTE 引擎（默认）
bash npu_patch/scripts/run_test.sh
# urma 引擎
bash npu_patch/scripts/run_test.sh --use_urma
```

`run_test.sh` 覆盖三类测试：

1. **Buffer 公共 API（`test_npu/buffer_test`）**：`test_buffer_*` pytest 文件按
   `--use_urma` 用 `pytest -k` 筛选 parametrize 出的 `urma` / `mte` 变体；脚本式
   C / cpython 接口测试（`test_buffer_c_api` / `test_buffer_cpython_api`）引擎
   不可切、原样运行；分组 e2e（`test_buffer_grouped_e2e`）单独一节、不带 `-k`。
2. **urma 算子（`test_npu/urma`，仅 `--use_urma` 且仅 A5）**：`test_*_udma` +
   各 UDMA 脚本式算子测试（`test_moonep_combine_udma_with_plan` /
   `test_moonep_dispatch_udma_with_plan` / `test_dispatch_combine_udma_joint`）。
3. **mte ubmem 算子（`test_npu/mte`，默认）**：`test_*` + `test_planning` /
   `test_moonep_combine` / `test_moonep_dispatch_with_plan` /
   `test_dispatch_combine_joint`。

多进程下所有 rank 使用相同的 shmem bootstrap：单机走 UID bootstrap
（`ShmemRuntime.init_with_buffer`），多机走 ip_port/DEFAULT bootstrap（见
「多机运行（buffer-backed + ip_port bootstrap）」）。

## Python API

`moonep` 顶层已按后端替换，统一
`from moonep import ...` 即可拿到Ascend实现（当 NPU 后端启用时，`moonep.Buffer` /
`moonep.MoonEPCommPlan` / `moonep.ShmemRuntime` 即 `npu_patch` 的同名对象，并保留 `moonep.init_shmem`，见 `MoonEP/moonep/__init__.py`）。

对外可用API：

- **MoonEP 风格 Buffer API**（推荐，Python + shmem 对称内存）：`Buffer` /
  `MoonEPCommPlan` / `ShmemRuntime`，`dispatch` / `prefetch_weight` / `combine` /
  `reduce_grad` 用法见下一节。

```python
# Buffer / Plan / shmem / init 一律从 moonep 顶层入口拿（NPU 上已替换为 npu_patch 实现）
from moonep import Buffer, MoonEPCommPlan, ShmemRuntime, init_shmem
# 框架侧对称张量助手统一走 moonep.buffer，下列都是 NPU 专有名字
# （GPU 进程里不存在，也无需使用）：
from moonep.buffer import (
    create_full_weight,                  # 创建并填充 full weight（[E+B,H,H']）
    create_full_grad_and_reduce_buffer,  # reduce_grad 的 (full_grad, reduce_buffer)
    remote_home_rows,                    # 按 pe 点读 home chunk（免 bind，init 后可用）
    read_reduce_slot,                    # 按 pe 点读 reduce slot chunk（免 bind）
    expert_pool_row,                     # 全局 expert 的 [H, H'] 源行（init 后免 bind）
)
# 说明：GPU 的 create_nvl_dist_tensor / create_nvl_single_owner_tensor /
# pad_dim0_for_alignment 在昇腾侧没有对应实现——NPU 不需要 NVLink 映射，
# 跨 rank 张量由 shmem 对称堆 + 上面的构造器/点读助手覆盖。
# 说明：buffer_c.create_vmm_physical / launch_* 等是低层内部实现，应用侧无需
# 直接 import npu_patch —— 与 tests/test_e2e_npu_gpu.py 的写法一致。
```

## MoonEP Buffer API（shmem 对称内存）

`moonep.Buffer`（NPU 上即 `npu_patch` 的实现，见前节）是 `moonep/api.py`
Buffer 类的昇腾移植：在 `Buffer` 构造时
一次性分配 shmem 对称通信内存（`hidden_buf` / 各 meta 独立缓冲 / plan 广播区，
各变量单独分配 shmem 地址，无合并 `meta_buf`、无 `xxx_off`）并实例化一组
`KernelOp` 算子对象，`dispatch` / `prefetch_weight` / `combine` / `reduce_grad`
调用时执行 `op.compute`。跨 rank 访问走对称寻址：host 侧 `put/get`（v1），
kernel 侧 `aclshmem_ptr`（AscendC 迁移接缝）。

Ascend多卡传输每个对称块按实际 shape 精确分配
（各 meta 变量单独分配 shmem 地址），buffer-backed 对称堆的 descriptor
总大小按 VMM 粒度（2 MiB）对齐，这个对齐目前影响create_vmm_physical不支持创建epn * H * Hp * element_size < 2,097,152的场景。对称块不额外创建 shmem Buffer 对象：tensor
本身即对称块，下游算子通过 `tensor.data_ptr()` 获取对称基址（host 侧
put/get/signal/free 需要 Buffer 时现场构造）。

### 使用流程

当前只有 buffer-backed 初始化（`ShmemRuntime.init_with_buffer` /
`init_with_ip_port`）：**先构造 Buffer 与框架侧全部对称 VMM chunk（登记进全局
descriptor 注册表），再做一次对称堆 init，之后 Buffer 在首次
dispatch/combine/prefetch/reduce_grad 时把 chunk 翻译成内核要的对称地址**。
顺序不可颠倒：init 的 descriptor 集在 init 时冻结，之后创建的 VMM chunk 不会
进入当前对称堆。

```python
import torch.distributed as dist
from moonep import Buffer, ShmemRuntime

dist.init_process_group(backend="hccl")
R = dist.get_world_size()
E, S, H, K = 64, 4096, 256, 16
epn = E // R
B = epn                    # prefetch / 归约槽位数，默认等于 epn

# 1. 构造 Buffer（此时仅登记 VMM descriptor，shmem 尚未 init）
use_udma = True            # False = MTE（AscendC）；True = URMA/UDMA（仅 A5）
buffer = Buffer(S, H, K, E, R, B=B, num_sms=32, group=dist.group.WORLD,
                explicitly_destroy=True, use_udma=use_udma)

# 2. 构造框架侧 full weight / full grad / reduce buffer（见下节，
#    同样必须在 init 之前登记 descriptor）。

# 3. 一次性 buffer-backed 对称堆 init（单机 UID bootstrap；多机见下）。
#    enable_udma 必须与 buffer 的 use_udma 一致；URMA 参考 qp_count=2。
ShmemRuntime.init_with_buffer(group=dist.group.WORLD,
                              enable_udma=use_udma, qp_count=2)

# 4. 首次 dispatch/combine/prefetch/reduce_grad 时 Buffer 自动翻译对称地址。

# 5. 释放顺序：先 finalize 再 destroy（VMM 逆操作）。
ShmemRuntime.finalize()
buffer.destroy()
```

### ubmem（MTE）与 urma（URMA/UDMA）路径：初始化与调用

planning / dispatch / combine / prefetch / reduce_grad 各有两条 kernel 路径，
通过 `Buffer` 构造时的 `use_udma`（及每算子 `*_use_udma` 覆盖）选择：

- **ubmem / MTE 路径**（`use_udma=False`，默认）：AscendC kernel 走 MTE2/MTE3
  搬运，无 UDRMA。无论传不传 `use_udma`，非 A5 芯片都会自动回落 MTE。
- **urma / UDMA 路径**（`use_udma=True`，仅 Ascend950/A5）：dispatch / combine /
  prefetch / reduce_grad 走 UDRMA RDMA 直写；planning 默认保持 MTE
  （`planning_use_udma=False`）。

`Buffer` 构造时会设置进程级环境变量（`ASCEND_MOONEP_USE_UDMA` 与每算子
`ASCEND_MOONEP_{OP}_USE_UDMA`），kernel launcher / 算子据此选择 native 或 UDMA
变体；无需手动 export，但可用环境变量临时覆盖。

```python
# urma：dispatch/combine/prefetch/reduce_grad 走 URMA，planning 留 MTE
buffer = Buffer(S, H, K, E, R, B=B, num_sms=32, group=dist.group.WORLD,
                explicitly_destroy=True, use_udma=True, planning_use_udma=False)
# ubmem/mte 全路径：全部算子走 AscendC
buffer = Buffer(S, H, K, E, R, B=B, num_sms=32, group=dist.group.WORLD,
                explicitly_destroy=True, use_udma=False)
```

初始化时 `enable_udma` 必须与 `use_udma` 一致（URMA 对称堆带 UDRMA/QP 配置），
URMA 用 `qp_count` 指定每 rank 的 Dispatch UDMA QP 数：

```python
# urma：enable_udma=True + qp_count
ShmemRuntime.init_with_buffer(group=dist.group.WORLD, enable_udma=True, qp_count=2)
# ubmem/mte：enable_udma=False
ShmemRuntime.init_with_buffer(group=dist.group.WORLD, enable_udma=False, qp_count=0)
```

测试侧统一入口是 `test_npu/kernel_test_utils.py::init_shmem`（自动选单机 UID /
多机 ip_port bootstrap，`enable_udma` 透传）；调用仍走公共 Buffer 方法
`buffer.dispatch(...)` / `buffer.prefetch_weight(...)` / `buffer.combine(...)` /
`buffer.reduce_grad(...)`。端到端参考：`test_npu/mte/test_e2e.py`（ubmem）与
`test_npu/urma/test_e2e_urma.py`（urma）。

### 框架层要创建的对象：Full Weight / Full Grad / Reduce Buffer

调度侧（dispatch/combine）的通信 chunk 与 plan 由 `Buffer` 内部管理；框架层
需要为 **prefetch_weight（权重侧）** 和 **reduce_grad（梯度侧）** 准备三组对称
张量。下列构造器是 **NPU 专有**名字（仅由 `moonep.buffer` 在 NPU 后端
re-export，GPU 进程中不存在——GPU 侧等价的稠密 `[E+B]` / `[R,B,H,H']`
张量由框架/测试直接以 `torch.empty` 构造）；在 NPU 上它们会登记进全局
descriptor 注册表并在首次算子调用时翻译成对称基址，
应用侧无需直接操作低层 `npu_patch.buffer_c`：

- **Full Weight（pool）**：`buffer.prefetch_weight` 的 `full_gate_weight` /
  `full_up_weight` / `full_down_weight` **就是该 full_weight**——由
  `create_full_weight` 一步创建并填充的逻辑 `[E+B, H, H']` bf16 权重池（NPU 上即
  本 rank home/slot chunk 的 `[E+B]` 投影，附 `_npu_home_chunk` / `_npu_slot_chunk` /
  `_projection`）：从全局 expert 表把本 rank 拥有的权重写入 home 行、slot 清零，
  `source_offset` 为逐投影偏置（gate/up/down 取 0/1/2）：

  ```python
  B = E // R                            # prefetch / 归约槽位数，默认等于 epn
  full_weight = create_full_weight(E, B, H, Hp, rank, R, remote_expert,
                                   source_offset=source_offset,
                                   dtype=torch.bfloat16,
                                   device=torch.npu.current_device())
  ```

- **Full Grad + Reduce Buffer**（`buffer.reduce_grad` 的 `full_gate_grad` /
  `full_up_grad` / `full_down_grad` 与 `gate_reduce_buffer` / `up_reduce_buffer` /
  `down_reduce_buffer`）：用 `create_full_grad_and_reduce_buffer` 一并创建，返回
  `(full_grad, reduce_buffer)`；`full_grad` 为 `[E+B, H, H']` **fp32**（附
  `_npu_home_chunk` / `_npu_slot_chunk` / `_projection`），Ascend reduce_grad 只读
  本 rank slot chunk 的对称基址。`grad_table` / `reduce_table` 分别是 rank 不变的
  `[E, H, H']` 梯度参考表与 `[R, B, H, H']` slot 参考表，本 rank 只拷贝自己的
  home / slot 切片：

  ```python
  full_grad, reduce_buffer = create_full_grad_and_reduce_buffer(
      E, B, H, Hp, rank, R, grad_table, reduce_table,
      dtype=torch.float32, device=torch.npu.current_device())
  ```

  示例见 `tests/test_e2e_npu_gpu.py`：权重侧用 `make_prefetch_args`，梯度侧用
  `make_grad_reduce_args`。

要点：

- 三组对称张量（full_weight / home-slot 投影）必须在 `moonep.init_shmem`
  （`ShmemRuntime.init_with_buffer` / `init_with_ip_port`）**之前**创建——init 的
  descriptor 集在 init 时冻结，之后的 VMM chunk 不会进入当前对称堆。
- **init 之后**跨 rank 读取**免 bind**：`remote_home_rows` /
  `read_reduce_slot` / `expert_pool_row` 按需解析对端对称地址
  （`buffer_c.remote_view`），返回 live 零拷贝视图。
- NPU 上 full_weight / full_grad 的投影视图基址**未登记**
  buffer-backed init，不能直接作对称基址用；算子只读 home/slot chunk 的对称基址，
  不读投影视图本体（投影段别名同一份物理内存）。
- `B`（prefetch / 归约槽位数）默认等于 `epn`；`Buffer` 内部的组合 `[E+B]` slot
  布局假定 `B == epn`（benchmark runner 构造时会断言），而 pool 投影路径对 `B`
  无此约束。
- 完整参考：`tests/test_e2e_npu_gpu.py` 的 `make_prefetch_args` /
  `make_grad_reduce_args`；benchmark 见「基准测试」一节。

### Buffer dispatch / combine / grad reduce

```python
# hidden_sh [S,H] bf16, route_weights_sk [S,K] fp32,
# topk_experts_sk [S,K] int32, tokens_per_expert [E] int32
hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
    hidden_sh, route_weights_sk, topk_experts_sk, tokens_per_expert,
)
assert isinstance(plan, MoonEPCommPlan)

# GEMM 就地在 hidden_nvsh 上执行后：
out_sh, out_sk, _ = buffer.combine(
    plan=plan, hidden_nvsh=hidden_nvsh, route_weights_nvs=route_weights_nvs,
)

# 复用 plan（跳过 planning，只重放 dispatch；返回同一个 plan 对象）：
hidden_nvsh2, w2, cu2, plan2 = buffer.dispatch(hidden_sh, plan=plan)

# 权重侧 prefetch（先构造好三份 full weight，见上节）：
buffer.prefetch_weight(
    plan=plan,
    full_gate_weight=full_gate_weight,
    full_up_weight=full_up_weight,
    full_down_weight=full_down_weight,
)

# 梯度侧 reduce（先构造好三份 full grad + reduce buffer）：
buffer.reduce_grad(
    plan=plan,
    full_gate_grad=full_gate_grad, full_up_grad=full_up_grad,
    full_down_grad=full_down_grad,
    gate_reduce_buffer=gate_reduce_buffer, up_reduce_buffer=up_reduce_buffer,
    down_reduce_buffer=down_reduce_buffer,
)
```

- `zero_copy=True`：dispatch 返回本地对称分片的视图（`hidden_buf_local` /
  `weights_buf_local`，远程 rank 已直接写入）；combine 的 `hidden_nvsh` 必须
  别名该视图（`data_ptr` 相同），否则断言失败。
- `async_finish=True`：返回 `torch.npu.Event`。当前实现是兼容 shim——仍在
  current stream 上执行，随后设备同步 + 跨 rank 重排，避免连续 async 操作共享
  shmem 栅栏代际而死锁（跨 rank kernel 内含设备侧全 rank 栅栏）。
- 断言级参考见 `test_npu/mte/test_e2e.py` 与 `test_npu/urma/test_e2e_urma.py`
  （含 plan 复用、zero_copy、async、prefetch 投影读回、grad reduce 参考校验）。

依赖 `shmem` Python 包（cann-shmem `src/python`）。Buffer 路径与 `_C` 必须复用
同一份 SHMEM runtime，不能在同一进程中再初始化 C++ `ShmemInstance`。

### 多机运行（buffer-backed + ip_port bootstrap）

单机默认走 UID bootstrap（`ShmemRuntime.init_with_buffer`：rank0 生成 UID 经
`torch.distributed` 广播）。多机（多节点）改用 ip_port / DEFAULT bootstrap：
全体 rank 传同一个 `tcp://<rank0_ip>:<port>`，rank0 建 config store server、
其余连 client——对称空间可跨节点。

```python
from moonep import Buffer, ShmemRuntime

# 构造完所有 Buffer / 对称 pool（moonep.buffer，登记 VMM descriptor）后：
ShmemRuntime.init_with_ip_port(
    group=dist.group.WORLD,
    ip_port="tcp://192.168.1.10:29501",   # 缺省时从 MASTER_ADDR/MASTER_PORT 推导（端口 +1）
    descriptors=None,                      # None = 进程内已登记的 VMM 分配
)
# 或等价：ShmemRuntime.init_with_buffer(ip_port="tcp://192.168.1.10:29501")
```

### 基准测试（npu_patch/benchmarks）

`npu_patch/benchmarks` 提供 Ascend benchmark 集：`bench_comm`、
`bench_grad_reduce`、`bench_prefetch`、`bench_vs`、`bench_vs_all2all`、
`bench_barrier`、`bench_dispatch_maxvio`、`bench_grad_reduce_maxvio`，共享 `_bench_common.py`
助手与 `run_bench.sh`（torchrun wrapper）。每个 benchmark 都带
`--engine {ubmem,urma}`（默认 `ubmem` = MTE/UB 原生路径；`urma` = UDMA/URMA
路径，仅 A5/Ascend950），作用到 Buffer、原始 `launch_*` 入口与 SHMEM 堆 init。
`bench_vs_all2all` 是 dispatch/combine 的对手实验：以 `bench_vs.py` 为模板，
把 MoonEP 与直接基于 `torch.distributed` `all_to_all_single` 的“无 plan、无
prefetch”实现放在同一组 routing/计时下对比（`--libs moonep all2all`，也可
`--libs all2all` 单独验证该集合通信，此时不加载 Buffer / shmem 堆）。

先构建，再运行：

```bash
bash npu_patch/scripts/build_a5.sh

# 单机 8 卡，默认脚本 + 默认引擎（ubmem）
bash npu_patch/benchmarks/run_bench.sh
# 指定脚本
bash npu_patch/benchmarks/run_bench.sh bench_prefetch.py
# urma 引擎（仅 A5/Ascend950）
ENGINE=urma bash npu_patch/benchmarks/run_bench.sh bench_comm.py
```

多机启动（`NODES>1` 各节点各执行一次，`MASTER_ADDR` 指向 rank0）；多机 shmem
ip_port 缺省从 `MASTER_ADDR` / `MASTER_PORT` 自动推导（端口 +1），需要时可显式
指定 `SHMEM_IP_PORT="tcp://<ip>:<port>"`：

```bash
# node 0
NODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.10 bash npu_patch/benchmarks/run_bench.sh bench_vs.py
# node 1
NODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.10 bash npu_patch/benchmarks/run_bench.sh bench_vs.py

# MoonEP vs torch.distributed all_to_all（同样支持 NODES/NODE_RANK 多机）
bash npu_patch/benchmarks/run_bench.sh bench_vs_all2all.py
```
