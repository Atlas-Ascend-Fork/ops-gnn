# torch_sparse SpMM 算子设计（Ascend 950）

社区任务设计评审见 [设计 PR #1409](https://gitcode.com/cann/cann-ops-competitions/pull/1409)。
本实现随 ops_gnn 统一构建，代码位于 csrc/npu/torch_sparse_spmm，
Python 适配位于 python/ops_gnn/torch_sparse_spmm.py。

## 需求与接口

实现 COO/CSR 稀疏矩阵与稠密矩阵相乘。COO 支持 sum；CSR 支持
sum/add、mean、min、max。索引为 int64，矩阵输入形状为 [..., N, K]，
输出为 [..., M, K]。边权可选，每条边支持一个权重或 K 个特征权重。
mean 按行内边数求平均，重复边分别计数。

导入 ops_gnn.torch_sparse_spmm 后，注册 torch_sparse.spmm、
torch.ops.torch_sparse 的 CSR 接口以及 SparseTensor 的稠密乘法路径。
CPU/CUDA 输入使用已有上游实现，NPU 输入使用本算子。
Python 签名和 torch_sparse 保持一致；辅助 csr_spmm 可显式选择输出精度。

## 类型与边界

支持 FP32→FP32、FP16→FP16/FP32、BF16→BF16/FP32、INT8→INT32/FP32
共七种输入输出组合。浮点使用 FP32 累加；INT8→INT32 使用整数累加。
空行输出零，min/max 的 arg 为边数 E；最值忽略 NaN，平局保留第一条边，
全 NaN 行输出 0/E。整数 mean 向零截断。

支持批量及非连续输入；Host 规范化连续布局后启动内核。
Python/Host 检查设备、类型、形状、长度、索引范围和输出大小溢出。
设备验证 CSR 指针单调性、首尾值及列索引边界；错误在计算内核启动前抛出。
梯度入口同样检查 grad 的设备、浮点类型和预期输出形状，并限制权重宽度为 1/K。

## 内核方案

参考 cann/ops-sparse 的 sparse/spmm/arch35/spmm_kernel.cpp，提交
9a514470c2cc99e469c34a615048eb8b30a66694，保留来源及许可证声明。
CSR 内核由 SIMT 线程遍历行和特征，按输入类型、归约和索引宽度选择模板。
规则维度路径进行多特征处理和边预取；通用路径覆盖其他维度、批量、最值及整数。
直接启动 Ascend C Kernel，使用当前 NPU Stream；不调用 aclsparseSpMM 接口。

COO 在 NPU 上排序并转换为 CSR，按置换重排边权后调用 CSR 内核。
Python 缓存拓扑验证、CSR/CSC 元数据和 COO 转换。缓存键检查 Tensor 身份、
版本和相关元数据，数值不缓存；原地修改使缓存失效。无版本号的 inference Tensor
不缓存。外部裸指针修改后调用 clear_validation_cache 清理缓存。

## 自动求导

浮点 sum/mean 支持 value 和 mat 的一阶梯度。稠密梯度通过转置图聚合；
边权梯度计算邻接源特征与 grad 的内积或按特征乘积，并处理 batch 和 mean。
保存输入 Tensor 以检测计算后原地修改，所需 CSC 元数据在梯度路径按需生成。
min/max 梯度、二阶梯度、整数梯度及稀疏乘稀疏不支持，调用时明确报错。

## 工程与验证

op_host 提供 ATen 接口；op_kernel/arch35 分离启动、计算模板与形状定义；
六个内部接口由 csrc/pybind.cpp 统一绑定。仓库架构扫描仅在 dav-3510 构建中
包含该算子，打包使用仓库 MANIFEST.in 的通用规则。

测试位于 test/torch_sparse_spmm/arch35，仅包含三个文件：
benchmark_torch_sparse_spmm.py 在脚本内生成输入并报告平均调用耗时，
golden.py 提供 CPU FP64 参考及误差规则，test_torch_sparse_spmm.py
合并功能、参数防护、缓存与 PyG 测试。缺少 PyG 依赖时仅跳过 PyG 用例。
覆盖七种精度、四种归约、空输入、批量、非连续、梯度与最值 arg。
任务验收用例、ATK 和性能复现工具保留在 PR 独立验证附件，执行报告不随
算子源码提交。COO benchmark 测量预热后的调用，包含边权重排和 SpMM。

接口限制见 [API 文档](../../../zh/api_reference.md#torch_sparse_spmm)，构建与测试见 [开发指南](../../../zh/development_guide.md#torch_sparse_spmm)。
