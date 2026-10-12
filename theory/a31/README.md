# A31 Research Package

日期：2026-10-11。定位：理论推导、原始代码/文献审查、小型核验与后续实验规格。

## 阅读顺序

`A31_THEORY_REPORT_ZH.md` 是完整中文报告，含 I–VII 七项交付、定理/反例、两种保留架构、风险与成本预测、最多三个实验、文献和版本来源。

`CODEX_A31_NEXT.md` 是可交给 coding agent 的下一轮任务。默认先做 CPU algebra 与已有缓存的能量审计，不启动大型训练，不重新做 current selection。真实小型成像由输入完整性、功率形式和预算 gate 控制。

`a31_matrix_checks.py` 与 `A31_MATRIX_CHECKS.json` 是本轮实际运行的22项小型核验。运行：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python a31_matrix_checks.py
```

依赖 NumPy 和 SciPy。脚本仅使用固定seed自行构造的小矩阵、解析平板和9维辐射修正点偶极系统，不读取用户材料真值，不调用GPU或原生网格Maxwell后端。运行会更新同目录结果JSON；保留原结果副本后复现。

## 结论边界

22/22 PASS 表示所列恒等式数值吻合、预设反例成功反驳无条件主张；不代表真实复杂材料成像或在线加速已经通过。主方向是耗散认证的补偿感知材料决策推断；相对于指定精确模型后验的数值决策regret，不等于错误prior下对未知真值的绝对风险证书。

A17最新closure已有有限非线性证据，仍应独立投稿；OPM不作为新路线必须保留的结构。全文不可得的竞争文献在报告中明确标注。

`MANIFEST.json` 保存交付时各文件的大小与SHA256；重跑JSON后其校验值会改变。