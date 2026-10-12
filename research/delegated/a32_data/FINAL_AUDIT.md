# A32 最终只读实现审计

审计结论：未发现会改变当前 **E0 FAIL（主方法 MSE 恶化39.46%）** 的实质 packing、质量坐标、白化或当前批次 truth 泄漏错误。原生 E1／3D 的 **NOT_RUN** 表述成立。科学裁决由父 Codex 负责；以下为代码、文本与可复现性检查。

这是父线程修复前的审计快照。父线程已接收第2–5项，计划修复／发布；本审计没有重新核验这些后续修改。

1. **当前物理数组口径一致。** `src/a32/preflight.py:10–16,37–40` 使用逐源 `[Re32,Im32]` 和材料 `[Re576,Im576]`；`src/a32/e0.py:43–47,55–71` 正确把质量 Jacobian 乘 `sqrt(1/576)*T`，采用 `sqrt(2)/sigma` 白化，并用原 N3 的同锚 `F(N3)+L(chi)-L(N3)` 偏移。`src/a32/diagnostics.py:30–37` 的诊断目标与在线目标一致。TRAIN prior 使用物理 amplitude1.2 的系数／质量坐标，未见遗漏或重复 amplitude／mass 因子。

2. **N3 作用范围的文字应补齐。** `src/a32/report.py:79`、`docs/MATHEMATICAL_SCOPE.md:7` 的“N3只作初值／支持”遗漏了 `src/a32/e0.py:45–47` 的合法校准锚用途。准确表述应包括“初值、条件支持及同锚偏移校准；不作为独立prior”。这不改变已执行算法或当前结果。

3. **官方 gate 应复用严格 coverage guard。** `src/a32/e0.py:158–168` 另写的 completeness 检查比 `src/a32/statistics.py:312–329` 弱，没有完整复用缺失／重复背景、失败 reference、unexpected parents 等检查。当前两份 gate 均为 FAIL，gain 只差浮点舍入；现有32条件结果没有因此改变。父线程已接收统一 gate 的修复。

4. **文档复现步骤有实际缺口。** `README.md:23–30` 使用已存在的 job IDs，而 `tools/a32_job.py:11` 对已有 job 目录拒绝执行。应采用独立复现 IDs／明确独立输出与预算。README 同时遗漏 `python -m src.a32.diagnostics` 和 `python tools/plot_actual.py`；前者为 `src/a32/report.py:10` 所需归因文件的生成入口，后者在 `tools/plot_actual.py:15,25–29` 生成 parent 表与实际图。

5. **报告链接检查：有一处断链，其余已检查本地目标存在。** `src/a32/report.py:156`／`A32_E0_E1_DECISION.md:117` 的 `PER_PARENT_METRICS.csv` 不存在；真实文件是 `results/a32/offline/tables/E0_PARENT_METRICS.csv`（`src/a32/statistics.py:366`）。报告引用的数学说明、README、E0指标、失败归因、decision／费用／失败账本及5张嵌入图片均存在。以下5个来源链接可匿名打开：[A17理论](https://github.com/migodam/a17-maxwell-certified-exchange/blob/df61a1b533682a883cdaf65259b4528c354bd832/closure_final/manuscript/CORE_THEORY.tex)、[A17效率限制](https://github.com/migodam/a17-maxwell-certified-exchange/blob/df61a1b533682a883cdaf65259b4528c354bd832/efficiency_followup/GPT_REVIEW_CONTEXT.md)、[A30报告](https://github.com/migodam/a30-maxwell-observable-material-manifold/blob/717d3d26a290248bd2b9aeb55f3d03694b407b84/A30_E0_E1_DECISION.md)、[A29 Release](https://github.com/migodam/a29-material-hypothesis-coverage/releases/tag/a29-v0.1.0)、[OAS论文](https://arxiv.org/abs/0907.4698)。

6. **数组发布是尚待完成的独立 gate。** 审计时 `publication/` 为空，checkout 无 remote／tracked files，因此尚不能验证 A32 公共报告或数组 Release。`src/a32/report.py:156` 的“大数组另放Release”当时仍是待实现的发布声明。若 NPZ 从 Git 排除，必须实际上传数组到 Release、提供匿名下载与解压说明：`src/a32/data.py:298–305` 的评分访问门需要32份 online NPZ，`tools/plot_actual.py:25` 需要 `RECONSTRUCTIONS.npz`。仅发布指标／token而缺数组不能重评分或重建实际图。父线程已承诺实际上传并匿名核验；发布成功不改变科学 FAIL。

7. **未发现凭据类公开敏感内容。** 对公有候选文本的典型 API／GitHub／AWS token 和私钥模式扫描无匹配；未发现 `.env`、credentials、private-key 文件。该结论是有限模式检查，不是无条件安全保证。报告科学范围区分充分：`A32_E0_E1_DECISION.md:54,66,90,96–103` 明确保留 exposed bootstrap、OFFLINE capacity、未完成因果归因以及 native E1／加速 NOT_ESTABLISHED。

本次审计仅阅读源代码、已生成的 Markdown／JSON／CSV／文件目录并检查引用网页；没有重跑测试、物理、优化、评分、图表或统计，没有打开 target NPZ／locked inputs，没有 hash 检查、DeepSeek 或额外 agents。仅按父线程最后授权写入此审计文件。
