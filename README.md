# A32 — 连续材料恢复与材料风险反馈

完整研究入口：[A32_E0_E1_DECISION.md](A32_E0_E1_DECISION.md)。协议：[a32_config.json](a32_config.json)。公式边界：[数学说明](docs/MATHEMATICAL_SCOPE.md)。原始表、图、失败和费用均在 `results/a32/`。

**结果：E0 FAIL，主方法使复杂材料MSE增加39.46%；E1与三维验证NOT_RUN。** 同chart offline truth-fit有容量，当前在线恢复未利用它。新增全波状态、GPU、训练均为零。

本研究先检验 measurement-conditioned 连续材料拟合能否利用 A30 离线容量，再条件性检验 A17 成对反馈的材料风险价值。主平台为已知背景二维 TM，24×24、8源、32接收器；不是三维矢量成像或新 blind benchmark。没有训练网络。

## 来源

- [A30 已冻结报告](https://github.com/migodam/a30-maxwell-observable-material-manifold/blob/717d3d26a290248bd2b9aeb55f3d03694b407b84/A30_E0_E1_DECISION.md)：Sparse/PCA 结果是 offline capacity，不是已实现在线恢复。
- [A17 理论](https://github.com/migodam/a17-maxwell-certified-exchange/blob/df61a1b533682a883cdaf65259b4528c354bd832/closure_final/manuscript/CORE_THEORY.tex)：current-space factor 与 material paired closure。
- [A17 效率限制](https://github.com/migodam/a17-maxwell-certified-exchange/blob/df61a1b533682a883cdaf65259b4528c354bd832/efficiency_followup/GPT_REVIEW_CONTEXT.md)。
- 原 A31 附件保留在 `theory/a31/`；A32 用户协议优先，不重开 compensation／uniform certificate 路线。

`src/a32/frozen_representations.py`、`frozen_metrics.py`、`frozen_physics.py` 及 `vendor/frozen_a29/` 直接复用已公开 A30/A24/A2 代码；`a17_factors.py` 复用 A17 bounded dense algebra。未修改 Maxwell solver，没有执行原 loader 的哈希检查。

## 复现

安装 `requirements.txt`，从 [A29 Release](https://github.com/migodam/a29-material-hypothesis-coverage/releases/tag/a29-v0.1.0) 下载 `A29_REPRO_INPUTS.zip` 与 `A29_RAW_EVIDENCE.zip`，分别解压到同一 `inputs/a29_release/`。配置 `a32_config.json` 的 `input_root` 指向该目录，不需要下载或打开 locked 文件。

在独立复现checkout内运行。使用单 BLAS 线程；每个命令用预算 wrapper 和新job ID运行，保留已发布账本，不删除历史费用。以下命令会重新计算缓存E0，写入该checkout的结果；不执行新Maxwell状态。协议的1800秒仍包括历史累计费用。

```sh
a32_run="reproduce-$(date +%Y%m%d-%H%M%S)"
python tools/a32_job.py --id "$a32_run-tests" -- python -m unittest discover -s tests -v
python tools/a32_job.py --id "$a32_run-preflight" -- python -m src.a32 preflight
python tools/a32_job.py --id "$a32_run-e0" -- python -m src.a32 e0
python tools/a32_job.py --id "$a32_run-score" -- python -m src.a32 score
python tools/a32_job.py --id "$a32_run-diagnosis" -- python -m src.a32.diagnostics
python tools/a32_job.py --id "$a32_run-shadow" -- python -m src.a32 shadow
python tools/a32_job.py --id "$a32_run-trajectories" -- python -m src.a32 trajectories
python tools/a32_job.py --id "$a32_run-vector-check" -- python -m src.a32 vector-check
python tools/a32_job.py --id "$a32_run-figures" -- python tools/plot_actual.py
python tools/a32_job.py --id "$a32_run-report" -- python -m src.a32 report
```

E0 先保存全部32条在线输出，再允许评分器读取 development truth。E0 未通过时，后续入口只生成 NOT_RUN，不执行任何 full state。新一次物理复现实验必须声明独立预算与输出；不能删除现有账本后重复当前 paid campaign。

实现止于本次实际达到的E0、独立缓存诊断和数学测试。后续CLI是gate拒绝入口；原生E1轨迹驱动因资格失败未接线，不能通过改写gate直接运行一个已验证的E1算法。

大数组放在 [A32 Release](https://github.com/migodam/a32-material-risk-directed-feedback/releases/tag/a32-e0-v0.1.0) 的 `A32_E0_RAW_ARRAYS.zip`；解压到仓库根目录可复核已发布在线预测和offline truth-fit。ZIP不包含locked数据。重新计算需要上面的A29公共输入。

## 判决原则

主方法预先固定 PCA64＋TRAIN Gaussian＋calibrated-L。E0 要求12复杂 parent 的等权平均固定材料 MSE相对N3降低至少5%；bootstrap只是描述。Sparse、ridge、多尺度、smooth 的结果不替代主门槛。局部风险、材料图像和完整成本分别评价，不能以残差或 GN-gap 替代材料恢复。
