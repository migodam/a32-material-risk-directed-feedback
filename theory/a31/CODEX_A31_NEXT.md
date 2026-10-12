# A31.1 — Dissipation-Certified Material Decision Inference
# Codex 下一轮执行规格

## 任务与边界

读取同目录 `A31_THEORY_REPORT_ZH.md`、`a31_matrix_checks.py`、`A31_MATRIX_CHECKS.json`。本轮不是训练新 NN，不重新做 OPM/current selection，不默认复制 A29 的固定小候选 bank。目标是检验：真实 Maxwell 耗散结构能否低成本控制补偿后的材料决策误差，并建立会正确拒绝失败场景的性能预测器。

开始时使用 plan mode 核对依赖、数据访问边界与预算，再按已冻结阶段执行；不因已有 package 名称推断所有原数组均存在。默认仅执行 E1 CPU/缓存核验；E2 必须满足下述 gate，E3 保持关闭，直到独立配置显式开启。不要自动发布、push、删除原结果或启动后台任务。

## 1. 原始项目输入

从用户提供的实际本地路径读取，或读取明确允许的原仓库文件。不扫描整个磁盘，不自动搜索 locked 数据。

- A17 commit `df61a1b533682a883cdaf65259b4528c354bd832`：`closure_final/manuscript/CORE_THEORY.tex`、`closure_final/START_HERE.md`、`efficiency_followup/GPT_REVIEW_CONTEXT.md`，以及用户确实提供的原生 DDA/polarizability/backend。
- A23 commit `a47cda424dfb55792e69d952a25f54ce9a2a5613`：`src/a20/opm.py`、原生 polarizability/Green adapter。
- A24 commit `ca80ca2a29867e1b03589061dd5c17688c9acbe0`：`src/a24/relative_feedback.py`、`physics.py`、原有计算账本。
- A28 commit `c7819fa142bb70f9920835e8e91ba145f15ed8e3`：`E1_DECISION.md`、`src/a28/physics.py`/`pipeline.py`，用户实际提供的已付费 cache manifest。
- A29 commit `5d5e31fcfc56ac8ef9d64c46b4b7bbdedef96bb4`：`A29_DECISION.md`、`src/a29/proposals.py`，仅用来保持原 baseline/信息边界，不重新进行该有限候选 selector。
- A30 报告与 protocol；原32条件全 pool 若缺失，标记缺失。不得把三图 hull 或四例交接结果当完整 A29.1。

先写 `INPUT_MANIFEST.json`：路径、SHA256、权限、来源commit、shape、dtype、物理含义。没有可确认的原生物理输入则 `NATIVE_PHYSICS_NOT_RUN_MISSING_INPUT`；CPU algebra 部分仍可完成。严禁导入模块时隐式执行 full_state。

## 2. 必须实现的数学接口

使用 float64 实材料/白化数据坐标与 complex128 物理变量。质量坐标 W 和噪声协方差只计一次；proper complex Gaussian 的实虚方差为复方差一半。零噪声不设置隐藏 sigma floor。

### 2.1 非线性 source-field 缺陷

实现 `nonlinear_defect_identity(chi, j_tilde, direction, backend)`：

`rho = b(chi) - L(chi) @ j_tilde(chi)`

`D(F-F_tilde)[h] = S R D(rho)[h] + S R D(X)[h] G R rho`

对于线性 contrast-source，`D(X)[h]=M_h`。DDA 必须使用真实 polarizability derivative，不能直接代为 h。

固定 basis 的 Galerkin 模型还要分别验证：

`J_F - J_A = O_A R I_A + S R_A (B_F-B_A)`

`I_A=(I-L R_A)B_F`, `O_A=S(I-R_A L)`。

必须在同一非线性模型上做 analytic JVP 与 finite difference；不把 frozen B 的 reduced derivative 当作实际 nonlinear ROM derivative。basis 若依赖 chi，要计其导数，或者明确固定 basis 的作用域。任意近似逆不满足 `R_A L R_A=R_A` 时应加入对应项或拒绝这个简化恒等式。

### 2.2 物理功率接口

创建 `PowerConvention`，至少保存：time_sign、physical current definition、material law、cell mass、background、boundary flux、radiation self convention、active support、source normalization、SI prefactor/normalized prefactor。

只在合法时返回：

`Z = X^{-1} - G`

`D = -Im_H(Z) = D_abs + D_rad`

`D_abs = X^{-H} Im_H(X) X^{-1}`

对 DDA 区分准静态 α 与 radiation-corrected α；自项只能计一次。若 radiative loss 已包含在 α 内，不再重复加入相同自项。零 X 不能加 jitter 求逆；使用固定活跃支撑、场变量或混合形式，并记录其适用边界。

多源完整 Gram 核验：

`(Einc^H J - J^H Einc)/(2i) = J^H D_abs J + J^H D_rad J`。

给每个功率量物理解释，不能将 `norm(y)**2` 当全散射功率。无独立完整边界/端口功率测量时，不添加伪 energy likelihood。

`loss_lower_bound(chi)` 只可返回已证明的 `0 < D0 <= D` 或 failure；不能人为给材料增加 Imχ 下限。允许一个明确声明为有损子类的实验配置，但不能声称该子类代表原box全部材料。

### 2.3 目标残差证书

接口 `goal_error_interval(chi, current_approx, dual_approx, q, scope)` 返回：

`rho=b-Z j_tilde`

`tau=S^H q-Z^H psi_tilde`

`center=psi_tilde^H rho`

`radius=norm(tau,D0^{-1})*norm(rho,D0^{-1})`。

字段包括 physical_condition、uniformity_scope、roundoff_allowance、source_RHS、operator_matvecs、bytes、wall_time。8源堆叠 q 一般对应8个dual RHS，不能记为1。精确 full dual 只能当离线验证，不得假装 cheap certificate。

有限材料域的 uniform bound 需要额外 Lipschitz/Taylor/interval covering 证明；仅点残差通过不能给整个 branch 授予认证。严格区间算术未实现时明确写 `FLOATING_POINT_EVALUATED_ANALYTIC_BOUND`，不写 machine-verified mathematical proof。

### 2.4 补偿与概率

实现 bounded compensation convex problem及其 feasible primal witness/dual lower bound。`U`含哪些contrast/boundary/calibration自由度、box/radius及来源必须保存。自由投影只是诊断，不能替代有界合法补偿。

proper Gaussian 线性 nuisance 模型：

`V(chi)=Sigma+B(chi) Gamma B(chi)^T`

`loglike=-0.5*r^T V^{-1} r-0.5*logdet(V)`。

B依赖材料时不能漏掉logdet或冻结V而仍称exact marginalization。独立源/收增益共同未知时为bilinear，只有明确线性化作用域才可用上述解析式；其余用小型积分/采样并标注精度。

同一个固定 V 的 signed pair loglike defect：

`delta_1-delta_0 = rbar^T P e_minus - secant^T P e_plus - e_plus^T P e_minus`, `P=V^{-1}`。

shared offset也可能改变log odds；不自动删除。每个证书区间必须说明是 absolute score、relative score还是noise-whitened field误差。

### 2.5 材料行动证书

正确区分：posterior support、quadrature nodes、最终有限报告 actions。

已归一化cheap weights q_i及 `delta_i in [lo_i,hi_i]`，实现变量 `(p,t)` 的约束：

`sum p=1, p_i>=q_i exp(lo_i)t, p_i<=q_i exp(hi_i)t, t>=0`。

采用稳定的shared log shift避免overflow；不同参考常数必须在全support一致。计算

`U(a)=max_b max_(p,t) sum_i p_i [loss(a,chi_i)-loss(b,chi_i)]`。

以SciPy HiGHS等常规LP实现，不需要NN。测试LP在小例子中与顶点/稠密枚举一致。finite action无法覆盖全连续Bayes action时加action approximation项，否则限定结论。

严格支持误差未控制时标签必须为 `CONDITIONAL_ON_QUADRATURE`。不得用ESS当missing-topology质量上界。材料均值平方风险、NRMSE、IoU、topology loss各自独立定义，不混用其最优行动。

## 3. E1：默认执行，0新增原生状态

运行随包22checks；保存原结果，不覆盖用户已有历史输出。新增 native checks 仅用清单中现成缓存，禁止缺失时full fallback。

测试正常有损、loss→0、high contrast、nonnormal、oblique projection、radiative self遗漏、source一致性破坏、nuisance吸收结构、错误prior与Dirac determinant。

E1 gate字段：`algebra_ok`, `physical_power_form_established`, `native_cache_complete`, `D0_valid`, `nonvacuous_material_goal_bound`, `predicted_certificate_cost_below_avoided_F`, `counterexamples_correctly_rejected`。

没有原生 cache 或无法确认self convention，不开启E2。D0失效要准确拒绝，不以全矩阵eig/逆免费替代。可以将dense eig用于离线核验，但费用/权限必须标为OFFLINE_ONLY。

## 4. E2：小型成像；通过E1后才可启用

默认配置 `enable_imaging=false`。满足E1且用户已有本任务的小规模实验授权时，按报告V.2固定最多8parents、128次全物理状态上限。若环境只有当前package、没有原生backend，停止在完整可执行spec/CPU测试，不自行造一个后端冒充原项目。

同一连续材料表示/先验比较 `calibrated_L`, `ordinary_full`, `decision_adaptive`。在线表示至少允许连续边界和独立复contrast，能通过明确的birth/death步骤处理空腔；不把全部truth family当chart index输入。proposal可依赖y，但importance weight必须含 `p_prior*likelihood/q_proposal(y)`；或者仅用proposal初始化正确posterior kernel。

新 clean label生成同样计state budget。真值由独立evaluator读取；不要把根据当前truth拟合的oracle参数回传在线初始化。先输出预测再评分。主指标固定mass squared material error与topology loss，兼报原Δχ NRMSE/edge/IoU。

预算不能保证高精度reference posterior：达不到则标注reference uncertainty，不发布posterior calibration PASS。pilot只决定路线是否值得继续，不作独立大样本显著性声明。

## 5. E3：本轮不默认运行

按报告V.3另行冻结后启用，最多16新parents、256额外state calls。E2+E3总上限384。超预算停止保存，不缩减reference精度抹去失败。

同一个inverse algorithm比较不同physics query policy，完整primitive费用计入。既要报告预测数值decision regret，也要报告真实材料误差；两者不混为一谈。OOD时可以正确宣布无证书，不能强迫给高置信度图像。

## 6. 成本账本与禁止行为

每次物理动作前原子预约。至少记录：total state calls、unique chi、LU count、forward/adjoint/JVP RHS、cached matvec、dual certificate、posterior integration、setup、传输、CPU/GPU active wall、peak allocated/reserved/process memory、失败/重试/fallback、cold/warm与共享cache次数。

文件加载、encoder、proposal、optimizer、LP不属于免费操作。强制保存预跑成本预测表，其primitive时间来自既有合法profile，不填写未经测量的秒数。相同预算指raw-data-to-final-action而非只比core solve。

禁止：新大网络训练、current basis selection、rank sweep、隐藏jitter、默默pinv、clip掉非法模型后不记损失、以free gains吸收模型误差后宣称材料准确、打开locked、重写A17/A29旧裁决、用已暴露开发真值认证独立预测。

## 7. 输出

生成 `A31_1_DECISION.md`、`INPUT_MANIFEST.json`、`FROZEN_PROTOCOL.json`、`POWER_CONVENTION.md`、`ALGEBRA_NATIVE_CHECKS.json`、`PRE_RUN_COST_PREDICTION.json`、`QUERY_LEDGER.jsonl`、`RISK_PREDICTIONS.jsonl`、`OFFLINE_MATERIAL_SCORES.csv`（仅E2实际运行时）、`FAILURES_AND_LIMITS.md`。

每条记录标明 RIGOROUS_ANALYTIC_BOUND / FLOATING_POINT_EVALUATED_ANALYTIC_BOUND / CONDITIONAL_ON_QUADRATURE / LOCAL_GAUSSIAN_APPROXIMATION / EMPIRICAL_INTERVAL / HEURISTIC。未运行文件用明确NOT_RUN，不填假数据。

最终裁决只回答：物理功率形式是否成立；证书是否非空且可支付；是否预测材料而非残差；full F相对calibrated-L有何材料价值；完整费用是否净改善；下一步GO/NO_GO及其具体原因。不要为了保护OPM或A31假说而改变失败标准。