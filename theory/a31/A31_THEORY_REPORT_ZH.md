# A31 — Maxwell 物理原生推断：代数、耗散、对称性与材料决策风险

**研究日期：2026-10-11。性质：理论研究、原始代码审查、文献竞争审查与小规模核验；不是新的完整成像实验报告。**

## 结论先行

推荐的主线不是“再做一个 OPM 网络”，而是：

> **耗散认证的、校准感知的材料决策推断（Dissipation-certified, calibration-aware material decision inference）。**
>
> 把 Maxwell 反馈用于计算“当前物理近似会怎样改变材料后验与材料决策”，而不再把更准确的电流或更小的数据残差当作材料成像收益。

这条线的核心对象应从 `current approximation error` 转为 **decision-relative likelihood defect：与材料决策有关的似然差误差**。物理耗散提供数值误差证书；校准消元去掉仪器可以吸收的分量；材料后验决定剩余误差是否值得支付 full F。

本轮证明了若干有限维恒等式和有条件证书，形成了一个无需大型网络即可检验的算法。**尚未证明它在真实复杂材料成像上同时优于 calibrated-L 与普通 full-adjoint，也未证明证书在弱损耗、近共振状态中足够便宜和紧。** 这些是下一轮的决定性实验，而不是可以由理论恒等式直接宣布的结论。

A17 应继续作为独立论文。OPM 没有在本轮显示不可替代性；保留的是 Maxwell 的反馈和源一致性，而不是 O/P/M 三个字母。

### 状态标记

- **PROVED**：正文给出证明的数学陈述；不自动表示首次发表。
- **CONDITIONALLY PROVED**：在明确的物理、概率、离散化或支持集条件下成立。
- **KNOWN RESULT**：已有理论或直接标准推论，不计作 A31 独立创新。
- **COUNTEREXAMPLE**：推翻一个不加条件的说法。
- **CONJECTURE**：需要新物理实验检验的科学假说。

## 0. 证据校正与审查范围

读取并核对了 A17 的最新 closure/efficiency 报告、主稿理论；A23 的原始 OPM/Schur 代码；A24 的 relative-feedback 原始实现；A28 的执行与成像更新代码；A29 的 proposal 实现及裁决；Library 中的 A30 中文理论报告与实验协议。没有取得 A29 全部原始 NPZ、没有独立重跑 A17–A29 GPU campaign，也没有把历史 development 对象当成新的盲测。

| 项目 | 核实后的结论 | 对 A31 的约束 |
|---|---|---|
| A17 | 最新 closure 已有三组完整非线性对照：复材料误差改善 10.84%、0.22%、7.16%；时间比 2.68、1.74、1.46。并非只有冻结 GN 证据 | 可以支持机制论文，不能宣称普适加速 |
| A17 efficiency | top-1/cap-2 保留约 99% 冻结 GN 收益，却使 smooth 最终材料误差恶化 5.95%；验证 Jd 仅占原额外耗时约 1.19%/3.82% | GN fidelity 不是完整材料风险预测器；“减少认证 solves 就会快”也不成立 |
| A23 | 3D DDA，1728 cells、3456 实材料自由度；cached Schur 冷启动只节约约 0.35–0.86%；大量材料不可行 | 不能把原始误差降低和可部署成像混为一谈 |
| A24 | 24×24、2D TM、8 源/32 接收器；R16 任务误差与质量门槛未过；full F + 同样更新与普通 full-adjoint 等价 | 新方法必须胜过 ordinary adjoint，而不只是自己的劣化对照 |
| A28 | 六次 F 对复杂材料仅约 1.01% 改善，对数据残差约 27.33%；事后中心/中点校准消除相当部分 L-on-F 判别错误 | 必须保留 calibrated-L，区分 offset 与结构信息 |
| A29 | hard Oracle coverage 改善 2.1468%，不是凸包或连续流形的容量上限 | 不允许在原有限 bank 上再花大钱做 selector |
| A30 | 已有 stratified manifold、bounded compensation、一般 posterior stability；全 32 条件 hull 尚未完成 | 这些不能重新计作 A31 创新 |

代码层面还有两点重要区别。[A17–A30]

**第一，A17 的 3D 实现使用 contrast-dependent polarizability 及其导数，不能直接把所有离散公式写成 X=χ。** 本报告的连续 contrast-source 公式在 DDA 中要用真实的 X(χ)、DX(χ)，并保留 radiation reaction/self term。

**第二，A24 使用相对源变量**

\[
w=M_h e,\quad w=(I-M_h G R_a)^{-1}M_h e_a,\quad \Delta j=R_aw.
\]

所以代码中的反馈近似是 `G R_a`，而用户给定的 Δj 公式包含 `R_a M_h G`。两者由变量变换联系，不能因为乘法次序不同就认定实现错误，也不能把它们当成同一个矩阵去做能量投影。

## 1. 八个候选机制的筛选

| 候选 | 数学对象 | 裁决 | 最直接反例/失败条件 |
|---|---|---|---|
| M1 current Schur→material covariance | H 的 rank-2r 有符号更新 | 保留为诊断工具，不计独立创新 | 更大的错误 Jacobian 给出更小 Cpost，却有更大真实风险 |
| M2 非线性 residual–feedback 微分分解 | D(F−F̃)=SRDρ+SRM_hGRρ | 保留；把 A17 冻结理论接到真实非线性近似 | ρ 很小但 Dρ 很大 |
| M3 耗散–Schur–dual residual | −Im Z_eff=E*DE；目标误差双残差界 | 保留为物理认证核心；组成部分已有先例 | 损耗下界消失，或遗漏辐射自项 |
| M4 补偿后分离的目标函数证书 | 一个 separating witness q，而非全数据误差 | 保留为主线定理候选 | 结构差完全位于 nuisance orbit |
| M5 材料决策后验扰动 | signed likelihood defect、tilt oscillation、robust regret | 保留为主线决策核心 | 未覆盖 posterior support；错误 prior |
| M6 势的 gauge 产生材料先验 | A,φ 的规范自由度 | 淘汰 | 势改变而 E,H,χ 完全不变 |
| M7 Maxwell+能量双重 likelihood | 重复加入同一个物理约束 | 淘汰 | 改变参考测度后产生虚假的共振偏好 |
| M8 passive rational feedback | pH/positive-real ROM | 仅备选，不计“被动 ROM”本身为创新 | setup、material-dependent assembly 或证书成本吃掉收益 |

这些机制改善的指标不同：M2–M3 改善对数值近似的可控性；M4 改善可辨识性判断；M5 控制相对于指定精确后验的材料决策损失。**没有一个机制自动增加实验中已有的信息量。**

# I. Algebraic Discovery

## I.1 冻结 A17 到非线性成像：必须补上的总场缺陷

设

\[
L_\chi=I-M_\chi G,\quad b_\chi=M_\chi e^{inc},\quad R_\chi=L_\chi^{-1}.
\]

对任意可微近似电流 j̃(χ)，定义**方程残差**而不是接收数据残差：

\[
\rho(\chi)=b_\chi-L_\chi\widetilde j(\chi),\qquad
\varepsilon_j=j-\widetilde j=R_\chi\rho.
\]

假设 S 与材料无关。对真实材料方向 h，

\[
\boxed{
D(F-\widetilde F)[h]
= S R_\chi D\rho[h]
+S R_\chi M_h G R_\chi\rho.
}
\tag{T1}
\]

**PROVED。** 对 SRχρ 求导，使用 DRχ[h]=RχM_hGRχ 即得。第一项是残差本身的材料导数；第二项是已有状态误差经材料扰动再次激发并反馈到接收器。对 DDA，用 DX[h] 代替 M_h；若 S、源或 basis 依赖材料，必须加入其导数。

这不是声称一个新的 resolvent identity，而是指出**实际非线性 surrogate 的导数缺陷有两个不同的物理来源**。它直接解释为什么仅认证一个冻结 B 的电流模型，不能认证完整 nonlinear imaging。

对固定 basis 的 Galerkin ROM，令

\[
R_A=V(V^*L_\chi V)^{-1}V^*,\quad
B_Fh=M_h(e^{inc}+Gj),\quad
B_Ah=M_h(e^{inc}+Gj_A).
\]

其真正导数为 J_A=SR_AB_A，而不一定是 SR_AB_F。因此

\[
\boxed{
J_F-J_A
=O_A R_\chi I_A+SR_A(B_F-B_A),
}
\tag{T1b}
\]

其中 I_A=(I−LR_A)B_F，O_A=S(I−R_AL)。第二项正是**源依赖总场错误造成的共同材料注入错误**。它在本轮小矩阵中非零；只用冻结项的相对缺陷约 3.90%，补齐后误差约 3.31×10⁻¹⁶。

此外，原双侧恒等式需要 R_ALR_A=R_A。对任意近似逆，应使用

\[
R-R_A=(I-R_AL)R(I-LR_A)+(R_A-R_ALR_A).
\]

不要把适用于 Galerkin 广义逆的公式直接套给任意 NN closure。

**材料结论。** 近似 likelihood gradient 的误差含 Dρ 和状态反馈项；仅有小 ρ、好的 current norm 或良好 frozen GN gap 都不足以预测材料优化路径。

**成本。** ρ 需要真实 L 的 matvec；Dρ 需要 surrogate 的一致 JVP 与真实 DL，不必先形成完整 J。通过物理耗散可以界定 R 的相关作用，但如果只能通过 full R 求出误差，就不能把这部分当成在线免费预测器。

**决定性反例。** 在一维参数中令 ρ(z)=ε sin(z/ε²)v。残差一致趋零，而导数振幅为 1/ε。这个例子推翻“forward residual 小就能安全保留 derivative”的无条件说法。

## I.2 Material-risk-aware Schur：能精确计算，但没有单调性

采用白化的实坐标，联合未知量 θ=(z,γ)，先验 C_pr≻0，局部联合观测矩阵 A。设一次有结构的 feedback 修正给出

\[
A_1=A_0+UV^T,\quad H_i=C_{pr}^{-1}+A_i^TA_i,\quad C_i=H_i^{-1}.
\]

定义

\[
Q=[A_0^TU,V],\qquad
K=\begin{bmatrix}0&I\\I&U^TU\end{bmatrix}.
\]

则

\[
H_1-H_0=QKQ^T,
\]

\[
\boxed{C_1=C_0-C_0Q(K^{-1}+Q^TC_0Q)^{-1}Q^TC_0.}
\tag{1}
\]

若 Tθ 为材料切线，W 为固定材料度量，局部风险变化恰为

\[
R_0-R_1=
\operatorname{tr}\!\left[WT C_0Q(K^{-1}+Q^TC_0Q)^{-1}Q^TC_0T^T\right].
\tag{2}
\]

**KNOWN RESULT：Woodbury/Schur 的直接材料风险推论。** 不能作为 A31 独立的“新 Schur 定理”。它的用途是直接计算 signed material-risk change，而不是用 Fisher trace 代替。K 是不定的，因此风险变化不保证为正。联合做此更新再消去 nuisance，与一致的块 Schur 消元等价。

在 A17 中，U、V 来自 Y_C Γ_C⁻¹N_C 的正确实化。一个复 current mode、8 个源，不等于一个实 rank-one material 更新；必须先堆叠各源、再对共同材料变量实化。

### 协方差不是错误模型下的实际风险

真实模型为 y=f_F(m)+J_F(z−m)+n。近似后验均值采用

\[
\mu_A=m+K_A[y-f_A(m)],\quad d=f_F(m)-f_A(m),
\]

真实 Bayes 增益为 K_F，数据协方差 Ω_F=J_FC_prJ_Fᵀ+Σ。在相同真实先验、线性材料映射 T 下，

\[
\boxed{
\mathbb E\|T(\mu_A-z)\|_W^2
=R_F+
\operatorname{tr}\{WT(K_A-K_F)\Omega_F(K_A-K_F)^TT^T\}
+\|TK_A d\|_W^2.
}
\tag{3}
\]

**PROVED，标准 Bayes 正交分解。** 这区分了不可约风险、错误斜率和错误 offset，正适合 A28 的失败模式。

反例：J_F=0.01，J_A=0.1，先验和噪声方差均为 1。近似报告的 C_A=0.990099 小于正确 C_F=0.999900，但近似估计器真实 MSE=1.007824，大于正确 Bayes 风险。所谓“信息更强”在这里只是模型过度自信。

## I.3 真正的 Maxwell 功率形式：用阻抗 Z，不是随意对称化 L

以下先限定：e⁻ⁱωᵗ、均匀无损外部背景、非磁介质；j=χE 为归一化极化量，物理极化电流 J_p=−iωε₀j；内积为实际体积积分或一致的质量矩阵内积。在 χ 可逆的活跃支撑上，

\[
Z_\chi=M_\chi^{-1}-G,\qquad Z_\chi j=e^{inc}.
\]

令 Im_H A=(A−A*)/(2i)，则

\[
\boxed{
D_\chi=-\operatorname{Im}_H Z_\chi
=M_\chi^{-*}(\operatorname{Im}_HM_\chi)M_\chi^{-1}
+\operatorname{Im}_HG
=:D_{abs}+D_{rad}\succeq0.
}
\tag{4}
\]

并有

\[
P_{ext}=\frac{\omega\epsilon_0}{2}\operatorname{Im}\langle e^{inc},j\rangle
=\frac{\omega\epsilon_0}{2}j^*D_\chi j
=P_{abs}+P_{scat}.
\tag{5}
\]

**KNOWN RESULT：VIE 功率/光学定理形式。** 吸收限制光响应及 |χ|²/Imχ 的尺度已经出现在 Miller 等的光学响应极限理论中，不能声称这部分新。[L10]

必须满足三个实现条件：G 包含正确的 outgoing radiation/self contribution；DDA 的 radiation reaction 在 α⁻¹ 与 G 的拆分中只计一次；质量矩阵和变量缩放与实际功率一致。对 lossy host、PML 或只含 off-diagonal G 的模型，不能直接把 ImG 当成正确的完整辐射功率。

### 多源功率 Gram

令 J=[j₁,…,j_s]，E_inc=[e₁,…,e_s]，则

\[
\boxed{
\frac{E_{inc}^*J-J^*E_{inc}}{2i}
=J^*D_{abs}J+J^*D_{rad}J.
}
\tag{6}
\]

这是所有相干源组合的功率关系，不只是逐源的标量检查。它验证源间一致性；但对已准确满足 Maxwell 的候选自动成立，不能重新算一份独立材料数据。

### 被动 Schur 的准确条件

把 Z 分块为

\[
Z=\begin{bmatrix}A&B\\C&D_h\end{bmatrix},\quad
E=\begin{bmatrix}I\\-D_h^{-1}C\end{bmatrix},\quad
Z_{eff}=A-BD_h^{-1}C.
\]

当隐藏块可逆，

\[
ZE=\begin{bmatrix}Z_{eff}\\0\end{bmatrix},\quad
Z_{eff}=E^*ZE,\quad
\boxed{-\operatorname{Im}_HZ_{eff}=E^*D_\chi E.}
\tag{7}
\]

**PROVED；属于已知 passive Schur/Kron reduction 原理。** 吸收和辐射两项分别按相同 E 合同变换，因而保留真实功率分解。隐藏端有外部激励时，必须同时保留消元后的仿射驱动及其功率交叉项，不能只保留 Zeff。

V*ZV 保留该耗散结构。V*LV 一般不等价；若 L=MχZ，要实现同一个 Z-Galerkin，L 的测试空间应取 W_test=Mχ⁻*V。任意 Petrov–Galerkin 或 SVD feedback closure 不自动保能。互易性使用转置而非共轭转置；复 basis 下需正确的 reciprocal dual/metric，不能直接要求所有 reduced 矩阵都普通对称。

反例：

\[
Z=\begin{bmatrix}1-i&10\\10&1-i\end{bmatrix},\quad V=e_1,\quad W=e_1-ie_2.
\]

完整模型 −ImZ=I；W*V=1，但 W*ZV=1+9i，对应 reduced dissipation=−9，已变成主动系统。

## I.4 从物理耗散到不调用 full resolvent 的目标误差界

假设有**真正可计算的下界** 0≺D₀≼−Im_H Z。标量严格有损材料可使用 D_abs=diag(Imχ/|χ|²)；不允许人为加 loss floor 后还声称证明了原材料。

由 x=Z⁻¹v，有

\[
\|x\|_{D_0}^2\le-\operatorname{Im}(x^*Zx)
\le|x^*v|\le\|x\|_{D_0}\|v\|_{D_0^{-1}},
\]

故

\[
\|D_0^{1/2}Z^{-1}D_0^{1/2}\|_2\le1.
\tag{8}
\]

对 j̃ 与任意近似 dual ψ̃，定义

\[
\rho=b-Z\widetilde j,\qquad
\tau=S^*q-Z^*\widetilde\psi.
\]

则

\[
q^*S(j-\widetilde j)=\widetilde\psi^*\rho+\tau^*Z^{-1}\rho,
\]

\[
\boxed{
|q^*S(j-\widetilde j)-\widetilde\psi^*\rho|
\le\|\tau\|_{D_0^{-1}}\|\rho\|_{D_0^{-1}}.
}
\tag{T2a}
\]

**CONDITIONALLY PROVED。** 它是 dual-weighted residual/coercivity 的物理专门化，绝非首次提出 Maxwell output certification。Chen 等 2010 已有 harmonic Maxwell 的 rigorous output bounds。[L5]

其值得检验的区别在于：不通过逐状态 full resolvent 或昂贵 inf-sup 搜索取常数，而使用真实损耗直接构造 D₀；q 由材料分支/后验决策选定，不是所有接收器逐一认证。

计算包括近似 primal/dual、真实 Z/Z* matvec、D₀⁻¹加权，必须计费。8 源有8组残差；一个堆叠的数据目标 q 通常需要8个源对应的 dual RHS，而不是“所有源只需一个 RHS”。可以用低阶 dual，也必须计其构造费用。

**失败条件。** Imχ→0、|χ|大而损耗固定、支撑改变产生零 χ、radiation term 未正确实现、PML 的伪耗散被当成物理吸收，均可使证明失效或界变得无用。接近共振不直接推翻不等式，但会使低损耗界非常松。能量半正定不等于有严格稳定常数。

## I.5 补偿后的分离证书：不必先认证整个数据向量

考虑两条材料分支的 cheap affine 差

\[
\widetilde\delta(u)=\delta_0+Bu,\qquad u\in\mathcal U,
\]

U 是明确的 bounded nuisance/material compensation 集。支持函数记 h_U(v)=sup_{u∈U}vᵀu。任取 ||q||₂≤1，有

\[
\inf_{u\in U}\|\delta_0+Bu\|
\ge q^T\delta_0-h_U(-B^Tq).
\]

若对整个被认证区域，两个分支的真实非线性/物理误差满足

\[
|q^T(F_i(u_i)-\widetilde F_i(u_i))|\le\beta_i(q),
\]

则

\[
\boxed{
 d_{comp,F}\ge
 [q^T\delta_0-h_U(-B^Tq)-\beta_0(q)-\beta_1(q)]_+.
}
\tag{T2b}
\]

**CONDITIONALLY PROVED。** 对每个可行 u 使用 ||v||≥qᵀv，再下确界即可。β 可由 T2a、有限位移余项、decoder 曲率与 calibration remainder 组成。一个点上的 residual bound 不能直接当成整个 U 的 β。

此处的计算收益假说很具体：用 cheap compensated problem 找到一个分离方向 q，只认证这个方向的 Maxwell 输出，避免为一个 topology 决策认证全部场和全部 receiver responses。

这比“先算完整 F 再判断 F 是否值得算”更合理，但仍然可能失败：统一 β 构造比 full F 更贵；q 在真实模型下接近无效；nuisance orbit 正好吸收结构差；多分支的巨大 evidence volume 使一个两点距离不足以决定 posterior odds。

A17 的直接联系是

\[
q^*(J_F-J_A)h=q^*O_AR I_Ah
\]

及 T1b 的额外总场项。有限结构差需要沿材料路径积分，且 q 的 calibration horizontal component、路径上的 injection、resolvent 和额外状态项必须一起处理。只在一个冻结点使用 O_AR I_A，不能认证一次 topology change。

## I.6 后验只关心材料相关的似然误差

在固定、共同的线性 Gaussian nuisance 模型中，噪声已白化，γ∼N(0,Γ)，

\[
P=(I+B\Gamma B^T)^{-1}.
\]

设 f̃₀、f̃₁ 为 cheap means，e_i=f_i−f̃_i，δ_i 为 exact 与 cheap log likelihood 的差。定义

\[
s=\widetilde f_1-\widetilde f_0,\quad
e_-=e_1-e_0,\quad e_+=(e_1+e_0)/2,\quad
\bar r=y-(\widetilde f_1+\widetilde f_0)/2.
\]

直接展开得

\[
\boxed{
\delta_1-\delta_0
=\bar r^TPe_- -s^TPe_+-e_+^TPe_-.
}
\tag{T3a}
\]

**PROVED。** 这是材料分支 posterior odds 的误差，而非两个 current errors 的简单相加。q=Pr̄ 与 q=Ps 正是可交给 T2a 的目标。

**共同 offset 并非总能忽略。** 若 e₀=e₁=c，仍有 δ₁−δ₀=−sᵀPc；仅当 c 与有效 secant 正交，或在合法 flat-nuisance 极限中被完全消去，才不影响赔率。offset 校准可以大幅改善判别，但不能据此声称新恢复了结构可观测方向。

若 B、Γ、Σ 依赖材料，两分支 P_i 和 log det V_i 不同，上式必须改为各自完整的二次型加 determinant 差。双线性复 gains 的边缘化不因写了 Gaussian prior 就自动解析。

### 精确的材料后验输运

以 q(x) 表示 cheap posterior，x=(τ,z,γ)，δ(x)=log L_F−log L̃，p_t∝q exp(tδ)。在支配收敛等常规可积条件下，

\[
\boxed{
\mu_F-\mu_q=\int_0^1\operatorname{Cov}_{p_t}(\chi,\delta)\,dt.
}
\tag{T3b}
\]

**PROVED；属于 Bayesian linear response/score covariance 的已知原理。** 它说明大 forward error 可以几乎不改变材料均值，只要似然误差与材料差异几乎不相关；相反，弱但材料相关的误差也可能改变分支。

小误差下有 μ_F−μ_q≈Cov_q(χ,δ)，故 squared-loss excess Bayes risk 的第一非零项是 ||Cov_q(χ,δ)||²_W，而不是 ||e||² 或 Fisher trace。若 δ 只是后验支持上的常数，后验完全不变。

## I.7 从误差包络到材料决策证书

令 h=ess sup δ−ess inf δ，且精确与近似后验使用同一个 proper prior、覆盖同一支持。则

\[
\boxed{\operatorname{TV}(p_F,q)\le\tanh(h/4).}
\tag{T3c}
\]

这是 bounded exponential tilt 的尖锐概率界，不计作全新统计定理。可将 δ 减去任意常数，说明**应该认证相对似然差，而不是浪费成本认证共同常数误差**。

若材料支持在 W 范数下的直径为 D_W，posterior mean 是允许的材料行动，则

\[
\|\mu_F-\mu_q\|_W\le D_W\tanh(h/4),
\]

\[
\boxed{
\mathbb E_{p_F}[\|\chi-\mu_q\|_W^2-\|\chi-\mu_F\|_W^2]
\le D_W^2\tanh^2(h/4).
}
\tag{T3d}
\]

**CONDITIONALLY PROVED。** 后一式直接控制材料决策后悔值。它不保证 μ_F 接近某个实际 OOD 真值；这是相对于指定正确后验的计算误差证书。

对 topology 事件 A，|p_F(A)−q(A)|≤TV。若 cheap posterior 的第一、第二名概率差大于 2TV，最可能分支不会交换。MAP 参数点的胜出、分支总概率胜出和材料均值稳定是三个不同问题。

### 更紧的有限行动证书：不是只有一个 TV 大常数

若有限离散后验或已认证积分单元给出 q_i，且 δ_i∈[l_i,u_i]，定义

\[
\mathcal P=\{p:\exists t\ge0,\ \sum_i p_i=1,
\ q_i e^{l_i}t\le p_i\le q_i e^{u_i}t\}.
\]

对事先定义的有限材料行动 a∈A、损失 L_ai，

\[
\boxed{
U(a)=\max_{b\in A}\ \max_{p\in\mathcal P}
\sum_i p_i(L_{ai}-L_{bi}).
}
\tag{T3e}
\]

每个内层是线性规划，U(a) 恰为该误差集合中行动 a 的最坏后悔值。以状态误差区间、材料距离和 posterior mass 共同决定是否值得追加 F，避免“最大残差优先”。

**条件边界。** 有限粒子上的精确 LP 不等于连续材料后验的严格证书。连续情形还要计 cell 内材料变化、积分误差、未覆盖 support/tail 和漏掉的拓扑质量。普通 SMC 的 ESS 或若干随机 samples 不能代替这些证明。有限行动也不是 posterior support：允许 broad/adaptive continuous prior，而只把最后的少量报告行动做离散决策比较。

**独立创新裁决。** T1–T3 构成值得检验的物理—概率接口；其基本代数、coercivity、TV、LP 部件均不是新领域。潜在论文贡献必须来自：Maxwell 可支付的目标证书、校准后仍有意义的材料证书、以及实际比基线便宜的同风险计算政策。单独把已知公式排列在一起不够。

# II. Physics Prior Design

## II.1 六种物理关系不能混为一谈

统一采用 e⁻ⁱωᵗ。频域 Maxwell 为

\[
\nabla\times E=i\omega\mu H,\qquad
\nabla\times H=J_{ext}-i\omega\epsilon E.
\]

导电损耗可合并进 ε_eff=ε+iσ/ω。对局域被动材料，Im_H ε_eff≽0、Im_H μ≽0。在有限区域 Ω 上，复 Poynting 恒等式给出

\[
\frac12\operatorname{Re}\int_{\partial\Omega}(E\times H^*)\cdot n
+\frac\omega2\int_\Omega
(E^*\operatorname{Im}_H\epsilon E+H^*\operatorname{Im}_H\mu H)
=-\frac12\operatorname{Re}\int_\Omega E\cdot J_{ext}^*.
\tag{9}
\]

右端按此符号约定定义为 impressed source 向场输入的净功率。没有区域内 impressed source 时，就是净流入等于吸收；存在端口时，改写为 P_in−P_out=P_abs。

| 关系 | 正确含义 | 不能推导什么 |
|---|---|---|
| lossless conservation | P_abs=0，完整边界输入输出相等 | 不是任意 receiver matrix 的酉性 |
| passive dissipation | P_abs≥0 | 不保证 material image 唯一 |
| Poynting balance | 输入、输出、内部耗散的局部/整体平衡 | 不等同于“散射功率守恒” |
| optical theorem | 适用散射背景下 extinction 与 forward amplitude/功率的关系 | 不能原样搬到一般 lossy host 的差分功率 |
| reciprocity | 对称本构条件下交换源与接收的双线性关系 | 不等于无损，也不使用共轭转置 |
| causality/dispersion | 时间响应因果性及多频解析结构 | 单频 Imε≥0 不足以证明整个频谱因果 |

互易性在 εᵀ=ε、μᵀ=μ 等相应条件下给出

\[
\int_{\partial\Omega}(E_a\times H_b-E_b\times H_a)\cdot n
=\int_\Omega(E_b\cdot J_a-E_a\cdot J_b).
\tag{10}
\]

有损但 reciprocal 的材料仍满足此关系；lossy scattering 不必 unitary，也不因 reciprocal 就在只观测场的系统中可作简单时间反演。含 bianisotropy 时应使用完整本构互易条件，而非仅检查一个 ε。

### 8 源/32 接收器实际可以使用什么

可以利用已知材料/几何的被动性、绝对材料范围、正确的 forward 残差，以及源和接收器真正可交换时的对应 reciprocity。若测到的是已校准的远场 forward amplitude，也可能由 optical theorem 获得 extinction；这需要明确的方向、归一化和相位，而不是“任意32个复数”。

不能直接把 ∑|y_rs|² 当成全部散射功率。有限点场采样通常既不是完整辐射通道，也不是 power-orthonormal ports。即使完整传播通道上的 scattering operator 是 unitary/contractive，evanescent 通道还涉及不同的通量关系；Byrnes–Foreman 已明确处理这种区别。[L7]

对于独立测得的端口反射/输入功率、完整或受控的辐射功率、热吸收等，能量关系可成为**新增数据**；对已经给出的8×32场值，它通常只是模型关系。一个人为规定的源场振幅也不自动等于材料变化下保持恒定的实际馈电功率。

## II.2 能量约束进入四个位置，只有一部分会新增材料信息

| 用法 | 真正减少的集合/误差 | 额外成本 | 与旧方法的区别 |
|---|---|---|---|
| A. material feasibility | 排除主动材料、不允许的色散、未知但可由独立知识确定的几何/材料 | 多为逐点或低维参数投影；通常0次F | 若只是 Imχ≥0，基本就是已有 physical box |
| B. current–field compatibility | 在任意 latent current 中排除违背 Maxwell 的场 | G/L matvec；若求精确场则一次F | 精确求解后能量自动成立，不应再当独立 likelihood |
| C. structure-preserving operator | 防止 surrogate 非物理增益、提供耗散范数稳定界 | pH/ROM setup、assembly、残差/dual成本 | 改善数值模型及证书，不是新实验信息 |
| D. probabilistic energy model | 若有独立物理材料先验，可约束 posterior；若仅重复方程则无新增信息 | evidence、校准边缘化、积分常比网络本身贵 | 必须区分 physical energy 与 negative log posterior |

直接设置 p(χ)∝exp[−β P_abs(χ)] 并没有普适的材料先验依据：它会偏向低损耗/低场响应，而不是“更正确的材料”。把 P_abs、储能、Bayesian energy、神经 EBM energy 都叫 energy，并不能建立它们之间的概率等价性。

### 两个材料都守能，但观测不可区分

**COUNTEREXAMPLE 1：精确一维 Maxwell slab。** 在单一 TEM 通道/正常入射下，令两块无损 slab 厚度同为真空波长 λ₀，折射率分别为 n=1、2，μ_r=1。两者内部传播相位分别为2π、4π，传输矩阵都为 I，故同频的反射和传输完全相同。两者 ε_r 分别为1与4，都满足 Maxwell 与能量平衡。这是指定单通道设置的精确反例，不是声称任意3D环形阵列都有同一个 slab 歧义。

**COUNTEREXAMPLE 2：当前有限数据维数。** 在24×24网格上，即使把所有损耗固定为严格正值，仅让576个实介电常数在一个开集变化，8×32复数据也只有512个实维度。若 forward 连续，则它不可能在整个576维开集上单射：否则把输出嵌入 R⁵⁷⁶ 后，invariance of domain 要求像集是开集，但它实际上落在512维子空间中。这说明即便全部候选都被动并严格满足能量方程，仍存在不可区分材料。它不证明某个特定 cavity 对不可区分，也不否定低维目标函数能被准确恢复。

## II.3 真正有内容的 dispersion prior

在声明的频率范围内，可以采用被动 Lorentz/Debye 等材料模型，例如

\[
\epsilon(\omega)=\epsilon_\infty+
\sum_\ell\frac{f_\ell}{\omega_\ell^2-\omega^2-i\gamma_\ell\omega},
\qquad f_\ell\ge0,\ \gamma_\ell\ge0.
\tag{11}
\]

这把多频材料自由度耦合起来，与逐频独立的 box/TV 不同；其价值来自**已知的材料色散类别**和多频数据，而非单频神奇增强。有限频带上不能在没有尾部模型的情况下完整施加 Kramers–Kronig。未知色散类型或错误共振项会带来 prior misspecification。

尤其要区分：χ 与 a 都被动，不代表 h=χ−a 被动。例如 Imχ=0.1、Ima=0.3，则 Imh=−0.2。这不表示材料有源。对 relative feedback 的 h 强加 Imh≥0，会错误删掉合法材料。

## II.4 Port-Hamiltonian 和 positive-real：有价值，但不是频域正定魔法

对非色散、实正定 ε、μ 的时域问题，可用状态 x=(D,B)，Hamiltonian

\[
\mathcal H(x)=\frac12\int(D^*\epsilon^{-1}D+B^*\mu^{-1}B).
\]

curl 耦合在保留正确边界端口后是 skew-adjoint；其能量平衡包含边界通量。合适 mixed discretization 后可写成

\[
M\dot x=(J-R)x+Bu,\quad y=B^Tx,
\quad M\succ0,\ J^T=-J,\ R\succeq0,
\]

\[
\dot{\mathcal H}=u^Ty-x^TRx.
\tag{12}
\]

含色散时需要引入极化/振子等辅助状态；不能直接把复杂 ε(ω) 塞进静态正定 Hamiltonian。频域 pencil `sM−J+R` 在 s=−iω 时也不成为普通 SPD 系统。

合同降阶保留 M_r、J_r、R_r 的结构。真正 power-normalized 的端口变量可以通过 Cayley 变换联系 positive-real impedance 与 contractive scattering。但这是**完整且合法的端口模型**，不自动适用于32个点场读数。

Gouzien 等的全文明确指出一般 Loewner SVD 压缩会丢失 passivity，需额外 spectral-zero interpolation；其示例先用了500个频率响应，且对正 feedthrough shift 做了灵敏度处理。其主结果是2D TE waveguide 的 forward ROM，并非材料风险保证。[L6] Benzaouia 等与 QNM-Net 已有有限模态物理约束结构，因此“把反馈改为 passive rational layer”不能单独当创新。[L8,L9]

## II.5 Gauge、校准与商空间

## 四种不唯一性

1. **真实势规范**：A→A+∇λ、φ→φ+iωλ，保持 E=iωA−∇φ、B 不变；不直接变换材料 χ。
2. **材料等价类**：不同材料在指定测量下具有同一 likelihood。
3. **仪器 nuisance symmetry**：gain、phase、clock、位置等同时作用于观测。
4. **表示冗余**：同一材料被多个 chart/latent code 表示。

只有后三种与材料推断的有效信息直接有关。给材料 latent 冠以 gauge-equivariant 的名字，并不能把第一种规范自由度变成材料先验。

### 自由可分复 gain 的可计算不变量

在非零、无噪声数据上，

\[
y_{rs}=g_rh_sF_{rs}(\chi),\qquad
I_{rr'ss'}=\frac{y_{rs}y_{r's'}}{y_{rs'}y_{r's}}
\]

消去可分 gains。完整 bipartite graph 的独立复不变量数为

\[
RS-(R+S-1)=(R-1)(S-1).
\]

8源32接收器对应217个复不变量，gain orbit 为39个复自由度。一般图是 `E−V+c`；森林没有 cycle invariants。零元素、低SNR比值与多分量极化需要分层/更稳健坐标，不能简单全取 logarithm。

与 co-polar interferometry 的 `G_a S_ab G_b*`、Hermitian 同阵列结构不同，这里是独立源/接收器二部图，不可照搬其 invariant count。其把仪器群称作 gauge 的方法已有文献，不是电磁势规范。[L19]

**噪声警告。** 对 `y=ghF+n` 且 n 是增益之后加入的固定方差噪声，作用于 y 的 gain 群也会缩放噪声。无噪声均值的 cross ratio 不变，不等于 noisy ratio 的完整分布与 gain 无关；其方差、偏差及SNR仍依赖 gain。因此首选 raw-data calibration-aware likelihood，而不是把 ratio 输入网络后假装得到了 iid Gaussian invariant measurements。

### 平移为什么会被吸收

均匀背景中的远场平移 t 满足

\[
F_{rs}(\chi(\cdot-t))=
 e^{ik(d_s-\widehat r_r)\cdot t}F_{rs}(\chi).
\tag{13}
\]

相位恰好可分为 source factor 与 receiver factor；自由 gains 因而可吸收绝对平移。这是所有散射阶的远场变换律，不只是 Born 一阶。近场一般不具有此严格关系。多频仅在 gain/clock 跨频受到约束时可能打破等价性；逐频自由 gains 或自由延迟仍可保留歧义。

### 有效材料信息

在白化实坐标中，材料 tangent 为 J_z，仪器 tangent 为 B。独立 Gaussian prior precision 为 Λ_z、Λ_γ 时，

\[
H_{eff}=\Lambda_z+J_z^T\left[I-B(B^TB+\Lambda_\gamma)^{-1}B^T\right]J_z.
\tag{14}
\]

Gaussian nuisance covariance Γ=Λ_γ⁻¹ 给出 P=(I+BΓBᵀ)⁻¹。flat limit 才成为正交 projection；有界补偿不是无界投影。联合 prior 有交叉相关时，应对完整联合 precision 做 Schur，而不是继续使用独立式。

“horizontal tangent”在这个线性层面就是该消元结构，没有额外新定理。只有全局非线性群作用、跨 chart 转换、正确噪声推前和商空间 measure 被处理，几何语言才可能带来超出 Jacobian projection 的结果。

### 架构取舍

- **invariant encoder** 去掉 gain orbit 内的信息，可能抗未标定仪器，但也永久丢掉已校准振幅和绝对位置；低SNR时统计模型更难。
- **equivariant calibration latent** 应把材料与仪器变换律分开，保留可通过 reference 恢复的信息；网络不应把 nuisance 当材料纹理。
- **quotient topology posterior** 对合法 nuisance 边缘化，在轨道重叠时报告多假设，而不是自动选择一个 chart。

Gauge fixing 只是移除坐标重复；Bayesian marginalization 积分保留 nuisance uncertainty。给某个源幅度设定真实已知值是额外校准信息，而不是免费 gauge fixing。

# III. Architecture Designs

## III.1 Architecture A — Maxwell Energy Posterior：正确的概率基础，但不单独推荐为新论文

设 χ=D_τ(z)，拓扑 prior π_τ、连续 prior p_τ(z)、proper calibration prior p_γ(γ)。联合模型最清楚的定义是

\[
\pi_\tau p_\tau(z)p_\gamma(\gamma)\,dz\,d\gamma\,
\delta_{j(\chi)}(dj)\,
\mathcal N(y;\mathcal S_\gamma j,\Sigma).
\tag{15}
\]

这里 Dirac 是确定性求解映射的条件概率，不是额外的“场先验”。被动材料指示函数放进 p_τ(z)。Maxwell 已作为硬约束时，适用的 Poynting balance 自动成立。

若改写成关于复 current 的 Lebesgue density，应为

\[
p(j\mid\chi)=|\det_{\mathbb C}L_\chi|^{2s}
\prod_{r=1}^s\delta_{\mathbb R}(L_\chi j_r-b_r).
\tag{16}
\]

s 是源数。遗漏 determinant 后，积分会人为乘上 |detL|⁻²ˢ，偏爱近奇异材料。一个复维度中 L=0.1 与 L=1，若无测量且材料 prior 等权，错误写法给出约99.01%对0.99%，正确权重仍50%对50%。**这不是物理共振先验，而是参考测度错误。**

有理由使用 soft physics 时，应明确 discrepancy 的来源。例如 η∼N(0,Q)、Lj=b+η，才得到规范化的 p(j|χ)=|det_R L|N(Lj−b;0,Q)。不能同时把 exact Maxwell 和同一 residual 当两份独立证据。数值误差可能跨材料、源和频率相关，不应随便按独立传感器噪声处理。

### 解析消元与 evidence

消去 j 只意味着采用 F(χ)，并不意味着免去 forward solve。若 calibration 在所用区域确实是仿射 Gaussian，则

\[
y\mid\chi\sim N(f_\chi+B_\chi m_\gamma,
V_\chi),\quad V_\chi=\Sigma+B_\chi\Gamma B_\chi^T,
\]

\[
E_\tau(z)=\tfrac12r_\chi^TV_\chi^{-1}r_\chi
+\tfrac12\log\det V_\chi-\log p_\tau(z)-\log\pi_\tau.
\tag{17}
\]

对真正 bilinear gains 要积分、联合求解或作有误差说明的局部近似，不能称“已解析消去”。

拓扑概率必须来自

\[
Z_\tau=\pi_\tau\int p_\tau(z)p_\gamma(\gamma)L(y\mid\tau,z,\gamma)\,dz\,d\gamma.
\tag{18}
\]

不同维度的 chart 不能比较 MAP peak。Laplace evidence 含 (2π)^{d/2}/sqrt(detH)，且只适用于内部、单峰、非退化局部区域；birth/death、边界和多模态处可能失效。

若 prior 原本定义在 latent Lebesgue measure 上，不应额外乱乘 geometry Jacobian；若 prior 定义为材料流形体积密度 ρ，则 latent density 才含 sqrt(det(TᵀWT))。重复 chart 还需要唯一参数化或 partition of unity，避免重复计数。

**训练和成本。** 可以完全无训练；材料 prior 可由独立材料样本建立，不必有全波标签。每个新的 χ 节点仍可能需要一次 LU+8 primal RHS，梯度另需8 adjoint RHS。多模态采样和 evidence 计算可能很贵。该架构的概率含义清楚，但单独的 hierarchical Bayesian shape/contrast + Maxwell 不具备足够新增贡献。

## III.2 保留主架构 C/A — 校准感知、多拓扑、决策目标认证

### 目标与表示

保留 A 的规范化 posterior，但用分层材料模型而非固定小 chart：

\[
\chi=\chi_b+\sum_{k=1}^{K_\tau}c_k m_{\tau,k}(z_{geom})+\Psi_{\tau}u.
\tag{19}
\]

m 为 cell-average 或可微 occupancy，c 为共享于各源的复材料参数，u 为可扩展细尺度系数。固定分支中的 u 需遵守该分支的界面/支撑约束；允许 birth/death、分裂、合并和新 chart，而不是靠细尺度残差偷偷改变 τ 的定义。未建模材料应有实际参与推断的 broad component，不能只给一个名为“other”的空类别。

最低尺寸、最低 contrast 或色散范围只有在真实先验或明确的实验假设中才可采用；否则 topology 在特征消失处本来就不稳定。hard indicator 不在 L² 中产生普通可微 material tangent，应使用一致的 shape calculus 或 cell-average/smooth occupancy，并记录几何近似误差。[A30]

### 计算图

`raw y + calibration specification → broad prior / measurement-dependent proposal → continuous states within strata → cheap physical model → nuisance integration → signed likelihood / goal residual intervals → topology evidence + material actions → risk/cost gate → selected full F / derivative refresh → updated posterior`。

proposal 可以用 BP/N3、优化得到的模态、现成流模型；其重要性权重必须为 `prior × likelihood / proposal(y)`。不能把由 y 产生的 proposal 又当作独立 unconditional prior。原 A29 bank 只作失败对照，不作为主架构的 posterior support。

### 更新规则

1. 以 cheap/corrected-L 初始化多个材料分支；同一 prior、box、decoder 和优化规则用于所有物理模型。
2. 在每个分支内执行 trust-region/Laplace 或 SMC；拓扑跳跃使用明确的 proposal ratio、维数变化 Jacobian 与 prior mass。
3. 计算 Eq.17，包括 material-dependent calibration covariance 的 logdet。
4. 用 T1 判断是 offset、derivative 还是 propagation 错误；用 T2a 只认证决策目标，用 T2b 检验补偿后分离。
5. 输出 topology probability intervals、within-topology covariance、posterior mean、constrained Bayes action，以及尚未覆盖的质量。
6. 根据 T3 的 posterior/decision intervals 选择是否追加 full F；不能把 interval 宽度当成“保证有收益”。

### 后验均值会填平空腔

若空腔与实心两个解释都占显著概率，pixel-wise mean 会把空腔变成中间介电常数。它可能仍是 squared-loss 最优、也仍在 convex physical box 内，却不代表一个最可能的拓扑。应同时给出分支条件图像与概率；要求合法结构时用 constrained Bayes action。

在多分支混合中，材料风险含

\[
R_{post}=\sum_\tau p_\tau\operatorname{tr}(W C_{\chi\mid\tau})
+\sum_\tau p_\tau\|\mu_\tau-\mu\|_W^2.
\tag{20}
\]

第二项不能被单个 GN covariance 代替。

### 哪些解析、哪些学习

物理 current 唯一求解、局部 Gaussian nuisance 积分、低秩 covariance 更新、finite-action robust regret 可解析/确定性计算；复杂形状 prior 与 proposal 可以学习，但不是必须。首轮不训练网络，也不预测 arbitrary current。

训练若使用，只允许 TRAIN 材料图像、物理允许的 calibration 统计和独立模拟数据；历史 development 真值不能继续调 predictor 后改称独立验证。

### 成本与失败

每个 cheap state 要计 reduced solve、state-dependent nuisance matrix、logdet、quadrature/SMC、证书和行动优化。新增 full F 计一次物理状态 LU+8 RHS；Jacobian 可选32 receiver adjoint 或材料方向的8d RHS。主架构的原创新颖性只可能在 T1–T3 的**材料决策目标与实测成本优势**，不是“多拓扑 + Bayes”的组合名称。

主要失败：prior/representation 不覆盖；多模态采样漏支；gain 模型错误；能量证书无用；证书/积分贵于 full F；高对比细尺度导致 chart curvature 太大；没有可验证的 posterior tail；actual material risk 与 predictor 在新家族上失配。

## III.3 保留备选 B — Passive Feedback Core + 显式数值不确定性

这里不把 h=χ−a 当成独立 passive medium，也不强行令 (I−K_h) 具有正定 Hamiltonian。应在**绝对物理阻抗/合法 pH realization** 上做降阶：

\[
Z_r(\chi,\omega)=V^*Z_\chi(\omega)V,
\quad -\operatorname{Im}_HZ_r=V^*D_\chi V\succeq0,
\]

或使用含色散辅助状态的

\[
M_r\dot x=(J_r-R_r)x+B_ru.
\]

相对源 anchor 用于去掉已知背景响应；rational core 保留反馈传播；材料相关少量 correction 只能进入允许的结构参数，例如 PSD R_r 的 Cholesky 参数、skew J_r、合法 Lorentz 参数。forward 与 adjoint 必须绑定同一模型。

数值误差通过 residual-based deterministic intervals 或明确的 correlated probabilistic discrepancy 建模。若设 f= f̃+U_eη，η 的 prior、均值偏差与跨候选相关性必须校准；加入 C_num 后 likelihood 包含 logdet(Σ+C_num)，且 full refinement 应使数值不确定性收缩。误差 bound 本身并不唯一决定概率分布。

**训练：不需要。** 首轮以 structure-preserving projection/插值和解析 error control 实现；学习至多预测少量可解释参数或 proposal，不预测全 current。

**在线成本：** 一般参数装配 O(nr²)、r 阶 factorization、8源与dual RHS、接收输出和证书；不能只报 r³。材料多变、basis 跨拓扑不适用或要频繁重建时，setup 和 memory 可能超过 ordinary adjoint。χ=0 时需采用场变量、活跃支撑消元或正确的混合形式，不能直接逆 χ 后加数值抖动。

**保留条件：** 多频/多图像摊销显著；真实耗散结构已验证；同风险总成本小于 calibrated-L/普通 ROM；比单纯 pH/QNM 已有工作额外得到材料风险控制。否则不进入大型训练。

# IV. Performance Prediction Theory

## IV.1 先把五种风险分开

| 层次 | 定义/含义 | 需要什么证据 |
|---|---|---|
| fundamental Bayes/minimax | 给定物理数据与材料类，本来就无法消除的风险 | 指定 prior 或 minimax class、两点/多点下界 |
| representation-limited | 限定行动/表示集合后可达到的最好风险 | coverage、projection bias、几何曲率 |
| algorithmic | 某个优化/采样过程实际达成的风险 | 收敛/积分误差、漏模态、初始化、实际 paired outcome |
| numerical-model excess | F̃ 代替 F 所增加的决策损失 | T1–T3、correct likelihood discrepancy |
| task risk | 完整材料、空腔、边界、平均 ε、吸收率等不同目标 | 明确固定的损失与度量，不可混用 |

本报告优先使用固定 W 下的 squared material loss。NRMSE、SSIM、edge F1、topology 0–1 loss 单独报告；它们没有自动共享同一个 Bayes action 或风险公式。特别是 truth-dependent NRMSE denominator 会改变决策目标，不能直接套 tr(WC)。

## IV.2 比三角不等式更有内容的 representation 分解

若允许的行动集合 M 为仿射子空间，p=Π_Mχ*，则对 a∈M 有严格 Pythagoras：

\[
\|a-\chi^*\|_W^2=e_M(\chi^*)^2+\|a-p\|_W^2.
\tag{21}
\]

若 M 是远离边界/奇异点、reach 为 ρ>0 的光滑子流形，唯一 projection 距离 e<ρ，则 normal–chord 几何给出

\[
e^2+(1-e/\rho)\|a-p\|_W^2
\le\|a-\chi^*\|_W^2
\le e^2+(1+e/\rho)\|a-p\|_W^2.
\tag{22}
\]

**CONDITIONALLY PROVED，使用已知 positive-reach 几何。** 对 n=χ*−p，用 |〈n,a−p〉|≤e||a−p||²/(2ρ)，展开平方即可。拓扑变化、边界、退化 contrast 附近 reach 可为0，不能使用此局部保证。

该界把 coverage error 与 inference error 分开，并显示曲率如何把它们耦合。e_M 是真值相关量，不能在线假装已知；只能由独立 prior-predictive/held-out 审计或物理 class 上的最坏界获得。

如果只把 prior support 限制在 M，但允许输出 posterior mean，则行动可能位于 conv(M) 而非 M。此时 hard Oracle coverage 不是该均值的风险下限。A29 的约2.15%因此不能直接成为 posterior averaging 的理论天花板，但也完全不能证明 averaging 会解决真实 cavity。

## IV.3 局部 Gaussian 风险与非线性/多模态修正

在局部正确模型中，

\[
C_z=(C_0^{-1}+J_z^TP_\gamma J_z)^{-1},\qquad
R_{local}=\operatorname{tr}(WT C_zT^T).
\tag{23}
\]

这必须带有物理材料度量 TᵀWT、proper prior、正确白化和 nuisance Schur。单独看 J 的 condition number、rank 或 trace 不够；QoI-optimal covariance/mean approximation 已有 goal-oriented Bayesian inverse problem 理论。[L14]

非线性时，可在 posterior credible region Ω 内比较真实 log density 与局部二次近似。若误差绝对值≤κ，且真实/近似尾部质量分别≤η_p、η_q，则在 Ω 上使用 T3c，有整体 TV 的保守界

\[
\operatorname{TV}(p,q)\le
\min\{1,\tanh(\kappa/2)+\eta_p+\eta_q\}.
\tag{24}
\]

κ 可以由三阶/四阶 remainder、decoder 曲率和 T1 的 derivative defect 给出；但这些常数本身要计算，不能用一个未经证明的 Taylor radius 代替。弱可观测方向使 posterior 展宽，正是局部 Gaussian 常失效的位置。多拓扑还必须加 Eq.20 的 between-topology covariance。

非Gaussian likelihood 下，posterior tilt 与 robust regret 仍成立；需要替换的是 δ 的具体误差界。第一阶项由真实 likelihood 的 score 对 e 的作用决定，而不一定是 rᵀΣ⁻¹e。

## IV.4 两点材料风险下界：先识别确实不可能恢复的结构

取两个可行材料 χ₀、χ₁，W 距离 D；相同 Gaussian covariance 下的白化 mean 距离为 d。等 prior 的最佳二点错误率为 P_e=Φ(−d/2)。若 π(Y) 为一个分支的 posterior probability，平方材料 Bayes 风险为

\[
R_{2pt}=D^2\mathbb E[\pi(1-\pi)].
\]

因为 π(1−π)≥min(π,1−π)/2，

\[
\boxed{R_{minimax}\ge R_{2pt}\ge\frac{D^2}{2}\Phi(-d/2).}
\tag{25}
\]

**PROVED，Le Cam/binary Bayes 原理的一个直接形式。** 在 d=0 时得到 D²/4，恰为不可区分二点的均值风险。用任务距离 D_task 替换 D，则只约束该任务，不必约束整幅图。

要认证“某结构无法可靠区分”，应找**可行补偿后的近似相同响应对**，并用误差界得到真实 d 的上界；T2b 的距离下界用于证明确实存在分离，是另一方向。对于复合拓扑类，Φ(−d_comp/2) 不能直接宣称是完整 Bayes classification error，因其还包含 nuisance prior、参数体积和多模态。

小空腔的有限体积 signature 在均匀场/稳定 resolvent 条件下可为 O(v)，而 material L² difference 为 O(√v)，因此无最小尺度的类通常不具备统一可观测常数。近共振会破坏此处的统一常数假设，不能一边假设 resolvent 有界一边宣称结论覆盖任意共振。

## IV.5 prior misspecification 与实际算法风险

若真实 prior 与采用的 prior 在共同支持上有已知、有限的 log-density-ratio oscillation，它可与 likelihood defect 合并进入 T3；这是有限且明确的 robust Bayes 情形。若真实材料在 prior support 之外，或数据对某方向完全无感，任何 posterior covariance 都可能严重失准。

**决定性反例。** y 只依赖 z₁，prior 令 z₂=0，而真实 z₂=1。无论把 F 解得多精确，posterior 对 z₂ 仍可零方差且完全错误。这同时推翻“物理一致性自动校准 OOD uncertainty”和“再算几次 full F 总能纠正材料”的说法。

优化方面，MAP objective gap 不等于材料误差。只有在明确的局部强凸性/受限可观测性条件下，目标 gap 才能转换成参数误差；该条件在 topology branch 或 flat nuisance 上可能不存在。对 squared-loss Bayes action，更直接的 identity 是

\[
R_p(a)-R_p(\mu_p)=\|a-\mu_p\|_W^2.
\tag{26}
\]

因此应同时审计 posterior approximation、数值物理误差和最终行动是否正确计算，不能用“优化收敛”包办三者。

## IV.6 如何判断 full F 值不值得支付

### 1. 点似然、MAP 与 topology

若白化 model defect 满足 ||e(χ)||≤b(χ)，以 cheap residual r̃=y−F̃(χ)，则

\[
|\log L_F-\log L_{\widetilde F}|
\le\|\widetilde r\|b+\tfrac12b^2=:B(\chi).
\tag{27}
\]

这是 A30 已有的基础界。两个离散候选的 cheap log-posterior gap 大于 B_i+B_j，才可认证该二者 winner 不变。连续 MAP 需要整个相关区域的界和优化 gap；topology probability 需要积分，不是比较两个 MAP peak。

对分支 τ，若证据修正 δ_τ∈[l_τ,u_τ]，其精确概率区间为

\[
p^-_\tau=\frac{q_\tau e^{l_\tau}}{q_\tau e^{l_\tau}+\sum_{k\ne\tau}q_k e^{u_k}},\quad
p^+_\tau=\frac{q_\tau e^{u_\tau}}{q_\tau e^{u_\tau}+\sum_{k\ne\tau}q_k e^{l_k}}.
\tag{28}
\]

关键是先获得合法的分支**证据**界；不能直接把分支 MAP 点的误差界拿来代替。

### 2. offset / derivative / propagation 的不同触发条件

- **只需 offset calibration**：误差主要为局部共同值偏差，且在当前 posterior region 上 derivative/finite-secant defect 已足够小。单个 F(a) 只能校正一点或常数项。
- **需要 derivative refresh**：T1 的 Dρ 项大，或 calibration 后 log-odds 仍对材料方向敏感。支付 JVP/adjoint，而非重复同一点 F。
- **需要 feedback/anchor 更新**：状态误差经 R M_h G 放大，旧 reference 的 relative perturbation 大，或 support/topology 改变使 reduced closure 不适用。
- **不值得算**：U(a) 已小，剩余误差只改变 nuisance/共同 likelihood 常数，或主要风险来自未覆盖材料与实验不可辨识性。

这些触发是对证据类型的要求，不是仅靠几个经验 threshold 自动作出的保证。

### 3. 增加 F 计算不等于增加实验数据

实验 information I(χ;y) 不因再解一次 Maxwell 而增加。full F 是减少我们对 forward computation 的不确定性，或修复错误 likelihood。

若另外采用一个**明确且校准过的 probabilistic numerical model**，令 D 为已有计算记录，Z_c=F(χ_c) 为一次待揭示的确定性计算结果，则对无约束 squared-loss Bayes action，基于该数值模型的预期计算收益为

\[
\boxed{
\operatorname{EVC}(c)=
\mathbb E_{Z_c\mid D,y}\|\mu_{D,Z_c}-\mu_D\|_W^2.
}
\tag{29}
\]

它来自 conditional total variance，是减少数值认知不确定性的 value of computation，不是新实验的 value of information。概率数值方法和随机 forward posterior perturbation 已有基础文献。[L18,L17]

只有误差上界 b，并不足以给出 EVC 的正下界：真实误差可能恰好为零。因此正确的策略是

\[
\text{skip if maximum plausible material regret is below tolerance/cost;}
\]

\[
\text{pay only if calibrated predicted EVC exceeds measured cost, or a mandatory safety tolerance requires refinement.}
\]

严格 U(a) 是“最多还可能损失多少”的证书，不是“这次计算必然赚多少”的保证。U 大而缺少可靠 EVC 时，应报告 unresolved/budget-limited，而不是虚构收益。

### 4. 停止准则

停止可以表示“相对于指定 posterior，数值计算造成的材料行动 regret≤ε_dec”，也可以表示“物理/先验歧义仍大但 additional F 不会解决”。二者不能都写成 recovered。返回状态至少区分 CERTIFIED_NUMERICAL_ACTION、STATISTICAL_AMBIGUITY、PRIOR_COVERAGE_UNKNOWN、BUDGET_LIMIT、UNCERTIFIED_PHYSICS。

## IV.7 运行成本的事前预测

必须使用

\[
T_{total}=T_{setup}+T_{encoder}+T_{proposal}+T_{certificate}
+T_{F/adj}+T_{update}+T_{fallback}+T_{transfer}.
\tag{30}
\]

实际需要预测的是一个操作计数向量和目标硬件上的 primitive timing，而不是只算 FLOPs。

| 操作 | 稠密 n-current 模型的主成本 | 8源32接收器时要注意 |
|---|---|---|
| 一个新材料状态 F | complex LU O(n³)+O(sn²) | 一次 LU 通常复用8个 primal RHS；换材料一般要重分解 |
| 一个 scalar objective gradient | O(sn²) adjoint，若 LU 已有 | 通常再8个 source-specific adjoint RHS |
| 完整 material J | receiver-adjoint 或每方向 JVP | 可用32 receiver adjoint 与 source contraction；不要默认8p次 RHS |
| r 阶 ROM | setup+O(nr²) assembly+r³+sr² | 不能漏 material-dependent projection 与 lifting |
| T2 证书 | primal/dual construction+Z/Z* matvec+加权范数 | matvec 与所有源/目标的数量都要记录 |
| calibration marginalization | B构造、small Schur、logdet | 不用F也可能占可观时间 |
| posterior inference | proposal、积分/SMC、图表与行动优化 | 可能比物理本身更贵，尤其n=576 |
| fallback | 失败概率×完整重算与缓存失效 | 不能只报成功样本 |

24×24标量模型的一块 complex128 n×n 矩阵约5.06 MiB；实 J(512×1152)约4.50 MiB。1728个3D矢量单元对应 n=5184，一块 complex128 矩阵约410.06 MiB；G、L、LU及工作区同时存在时峰值远高于一块矩阵。

**事前预测表必须在解锁新 truth 前冻结：**

| 方法 | 预先填写的次数 | 时间预测 | 何时拒绝该路线 |
|---|---|---|---|
| calibrated-L | setup anchors、候选/迭代数、一次/多次offset费用 | 以测得 primitive 做 cold/warm 合计 | 作为不能删除的强基线 |
| ordinary full-adjoint | 每状态LU/8源+adjoint、迭代及失败概率 | 同硬件、同 stopping rule | 作为 exact 物理基线 |
| proposed decision-certified | cheap eval、目标证书数、预计 full/refinement 数、积分费用 | 给区间，不报未经运行的秒数 | certificate+posterior overhead≥预期省下的F费用 |
| passive ROM backup | basis/setup、material assembly、freq/image复用数 | Tsetup/Nimages+Ton | break-even 不发生，或需要频繁重建 |

warm amortized time 可写 E[T]=Tsetup/N_img+Tcheap+N_cert t_cert+E[N_F]t_F+p_fail T_fallback。分位数和 peak memory 同时记录。物理 cache key 至少包含材料、频率、几何、源、边界、polarizability/self convention、精度和归一化。

A17 efficiency 的实际证据已说明，只削减 Jd 检查不一定触及主要时间；anchor/candidate/workspace 重建才可能主导。应首先测量，而不是预设“认证最贵”。[A17-efficiency]

## IV.8 检验 predictor 本身

| 输出 | 与什么比较 | 保证层级 |
|---|---|---|
| predicted material risk | 新 parent / 重复噪声下 realized squared material loss | Gaussian/local、经验或严格模型风险需分别标记 |
| predicted F gain | 同一初始材料、同一约束/优化规则的 cheap vs exact paired结果 | 不得用残差 gain 替代 |
| U(a) / posterior intervals | 独立高精度 posterior/reference 与所有 accounting terms | 只有支持/积分/物理误差均被控制时才是严格证书 |
| certificate violations | 真误差是否越过 numerical bound | 浮点残差检查不是 interval-arithmetic proof |
| false acceptance | skip之后 exact-posterior regret 超过容许阈值 | 与某个单次真值的偶然误差增减不同 |
| false rejection | 本可安全skip却支付F | 决定实际效率 |
| OOD | 新形状、contrast、loss、calibration、feature scale 分组 | 不能只看总平均 |
| predictor overhead | 构造证书与预测所花全部时间 | 必须扣除后仍有正净收益 |

冻结 prior、loss、threshold、预算、decoder 与 predictor；开发只用 TRAIN/历史 exposed 数据。新的评估按 parent 分组，不能把同一 parent 的两背景、多噪声、多个切片当独立样本制造窄置信区间。家族最差表现单列。

正确区分四种标签：RIGOROUS_ANALYTIC_BOUND、LOCAL_GAUSSIAN_APPROXIMATION、EMPIRICAL_INTERVAL、HEURISTIC。若只有有限 samples 上的 physics certificate，标签应为 CONDITIONAL_ON_QUADRATURE；不能写 GLOBAL_POSTERIOR_CERTIFIED。

有限试验的零违例只表明没发现实现问题，不证明普适可靠。已有 development 真值反复参与调整后，最多报告 reanalysis，不得称 independent validation。

# V. Minimal Decisive Experiments：最多三个最小实验

以下是下一轮 specification，不是本轮已获得的成像结果。三个实验按信息价值排序；后续阶段必须由前一阶段的机器可读 gate 解锁。禁止启动大型 NN 训练，也不把更多 current modes 或更大有限候选 bank 当作失败后的默认补救。

## V.1 E1：原生物理能量与决策缺陷核验

**回答的问题：** 当前真实离散算子能否提供合法且不空泛的耗散度量？T1 漏项和 T2/T3 的组合能否在不调用全部 full resolvent 的情况下预测材料决策误差？

本轮随包已有 22 项 checks，其中通用矩阵、解析平板和辐射修正的 9 维矢量偶极模型严格分列。下一步优先读取 A24/A28 已付费 cache 和 A17/A23 原生 polarizability/self-term 实现，核对单位、体积、时间因子、radiative self convention 与实际 loss。不能把任意正定矩阵命名成 Maxwell energy。

首先检查 Z 形式的功率 Gram 恒等式；在源相关场、共享材料条件下同时核对吸收与辐射。对原生 self-cell 或 CM/RR 极化率，辐射反作用只能计一次。若某个离散模型没有足够信息恢复物理功率形式，输出 ENERGY_FORM_NOT_ESTABLISHED，不修改求解器制造通过。

随后用少量冻结材料方向核对 T1 的两个缺陷通道，比较只保留 frozen-injection 项和完整导数。构造三种有明确作用的扰动：主要改变共享 offset、主要改变 nuisance 可吸收分量、主要改变材料分支 log odds。比较普通全数据误差界与 decision-relative 界；用同一真响应离线审核界，不把离线真响应用于在线 certificate 构造。

预先包含以下反例：loss 趋零、非正规 K、近共振、零 contrast 支持、遗漏辐射自项、一般 oblique projection、错误的共享源材料处理，以及 nuisance 完全吸收结构割线。正例不是唯一需要通过的样本；反例被正确拒绝也是验收内容。

**预算与 gate：** 原则上新增真实 Maxwell 状态为 0，只读缓存；缺 cache 即停止相应 native test。通用矩阵可用 CPU。统计 certificate 的构造、dual RHS 与内存费用，形成后续成本预测表。需要实际验证 D0 合法、所有报告的严格界无已知违例，并且存在材料相关目标方向的非空界。只有普通 norm 界 PASS 而材料决策界始终空泛，不通过主机制 gate。

## V.2 E2：小型材料推断，而非残差改善实验

**回答的问题：** 表示有足够容量时，补偿后的非线性物理是否真的改变材料推断？其收益能否由预先给出的 risk predictor 解释？

使用 24×24、8源/32接收器的原 2D TM 后端，先运行不带 NN 的多拓扑连续材料表示。不是把 A29 的 58–91 张有限图像重新排序。最低实现包含连续外边界、一个可出生/消失的空腔、分区独立复 contrast，以及受到先验控制的细尺度余项；更一般结构通过明示的 broad component/模型不足标志处理，不声称这个最小 pilot 覆盖所有复杂对象。

预先生成至多 8 个新 parent，包含空腔、非对称边界、非均匀复材料和一个超出主表示的负控制。训练/先验规则、尺度、噪声与标定模型先冻结，再生成独立评价样本。公开真值文件只供标签产生与离线评价；在线 proposal、action policy 与 certificate 不得读取。所有算法使用相同表示、先验、初始化和推断近似精度。

核心对照为：single-anchor calibrated-L；ordinary full-state adjoint/likelihood；主架构的 decision-certified inference。另用无需新物理的 ablation 将 log-likelihood defect 的完整值、共享偏差、补偿投影和 signed relative defect 分开。full F 与 ordinary full adjoint 不人为算成两种不同物理方法。标定已知与有 proper prior 的 nuisance 分别运行；free gains 作为明示反例，不据其结果重新调 prior。

**主指标：** 固定质量度量的 squared material error、topology loss、区内实虚部误差；Δχ NRMSE、edge/IoU 为连续性指标，不能混同不同损失的 Bayes action。记录 posterior within/between-topology variance、predictor 的预期材料收益及其 realized gain。所有真值 oracle 仅用于离线确认表示容量，不是在线推断输入。

**硬预算：** 至多 128 次新增真实 material-state evaluations，包含 clean labels、anchors、测试 fixture、有限差分、失败、重试和计时。RHS 另计，CPU wall 硬上限由 E1 primitive profiling 写入配置；不启用 GPU 训练。每次实际调用都先预约，不能按 unique χ 掩盖重复费用。

**决定性否证：** 若表示容量足够、后验积分误差得到控制，而 full F 相对 calibrated-L 在材料风险上仍无有意义收益，则不能继续声称 feedback 提供必要的材料收益。若 predictor 总把大 residual reduction 错判为材料收益，否决 predictor。若表征容量不足，停止物理选择并标为 REPRESENTATION_LIMITED，不归咎 Maxwell 信息。8 parents 只支持 pilot 级判断，不制造确认性统计显著性。

## V.3 E3：冻结策略后的独立预测与计算验收

**回答的问题：** 能否在看到真值之前预测 material risk、是否值得 full F，以及完整 time-to-image？

只有 E1/E2 达到可行性 gate 才开启。冻结材料 prior、表示规则、误差模型、loss、阈值、求解终止条件、cost conversion 和 EVC/skip policy。另取未参与开发的新 parent；计划至多 16 个，按 parent 而非噪声重复计独立样本。预设 moderate-loss ID、低损耗/高 contrast、nuisance misspecification、细尺度/拓扑 OOD 子组。

对每个在线 query，先写入预测的 posterior/action risk 区间、F 的潜在收益、证书类型与 primitive cost；之后才执行被选择的动作，最终再由独立 evaluator 读真值。比较 always-F、calibrated-L 和 adaptive-F。在相同 inverse algorithm 上分别替换物理计算 policy，避免把不同优化轨迹/先验同时改变后归因于 certificate。

若使用已存响应模拟 lazy query，必须标为 OFFLINE_POLICY_REPLAY；没有实际 timed end-to-end run 就不能称在线加速。reference posterior 若只得到有限 SMC/quadrature 近似，单列其误差或不确定性，不能将它当无误差真后验。

**硬预算：** 至多 256 次额外真实状态，与 E2 合计不超过 384；不足以完成预定精度时输出 BUDGET_LIMIT，不自动增加预算。所有 full/adjoint/RHS、posterior积分、certificate、CPU–GPU传输、fallback 与 setup 计入时间。禁止用更低精度 reference 为 proposed 制造优势。

报告 predicted/realized material risk 与 gain、区间 coverage、analytic certificate violations、false skip/false pay、各 family 最差值、OOD 失效和 predictor overhead。工程可行性门槛可预设为：达到与 always-F 相同的声明数值决策容差时，总时间至少降低 20%；或在相同完整时间预算下材料风险出现预先声明的有意义下降。这里的 20% 是投入筛选约定，不是数学常数。若 F 本身不优于强 calibrated-L，应以更便宜的 calibrated-L 作为部署结论，而不是保护复杂方案。

# VI. Paper-Level Novelty Verdict

## VI.1 文献竞争：哪些空间已经有人占据

“全文”表示已读取正文或作者全文/全文分析，不等于独立复现全部实验。PDF 图像截图请求在本环境失败，因此不声称逐图视觉核验；正文公式、作者页面和可读取的 HTML 为主要依据。以下每篇论文的衍生描述保持在其实际读取范围内。

| 文献 | 已覆盖的机制 | 本轮读取程度与对 A31 的限制 |
|---|---|---|
| SOM；TSOM；FFT-TSOM [L1–L3] | receiver-derived deterministic/ambiguous current；内部传播的第二重处理；Fourier current 表示和求解效率 | 原始出版记录/摘要核对；三篇原文未取得。不能宣称穷尽它们全部定理，也不能把 A23 的 O/P/M seeds 直接视为经典两重子空间 |
| SOM-Net [L4] | learned current unrolling、解析共享材料更新、current/field/material 联合监督 | 全文。其物理结构不是 generic PnP；正文没有提供本报告所述的 calibrated material-decision posterior 证书 |
| Chen et al. certified harmonic Maxwell RB [L5] | primal/dual output approximation、严格 a posteriori output bounds、offline/online 与共振算例 | 出版社摘要及参考文献，未取得正文。已足以排除“首次 Maxwell 输出误差认证”的主张；具体定理范围仍有新颖性不确定性 |
| Gouzien et al. port-Hamiltonian Maxwell ROM [L6] | 2D 时域/传递系统的功率结构、Loewner/结构保持模型约化 | 作者全文。普通 Loewner 压缩不自动保被动；其结构处理和多频训练/查询成本不能被略去。不能把 passive ROM 本身认作 A31 原创 |
| Byrnes–Foreman vector scattering constraints [L7] | propagating/evanescent 通道上的能量、互易与时间反演约束 | 全文。完整功率通道的约束不等于任意 8×32 场采样矩阵的 unitarity |
| Benzaouia et al. constrained QNM expansion [L8] | 截断 QNM 展开中保留散射矩阵的物理约束 | 全文 HTML。主要 lossless 约束及吸收扩展有适用条件；不是任意强损耗闭包的无条件定理 |
| QNM-Net / knowledge integration framework [L9] | 学习少量模式/背景参数，再经物理 QNM 层生成谱 | 全文。结构化散射预测已有直接先例；本报告不将“physics layer + 小网络”视为新贡献 |
| Miller et al. optical response limits [L10] | 以能量守恒/耗散得到光学响应界，材料因子与损耗紧密相关 | 全文 HTML。耗散加权控制早已存在；新方向只能是把它用于补偿后的材料决策误差，而非重新命名光学界 |
| Deep Injective Prior [L11] | 材料-only learned prior、低维生成表示、Maxwell MAP 与变分 Gaussian UQ | 全文。不能以“材料流形+物理+后验”作新意；其受限生成像空间偏差与本项目覆盖问题直接竞争 |
| LaMP [L12] | 题名/出版记录直接涉及 latent manifold/PnP EM inference | 仅取得元数据，无全文；不能声称它缺少某个具体理论。保留新颖性不确定性 |
| L-DPnP [L13] | latent diffusion prior 与 nonlinear likelihood 交替采样、解码样本均值 | 全文。有限实现并未建立本任务的全支持材料决策认证；不能把 diffusion+physics+MMSE 当新贡献 |
| Spantini et al. goal-oriented Bayes ROM [L14] | 线性 Gaussian 下任务协方差与后验均值的最优低秩近似 | 全文。prior/noise/task covariance-aware reduction 属于已知理论；本轮 signed model-change risk 不是替代其定理的首创宣称 |
| Bayesian level set [L15] | 几何 prior、后验良定性、界面不确定性和 MCMC | 出版社摘要；支持“Bayesian shape+topology”并非空白。没有把其结果无条件转成固定 Maxwell 采样的唯一性 |
| Certified nonlinear Bayesian dimension reduction [L16] | 带假设的 posterior approximation/KL 误差控制 | 公开摘要/作者出版记录。避免“首次后验误差认证”主张 |
| Random forward-model posterior stability；Bayesian probabilistic numerical methods [L17–L18] | 近似求解器误差与后验稳定性；计算不确定性的概率建模 | 原始摘要/出版记录。误差模型和 adaptive computation 一般思想并非新；必须证明其在本任务中的材料决策效益 |
| Co-polar interferometric invariants [L19] | gain group、closure 不变量及其统计处理 | 全文。其同站点 g_a g_b* 模型与本项目 bipartite g_r h_s 模型不同；217 的计数是本报告对后者的独立推导，不挪用前者计数 |

最重要的区分是：**现有工作已经覆盖了几乎所有“模块名”；尚值得研究的是物理残差如何在 nuisance 消元后改变材料后验决策，以及能否用合法功率度量低成本界定这种改变。** 这是待建立的具体机制，不是以关键词组合替代理论。

## VI.2 A17：值得独立推进投稿，但范围必须准确

裁决：**GO for a focused standalone submission；不是对接收结果或“强 TAP 必收”的保证。**

A17 已有可辨认的物理问题与理论闭环：材料注入、retained feedback、receiver return、paired material update-space，以及固定 current dimension 的 verified conditional exchange。最新 closure 增加了受限的非线性和噪声证据，不能再把它描述成完全没有 nonlinear experiment。[A17-closure,A17-theory]

应保留其窄而清楚的贡献：specified current modification 对共同 frozen quadratic 的材料效用，以及受限三维矢量 Maxwell 实证。必须明确 verified numerical check 不是 deterministic output-error certificate；nonlinear improvement 不普适，end-to-end 仍更贵。效率 follow-up 的“99% frozen fidelity 但最终图像变差”是诚实且有价值的限制，不应删去。[A17-efficiency]

A31 不应该并入 A17 挽救成像/加速叙事，也不应将尚未证明的概率模型加入已封口主稿。A17 的投稿成熟度与 A31 新机制的实验资格分别裁决。

## VI.3 OPM：没有证据支持不可替代性

裁决：**不再把 OPM 作为必须保留的平台或论文核心品牌。**

A23 的 O/P/M 是具体 residual/source/material seed 生成与传播组织；source anchor 可使某个 forcing seed 消失，但不使实际物理 feedback 消失。A24 的 exact feedback 与普通 full-state adjoint 是同一物理映射，低秩命名不能制造独立材料信息。当前没有证据证明 OPM 比合适的 ordinary adjoint、calibrated-L、已认证 ROM 或不同 task basis 具有不可替代的同成本优势。[A23-code,A24-code]

可以保留 OPM 作为某个实现选项或消融对照，不将其设为 A31 先决条件。保留下来的是 Maxwell entry–propagation–return 与 residual/dual algebra，而不是固定三字母分解。

## VI.4 A31：何时足以成为强 TAP 贡献

**本轮结论：CONDITIONAL GO for mechanism/predictor development；NO established Strong-TAP performance claim。**

已经严格推导并核验的是：非线性模型缺陷的额外源场通道；功率结构的条件消元/目标误差界；相对 log-likelihood defect 对材料 Bayes action 的传递；有限后验集合中的 exact robust decision LP。部分属于已知基础定理的专门化，部分是本报告明确组合的命题。不能只把这几条写在一起就宣称首创强论文。

真正能支撑论文的新内容需要同时满足：

1. 原生 Maxwell 耗散/辐射结构产生可计算、明显优于普通全数据误差的 decision-relative 界；对强散射、nonnormality、近共振的边界处理真实有效。
2. nuisance 补偿和材料 prior 的作用被分离，能解释至少一类此前 residual/material 脱钩的真实机制，而不是只拟合历史错误。
3. predictor 在未见 parent 上预测材料风险/数值决策损失，并正确拒绝 prior OOD、不可辨识与证书失效。
4. 完整 wall-time 和材料质量至少形成一种明确的 Pareto 优势：不是仅减少 full-call 数或增加 Fisher trace。

若其中第1项失败，能量方向退为正确性检查；若第2项失败，补偿不是当前失败的主解释；若第3项失败，算法还不能称 predictive inference；若第4项失败，可能保留理论与负结果，但不能宣传部署改进。

# VII. One Recommended Research Direction

## VII.1 唯一主方向

**耗散认证的补偿感知材料决策推断（DC-MDI，Dissipation-Certified Material Decision Inference）。**

**核心科学假说。** 在材料表示覆盖足够、nuisance prior 合法且统计模型不过度失配的区域，真正决定 full Maxwell 计算价值的是“补偿后、跨材料状态的相对似然缺陷与材料损失的耦合”，而不是 current/field/data residual 误差。Maxwell 的真实耗散度量有机会以少量目标 dual actions 控制这一耦合，避免不必要的全精度物理计算。

这个假说包含两个可分别否证的部分：材料收益机制存在；其认证比直接求解更便宜。第一部分成立而第二部分失败，不能叫加速。第二部分成立但 full F 不优于 calibrated-L，不能叫物理增强材料恢复。

**最重要的定理。** 在 T2a 的真实 D0 下界、合法 nuisance marginalization、统一 log-likelihood-relative bounds 和后验积分/支持误差控制成立时，T3d/T3e 给出使用 cheap posterior action 相对 exact-model posterior Bayes action 的材料损失上界。它是 solver-induced decision regret 定理，明确不声称对未知真值的绝对误差证书。T1 保证在构造 derivative/curvature certificate 时不遗漏材料改变总场的通道。

**明确算法。** 连续多拓扑 prior + 原始数据的 calibration-aware likelihood + single-anchor cheap model；用 T1/T2 构造目标物理误差，在 T3 的 posterior-relative 层计算 material-action regret；足够安全时停止，不足时仅细化影响动作的物理状态/目标；必要时 re-anchor、更新导数或调用 full F。概率误差模型有经过验证的条件分布时才使用 EVC，否则只作保守数值容差决策。

**关键反例。** 若两个材料在 nuisance 后的测量分布相同，它们完全可以同时满足 Maxwell、Poynting 与被动性；任何更精确求解都不能恢复缺失信息。另一个独立反例是 loss 趋零导致 D0 失去强正性，此时耗散证书可能无用，即使 full Maxwell 实际仍可逆。不能用人为 loss floor 规避这两类反例。

**最小证据。** 本轮 22 checks 是代数前置；还需要 E1 原生能量/源场验证，E2 材料风险而非残差的正负对照，以及 E3 独立 predictor 与完整费用验收。三个层级不能相互替代。

**完整失败条件。** 能量变量/自项不合法；posterior 支持包含无法处理的零 contrast 或 lossless 状态；目标 dual certificate 持续空泛；near resonance/nonnormality 使稳定性成本过高；表示漏掉真结构；prior/gain/noise model OOD；有限积分看不到重要分支；free nuisance 吸收全部结构；代价主要在表示/积分而非 forward；fallback 频繁；calibrated-L 已达到同样材料风险。任一发生都应输出对应失败原因，不用新 NN 或 rank sweep 掩盖。

## VII.2 唯一备选

**多查询/多频 passive rational ROM，服务于同一个材料 posterior。** 只有主要困难明确是重复物理成本、且真实 passive realization 能跨足够多图像/频率摊销时推进。它不以相对 h 的被动性为约束，不将 ordinary Loewner 或任意复投影自动当被动。必须与现成 certified/goal-oriented ROM 和普通缓存 adjoint 比较。它是备选计算内核，不是另开一个大型网络方向。

## VII.3 给 Codex 的交付与执行边界

完整下一轮任务见 `CODEX_A31_NEXT.md`。先运行 CPU algebra checks 和 cached native-energy audit；禁止从导入测试隐式启动真实 solver。只有 manifest、能量约定、预算预约与 gate 都通过，才开启明确授权的小型 E2。E3 保持关闭，直至 E2 结果按预先规则解锁。没有任何默认自动扩展、自动公开仓库或后台 GPU 训练。

**最终裁决：继续研究 Maxwell feedback，但把它从“更好预测电流”的承诺，转成“解释和控制材料决策中哪些物理误差值得消除”的问题。这个方向有可证明的数学骨架；是否有足够便宜、足够紧、足够真实的 Maxwell 实现，仍是下一轮必须被实验决定的核心。**# 附录 A. 本轮核验的可复现结果

`a31_matrix_checks.py` 使用固定 seed=311011、NumPy/SciPy，在 CPU 上执行。`A31_MATRIX_CHECKS.json` 保存环境、每项结果与作用域。共22项 PASS 包含应当失败的无条件说法被成功反驳，不意味着22项新物理成像成果。

| 检查 | 结果 | 含义 |
|---|---:|---|
| 相对 feedback 恒等式 | 相对误差约1.44e−15 | 源 anchor 坐标变换一致 |
| A17 类型 Galerkin 双侧缺陷 | 约3.00e−16 | 需要 R_A L R_A=R_A；任意近似逆有额外修正 |
| 非线性 ROM 导数完整双缺陷 | 约3.31e−16 | 忽略 source-field 项时，此例相对缺陷约3.90% |
| signed covariance/risk 更新 | 协方差相对误差约1.44e−15 | 一个例子的 R0−R1=−0.0351，风险并非必降 |
| 错误前向的协方差反例 | C_A≈0.99010，实际MSE≈1.00782 | 错误模型的自报协方差可低于其真实材料风险 |
| 功率 Schur 合同关系 | 约2.79e−16 | 与真实耗散形式兼容的消元保功率 |
| oblique projection 反例 | 完整D=I；reduced dissipation=−9 | W*V=1 仍不够保证被动性 |
| 目标 dual 误差界 | 实际误差7.5179，界52.0904 | 有效但可能松；另有构造例达到等号 |
| nonnormal resolvent 反例 | ρ(K)=0.5；||(I−K)⁻¹||≈120.03 | 只用1/(1−ρ)=2不合法 |
| 32×8 gain 不变量 | 相对误差约1.93e−16 | 完整非零 bipartite 数据有217复不变量 |
| signed log-odds defect | 约1.11e−16 | 共享 offset 也可改变判别，不能盲目删除 |
| sharp posterior tilt bound | TV=0.3363755=tanh(1.4/4) | h/4 常数的两点极值检验 |
| robust finite-action LP | 上界1.34830；随机试探最大约0.89719 | LP 不被这些试探反例违反；不是全空间物理证书 |
| posterior mean transport | 约1.22e−16 | Cov(χ,δ)控制均值随似然的变化 |
| Dirac 归一化反例 | 错误权重0.9901/0.0099；正确0.5/0.5 | 漏掉det可制造虚假先验 |
| 解析平板材料不可区分 | 两传递矩阵差约5.20e−16 | εr=1与4在指定单频通道可相同；不是原ring geometry |
| goal-only 补偿分离 | 分离下界2，完整误差100 | 大的任务无关输出误差可以不影响此决策 |
| 混合后验的 cavity 均值 | 均值risk0.25，硬填充risk0.5 | 平方材料损失与拓扑行动不同 |
| positive-reach 表示界 | 真实loss0.76790位于所证区间 | 需要reach/投影条件，不能推广到stratum奇异边界 |
| 9维辐射矢量偶极功率 | Gram相对误差约6.55e−18；辐射Gram最小特征值约6.09e−4 | 真实dyadic点偶极方程的功率检查，与通用随机矩阵分列 |

完整22项记录见JSON；表中合并了同一机制的若干检查。实际运行包含2个解析平板材料例、1个9维辐射矢量偶极状态/2个源RHS；没有网格Maxwell求解、GPU调用、训练或用户真值数组读取。时间字段只描述该脚本本次执行，不是所提算法time-to-image预测。

# 附录 B. 来源与版本

## B.1 项目原始证据

下列记录均为原始仓库或用户 A30 文件。核验只涉及已列出的理论、实现和报告；没有宣称逐行重审全部历史代码。报告里的历史数字由这些原报告提供，本轮未重算原始全批量结果。

- **[A17-closure]** `migodam/a17-maxwell-certified-exchange`，commit `df61a1b533682a883cdaf65259b4528c354bd832`，`closure_final/START_HERE.md`。
- **[A17-theory]** 同仓库/commit，`closure_final/manuscript/CORE_THEORY.tex`、`PAPER_TAP_FINAL.tex`。
- **[A17-efficiency]** 同仓库，`efficiency_followup/GPT_REVIEW_CONTEXT.md`；原效率发布标记 `a17-efficient-verification-2026-10-04`。原报告明确其不是独立新holdout campaign。
- **[A23] / [A23-code]** `migodam/a23-fast-material-imaging`，commit `a47cda424dfb55792e69d952a25f54ce9a2a5613`，`README.md`、`src/a20/opm.py`、`docs/BACKEND_MAP.md`。
- **[A24] / [A24-code]** `migodam/a24-source-anchored-om-feedback`，commit `ca80ca2a29867e1b03589061dd5c17688c9acbe0`，`README.md`、`src/a24/relative_feedback.py`。读取了 source anchor、low-rank/ROM构造及 forward/JVP/adjoint 实现。
- **[A28]** `migodam/a28-maxwell-material-information`，commit `c7819fa142bb70f9920835e8e91ba145f15ed8e3`，`E1_DECISION.md`、`src/a28/pipeline.py` 的缓存/审计/更新相关部分与事后bias诊断说明。
- **[A29]** `migodam/a29-material-hypothesis-coverage`，commit `5d5e31fcfc56ac8ef9d64c46b4b7bbdedef96bb4`，`A29_DECISION.md`、`src/a29/proposals.py` 的TRAIN-only规则和材料候选部分。
- **[A30]** 用户Library内 `A30_THEORY_REPORT_ZH.md`（2026-10-10）、`A30_MINIMAL_PROTOCOL.md`。完整理论报告已读取；其原始A29全数组缺口在本轮没有被补齐。
- **[A17–A30]** 为上面项目证据的集合引用，不表示访问了未列明的所有历史仓库。

## B.2 文献

以下DOI/arXiv足以定位原文。全文可得性与审阅范围见 VI.1；本轮不将未知全文内容作否定性新颖性证据。

[L1] X. Chen. Subspace-Based Optimization Method for Solving Inverse-Scattering Problems. IEEE TGRS, 48(1), 42–49, 2010. DOI: `10.1109/TGRS.2009.2025122`.

[L2] Y. Zhong and X. Chen. Twofold subspace-based optimization method for solving inverse scattering problems. Inverse Problems, 25, 085003, 2009. DOI: `10.1088/0266-5611/25/8/085003`.

[L3] Y. Zhong and X. Chen. An FFT Twofold Subspace-Based Optimization Method for Solving Electromagnetic Inverse Scattering Problems. IEEE TAP, 59(3), 914–927, 2011. DOI: `10.1109/TAP.2010.2103027`.

[L4] SOM-Net: Unrolling the Subspace-Based Optimization for Solving Full-Wave Inverse Scattering Problems. IEEE TGRS, 60, 2022. DOI: `10.1109/TGRS.2022.3217162`; arXiv: `2209.03567`.

[L5] Y. Chen, J. S. Hesthaven, Y. Maday and J. Rodríguez. Certified Reduced Basis Methods and Output Bounds for the Harmonic Maxwell's Equations. SIAM J. Sci. Comput., 32(2), 970–996, 2010. DOI: `10.1137/09075250X`.

[L6] M. Gouzien, C. Poussot-Vassal, G. Haine and D. Matignon. Port-Hamiltonian reduced order modelling of the 2D Maxwell equations. COMPEL. DOI: `10.1108/COMPEL-10-2024-0421`. 使用作者公开的23页全文；2025年接受/在线记录，卷期年份可能不同，不据预印本年份推断最终卷期。

[L7] N. Byrnes and M. R. Foreman. Symmetry constraints for vector scattering and transfer matrices containing evanescent components: Energy conservation, reciprocity, and time reversal. Physical Review Research, 3, 013129, 2021. DOI: `10.1103/PhysRevResearch.3.013129`; arXiv: `2011.14974`.

[L8] M. Benzaouia et al. Quasi-normal mode theory of the scattering matrix, enforcing fundamental constraints for truncated expansions. Physical Review Research, 3, 033228, 2021. DOI: `10.1103/PhysRevResearch.3.033228`.

[L9] A General Framework for Knowledge Integration in Machine Learning for Electromagnetic Scattering Using Quasinormal Modes. Laser & Photonics Reviews. DOI: `10.1002/lpor.202502769`. 本轮核对到出版商2026-03-17在线记录；与2025预印本/标识年份区分。

[L10] O. D. Miller et al. Fundamental limits to optical response in absorptive systems. Optics Express, 24, 3329–3364, 2016. DOI: `10.1364/OE.24.003329`.

[L11] A. Khorashadizadeh et al. Deep Injective Prior for Inverse Scattering. IEEE TAP, 2023. DOI: `10.1109/TAP.2023.3312818`; arXiv: `2301.03092`.

[L12] Sun et al. Latent Manifold Projection for Plug-and-Play Generative Priors in Electromagnetic Inverse Scattering. IEEE TAP, 2026, 元数据记录。DOI: `10.1109/TAP.2026.3722905`. 本轮未获得全文，出版页面直接读取也未成功；不作具体定理排除声明。

[L13] Guo et al. Plug-and-Play Latent Diffusion for Electromagnetic Inverse Scattering with Application to Brain Imaging. arXiv: `2509.04860`. 本轮使用原预印本全文，不未经核对宣称TAP出版。

[L14] A. Spantini, T. Cui, K. Willcox, L. Tenorio and Y. Marzouk. Goal-Oriented Optimal Approximations of Bayesian Linear Inverse Problems. SIAM J. Sci. Comput., 39(5), S167–S196, 2017. DOI: `10.1137/16M1082123`.

[L15] M. A. Iglesias, Y. Lu and A. M. Stuart. A Bayesian level set method for geometric inverse problems. Interfaces and Free Boundaries, 18(2), 181–217, 2016. DOI: `10.4171/IFB/362`; arXiv: `1504.00313`.

[L16] O. Zahm, T. Cui, K. Law, A. Spantini and Y. Marzouk. Certified dimension reduction in nonlinear Bayesian inverse problems. Mathematics of Computation, 2022. DOI: `10.1090/mcom/3737`; arXiv: `1807.03712`.

[L17] H. C. Lie, T. J. Sullivan and A. L. Teckentrup. Random Forward Models and Log-Likelihoods in Bayesian Inverse Problems. SIAM/ASA J. Uncertainty Quantification, 2018. DOI: `10.1137/18M1166523`; arXiv: `1712.05717`.

[L18] J. Cockayne, C. J. Oates, T. J. Sullivan and M. Girolami. Bayesian Probabilistic Numerical Methods. SIAM Review, 2019. DOI: `10.1137/17M1139357`; arXiv: `1702.03673`.

[L19] N. Thyagarajan et al. Invariants in copolar interferometry: An Abelian gauge theory. Physical Review D, 105, 043019, 2022. DOI: `10.1103/PhysRevD.105.043019`; arXiv: `2108.11399`.

## B.3 新颖性与验证的保留意见

没有对所有相关论文逐篇完成全文核对；尤其 L1–L3、L5、L12 的理论竞争不能通过“摘要没写”予以排除。所有本轮 PROVED 标签只意味着报告内陈述在给定假设下已给出数学证明，不意味着全球文献首创。需要投稿前继续做定理级 novelty audit，而不是增加无关实验替代。

本轮没有运行 E1 原生缓存审计、E2 材料成像或 E3 独立预测实验；实际完成的是已列出的代码/文献检查和22项小型核验。没有生成或承诺后台任务。# 附录 C. 三个实现前必须补齐的推导细节

## C.1 目标证书不是整个后验免费的通行证

T2b 的单一 separating goal 足以给特定分支分离的下界，但不能自动控制完整 posterior。对 T3a，令 P 为同一固定的边缘精度。如果两个目标证书分别给出

\[
\bar r^TPe_-\in[a-r_a,a+r_a],\qquad
\widetilde s^TPe_+\in[b-r_b,b+r_b],
\]

而交叉项满足 |e_+^TPe_-|≤κ，则

\[
\delta_1-\delta_0\in
[a-b-r_a-r_b-\kappa,\ a-b+r_a+r_b+\kappa].
\]

可用的通用 κ 是 ||e_+||_P ||e_-||_P，但这可能很松，且需要额外误差控制。只有误差具有可认证的共享结构、低秩相关性或很小的差分项时，目标证书才真正优于认证全部输出。材料相关的 nuisance covariance 还需额外控制 logdet 与不同精度矩阵，不包含在这个固定 P 公式中。

共享 anchor 提供一个准确的组织方法。记 e(χ)=F(χ)−F̃(χ)，沿可逆路径 χ(t)=a+t(χ−a) 有

\[
e(\chi)=e(a)+\int_0^1D(F-\widetilde F)_{\chi(t)}[\chi-a]dt.
\]

两个材料的误差差 e1−e0 消掉同一个 e(a)，但均值误差保留 e(a)。积分中的导数必须用 T1 的完整 residual/field 双通道，不能只用冻结 O_A R I_A。这个关系说明何时 offset 校准足够、何时还必须控制导数与传播；它不保证这些路径界计算便宜。

因此主方向的一个决定性实现 gate 是：**是否能便宜地控制相对似然中的二次交叉项和支持集误差，而不被全局粗 norm 上界吞掉收益。** 若不能，不能把“两次目标dual”宣传成完整 Bayes posterior 认证。

## C.2 真正功率端口上的 positive-real 到 scattering contractivity

这是已知网络/被动系统结论，用于明确物理变量，不作新颖性贡献。对有限维、实、被动 co-energy realization，设

\[
M\dot x=(J-R)x+Bu,\quad y=B^Tx+D_pu,
\]

其中 M=Mᵀ≻0，J=−Jᵀ，R=Rᵀ≽0，Re_H Dp≽0。H=xᵀMx/2 满足

\[
\dot H=u^Ty-x^TRx-u^T\operatorname{Re}_H(D_p)u\le u^Ty.
\]

在 Maxwell 实现中，u、y必须是与实际边界切向E/H配对、其乘积积分等于输入Poynting通量的端口变量。M必须由实际 ε、μ 和正确有限元质量矩阵得到；色散介质需要辅助状态。不能凭任意矩阵能写成 J−R 就认定 H 是物理电磁储能。

取 RMS 电压、电流端口 v=Zp(s)i，R0≻0为参考端口电阻，定义

\[
a={R_0^{-1/2}v+R_0^{1/2}i\over2},\quad
b={R_0^{-1/2}v-R_0^{1/2}i\over2}.
\]

则输入有功功率 Re(v†i)=||a||²−||b||²。令 Zn=R0⁻¹ᐟ² Zp R0⁻¹ᐟ²，

\[
S_p=(Z_n-I)(Z_n+I)^{-1},
\]

\[
I-S_p^*S_p=
4(Z_n+I)^{-*}\operatorname{Re}_H Z_n(Z_n+I)^{-1}\succeq0.
\]

这说明 proper normalized ports 上的正实阻抗如何产生 contractive scattering；Zp+I须可逆。若使用峰值phasor而非RMS，端口功率和波幅需加入一致的1/2系数。矩阵 S_p 是这里定义的完整端口散射，不是原数据接收矩阵 S，更不是任意32点场采样。一般evanescent和非正定flux通道也不能直接使用这一Euclidean contractivity形式。

## C.3 约束什么时候真正改变局部材料风险

被动性/physical box 是真实可行集合信息，但如果 posterior mass远离边界、约束局部不活动，它不会改变一阶切空间或给无约束Gaussian covariance凭空增加信息。活跃边界可使posterior成为截断分布，此时不能继续只用原Hessian逆当全部不确定性。这个效果与普通 physical-box prior一致，不是额外能量定理。

多频共享Lorentz/Drude参数、已测总吸收功率、独立标定参考等可以真实缩小允许材料/标定集合。前两者分别来自可证实的材料族先验和新实验观测；不能把它们与“对每个精确Maxwell候选重复检查Poynting恒等式”混为一谈。对有限频带，未知带外响应和模型阶数仍需计入prior misspecification。