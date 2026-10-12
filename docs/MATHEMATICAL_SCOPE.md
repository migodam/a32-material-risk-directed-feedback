# A32 数学接口与结论范围

## 材料与测量

沿用 A30 的固定 TRAIN 线性表示。B 在体积质量内积下正交，T=1.2B，故系数平方材料度量 M=TᵀWT=1.44I。物理材料始终是 χ=χb+Tc；Re/Im 共用一张材料分布，八源只分别产生电流。测量逐源 [Re32,Im32]，材料 [Re576,Im576]，实噪声白化 sqrt(2)/σ。

Gaussian prior 由独立64个 TRAIN parent 的系数均值和 OAS 协方差得到。信号依赖的 N3 用于初值、sparse 支持选择，以及已有 F(N3) 缓存的同锚偏移校准；不另作为独立 prior 乘上同一数据 likelihood。重心变换只把相同绝对先验 m 改写为 m−c_anchor。

## E0 的物理模型

背景线性 slope A 是已保存的质量 Jacobian 乘 T。校准模型为 F(N3)+L(χ)−L(N3)，而不是 F(投影N3)。校准只使用已有同材料 F(N3) 数据，不在 E0 付费审核新解。缓存读取和矩阵计算仍计时。

## A17 精确恒等式的边界

ΔJ=Y_C Γ_C⁻¹N_C 对应同一 L、S、B 下 retained Galerkin 与 [V,C] Galerkin 的差。它不等于 enlarged Jacobian 减 background J_L。A32 把 calibrated-L 保留为独立强基线。

全部源共同实化后，D=UZᵀ，Q=[J₀ᵀU,Z]，K=[[0,I],[I,UᵀU]]，ΔH=QKQᵀ。paired closure 只对所声明的无约束共同注入二次问题精确；active box 的 constrained 更新另作数值结果，不冒用这个 range 定理。

## 材料风险检查

在同一局部、未截断 Gaussian 模型下，H=J_FᵀJ_F+Λ、h=J_Fᵀ(y−f_anchor)+Λ(m−c_anchor)。对行动 b 与 b+αd，

G(α)=2αdᵀM(H⁻¹h−b)−α²dᵀMd。

令 qhat=Hhat⁻¹Md、ρ=Md−H_Fqhat、Ahat=2qhatᵀh−2dᵀMb，则精确局部恒等式为

G(α)=αAhat−α²dᵀMd+2αρᵀH_F⁻¹h。

以真正 prior precision Λ 得到 Bdual=2||ρ||_(Λ⁻¹)||h||_(Λ⁻¹)。数值 trust damping 不进入 covariance。这个下界只检查局部 Gaussian 材料风险；没有非线性 posterior score／边界矩控制时，不宣称完整材料风险证书。

## 非线性 source-field 缺陷

对真正可微 ROM current jtilde(χ)，ρ=b−Ljtilde，

D(F−Ftilde)[h]=SR Dρ[h]+SR DX[h]G Rρ。

固定 basis Galerkin 的真正导数使用 B_ROM=DX[h](e_inc+Gj_ROM)，并保留 J_F−J_ROM 的 frozen-injection 和 source-field 两项。仅用 full-field B_F 的 reduced tangent 不是该 nonlinear ROM 的真正导数。

三维 DDA 必须采用原 CM/RR α(χ)、α'(χ)、Goff 和 self convention。H≥Λ 的统计正定性不是 Poynting 耗散证书。功率审计无法建立时，状态是 ENERGY_FORM_NOT_ESTABLISHED，普通材料恢复仍可进行。

## 已有证据与新增证据

原 A31 的22项核验是历史有限矩阵／解析物理结果，原 JSON 保留。附件引用但未提供的 CORE_THEOREM.md 与 a31_r1_checks.py 记录缺失，未声称运行它们。A32 新矩阵测试只验证以上接口及反例，不当作实际材料成像结果。
