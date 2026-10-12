"""One Chinese research report, generated from actual frozen A32 results."""
from pathlib import Path
import csv,json,re
import numpy as np
from .budget import totals,read_rows,append_jsonl

def run(root):
    root=Path(root); out=root/'results/a32'
    cfg=json.loads((root/'a32_config.json').read_text());gate=json.loads((out/'E0_GATE.json').read_text())
    stats=json.loads((out/'offline/E0_SUMMARY.json').read_text());diag=json.loads((out/'offline/FAILURE_ATTRIBUTION.json').read_text())
    cases=list(csv.DictReader((out/'offline/E0_CASE_METRICS.csv').open()))
    physical=totals(root)
    jobs=read_rows(out/'COST_LEDGER.jsonl')
    test_job=next(r for r in reversed(jobs) if r.get('status')=='COMPLETE' and
                  r.get('command',[])[1:]==['-m','unittest','discover','-s','tests','-v'])
    test_log=out/'jobs'/test_job['id']/'OUTPUT.log'; tests=test_log.read_text()
    test_count=int(re.search(r'Ran (\d+) tests',tests).group(1))
    assert '\nOK\n' in tests
    assert physical['new_material_state_evaluations']==0 if gate['status']!='PASS' else True
    complex_rows=[r for r in cases if r['family'] in cfg['primary_families']]
    def mean(method,metric,rows=complex_rows):
        return float(np.mean([float(r[metric]) for r in rows if r['method']==method]))
    def stat(method,group='complex_families'):
        return stats['comparisons'][method+'_vs_N3'][group]
    labels=[('N3','N3：已有 BP+CNN'),('pca64_projected_N3','PCA64 投影 N3'),
            ('pca64_L_gaussian','PCA64 / L / Gaussian'),('pca64_calibrated_L_gaussian','PCA64 / calibrated-L / Gaussian（主方法）'),
            ('pca64_calibrated_L_ridge','PCA64 / calibrated-L / ridge'),('pca64_calibrated_L_multiscale','PCA64 / calibrated-L / multiscale'),
            ('sparse32_calibrated_L_gaussian','Sparse32 / calibrated-L / Gaussian'),
            ('sparse64_calibrated_L_gaussian','Sparse64 / calibrated-L / Gaussian'),
            ('sparse64_calibrated_L_multiscale','Sparse64 / calibrated-L / multiscale'),
            ('pca64_fixed_chart_OFFLINE_capacity','PCA64 固定 chart：OFFLINE truth-fit'),
            ('sparse64_fixed_chart_OFFLINE_capacity','Sparse64 固定在线 support：OFFLINE truth-fit')]
    table='|方法|材料 MSE|相对 N3 的 MSE 改善|Δχ NRMSE|IoU|edge F1|\n|---|---:|---:|---:|---:|---:|\n'
    for method,label in labels:
        gain=0. if method=='N3' else stat(method)['point_estimate']
        table+=f"|{label}|{mean(method,'fixed_mass_squared_error'):.6f}|{100*gain:+.2f}%|{mean(method,'delta_nrmse'):.4f}|{mean(method,'support_iou'):.4f}|{mean(method,'edge_f1'):.4f}|\n"
    family_table='|家族|独立 parents|主方法相对 N3 的 MSE 改善|PCA64 离线容量改善|\n|---|---:|---:|---:|\n'
    for family in ['shell','asymmetric_piecewise','mixed_scale','smooth']:
        family_table+=f"|{family}|4|{100*stat(cfg['primary_method'],'family_'+family)['point_estimate']:+.2f}%|{100*stat('pca64_fixed_chart_OFFLINE_capacity','family_'+family)['point_estimate']:+.2f}%|\n"
    ci=stat(cfg['primary_method'])['ci95']; groups=diag['groups']['complex']
    online_job=next(r for r in reversed(jobs) if r.get('command',[])[1:]==['-m','src.a32','e0'])
    score_jobs=[r for r in jobs if r['id'].startswith('e0-score')]
    ledger=read_rows(out/'FAILURE_LEDGER.jsonl')
    if not any(r.get('label')=='E0_FROZEN_MATERIAL_GATE' for r in ledger):
        append_jsonl(out/'FAILURE_LEDGER.jsonl',{'stage':'e0','label':'E0_FROZEN_MATERIAL_GATE',
                    'error':gate['reason'],'kind':'SCIENTIFIC_NO_GO_NOT_OPTIMIZER_FAILURE',
                    'material_MSE_gain':gate['relative_equal_parent_MSE_gain'],'required_gain':.05})
    decision={'status':'NO_GO_E1','E0':gate,'E1':{'status':'NOT_RUN','reason':'E0_FAILED'},
              'conditional_3d':{'status':'NOT_RUN','reason':'E0_FAILED_AND_NO_CONFIRMED_FULL_NATIVE_CACHE'},
              'A17_native_material_benefit':'NOT_ESTABLISHED_NOT_TESTED_IN_NATIVE_E1',
              'material_risk_prediction':'MATRIX_IDENTITY_TESTED_ONLY_NATIVE_PREDICTION_NOT_RUN',
              'same_quality_acceleration':'NOT_ESTABLISHED','physical_accounting':physical,
              'training_updates':0,'new_labels':0,'locked_access':False,
              'publication_status':'SCIENTIFIC_RESULTS_READY_PUBLICATION_SEPARATELY_RECORDED'}
    (out/'FINAL_DECISION.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False)+'\n')
    math={'status':'PASS','A32_tests':test_count,'test_log':str(test_log.relative_to(root)),
          'matrix_material_risk_and_source_field_tests':8,'generic_budget_guard_tests':7,
          'frozen_gate_coverage_tests':7,
          'data_and_chart_access_tests':14,'prior_and_QP_tests':14,
          'legacy_A31_22_checks':'HISTORICAL_JSON_PRESERVED_NOT_REEXECUTED',
          'native_3d': 'NOT_RUN','power_form':'NOT_ESTABLISHED_IN_THIS_CAMPAIGN',
          'full_Galerkin_native_ablation':'NOT_RUN_E0_FAILED',
          'missing_attachments':['CORE_THEOREM.md','a31_r1_checks.py']}
    (out/'MATHEMATICAL_CHECKS.json').write_text(json.dumps(math,indent=2)+'\n')
    text=f'''# A32 — 连续材料恢复与 A17 材料风险反馈：执行与科学裁决

2026-10-12。**E0 FAIL；E1及三维验证 NOT_RUN。** 全部32条件已完成，包含16个已暴露 parent。主方法的复杂材料固定平方误差相对 N3 **恶化 {100*-gate['relative_equal_parent_MSE_gain']:.2f}%**，没有达到预先固定的改善5%目标。没有支付新的全波状态，也没有用 full F 补救失败。

## 1. 物理问题与已有研究

材料表示能够拟合真值，并不意味着观测能够选出那个材料。A30 已经证明有限 bank 的凸组合、Sparse/PCA连续参数具有离线容量，而原 L-weighted bank 选择失败；它并没有执行连续系数反演。因此 A32 首先补上这个缺口，再判断是否值得研究 A17 的材料风险反馈。

[A17 的原始理论](https://github.com/migodam/a17-maxwell-certified-exchange/blob/df61a1b533682a883cdaf65259b4528c354bd832/closure_final/manuscript/CORE_THEORY.tex)描述同一物理状态下 current Galerkin空间改变如何同时改变材料曲率与残差拉回。[效率后续](https://github.com/migodam/a17-maxwell-certified-exchange/blob/df61a1b533682a883cdaf65259b4528c354bd832/efficiency_followup/GPT_REVIEW_CONTEXT.md)已经显示冻结 GN收益保留不等于最终图像改善或加速。[最新 A30](https://github.com/migodam/a30-maxwell-observable-material-manifold/blob/717d3d26a290248bd2b9aeb55f3d03694b407b84/A30_E0_E1_DECISION.md)是这里的容量依据；其13.12% hull、35.69% Sparse-64、21.91% PCA-64都是离线结果，旧补偿／uniform certificate路线未通过。

A32没有重做理论，也没有把附加物理检查变成成像资格。A31包的旧 DC-MDI主线和128状态计划被本次明确的 E0→条件E1协议取代。

## 2. 数据、材料与信息边界

平台为 **24×24二维标量 TM**、8源、32接收器、512维实测量、1152维全网格实材料坐标，complex128／float64。两个背景为 weak和ring；32条件均为既有 amplitude1.2、20dB观测。TRAIN64与已暴露 development16严格分开；未打开 locked ID/OOD，未生成新标签、未训练网络。

材料 box为 Reχ∈[−0.5,4]、Imχ∈[0,0.5]，cell mass=1/576。逐源[Re32,Im32] packing与sqrt(2)/σ白化得到缓存恒等式核对。既有标签固定复增量相位，本结果不代表任意独立实虚部材料恢复。

主表示是TRAIN-only非中心化 PCA64。Sparse32/64采用A30规则的第一个合法support；没有读取best_start_id、truth-selected支持或oracle系数。T=1.2B，因此实际材料度量M=TᵀWT=1.44I。初值是N3在同类中的box可行投影，未保留类外N3残差。

Gaussian先验由64独立TRAIN系数均值和OAS协方差形成；[OAS来源](https://arxiv.org/abs/0907.4698)是已知协方差估计方法。ridge匹配协方差trace，不能声称其precision强度相同；多尺度先验使用TRAIN Haar层方差的precision pullback。N3用于初值／支持，以及已有F(N3)缓存的同锚偏移校准；未当作独立prior重复乘相同数据。开发truth未参与先验、超参数或方法选择。

在线过程先保存32份预测和完成标记，再允许独立评分器读取truth。缓存F(N3)只用于F(N3)+L(χ)−L(N3)的偏移；不是F(投影N3)，也不是免费冷部署物理。

## 3. 实施与数值验证

实现了真实连续系数的受约束凸二次拟合，解析梯度、可行QR active-set、统一200迭代上限、KKT 1e−8、可行性1e−10；没有jitter、伪逆、像素后clipping或阈值放宽。

32条件×3表示×4拟合=384次在线measurement-conditioned QP；另有96次N3可行投影和96次独立offline固定chart truth-fit。**576/576 QP成功**。记录的最大KKT相对量为{max(float(r['kkt_relative']) for r in cases if r.get('kkt_relative')):.3e}；主方法独立重新组装检查最大stationarity残差为{diag['max_independent_primary_KKT_relative']:.3e}，最大raw可行性误差{diag['max_raw_primal_violation']:.3e}。这排除了当前已执行小凸问题的优化器未收敛解释，不保证材料正确。

{test_count}项A32测试通过，覆盖paired实化／Schur、signed Hessian、局部风险恒等式、一次HVP的对偶界、GN-gap与材料risk反例、完整source-field双缺陷、重心变换、数据隔离、box／KKT、统一预算拒绝及parent配对完整性。原A31的22项JSON只是保留的历史证据；缺失的CORE_THEOREM.md和a31_r1_checks.py没有被冒称已运行。

数学接口严格比较J_V→J_[V,C]。ΔJ=YΓ⁻¹N没有被当作对背景J_L的精确修正。active box的更新不冒用无约束paired closure，局部Gaussian风险不冒称非线性后验风险。详见[数学说明](docs/MATHEMATICAL_SCOPE.md)。

## 4. 连续材料恢复结果

先在每parent内平均weak/ring，再等权汇总12复杂parents；smooth单列。门槛是固定平方材料误差之均值比，不是平均parent NRMSE增益。全部失败或缺失都应计入；本次配对完整。

{table}

主方法的2000次parent配对bootstrap区间为 **[{100*ci[0]:.2f}%, {100*ci[1]:.2f}%]**；它仅描述已暴露小样本不确定性，不追加显著性gate。所有复杂家族均恶化：

{family_table}

Sparse64的smooth子组有约7–9%的MSE改善，却没有在复杂家族复制；不能用smooth替代主端点。多尺度prior是冻结对照，虽然比Gaussian稍好，仍未超过N3。没有事后切换主方法。

本表offline容量使用**当前同一在线固定support**及固定MSE口径，不能与A30 best-of-eight相对NRMSE数字直接相减。PCA64和Sparse64的容量改善分别为{100*stat('pca64_fixed_chart_OFFLINE_capacity')['point_estimate']:.2f}%、{100*stat('sparse64_fixed_chart_OFFLINE_capacity')['point_estimate']:.2f}% MSE；这些不是在线算法结果。

![真实固定四家族材料图](results/a32/figures/FIG01_ACTUAL_FIXED_FAMILY_MATERIAL.png)

图示预选250065／250069／250073／250077的weak条件，未挑最好图；统一|Δχ|色标0–1.2，完整数值保留于NPZ。右列明确OFFLINE truth-fit：说明chart有容量，不说明测量已恢复该结构。

![全部parent改善](results/a32/figures/FIG02_E0_ALL_PARENT_GAINS.png)

![冻结prior与校准偏移对照](results/a32/figures/FIG03_FROZEN_PRIOR_AND_OFFSET.png)

此图比较冻结方法彼此的增量；正值不表示优于N3，也不改变主门槛失败。

![容量与实际恢复](results/a32/figures/FIG04_CAPACITY_VS_ACTUAL_RECOVERY.png)

## 5. 失败机制：已验证什么，仍不能归因什么

**表示限制存在，但不足以解释全部失败。** PCA64投影N3使复杂MSE恶化11.74%，说明强制全局chart损失了N3的一部分结构；同chart truth-fit仍可以改善38.68%。因此“表示内没有更好材料”已被直接反驳。Sparse64的投影损失只有3.33%，其在线Gaussian拟合仍恶化30.09%，也不能仅归因于PCA投影。

**优化器不是已执行凸问题的主瓶颈。** 独立KKT复核通过；truth-fit与在线MAP在相同材料类内指向不同图像。更长地解同一个凸目标不会因此自动接近truth。

**偏移校准确实帮助数据模型，却没有解决材料选择。** 复用原clean labels，在真值材料处、按噪声每分量RMS比较，L模型偏差为{groups['background_model_bias_noise_RMS']:.3f}倍噪声；single-anchor calibrated-L降至{groups['calibrated_model_bias_noise_RMS']:.3f}倍。实际噪声白化RMS为{groups['measurement_noise_RMS']:.3f}。这没有付费新F。cal-L的MSE也比未校准L稍好，却仍明显输给N3。

**弱方向与先验有直接迹象。** 按固定TRAIN prior白化的背景Jacobian谱，64维中平均{groups['L_weak_prior_whitened_dimensions']:.2f}维的likelihood/prior曲率比低于1。主拟合与同chart oracle之差的{100*groups['in_chart_fit_minus_oracle_weak_material_energy_fraction']:.2f}%材料能量位于这一背景L弱空间；shell、mixed更集中。该“1”是诊断坐标的比较尺度，不是新gate；也不是full-F posterior uncertainty证书。

oracle的平均TRAIN Mahalanobis平方距离为{groups['oracle_prior_Mahalanobis_squared']:.2f}，MAP为{groups['primary_prior_Mahalanobis_squared']:.2f}。它们说明当前目标在数据／先验之间的取舍，并不能单独证明prior misspecification的因果比例。这里没有重调prior。

![缓存失败归因](results/a32/figures/FIG06_CACHE_ONLY_FAILURE_ATTRIBUTION.png)

剩余限制必须保留：低于单通道噪声的RMS偏差仍可能具有重要的相干材料方向；没有在新恢复图像处支付full-F，也没有比较full-F与L的局部弱子空间旋转。因此不能由上述结果宣布full-F derivative无价值、材料不可识别、或者A17理论被推翻。数据模型偏差和结构prior影响尚未被完全因果分离。

数字拓扑采用原4连通前景／8连通背景、truth-derived25%支持阈值，仅用于离线评分。主方法的components计数误差3.375，N3为2.667；hole计数均值有个别下降，却同时损失edge／IoU，不能把更光滑图像写成更好拓扑。完整逐条件计数在CSV。

## 6. E1停止与四个最终问题

冻结主方法MSE改善为{100*gate['relative_equal_parent_MSE_gain']:.2f}%，未达+5%，故**E1 shadow、短轨迹、条件3D全部NOT_RUN**。不存在原生paired材料收益、predicted-versus-realized真实材料风险或同质量加速数据。风险和source-field模块的矩阵核验不能填补它们。

实现范围止于缓存E0、独立诊断和小矩阵数学测试。后续CLI保留gate拒绝入口；因为资格失败，没有继续接线原生E1轨迹驱动。不能把这些NOT_RUN入口当成已经实现并验证的full-Maxwell材料反馈算法。

1. **连续拟合是否利用A30容量？** 当前冻结Gaussian/ridge/multiscale在线实现没有利用复杂结构headroom；同support的离线容量仍存在。
2. **A17是否提供普通material solver没有的收益？** 本实验尚未取得进入原生E1的资格，答案是NOT_ESTABLISHED；没有建立强算法论文的材料优势。
3. **风险判断是否优于GN-gap？** 小矩阵反例证明两种度量不同，接口通过；真实Maxwell更新的预测价值未运行。
4. **反馈是否有同质量成本优势？** 未建立，不能以小矩阵求解速度、零新增state或当前rank宣称加速。

**研究裁决：NO-GO进入当前A17反馈成像pilot。** 更合理的下一步假说是先建立能够保留N3结构信息的measurement-conditioned材料行动，而不是用一个全局Gaussian MAP把它覆盖；若将N3作为conditional estimator，必须明确同一测量的概率依赖，不能再充当独立prior重复likelihood。这个建议尚未实验，不在本任务追加训练、rank、current bank或新物理。

## 7. 成本、实现失败与复现

新增真实full-wave状态 **0/64**，新LU **0**，新LU RHS **0**，GPU占用 **0秒**，训练更新 **0**。实际背景／F(N3)缓存读入、矩阵应用、prior生成、QP、评分、图表和失败都有CPU记录。E0在线批次包含32条件及全部预登记对照，wall **{online_job['wall_seconds']:.3f}秒**；这是缓存批处理计算，**不是N3+full-anchor的部署推理时间**。

系数QP的kernel计时与原始测量到图像的完整耗时分开。N3已缓存，F(N3)来自历史paid state；首次部署取得它们的成本未在这里重新测量。没有共同质量命中，也没有实际冷／warm端到端提速结论。

![仅系数求解kernel耗时](results/a32/figures/FIG05_QP_KERNEL_TIME_ONLY.png)

CPU按每个实际job的wall／process CPU较大者记账，报告生成时记录 **{physical['charged_cpu_seconds']:.3f}/1800秒**；后续报告与发布费用以COST_LEDGER及publication记录为准。交互式少量读文件及外部模型推理未被这些host-job receipt完全覆盖，不把该数字当全部工作总耗时；预算远未接近上限。worker失败的NumPy／Matplotlib导入、解释器路径，以及一项过严roundoff测试断言均保留；数值solver／科学门槛没有因此改变。最终整合{test_count}测试通过。

材料与模型收益失败和实现失败分开记录。没有hash／SHA256检查，没有traceable备份或额外native测试campaign。A17/A30原代码、结果及NO-GO均未改写。

执行入口、依赖和公共输入下载见[README](README.md)。原始指标见[E0 CSV](results/a32/offline/E0_CASE_METRICS.csv)、[逐parent表](results/a32/offline/tables/E0_PARENT_METRICS.csv)、[失败归因](results/a32/offline/FAILURE_ATTRIBUTION.csv)。最终状态与完整计费分别见[FINAL_DECISION](results/a32/FINAL_DECISION.json)、[COST_LEDGER](results/a32/COST_LEDGER.jsonl)、[FAILURE_LEDGER](results/a32/FAILURE_LEDGER.jsonl)。大数组另放Release；在线预测与offline truth-fit分目录。

## 8. 证据层级

|层级|实际状态|
|---|---|
|A17／A31已有理论|读取、复用；没有因实验失败改写|
|A32矩阵、访问边界和QP测试|{test_count}项通过；不当作材料优势|
|已有真实Maxwell数据上的在线材料恢复|完整32条件，主方法FAIL|
|同chart offline capacity|完整32条件；truth拟合，不是在线算法|
|新原生Maxwell反馈／三维验证|NOT_RUN，新增state=0|
|盲测、NN、后续E2|NOT_RUN／未授权扩展|
|网页发布|独立记录；发布成功不改变科学NO-GO|
'''
    (root/'A32_E0_E1_DECISION.md').write_text(text)
    print(json.dumps({'scientific_status':'NO_GO_E1','E0_gain':gate['relative_equal_parent_MSE_gain'],
                      'CPU_so_far':physical['charged_cpu_seconds'],'new_states':physical['new_material_state_evaluations']}))
    return decision
