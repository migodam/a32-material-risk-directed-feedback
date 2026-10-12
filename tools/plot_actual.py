"""Render only the completed A32 experimental outputs; no physics or fitting."""
from pathlib import Path
import csv,json,sys
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from src.a32.statistics import read_metrics_csv,aggregate_parent_rows,write_e0_statistics,METRICS
from src.a32.plots import (plot_reconstruction_grid,plot_online_parent_gains,plot_method_comparisons,
                          plot_offline_capacity,plot_quality_runtime)
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

cfg=json.loads((root/'a32_config.json').read_text());out=root/'results/a32/figures';out.mkdir(exist_ok=True)
rows=read_metrics_csv(root/'results/a32/offline/E0_CASE_METRICS.csv')
summary=json.loads((root/'results/a32/offline/E0_SUMMARY.json').read_text());pr=summary['parent_rows']
write_e0_statistics(summary,root/'results/a32/offline/tables')
cp=[p for p in cfg['parent_ids'] if p not in range(250073,250077)]
labels={'N3':'N3: cached BP+CNN','pca64_projected_N3':'PCA64 projected N3','pca64_L_gaussian':'PCA64 L+Gaussian',
        'pca64_calibrated_L_gaussian':'PCA64 cal-L+Gaussian (PRIMARY)','pca64_calibrated_L_ridge':'PCA64 cal-L+ridge',
        'pca64_calibrated_L_multiscale':'PCA64 cal-L+multiscale','sparse64_calibrated_L_gaussian':'Sparse64 cal-L+Gaussian',
        'sparse64_calibrated_L_multiscale':'Sparse64 cal-L+multiscale',
        'pca64_fixed_chart_OFFLINE_capacity':'PCA64 OFFLINE truth fit',
        'sparse32_fixed_chart_OFFLINE_capacity':'Sparse32 OFFLINE fixed support',
        'sparse64_fixed_chart_OFFLINE_capacity':'Sparse64 OFFLINE fixed support'}
paths=[]
with np.load(root/'results/a32/offline/RECONSTRUCTIONS.npz',allow_pickle=False) as z:
    arrays={k:z[k] for k in z.files}
paths.append(plot_reconstruction_grid(arrays,rows,methods=['pca64_projected_N3','pca64_calibrated_L_gaussian',
     'pca64_calibrated_L_multiscale','sparse64_calibrated_L_multiscale','pca64_fixed_chart_OFFLINE_capacity'],
     output_dir=out,basename='FIG01_ACTUAL_FIXED_FAMILY_MATERIAL',color_limits=(0,1.2),method_labels=labels))
paths.append(plot_online_parent_gains(pr,candidate_methods=['pca64_projected_N3','pca64_calibrated_L_gaussian','sparse64_calibrated_L_gaussian'],
     reference_method='N3',parent_ids=cfg['parent_ids'],output_dir=out,basename='FIG02_E0_ALL_PARENT_GAINS',method_labels=labels,
     evidence_label='E0 cached online material fits'))
groups={}
for rep in cfg['representations']:
    groups[rep+' frozen priors / offset']=[(rep+'_calibrated_L_gaussian',rep+'_L_gaussian'),
        (rep+'_calibrated_L_ridge',rep+'_calibrated_L_gaussian'),(rep+'_calibrated_L_multiscale',rep+'_calibrated_L_gaussian')]
paths.append(plot_method_comparisons(pr,comparison_groups=groups,parent_ids=cp,output_dir=out,
    basename='FIG03_FROZEN_PRIOR_AND_OFFSET',method_labels=labels))
paths.append(plot_offline_capacity(pr,methods=['N3','pca64_fixed_chart_OFFLINE_capacity','pca64_calibrated_L_gaussian',
   'sparse32_fixed_chart_OFFLINE_capacity','sparse64_fixed_chart_OFFLINE_capacity','sparse64_calibrated_L_multiscale'],
   parent_ids=cp,output_dir=out,basename='FIG04_CAPACITY_VS_ACTUAL_RECOVERY',method_labels=labels))
timepr=aggregate_parent_rows(rows,cfg,metrics=(*METRICS,'online_wall_seconds'))
paths.append(plot_quality_runtime(timepr,methods=['pca64_L_gaussian','pca64_calibrated_L_gaussian','pca64_calibrated_L_multiscale','sparse64_calibrated_L_gaussian'],
     parent_ids=cp,output_dir=out,runtime_field='online_wall_seconds',
     runtime_label='Coefficient QP kernel wall seconds (setup, N3 and F-anchor excluded)',
     evidence_label='E0 cached algebra; NOT end-to-end inference',basename='FIG05_QP_KERNEL_TIME_ONLY',method_labels=labels))
diag=json.loads((root/'results/a32/offline/FAILURE_ATTRIBUTION.json').read_text()); families=['shell','asymmetric_piecewise','mixed_scale','smooth']
fig,axs=plt.subplots(1,2,figsize=(11.5,4.6),layout='constrained');x=np.arange(4)
axs[0].bar(x-.15,[diag['groups'][f]['background_model_bias_noise_RMS'] for f in families],.3,label='background L')
axs[0].bar(x+.15,[diag['groups'][f]['calibrated_model_bias_noise_RMS'] for f in families],.3,label='single-anchor cal-L')
axs[0].axhline(1,color='k',linestyle='--',label='one noise RMS')
axs[0].set_ylabel('Bias/noise RMS at truth (OFFLINE cached response)');axs[0].legend(fontsize=8)
axs[1].bar(x,[100*diag['groups'][f]['in_chart_fit_minus_oracle_weak_material_energy_fraction'] for f in families])
axs[1].set_ylim(0,100);axs[1].set_ylabel('Fit-minus-oracle material energy in L-weak space (%)')
for ax in axs:ax.set_xticks(x,families,rotation=18);ax.grid(axis='y',alpha=.2)
fig.suptitle('Cache-only failure attribution; L/prior local coordinates, not full-F posterior certainty')
base='FIG06_CACHE_ONLY_FAILURE_ATTRIBUTION'
fig.savefig(out/(base+'.png'),dpi=180);fig.savefig(out/(base+'.svg'));plt.close(fig)
with (out/(base+'.csv')).open('w',newline='') as f:
    keys=['family',*diag['groups']['shell'].keys()];w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
    for family in families:w.writerow({'family':family,**diag['groups'][family]})
paths.append({'png':str(out/(base+'.png')),'svg':str(out/(base+'.svg')),'csv':str(out/(base+'.csv'))})
(out/'FIGURE_INDEX.json').write_text(json.dumps(paths,indent=2)+'\n')
print(json.dumps({'actual_figures':len(paths),'new_physics':0,'root':str(out)}))
