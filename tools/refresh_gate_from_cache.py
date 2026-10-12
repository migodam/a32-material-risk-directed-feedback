"""Rebuild only summaries and eligibility from saved E0 metrics, never fitting."""
from pathlib import Path
import json,sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from src.a32.budget import Book
from src.a32.statistics import read_metrics_csv,e0_statistics,write_e0_statistics
from src.a32.e0 import gate_from_statistics,clean
book=Book(root,'report');cfg=book.config;out=root/'results/a32'
rows=read_metrics_csv(out/'offline/E0_CASE_METRICS.csv')
with book.span('refresh_authoritative_cached_statistics'):
    result=e0_statistics(rows,cfg)
    gate=gate_from_statistics(result,cfg)
    old=json.loads((out/'E0_GATE.json').read_text())
    assert old['status']==gate['status']=='FAIL'
    assert abs(old['relative_equal_parent_MSE_gain']-gate['relative_equal_parent_MSE_gain'])<1e-14
    (out/'offline/E0_SUMMARY.json').write_text(json.dumps(clean(result),indent=2)+'\n')
    (out/'E0_GATE.json').write_text(json.dumps(gate,indent=2)+'\n')
    write_e0_statistics(result,out/'offline/tables')
print(json.dumps({'gate':gate['status'],'MSE_gain':gate['relative_equal_parent_MSE_gain'],
                  'new_fits':0,'new_physical_states':0}))
