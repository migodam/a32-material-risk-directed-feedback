"""Import actual worker job costs once; alias receipts are not new jobs."""
from pathlib import Path
import json,sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from src.a32.budget import append_jsonl,read_rows
rows=read_rows(root/'results/a32/COST_LEDGER.jsonl');existing={r['id'] for r in rows}
names=['a32_data/INPUT_SCHEMA_RECEIPT.json','a32_data/TEST_RECEIPT.json','a32_data/WHITELIST_INPUT_RECEIPT.json',
       'a32_qp/CPU_RECEIPT.json','a32_figures/CPU_RECEIPT_001.json','a32_figures/CPU_RECEIPT_002.json',
       'a32_figures/CPU_RECEIPT_003_LAUNCH_FAILURE.json','a32_figures/CPU_RECEIPT_004.json']
for name in names:
    path=root/'research/delegated'/name
    if not path.exists():continue
    receipt=json.loads(path.read_text());eid='worker:'+name
    cpu=float(receipt.get('process_cpu_seconds') or receipt.get('cpu_seconds') or 0.)
    wall=float(receipt.get('wall_seconds') or receipt.get('launch_wall_seconds') or receipt.get('wall_seconds_reported_by_exec') or 0.)
    charged=max(cpu,wall,float(receipt.get('cpu_charged_seconds') or receipt.get('charged_seconds') or 0.))
    status=receipt.get('outcome',receipt.get('status','RECORDED'))
    if receipt.get('success') is False or receipt.get('failures',0) or receipt.get('errors',0):status='FAILED'
    if eid in existing:
        previous=next(r for r in rows if r.get('id')==eid)
        correction_id=eid+':RECEIPT_METADATA_CORRECTION'
        if correction_id not in existing and (previous['status']!=status or previous['charged_seconds']<charged):
            append_jsonl(root/'results/a32/COST_LEDGER.jsonl',{
                'event':'WORKER_RECEIPT_CORRECTION','id':correction_id,'corrects':eid,'status':status,
                'charged_seconds':max(0.,charged-previous['charged_seconds']),
                'known_launch_wall_seconds':wall,'process_cpu_seconds_original':receipt.get('process_cpu_seconds'),
                'gpu_occupied_seconds':0.,'source':str(path.relative_to(root)),
                'note':'Original receipt retained; no duplicate job or physical state.'})
        failure_ids={r.get('label') for r in read_rows(root/'results/a32/FAILURE_LEDGER.jsonl')}
        if 'FAIL' in str(status).upper() and eid not in failure_ids:
            append_jsonl(root/'results/a32/FAILURE_LEDGER.jsonl',{
                'stage':'implementation-tests','label':eid,'error':str(receipt.get('error') or
                    f"{receipt.get('failures',0)} failed assertions, {receipt.get('errors',0)} errors; see original receipt"),
                'scope':'SYNTHETIC_TEST_OR_IMPORT_NOT_PHYSICAL_EXPERIMENT','source':str(path.relative_to(root))})
        continue
    row={'event':'WORKER_JOB','id':eid,'status':status,'charged_seconds':charged,
         'process_cpu_seconds':cpu,'wall_seconds':wall,'gpu_occupied_seconds':0.,'source':str(path.relative_to(root))}
    append_jsonl(root/'results/a32/COST_LEDGER.jsonl',row)
    if 'FAIL' in str(status).upper():
        append_jsonl(root/'results/a32/FAILURE_LEDGER.jsonl',{'stage':'implementation-tests','label':eid,
                    'error':str(receipt.get('error',receipt.get('reason',status))),
                    'scope':'SYNTHETIC_TEST_OR_IMPORT_NOT_PHYSICAL_EXPERIMENT','source':str(path.relative_to(root))})
    print(json.dumps(row))
