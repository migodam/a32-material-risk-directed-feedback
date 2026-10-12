"""Run one bounded scientific job; charge inclusive wall/process CPU once."""
from pathlib import Path
import argparse, json, os, resource, subprocess, sys, time

def main():
    p=argparse.ArgumentParser(); p.add_argument('--id',required=True); p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args(); root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root)); from src.a32.budget import charged_cpu,append_jsonl
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not command: raise ValueError('COMMAND_REQUIRED')
    out=root/'results/a32/jobs'/a.id; out.mkdir(parents=True,exist_ok=False)
    cap=1800.-charged_cpu(root)
    if cap<=0: raise SystemExit('CPU_BUDGET_EXHAUSTED')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    before=resource.getrusage(resource.RUSAGE_CHILDREN); start=time.perf_counter(); status='FAILED'; code=1
    with (out/'OUTPUT.log').open('w') as f:
        try:
            proc=subprocess.run(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=max(.1,cap-2.))
            code=proc.returncode; status='COMPLETE' if code==0 else 'FAILED'
        except subprocess.TimeoutExpired: status='BUDGET_TERMINATED'; code=124
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    wall=time.perf_counter()-start; cpu=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
    row={'event':'JOB','id':a.id,'status':status,'command':command,'exit_code':code,
         'wall_seconds':wall,'process_cpu_seconds':cpu,'charged_seconds':max(wall,cpu),'gpu_occupied_seconds':0.,
         'memory_peak_rss_bytes_platform_native':after.ru_maxrss}
    append_jsonl(root/'results/a32/COST_LEDGER.jsonl',row)
    if status!='COMPLETE':
        append_jsonl(root/'results/a32/FAILURE_LEDGER.jsonl',{
            'stage':'bounded_job','label':a.id,'error':status,'exit_code':code,
            'output_log':str((out/'OUTPUT.log').relative_to(root)),
            'scope':'See command and original log; counted even on failure.'})
    (out/'manifest.json').write_text(json.dumps(row,indent=2)+'\n'); print(json.dumps(row))
    raise SystemExit(code)

if __name__=='__main__': main()
