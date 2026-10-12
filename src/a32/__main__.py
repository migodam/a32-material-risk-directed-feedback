from pathlib import Path
import argparse,json

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','freeze','e0','score','shadow','trajectories','vector-check','report'])
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[2]));a=p.parse_args();root=Path(a.root)
    if a.stage=='preflight':
        from .preflight import run;run(root)
    elif a.stage in {'freeze','e0'}:
        from .e0 import run_online;run_online(root)
    elif a.stage=='score':
        from .e0 import score;score(root)
    elif a.stage in {'shadow','trajectories','vector-check'}:
        gate=root/'results/a32/E0_GATE.json'
        if not gate.is_file() or json.loads(gate.read_text()).get('status')!='PASS':
            value={'status':'NOT_RUN','reason':'E0_PASS_REQUIRED','stage':a.stage,'new_physics_states':0}
            (root/'results/a32'/('E1_'+a.stage.replace('-','_').upper()+'.json')).write_text(json.dumps(value,indent=2)+'\n')
            print(json.dumps(value));return
        from .experiment import run_stage;run_stage(root,a.stage)
    elif a.stage=='report':
        from .report import run;run(root)

if __name__=='__main__':main()
