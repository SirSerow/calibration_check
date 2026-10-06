#!/usr/bin/env python3
"""Smoke all adapters, estimate cost from measured throughput, then infer."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT/'src'))
from model_calibration.io import read,write,sha256


def worker(run,hourly,remaining,smoke_images=50):
    run=Path(run)
    if hourly<=0 or hourly>1 or remaining<=1 or remaining>10:
        raise ValueError('Worker requires positive budget with $1 retrieval reserve')
    interpreters=read('configs/interpreters.json')
    registry=read('configs/models.json')
    deadlines=time.monotonic()+(remaining-1)/hourly*3600
    measurements={}
    os.environ['PYTHONPATH']=str(ROOT/'src')
    os.environ['HF_HOME']=str(ROOT/'weights/huggingface')
    os.environ['NVIDIA_TF32_OVERRIDE']='0'
    run.joinpath('logs').mkdir(parents=True,exist_ok=True)
    def execute(model,limit=None):
        argv=[interpreters[model],'-m','model_calibration','--run',str(run),'infer',model]
        if limit is not None: argv+=['--limit',str(limit)]
        timeout=deadlines-time.monotonic()
        if timeout<=0: raise RuntimeError('Budget runtime exhausted')
        env=os.environ.copy()
        site=Path(interpreters[model]).parent.parent/'lib/python3.11/site-packages'
        libraries=[str(p) for p in site.glob('nvidia/*/lib')]
        compatibility=Path(interpreters[model]).parent.parent/'cuda-compat/lib'
        if compatibility.exists(): libraries.insert(0,str(compatibility))
        env['LD_LIBRARY_PATH']=':'.join(libraries+[env.get('LD_LIBRARY_PATH','')])
        with (run/'logs'/f'{model}.log').open('ab') as log:
            subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout,env=env)
    for model in registry:
        start=time.monotonic()
        execute(model,smoke_images)
        elapsed=time.monotonic()-start
        state=read(run/'manifest.json')['models'][model]
        cumulative=state['elapsed_seconds']
        observed=max(smoke_images,state['completed_images'])
        pending=max(0,5000-state['completed_images'])
        measured=max(elapsed,cumulative)
        measurements[model]={'images':observed,'pending_images':pending,'elapsed_seconds':measured,'resume_seconds':elapsed,'estimated_full_seconds':measured/observed*pending}
        print({'stage':'smoke','model':model,'elapsed_seconds':measured},flush=True)
        write(run/'smoke.json',measurements)
    estimate=sum(m['estimated_full_seconds'] for m in measurements.values())*1.5
    available=deadlines-time.monotonic()
    write(run/'runtime_estimate.json',{'seconds_with_50_percent_margin':estimate,'available_seconds':available,
                                      'hourly_usd':hourly,'estimated_inference_cost_usd':estimate/3600*hourly})
    if estimate>available:
        raise RuntimeError('Measured smoke throughput exceeds remaining budget')
    for model in registry:
        print({'stage':'full_inference','model':model},flush=True)
        execute(model)
        # A compact artifact index allows local verification before termination.
        inventory={str(p.relative_to(run)):sha256(p) for p in run.rglob('*.json') if p.name!='artifact-index.json'}
        write(run/'artifact-index.json',inventory)
    write(run/'worker_complete.json',{'completed_at':time.time(),'models':list(registry)})
    inventory={str(p.relative_to(run)):sha256(p) for p in run.rglob('*.json') if p.name!='artifact-index.json'}
    write(run/'artifact-index.json',inventory)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='results/runs/coco-v1')
    p.add_argument('--hourly-usd',type=float,required=True);p.add_argument('--remaining-usd',type=float,required=True)
    p.add_argument('--verify-local',action='store_true')
    a=p.parse_args()
    if a.verify_local:
        run=Path(a.run)
        inventory=read(run/'artifact-index.json')
        for path,expected in inventory.items():
            if sha256(run/path)!=expected: raise ValueError(f'Local artifact differs: {path}')
        complete=read(run/'worker_complete.json')
        manifest=read(run/'manifest.json')
        if set(complete['models'])!=set(manifest['registry']):
            raise ValueError('Incomplete registered-model artifact coverage')
        ids={f'{i:012d}.json' for split in manifest['splits'].values() for i in split}
        for model,state in manifest['models'].items():
            paths=list((run/'predictions'/model).glob('*.json'))
            if state['status']!='complete' or {p.name for p in paths}!=ids:
                raise ValueError(f'Incomplete image coverage for {model}')
            if any(str(p.relative_to(run)) not in inventory for p in paths):
                raise ValueError(f'Prediction absent from retrieval index: {model}')
        print(f'Verified {len(inventory)} local artifact hashes')
    else: worker(a.run,a.hourly_usd,a.remaining_usd)

if __name__=='__main__': main()
