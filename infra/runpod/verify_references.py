"""Verify native reference agreement after inference, then refresh retrieval hashes."""
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
from model_calibration.io import read,write,sha256
run=Path('results/runs/coco-v1')
while not (run/'worker_complete.json').exists():time.sleep(10)
interpreters=read('configs/interpreters.json')
for model in read('configs/models.json'):
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src');env['HF_HOME']=str(ROOT/'weights/huggingface');env['NVIDIA_TF32_OVERRIDE']='0'
    site=Path(interpreters[model]).parent.parent/'lib/python3.11/site-packages'
    env['LD_LIBRARY_PATH']=':'.join([str(p) for p in site.glob('nvidia/*/lib')]+[env.get('LD_LIBRARY_PATH','')])
    subprocess.run([interpreters[model],'infra/runpod/reference_check.py',model],check=True,env=env)
write(run/'reference_complete.json',dict(models=list(read('configs/models.json')),completed_at=time.time()))
write(run/'artifact-index.json',{str(p.relative_to(run)):sha256(p) for p in run.rglob('*.json') if p.name!='artifact-index.json'})
