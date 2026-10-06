"""Run one isolated adapter alongside other models on the same GPU."""
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
from model_calibration.io import read
model=sys.argv[1];python=read('configs/interpreters.json')[model]
site=Path(python).parent.parent/'lib/python3.11/site-packages'
env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src');env['HF_HOME']=str(ROOT/'weights/huggingface');env['NVIDIA_TF32_OVERRIDE']='0'
env['LD_LIBRARY_PATH']=':'.join([str(p) for p in site.glob('nvidia/*/lib')]+[env.get('LD_LIBRARY_PATH','')])
subprocess.run([python,'-m','model_calibration','--run','results/runs/coco-v1','infer',model],check=True,env=env)
