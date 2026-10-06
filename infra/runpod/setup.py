#!/usr/bin/env python3
"""Set up isolated environments and save exact resolved package inventories."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
os.environ['PATH']='/usr/local/cuda/bin:'+os.environ.get('PATH','')
sys.path.insert(0,str(ROOT/'src'))
from model_calibration.io import read,write,sha256


def command(argv):
    subprocess.run(list(map(str,argv)),check=True)


def setup():
    start=time.time()
    interpreters={}
    environments=read('configs/environments.json')
    common=['scipy==1.15.3','pycocotools==2.0.10','matplotlib==3.10.3']
    for name,env in environments.items():
        folder=ROOT/('.venv-'+name)
        if not folder.exists(): command([sys.executable,'-m','venv',folder])
        python=folder/'bin/python'
        command([python,'-m','pip','install','pip==25.1.1','setuptools==80.9.0','wheel==0.45.1'])
        packages=env['packages']+common
        if env.get('torch_index'):
            torch_packages=[p for p in packages if p.startswith(('torch==','torchvision=='))]
            command([python,'-m','pip','install','--no-cache-dir','--index-url',env['torch_index'],*torch_packages])
        command([python,'-m','pip','install','--no-cache-dir',*packages])
        if env.get('extra_wheels'):
            command([python,'-m','pip','install','--only-binary=mmcv','-f',env['wheel_index'],*env['extra_wheels']])
        command([python,'-m','pip','check'])
        if name=='paddle':
            from cuda_compat import create_links
            create_links(folder)
        lock=subprocess.check_output([python,'-m','pip','freeze'],text=True)
        Path('configs/locks').mkdir(parents=True,exist_ok=True)
        (Path('configs/locks')/(name+'.txt')).write_text(lock)
        for model in env['models']: interpreters[model]=str(python)
    write('configs/interpreters.json',interpreters)
    repo=ROOT/'external/darknet'
    makefile=repo/'Makefile'
    text=makefile.read_text()
    for key,value in {'GPU':'1','CUDNN':'0','CUDNN_HALF':'0','OPENCV':'0','LIBSO':'1'}.items():
        import re
        text=re.sub(r'^'+key+r'=\d+',key+'='+value,text,flags=re.M)
    # Ada and Ampere are compiled explicitly. FP16/cuDNN half is disabled.
    import re
    text=re.sub(r'^ARCH=.*?(?=\n\n)', 'ARCH= -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89',text,flags=re.S|re.M)
    makefile.write_text(text)
    command(['make','-C',repo,'-j','4'])
    write('configs/darknet-build.json',{'library_sha256':sha256(repo/'libdarknet.so'),
          'makefile_sha256':sha256(makefile),'cuda_compiler':subprocess.check_output(['nvcc','--version'],text=True),
          'gpu':True,'cudnn':False,'cudnn_half':False})
    write('configs/interpreters.json',interpreters)
    write('results/setup.json',{'elapsed_seconds':time.time()-start,'environment_locks':{name:sha256(Path('configs/locks')/(name+'.txt')) for name in environments}})

if __name__=='__main__': setup()
