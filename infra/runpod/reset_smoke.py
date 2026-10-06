"""Archive a failed smoke attempt before an adapter correction, never final data."""
import argparse
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from model_calibration.io import read,write

p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--run',default='results/runs/coco-v1')
a=p.parse_args();run=Path(a.run);m=read(run/'manifest.json')
if (run/'selection.json').exists() or m['models'][a.model]['status'] not in ('failed','smoke_complete'):
    raise ValueError('Only an unselected smoke attempt can be archived')
folder=run/'predictions'/a.model
history=run/'logs'/f'{a.model}-archived-{int(time.time())}'
history.mkdir(parents=True)
write(history/'status.json',m['models'][a.model])
if folder.exists(): folder.rename(history/'predictions')
m['models'][a.model]={'status':'pending','archived_smoke':str(history)}
write(run/'manifest.json',m)
