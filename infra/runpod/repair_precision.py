import sys
from pathlib import Path
sys.path.insert(0,'src')
import torch
from model_calibration.io import read,write,sha256,digest
from model_calibration.adapters import build
from model_calibration.pipeline import normalize
run=Path('results/runs/coco-v1');m=read(run/'manifest.json');model='rfdetr_nano'
a=build(model,m['registry'][model],'cuda:0')
torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
ann=read('data/annotations/instances_val2017.json');imgs=sorted(ann['images'],key=lambda x:x['id'])
probe=imgs[51];native=normalize(a.predict(Path('data/val2017')/probe['file_name']),model,probe['id'])
assert native==read(run/'predictions'/model/f"{probe['id']:012d}.json")['detections'],'FP32 policy differs from full inference'
audit=[]
for img in imgs[:50]:
 p=run/'predictions'/model/f"{img['id']:012d}.json";old=read(p);before=sha256(p)
 records=normalize(a.predict(Path('data/val2017')/img['file_name']),model,img['id'])
 old['detections']=records;old['excluded_non_coco_class_slots']=a.excluded_records
 write(p,old)
 audit.append(dict(image_id=img['id'],before_sha256=before,after_sha256=sha256(p)))
write(run/'precision-repair.json',dict(model_id=model,reason='Initial 50 smoke images predated the NVIDIA_TF32_OVERRIDE=0 worker setting; regenerated using FP32 highest precision and verified against a full-run image',environment={'NVIDIA_TF32_OVERRIDE':'0','float32_matmul_precision':'highest','cudnn_allow_tf32':False},images=audit))
print('Repaired 50 initial RF-DETR smoke predictions under the full-run FP32 policy',flush=True)
