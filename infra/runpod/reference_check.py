"""Compare cached predictions with native APIs (Darknet uses its CLI)."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from model_calibration.io import read,write
from model_calibration.adapters import build


def check(model,run):
    run=Path(run);m=read(run/'manifest.json');cfg=m['registry'][model]
    ann=read('data/annotations/instances_val2017.json')
    categories=sorted(c['id'] for c in ann['categories'])
    images={r['id']:r for r in ann['images']}
    ids=sorted(images)[:2]
    a=build(model,cfg,'cuda:0')
    results=[]
    for image_id in ids:
        meta=images[image_id];path=Path('data/val2017')/meta['file_name']
        family=cfg['adapter'];contiguous=family in ('ultralytics','yolox','rtmdet','darknet')
        if family=='ultralytics':
            kw=dict(imgsz=cfg['input_size'],conf=.001,iou=.7,max_det=100,device='cuda:0',half=False,verbose=False,augment=False)
            if model=='yolo26n':kw['nms']=False
            r=a.model.predict(str(path),**kw)[0].boxes
            boxes,scores,labels=r.xyxy.cpu().numpy(),r.conf.cpu().numpy(),r.cls.cpu().numpy()
        elif family=='ssd':
            import torch
            from PIL import Image
            from torchvision.transforms.functional import to_tensor
            with Image.open(path) as im,torch.inference_mode():
                r=a.model([to_tensor(im.convert('RGB')).to('cuda:0')])[0]
            boxes,scores,labels=r['boxes'].cpu().numpy(),r['scores'].cpu().numpy(),r['labels'].cpu().numpy()
        elif family=='rfdetr':
            from PIL import Image
            with Image.open(path) as im:r=a.model.predict(im.convert('RGB'),threshold=.001)
            boxes,scores,labels=r.xyxy,r.confidence,r.class_id
        elif family=='rtmdet':
            from mmdet.apis import inference_detector
            r=inference_detector(a.model,str(path)).pred_instances.cpu().numpy()
            boxes,scores,labels=r.bboxes,r.scores,r.labels
        elif family=='yolox':
            import torch,cv2
            from yolox.data.data_augment import ValTransform
            from yolox.utils import postprocess
            im=cv2.imread(str(path));ratio=min(640/im.shape[0],640/im.shape[1])
            x,_=ValTransform()(im,None,(640,640))
            with torch.inference_mode():r=postprocess(a.model(torch.from_numpy(x).unsqueeze(0).float().cuda()),80,.001,a.exp.nmsthre)[0]
            if r is None:boxes,scores,labels=[],[],[]
            else:
                r=r.cpu().numpy();boxes,scores,labels=r[:,:4]/ratio,r[:,4]*r[:,5],r[:,6]
        elif family=='darknet':
            with tempfile.TemporaryDirectory() as folder:
                out=Path(folder)/'native.json'
                subprocess.run([str(Path(cfg['repo_dir'])/'darknet'),'detector','test',cfg['data_file'],cfg['config_file'],cfg['weight'],str(path),'-thresh','.001','-letter_box','-dont_show','-out',str(out)],check=True,stdout=subprocess.DEVNULL)
                objects=json.loads(out.read_text())[0]['objects']
            boxes=[];scores=[];labels=[]
            for obj in objects:
                b=obj['relative_coordinates'];cx=b['center_x']*meta['width'];cy=b['center_y']*meta['height']
                w=b['width']*meta['width'];h=b['height']*meta['height']
                boxes.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2]);scores.append(obj['confidence']);labels.append(obj['class_id'])
        else:raise ValueError('Unsupported active model')
        comparison_threshold=.005001 if family=='darknet' else .001
        native=[]
        for box,score,label in zip(boxes,scores,labels):
            category=categories[int(label)] if contiguous else int(label)
            if category not in categories or float(score)<comparison_threshold:continue
            native.append(dict(box=list(map(float,box)),score=float(score),category_id=category))
        native.sort(key=lambda r:-r['score']);native=native[:100]
        cached=[r for r in read(run/'predictions'/model/f'{image_id:012d}.json')['detections'] if r['score']>=comparison_threshold]
        assert len(native)==len(cached),(model,image_id,len(native),len(cached))
        box_error=score_error=0.
        for expected,actual in zip(native,cached):
            assert expected['category_id']==actual['category_id'],(model,image_id,'class mismatch',expected,actual)
            box_error=max(box_error,float(np.max(np.abs(np.array(expected['box'])-actual['box']))))
            score_error=max(score_error,abs(expected['score']-actual['score']))
        assert box_error<=.05 and score_error<=1e-4,(model,image_id,box_error,score_error)
        results.append(dict(image_id=image_id,comparison_threshold=comparison_threshold,detections=len(cached),maximum_box_error_pixels=box_error,maximum_score_error=score_error))
    write(run/'reference_checks'/f'{model}.json',dict(model_id=model,passed=True,native_reference='Darknet CLI with letter_box' if family=='darknet' else 'native package API; independent class mapping and serialization',images=results))
    print(model,results,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--run',default='results/runs/coco-v1')
    args=p.parse_args();check(args.model,args.run)
