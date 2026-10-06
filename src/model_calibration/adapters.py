"""Lazy native-reference adapters; inference environments never share imports."""
import importlib.metadata
import inspect
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from .io import sha256, read, digest

COCO_IDS = (1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90)


def packed(boxes, scores, classes, contiguous=True):
    records = []
    for b,p,c in zip(boxes,scores,classes):
        c = int(c)
        if contiguous:
            if not 0<=c<len(COCO_IDS):
                raise ValueError('Contiguous COCO class outside 0..79')
            c = COCO_IDS[c]
        records.append(dict(box=list(map(float,b)),score=float(p),category_id=c))
    return records


class Adapter:
    def __init__(self, model_id, settings, device):
        self.model_id,self.settings,self.device = model_id,settings,device
        self.weight = Path(settings['weight'])
        if not self.weight.exists():
            raise FileNotFoundError(f'{self.weight}: run fetch-models first')
        self.lock = read('configs/provenance.lock.json')
        entry = self.lock['models'][model_id]
        if entry['artifacts'].get(str(self.weight))!=sha256(self.weight):
            raise ValueError('Checkpoint differs from frozen provenance')
        for path,expected in entry['artifacts'].items():
            if sha256(path)!=expected:
                raise ValueError(f'Artifact changed: {path}')
        if settings.get('repo_dir'):
            revision = subprocess.check_output(['git','-C',settings['repo_dir'],'rev-parse','HEAD'],text=True).strip()
            if revision!=entry['repository_revision']:
                raise ValueError('Reference repository revision changed')
        if device.startswith('cuda'):
            # FP32 requires disabling TF32 as well as autocast/half.
            if settings['adapter']!='picodet':
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError('Requested CUDA device is unavailable')
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
        for package,version in settings['packages'].items():
            if importlib.metadata.version(package)!=version:
                raise ValueError(f'Pinned package version required: {package}=={version}')

    def provenance(self):
        packages = sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())
        hardware = {'platform':sys.platform,'device':self.device}
        if self.device.startswith('cuda') and self.settings['adapter']!='picodet':
            import torch
            hardware.update(gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda)
        elif self.settings['adapter']=='picodet':
            import paddle
            hardware.update(gpu=paddle.device.cuda.get_device_name(),cuda=paddle.version.cuda())
        return dict(checkpoint_sha256=sha256(self.weight),settings=self.settings,
                    lock=self.lock['models'][self.model_id],packages=packages,hardware=hardware,precision='FP32',
                    adapter_source_sha256=digest([inspect.getsource(type(self)),inspect.getsource(Adapter),inspect.getsource(packed),COCO_IDS]))


class UltralyticsAdapter(Adapter):
    def __init__(self,*args):
        super().__init__(*args)
        from ultralytics import YOLO
        self.model = YOLO(str(self.weight))
        self.model.model.float()

    def predict(self,path):
        kwargs = dict(imgsz=self.settings['input_size'],conf=.001,iou=.7,max_det=100,
                      device=self.device,half=False,verbose=False,augment=False)
        if self.model_id=='yolo26n':
            kwargs.update(nms=False)
        boxes = self.model.predict(str(path),**kwargs)[0].boxes
        return packed(boxes.xyxy.cpu().numpy(),boxes.conf.cpu().numpy(),boxes.cls.cpu().numpy())


class SSDAdapter(Adapter):
    def __init__(self,*args):
        super().__init__(*args)
        import torch
        from torchvision.models.detection import ssd300_vgg16
        self.model = ssd300_vgg16(weights=None,weights_backbone=None,score_thresh=.001,detections_per_img=100)
        self.model.load_state_dict(torch.load(self.weight,map_location='cpu',weights_only=True))
        self.model.float().to(self.device).eval()

    def predict(self,path):
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        with Image.open(path) as image,torch.inference_mode():
            result = self.model([to_tensor(image.convert('RGB')).to(self.device)])[0]
        return packed(result['boxes'].cpu(),result['scores'].cpu(),result['labels'].cpu(),False)


class RFDETRAdapter(Adapter):
    def __init__(self,*args):
        super().__init__(*args)
        from rfdetr import RFDETRNano
        self.model = RFDETRNano(pretrain_weights=str(self.weight),device=self.device,resolution=384)
        self.model.model.model.float().eval()

    def predict(self,path):
        from PIL import Image
        with Image.open(path) as image:
            result = self.model.predict(image.convert('RGB'),threshold=.001)
        # Official COCO checkpoints use the sparse 1..90 category convention.
        native=packed(result.xyxy,result.confidence,result.class_id,False)
        self.excluded_records=[d for d in native if d['category_id'] not in COCO_IDS]
        return [d for d in native if d['category_id'] in COCO_IDS]


class YOLOXAdapter(Adapter):
    def __init__(self,*args):
        super().__init__(*args)
        import torch
        sys.path.insert(0,str(Path(self.settings['repo_dir']).resolve()))
        from yolox.exp import get_exp
        self.exp = get_exp(None,'yolox-s')
        self.exp.test_size=(640,640)
        self.exp.test_conf=.001
        self.model=self.exp.get_model().float().to(self.device).eval()
        self.model.load_state_dict(torch.load(self.weight,map_location='cpu')['model'])

    def predict(self,path):
        import cv2
        import torch
        from yolox.data.data_augment import ValTransform
        from yolox.utils import postprocess
        image=cv2.imread(str(path))
        if image is None:
            raise ValueError('Cannot decode image')
        ratio=min(640/image.shape[0],640/image.shape[1])
        x,_=ValTransform(legacy=False)(image,None,(640,640))
        with torch.inference_mode():
            output=postprocess(self.model(torch.from_numpy(x).unsqueeze(0).float().to(self.device)),80,.001,self.exp.nmsthre,class_agnostic=False)[0]
        if output is None:
            return []
        output=output.cpu().numpy()
        return packed(output[:,:4]/ratio,output[:,4]*output[:,5],output[:,6])


class RTMDetAdapter(Adapter):
    def __init__(self,*args):
        super().__init__(*args)
        from mmdet.apis import init_detector
        from mmengine.config import Config
        cfg=Config.fromfile(self.settings['config_file'])
        cfg.model.test_cfg.score_thr=.001
        cfg.model.test_cfg.max_per_img=100
        self.model=init_detector(cfg,str(self.weight),device=self.device).float().eval()

    def predict(self,path):
        from mmdet.apis import inference_detector
        result=inference_detector(self.model,str(path)).pred_instances.cpu().numpy()
        return packed(result.bboxes,result.scores,result.labels)


class PicoDetAdapter(Adapter):
    def __init__(self,*args):
        super().__init__(*args)
        sys.path.insert(0,str((Path(self.settings['repo_dir'])/'deploy/python').resolve()))
        from infer import DetectorPicoDet
        import paddle
        paddle.enable_static()
        self.model=DetectorPicoDet(str(self.weight.parent),device='GPU' if self.device.startswith('cuda') else 'CPU',
                                  run_mode='paddle',threshold=.001)
        # Keep the exported FP32 graph without unsupported cuDNN fusion passes.
        self.model.config.switch_ir_optim(False)
        self.model.predictor=paddle.inference.create_predictor(self.model.config)
        # The no-postprocess export lets us lower hidden filtering in Python.
        import infer
        from functools import partial
        from picodet_postprocess import PicoDetPostProcess
        infer.PicoDetPostProcess = partial(PicoDetPostProcess, score_threshold=.001, keep_top_k=100)

    def predict(self,path):
        result=self.model.predict_image([str(path)],visual=False)
        boxes=result['boxes']
        return packed(boxes[:,2:6],boxes[:,1],boxes[:,0])


class DarknetAdapter(Adapter):
    def __init__(self,*args):
        super().__init__(*args)
        # Native Darknet handles letterboxing, objectness*class probabilities,
        # and classwise NMS. A shared-library build is required in its own env.
        sys.path.insert(0,str(Path(self.settings['repo_dir']).resolve()))
        previous=os.getcwd()
        try:
            os.chdir(self.settings['repo_dir'])
            import darknet
        finally:
            os.chdir(previous)
        self.darknet=darknet
        self.native_build=read('configs/darknet-build.json')
        if sha256(Path(self.settings['repo_dir'])/'libdarknet.so')!=self.native_build['library_sha256']:
            raise ValueError('Darknet library differs from saved build provenance')
        self.network,self.names,_=darknet.load_network(self.settings['config_file'],self.settings['data_file'],str(self.weight),batch_size=1)
        if darknet.network_width(self.network)!=416 or darknet.network_height(self.network)!=416:
            raise ValueError('Darknet cfg input must be 416')

    def provenance(self):
        return super().provenance() | {'native_build':self.native_build}

    def predict(self,path):
        import ctypes
        dn=self.darknet
        image=dn.load_image(bytes(str(path),'utf8'),0,0)
        try:
            dn.predict_image_letterbox(self.network,image)
            count=ctypes.c_int(0)
            detections=dn.get_network_boxes(self.network,image.w,image.h,.001,.5,None,0,ctypes.pointer(count),1)
            try:
                dn.do_nms_sort(detections,count.value,len(self.names),.45)
                output=[]
                for i in range(count.value):
                    box=detections[i].bbox
                    for c in range(len(self.names)):
                        score=detections[i].prob[c]
                        if score>=.001:
                            output.append(dict(box=[box.x-box.w/2,box.y-box.h/2,box.x+box.w/2,box.y+box.h/2],score=score,category_id=COCO_IDS[c]))
                return output
            finally:
                dn.free_detections(detections,count.value)
        finally:
            dn.free_image(image)


def build(model_id,settings,device):
    classes={'ultralytics':UltralyticsAdapter,'ssd':SSDAdapter,'rfdetr':RFDETRAdapter,
             'yolox':YOLOXAdapter,'rtmdet':RTMDetAdapter,'picodet':PicoDetAdapter,'darknet':DarknetAdapter}
    return classes[settings['adapter']](model_id,settings,device)
