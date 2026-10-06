from pathlib import Path
import pytest
from model_calibration.io import read,write
from model_calibration.pipeline import prepare,infer,evaluate,select,calibrate
from model_calibration.reporting import report


class FakeAdapter:
    def __init__(self,offset): self.offset=offset;self.calls=0
    def provenance(self): return {'synthetic':True,'offset':self.offset,'packages':[('synthetic-adapter','1.0')]}
    def predict(self,path):
        self.calls+=1
        return [dict(box=[0,0,10,10],score=.85,category_id=1),dict(box=[30,30,40,40],score=self.offset,category_id=1)]


def make_run(tmp_path):
    cfg=read('configs/experiment.json');cfg['split_sizes']={'selection':10,'calibration':20,'final':30};cfg['bootstrap_samples']=30
    registry=read('configs/models.json')
    ann=tmp_path/'annotations.json'
    write(ann,dict(info={},images=[dict(id=i,file_name=f'{i}.jpg',width=100,height=100) for i in range(60)],
                   categories=[dict(id=1,name='person')],
                   annotations=[dict(id=i+1,image_id=i,category_id=1,bbox=[0,0,10,10],area=100,iscrowd=0) for i in range(60)]))
    run=tmp_path/'run'
    prepare(ann,run,cfg,registry,synthetic=True)
    return run,ann,cfg,registry


def test_end_to_end_resume_report_and_freezes(tmp_path):
    run,ann,cfg,registry=make_run(tmp_path)
    for name in registry:
        adapter=FakeAdapter(.99 if name=='yolo26n' else .35)
        infer(run,tmp_path,ann,name,adapter=adapter,limit=5)
        assert adapter.calls==5
        infer(run,tmp_path,ann,name,adapter=adapter)
        assert adapter.calls==60
        infer(run,tmp_path,ann,name,adapter=adapter)
        assert adapter.calls==60
    summary=evaluate(run,ann)
    selection=select(run)
    assert selection['model_id']=='yolo26n'
    with pytest.raises(ValueError,match='frozen'): infer(run,tmp_path,ann,'yolo26n',FakeAdapter(.99))
    corrected=calibrate(run)
    for result in corrected['methods'].values():
        assert result['rankings_unchanged'] and result['fixed_precision_unchanged']
    article=report(run,tmp_path/'figures')
    assert article.exists() and 'SYNTHETIC' in article.read_text()
    assert len(list((tmp_path/'figures').glob('*.png')))==6
    assert len(list((tmp_path/'figures').glob('*.svg')))==6
    old=read(run/'summary.json')
    assert evaluate(run,ann)==old
    assert select(run)==selection
    assert calibrate(run)==corrected
    assert 800<=read(run/'article_validation.json')['words']<=1100


def test_incomplete_models_are_explicit(tmp_path):
    run,ann,cfg,registry=make_run(tmp_path)
    infer(run,tmp_path,ann,'yolo26n',FakeAdapter(.99),limit=2)
    summary=evaluate(run,ann)
    assert set(summary['models'])==set(registry)
    assert summary['models']['yolo26n']['status']=='smoke_complete'
    with pytest.raises(ValueError,match='complete'): select(run)


def test_prepare_disjoint_and_reject_changed_config(tmp_path):
    run,ann,cfg,registry=make_run(tmp_path)
    splits=read(run/'manifest.json')['splits']
    assert all(not set(a)&set(b) for i,a in enumerate(splits.values()) for j,b in enumerate(splits.values()) if i!=j)
    cfg['seed']=43
    with pytest.raises(ValueError,match='different inputs'): prepare(ann,run,cfg,registry,True)


def test_explicit_exclusion_preserves_cache_and_prevents_late_changes(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('exclude_model','infra/runpod/exclude_model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    run,ann,cfg,registry=make_run(tmp_path)
    adapter=FakeAdapter(.35)
    infer(run,tmp_path,ann,'yolo11n',adapter,limit=2)
    old=read(run/'manifest.json')
    reduced={k:v for k,v in registry.items() if k!='yolox_s'}
    module.exclude(run,reduced,'yolox_s','Explicit test exclusion')
    updated=read(run/'manifest.json')
    assert updated['splits']==old['splits'] and updated['settings']==old['settings']
    assert updated['protocol_hash']!=old['protocol_hash']
    assert 'yolox_s' not in updated['models']
    assert read(run/'exclusions/yolox_s.json')['reused_shards_count']==2
    infer(run,tmp_path,ann,'yolo11n',adapter,limit=2)
    assert adapter.calls==2
    altered=dict(reduced);altered['yolo11n']=dict(reduced['yolo11n'],input_size=999)
    with pytest.raises(ValueError,match='Only'): module.exclude(run,altered,'yolov3','Invalid change')
    write(run/'selection.json',{})
    with pytest.raises(ValueError,match='precede'): module.exclude(run,reduced,'yolov3','Late change')


def test_malformed_native_box_is_retained_as_false_positive(tmp_path):
    from model_calibration.pipeline import normalize
    from model_calibration.matching import evaluate_coco
    run,ann,cfg,registry=make_run(tmp_path)
    raw=[dict(box=[0,0,10,-1],score=.8,category_id=1)]
    detections=normalize(raw,'yolo26n',1)
    assert detections[0]['box']==raw[0]['box'] and detections[0]['invalid_box']
    rows,_=evaluate_coco(ann,[dict(image_id=1,detections=detections)],[1])
    assert not rows[.5][0]['correct'] and not rows[.5][0]['ignored']
