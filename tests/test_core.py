import numpy as np
import pytest
from model_calibration.metrics import metrics, bootstrap
from model_calibration.fitting import fit, Correction, grouped_validation
from model_calibration.pipeline import normalize, prepare
from model_calibration.adapters import packed, COCO_IDS
from model_calibration.io import write, read
from model_calibration.matching import evaluate_coco


def test_hand_calculations_and_boundaries():
    assert metrics([.5,.5],[1,0])['ece']==0
    assert metrics([.9,.9],[1,0])['ece']==pytest.approx(.4)
    assert metrics([.1,.1],[1,0])['ece']==pytest.approx(.4)
    m=metrics([0,.5,1],[0,1,1],2)
    assert [b['count'] for b in m['bins']]==[1,2]
    assert m['brier']==pytest.approx(.25/3)
    assert m['ece']==pytest.approx(.5/3)
    assert metrics([],[])['ece'] is None
    with pytest.raises(ValueError): metrics([float('nan')],[1])


def test_bootstrap_keeps_images_and_pairing():
    rows=[dict(image_id=1,score=.8,correct=1,ignored=False,detection_id='a'),dict(image_id=1,score=.8,correct=0,ignored=False,detection_id='b')]
    out=bootstrap(rows,[1,2],samples=100,corrected={'a':.8,'b':.8})
    assert out['valid_samples']<100 # Both draws can be the empty image.
    assert out['ci']['precision']==[.5,.5]
    assert all(interval==[0,0] for interval in out['paired_delta_ci'].values())


def test_matching_duplicate_wrong_class_crowd_empty(tmp_path):
    path=tmp_path/'gt.json'
    write(path,dict(info={},images=[dict(id=1,width=100,height=100),dict(id=2,width=100,height=100)],
                    categories=[dict(id=1,name='person'),dict(id=2,name='bike')],
                    annotations=[dict(id=1,image_id=1,category_id=1,bbox=[0,0,10,10],area=100,iscrowd=0),
                                 dict(id=2,image_id=1,category_id=1,bbox=[40,40,20,20],area=400,iscrowd=1)]))
    raw=[dict(box=[0,0,10,10],score=.9,category_id=1),dict(box=[0,0,10,10],score=.8,category_id=1),
         dict(box=[0,0,10,10],score=.7,category_id=2),dict(box=[42,42,48,48],score=.6,category_id=1),
         dict(box=[44,44,49,49],score=.5,category_id=1)]
    preds=[dict(image_id=1,detections=normalize(raw,'test',1)),dict(image_id=2,detections=[])]
    matches,quality=evaluate_coco(path,preds,[1,2])
    by_id={r['detection_id']:r for r in matches[.5]}
    assert by_id['test:1:0']['correct']==1
    assert by_id['test:1:1']['correct']==0
    assert by_id['test:1:2']['correct']==0
    assert by_id['test:1:3']['ignored'] and by_id['test:1:4']['ignored']
    assert quality['map50']==pytest.approx(1)
    matches,_=evaluate_coco(path,[dict(image_id=1,detections=[]),dict(image_id=2,detections=[])],[1,2])
    assert matches[.5]==[]


def test_class_map_and_schema():
    assert len(COCO_IDS)==80
    assert packed([[1,2,3,4]],[.8],[11])[0]['category_id']==13
    assert packed([[1,2,3,4]],[.8],[90],False)[0]['category_id']==90
    with pytest.raises(ValueError): normalize([dict(box=[0,0,1,1],score=.5,category_id=12)],'a',1)
    data=normalize([dict(box=[0,0,1,1],score=.001,category_id=1)]*110,'a',1)
    assert len(data)==100 and data[0]['detection_id']=='a:1:0'


def test_fit_improves_nll_and_preserves_ranking():
    rng=np.random.default_rng(42)
    z=rng.normal(1,1,3000)
    p=1/(1+np.exp(-z))
    y=rng.binomial(1,1/(1+np.exp(-(.6*z-.8))))
    before=metrics(p,y)['nll']
    for method in ['temperature','platt']:
        c=fit(p,y,method)
        assert c.a>0
        assert metrics(c.apply(p),y)['nll']<=before+1e-8
        assert np.array_equal(np.argsort(p),np.argsort(c.logits(p)))
    with pytest.raises(ValueError): fit([.5,.6],[1,1],'temperature')
    with pytest.raises(ValueError): Correction('platt',0)


def test_grouped_folds():
    rows=[dict(image_id=i,score=p,correct=y) for i in range(20) for p,y in [(.9,1),(.8,0)]]
    result=grouped_validation(rows,list(range(20)))
    folds=result['fold_image_ids']
    assert sorted(i for fold in folds for i in fold)==list(range(20))
    assert len(set(i for fold in folds for i in fold))==20
    assert result['selected_method'] in ['temperature','platt']


def test_rf_sparse_inactive_slots_are_recorded(tmp_path):
    from types import SimpleNamespace
    from PIL import Image
    from model_calibration.adapters import RFDETRAdapter
    path=tmp_path/'image.png'
    Image.new('RGB',(20,20)).save(path)
    native=SimpleNamespace(xyxy=np.array([[0,0,10,10]]*3),confidence=np.array([.9,.004,.8]),class_id=np.array([1,45,90]))
    adapter=object.__new__(RFDETRAdapter)
    adapter.model=SimpleNamespace(predict=lambda image,threshold:native)
    result=adapter.predict(path)
    assert [d['category_id'] for d in result]==[1,90]
    assert result[0]['score']==.9
    assert adapter.excluded_records[0]['category_id']==45
    assert adapter.excluded_records[0]['score']==.004
