import importlib.util
from pathlib import Path
import pytest
from model_calibration.environment import load_env
from model_calibration.io import write,read
from model_calibration.hashnode import create_draft

spec=importlib.util.spec_from_file_location('runpod_control','infra/runpod/control.py')
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)


def test_spend_includes_multiple_sessions():
    s={'pods':[{'created_at':0,'terminated_at':3600,'hourly_usd':.4},
               {'created_at':3600,'hourly_usd':.6}]}
    assert control.spent(s,7200)==pytest.approx(1)


def test_only_project_pods_can_terminate(tmp_path,monkeypatch):
    monkeypatch.setattr(control,'STATE',tmp_path/'state.json')
    write(control.STATE,{'pods':[{'id':'mine','created_at':1,'terminated_at':2,'hourly_usd':.4}]})
    with pytest.raises(ValueError): control.project_pod('unrelated')


def test_env_does_not_evaluate_shell(tmp_path,monkeypatch):
    monkeypatch.delenv('CALIBRATION_TEST_VALUE',raising=False)
    path=tmp_path/'.env'
    path.write_text("CALIBRATION_TEST_VALUE='$(echo secret)'\n")
    load_env(path)
    import os
    assert os.environ['CALIBRATION_TEST_VALUE']=='$(echo secret)'


def test_synthetic_cannot_be_uploaded(tmp_path,monkeypatch):
    monkeypatch.setenv('HASHNODE_PAT','synthetic-test-token')
    write(tmp_path/'manifest.json',{'synthetic':True})
    with pytest.raises(ValueError,match='Synthetic'):
        create_draft(tmp_path,'publication','https://example.com/methodology')


def test_active_registry_has_five_requested_detectors():
    registry=read('configs/models.json')
    assert set(registry)=={'yolo26n','yolo11n','rfdetr_nano','yolox_s','yolov3'}
    assert 'picodet_s' in read('configs/excluded-models.json')
    environments=read('configs/environments.json')
    assert {model for env in environments.values() for model in env['models']}==set(registry)
    assert set(read('configs/experiment.json')['selection_candidates'])=={'yolo26n','yolo11n','rfdetr_nano'}
