"""Apply explicit exclusions before selection, preserving valid predictions."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from model_calibration.io import read,write,digest,sha256


def exclude(run,registry,model,reason):
    run=Path(run);manifest=read(run/'manifest.json')
    if any((run/p).exists() for p in ('selection.json','calibrators.json')):
        raise ValueError('Exclusions must precede selection and calibration')
    removed=[model] if isinstance(model,str) else list(model)
    old=manifest['registry']
    if all(name not in old for name in removed):
        if old==registry:return manifest
        raise ValueError('Unexpected registry change')
    if {k:v for k,v in old.items() if k not in removed}!=registry:
        raise ValueError('Only the explicitly excluded models may change')
    if (run/'summary.json').exists():
        write(run/'exclusions'/f'preliminary-summary-{int(time.time())}.json',read(run/'summary.json'))
        (run/'summary.json').unlink()
    previous=manifest['protocol_hash']
    current=digest(dict(splits=manifest['splits'],settings=manifest['settings'],registry=registry))
    audit={}
    for name in registry:
        for path in sorted((run/'predictions'/name).glob('*.json')):
            shard=read(path)
            if shard['protocol_hash'] not in (previous,current):
                raise ValueError(f'Unexpected cache protocol: {path}')
            if shard['protocol_hash']==previous:
                before=sha256(path);unchanged=digest({k:v for k,v in shard.items() if k!='protocol_hash'})
                shard['protocol_hash']=current;write(path,shard)
                assert digest({k:v for k,v in read(path).items() if k!='protocol_hash'})==unchanged
                audit[str(path.relative_to(run))]=dict(before=before,after=sha256(path))
    audit_path=run/'logs'/f'exclusion-shard-hashes-{int(time.time())}.json'
    write(audit_path,audit)
    for name in removed:
        amendment=dict(model_id=name,reason=reason,recorded_at=time.time(),
            previous_protocol_hash=previous,protocol_hash=current,former_registry_entry=old[name],
            former_status=manifest['models'].get(name),
            excluded_prediction_shards=len(list((run/'predictions'/name).glob('*.json'))),
            reused_shards_count=len(audit),reused_shards_audit=str(audit_path.relative_to(run)),
            reused_shards_audit_sha256=sha256(audit_path))
        write(run/'exclusions'/f'{name}.json',amendment)
        manifest.setdefault('exclusions',{})[name]=dict(reason=reason,audit=f'exclusions/{name}.json')
        manifest['models'].pop(name,None)
    manifest['registry']=registry;manifest['protocol_hash']=current
    write(run/'manifest.json',manifest)
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='results/runs/coco-v1')
    p.add_argument('--registry',default='configs/models.json');p.add_argument('models',nargs='+');p.add_argument('--reason',required=True)
    a=p.parse_args();exclude(a.run,read(a.registry),a.models,a.reason)
