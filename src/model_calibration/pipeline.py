import csv
import json
import importlib.metadata
import platform
import time
from pathlib import Path
import numpy as np
from .io import read, write, digest, sha256
from .matching import evaluate_coco
from .metrics import metrics, bootstrap
from .fitting import Correction, grouped_validation


def prepare(annotations, run, config, registry, synthetic=False):
    run = Path(run)
    ann = read(annotations)
    ids = sorted(x['id'] for x in ann['images'])
    sizes = config['split_sizes']
    if len(ids) != len(set(ids)) or len(ids) != sum(sizes.values()):
        raise ValueError('Annotation image count must equal disjoint split sizes')
    if not synthetic and (len(ids)!=5000 or sizes!={'selection':1000,'calibration':1500,'final':2500}):
        raise ValueError('Production protocol requires all 5000 COCO val2017 images')
    np.random.default_rng(config['seed']).shuffle(ids)
    splits, offset = {}, 0
    for name, size in sizes.items():
        splits[name] = sorted(ids[offset:offset+size])
        offset += size
    manifest = dict(schema_version=1, synthetic=synthetic, annotations_sha256=sha256(annotations),
                    splits=splits, settings=config, registry=registry,
                    protocol_hash=digest({'splits':splits,'settings':config,'registry':registry}),
                    annotations_path=str(Path(annotations).resolve()),
                    created_at=time.time(), python=platform.python_version(), host=platform.platform(),
                    models={k:{'status':'pending'} for k in registry})
    if (run/'manifest.json').exists():
        old = read(run/'manifest.json')
        if old['protocol_hash'] != manifest['protocol_hash'] or old['annotations_sha256'] != manifest['annotations_sha256']:
            raise ValueError('Refusing to replace an existing run with different inputs')
        return old
    write(run/'manifest.json',manifest)
    return manifest


def normalize(raw, model_id, image_id, threshold=.001, maximum=100):
    from .adapters import COCO_IDS
    output = []
    for d in raw:
        box = list(map(float, d['box']))
        score, category = float(d['score']), int(d['category_id'])
        if len(box)!=4 or not np.isfinite(box).all() :
            raise ValueError('Invalid original-image xyxy box')
        if not np.isfinite(score) or not 0<=score<=1 or category not in COCO_IDS:
            raise ValueError('Invalid score or COCO category')
        if score >= threshold:
            record=dict(box=box,score=score,category_id=category)
            if box[2]<box[0] or box[3]<box[1]:
                record['invalid_box']=True
            output.append(record)
    output.sort(key=lambda d:-d['score']) # Stable native-order tie handling.
    for index,d in enumerate(output[:maximum]):
        d['detection_id'] = f'{model_id}:{image_id}:{index}'
    return output[:maximum]


def save_model_status(run, model_id, status):
    import fcntl
    run=Path(run)
    with (run/'manifest.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        current=read(run/'manifest.json')
        current['models'][model_id]=status
        write(run/'manifest.json',current)
        fcntl.flock(lock,fcntl.LOCK_UN)


def infer(run, images, annotations, model_id, adapter=None, limit=None, device='cuda:0'):
    run = Path(run)
    manifest = read(run/'manifest.json')
    if sha256(annotations)!=manifest['annotations_sha256']:
        raise ValueError('Annotations changed')
    if (run/'selection.json').exists():
        raise ValueError('Inference is frozen after model selection; use a fresh run')
    model = manifest['registry'][model_id]
    ids = sorted(i for group in manifest['splits'].values() for i in group)
    if limit is not None:
        ids = ids[:limit]
    metadata = {x['id']:x for x in read(annotations)['images']}
    status = manifest['models'][model_id]
    start = time.monotonic()
    try:
        if adapter is None:
            import os
            os.environ['NVIDIA_TF32_OVERRIDE']='0'
            from .adapters import build
            adapter = build(model_id, model, device)
            if model['adapter']!='picodet':
                import torch
                # Libraries can change global matmul defaults during import.
                torch.set_float32_matmul_precision('highest')
                torch.backends.cudnn.allow_tf32=False
        provenance = json.loads(json.dumps(adapter.provenance(),allow_nan=False))
        if status.get('provenance') and status['provenance']!=provenance:
            raise ValueError('Adapter provenance differs from cached predictions')
        status.update(status='running', provenance=provenance)
        save_model_status(run,model_id,status)
        for i in ids:
            path = run/'predictions'/model_id/f'{i:012d}.json'
            if path.exists():
                cached = read(path)
                if cached['image_id']!=i or cached['protocol_hash']!=manifest['protocol_hash'] or cached['provenance_hash']!=digest(provenance):
                    raise ValueError('Prediction cache identity mismatch')
                continue
            detections = normalize(adapter.predict(Path(images)/metadata[i]['file_name']), model_id, i,
                                   manifest['settings']['inference_threshold'],manifest['settings']['max_detections'])
            write(path,dict(image_id=i,detections=detections,protocol_hash=manifest['protocol_hash'],
                            provenance_hash=digest(provenance),excluded_non_coco_class_slots=getattr(adapter,'excluded_records',[])))
        all_ids = [i for group in manifest['splits'].values() for i in group]
        done = sum((run/'predictions'/model_id/f'{i:012d}.json').exists() for i in all_ids)
        status.update(status='complete' if done==len(all_ids) else 'smoke_complete',completed_images=done)
        status.pop('error',None)
    except Exception as e:
        status.update(status='failed',error=f'{type(e).__name__}: {e}')
        raise
    finally:
        status['elapsed_seconds'] = status.get('elapsed_seconds',0)+time.monotonic()-start
        save_model_status(run,model_id,status)


def load_predictions(run, model_id, ids):
    manifest = read(Path(run)/'manifest.json')
    if manifest['models'][model_id]['status']!='complete':
        raise ValueError(f'{model_id} inference is incomplete')
    records = [read(Path(run)/'predictions'/model_id/f'{i:012d}.json') for i in ids]
    for image_id,r in zip(ids,records):
        if r['image_id']!=image_id or r['protocol_hash']!=manifest['protocol_hash'] or r['provenance_hash']!=digest(manifest['models'][model_id]['provenance']):
            raise ValueError('Stale predictions')
        if normalize(r['detections'],model_id,image_id,manifest['settings']['inference_threshold'],100)!=r['detections']:
            raise ValueError('Prediction schema or stable ID mismatch')
    return records


def eligible(rows, threshold):
    return [r for r in rows if not r['ignored'] and r['score']>=threshold]


def evaluate(run, annotations):
    run = Path(run)
    manifest = read(run/'manifest.json')
    if sha256(annotations)!=manifest['annotations_sha256']:
        raise ValueError('Annotations changed')
    cfg = manifest['settings']
    analysis_start=time.monotonic()
    analysis_provenance=dict(python=platform.python_version(),host=platform.platform(),
        packages={name:importlib.metadata.version(name) for name in ('numpy','scipy','matplotlib','pycocotools')},
        source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},protocol_hash=manifest['protocol_hash'])
    summary = dict(protocol_hash=manifest['protocol_hash'],synthetic=manifest['synthetic'],models={})
    for model_id in manifest['registry']:
        status = manifest['models'][model_id]['status']
        if status!='complete':
            summary['models'][model_id] = dict(status=status,error=manifest['models'][model_id].get('error'))
            continue
        fingerprint=digest(dict(protocol_hash=manifest['protocol_hash'],annotations_sha256=manifest['annotations_sha256'],
            model_provenance_hash=digest(manifest['models'][model_id]['provenance']),
            source={name:sha256(Path(__file__).parent/name) for name in ('pipeline.py','matching.py','metrics.py')},
            packages={name:importlib.metadata.version(name) for name in ('numpy','pycocotools')},
            predictions={p.name:sha256(p) for p in sorted((run/'predictions'/model_id).glob('*.json'))}))
        cache_path=run/'evaluations'/f'{model_id}.json'
        if cache_path.exists():
            cached=read(cache_path)
            if cached['fingerprint']==fingerprint and cached['result_hash']==digest(cached['result']) and all(
                (run/path).exists() and sha256(run/path)==expected for path,expected in cached['matches'].items()):
                summary['models'][model_id]=cached['result']
                continue
        output = dict(status='complete',splits={})
        for split,ids in manifest['splits'].items():
            predictions = load_predictions(run,model_id,ids)
            matched,quality = evaluate_coco(annotations,predictions,ids)
            write(run/'matches'/model_id/f'{split}.json', {str(k):v for k,v in matched.items()})
            rows = eligible(matched[cfg['primary_iou']],cfg['primary_threshold'])
            primary = metrics([r['score'] for r in rows],[r['correct'] for r in rows],cfg['bins'])
            primary.update(bootstrap(rows,ids,cfg['bins'],cfg['bootstrap_samples'],cfg['seed']))
            sensitivity = []
            if split=='final':
                settings = [(t,.5,15) for t in cfg['sensitivity_thresholds']]+[(.25,.75,15)]+[(.25,.5,b) for b in cfg['sensitivity_bins']]
                for threshold,iou,bins in settings:
                    r = eligible(matched[iou],threshold)
                    m = metrics([d['score'] for d in r],[d['correct'] for d in r],bins)
                    m.update(bootstrap(r,ids,bins,cfg['bootstrap_samples'],cfg['seed']))
                    sensitivity.append(dict(threshold=threshold,iou=iou,bins_count=bins,metrics=m))
            diagnostics=dict(invalid_boxes=sum(bool(d.get('invalid_box')) for image in predictions for d in image['detections']),
                             invalid_boxes_above_primary=sum(bool(d.get('invalid_box')) and d['score']>=cfg['primary_threshold'] for image in predictions for d in image['detections']),
                             excluded_non_coco_slots=sum(len(image.get('excluded_non_coco_class_slots',[])) for image in predictions),
                             excluded_non_coco_slots_above_primary=sum(d['score']>=cfg['primary_threshold'] for image in predictions for d in image.get('excluded_non_coco_class_slots',[])))
            output['splits'][split] = dict(primary=primary,quality=quality,sensitivity=sensitivity,diagnostics=diagnostics)
        summary['models'][model_id] = output
        write(cache_path,dict(fingerprint=fingerprint,result=output,result_hash=digest(output),
            matches={str(p.relative_to(run)):sha256(p) for p in (run/'matches'/model_id).glob('*.json')}))
    write(run/'summary.json',summary)
    analysis_provenance['elapsed_seconds']=time.monotonic()-analysis_start
    analysis_provenance['completed_at']=time.time()
    write(run/'analysis_provenance.json',analysis_provenance)
    return summary


def select(run):
    run = Path(run)
    manifest,summary = read(run/'manifest.json'),read(run/'summary.json')
    if summary['protocol_hash']!=manifest['protocol_hash']:
        raise ValueError('Summary protocol mismatch')
    # Require all registered models, preventing silent selection from a partial benchmark.
    if set(summary['models'])!=set(manifest['registry']) or any(v['status']!='complete' for v in summary['models'].values()):
        raise ValueError('All registered models must complete before selection')
    candidates = manifest['settings']['selection_candidates']
    ranked = []
    for k in candidates:
        m = summary['models'][k]['splits']['selection']['primary']
        if m['ece'] is None:
            raise ValueError(f'No selection detections for {k}')
        ranked.append(dict(model_id=k,ece=m['ece'],nll=m['nll'],ci=m['ci']['ece']))
    ranked.sort(key=lambda x:(-x['ece'],-x['nll'],x['model_id']))
    winner = ranked[0]
    uncertain = [r['model_id'] for r in ranked[1:] if max(r['ci'][0],winner['ci'][0])<=min(r['ci'][1],winner['ci'][1])]
    decision = dict(model_id=winner['model_id'],ranking=ranked,overlapping_ece_intervals=uncertain,
                    protocol_hash=manifest['protocol_hash'],selection_summary_hash=digest(summary),
                    criterion='selection ECE descending; exact ties NLL descending, model ID ascending')
    path = run/'selection.json'
    if path.exists() and read(path)!=decision:
        raise ValueError('Selection is frozen; start a new run to change it')
    write(path,decision)
    return decision


def calibrate(run, annotations=None):
    run = Path(run)
    manifest,decision = read(run/'manifest.json'),read(run/'selection.json')
    cfg = manifest['settings']
    annotations=annotations or manifest.get('annotations_path','data/annotations/instances_val2017.json')
    if sha256(annotations)!=manifest['annotations_sha256']:
        raise ValueError('Annotations changed')
    if decision['protocol_hash']!=manifest['protocol_hash'] or decision['selection_summary_hash']!=digest(read(run/'summary.json')):
        raise ValueError('Frozen selection inputs changed')
    model_id = decision['model_id']
    data = read(run/'matches'/model_id/'calibration.json')
    rows = eligible(data[str(cfg['primary_iou'])],cfg['primary_threshold'])
    frozen = grouped_validation(rows,manifest['splits']['calibration'],cfg['folds'],cfg['seed'],cfg['tie_tolerance'])
    frozen.update(model_id=model_id,calibration_data_hash=digest(data),protocol_hash=manifest['protocol_hash'])
    path = run/'calibrators.json'
    if path.exists() and read(path)!=frozen:
        raise ValueError('Calibration is frozen; inputs changed')
    write(path,frozen) # Persist method BEFORE reading final matches.
    final = read(run/'matches'/model_id/'final.json')
    rows = eligible(final[str(cfg['primary_iou'])],cfg['primary_threshold'])
    before = metrics([r['score'] for r in rows],[r['correct'] for r in rows],cfg['bins'])
    results = dict(model_id=model_id,selected_method=frozen['selected_method'],before=before,methods={},
                   fixed_population_ids_hash=digest(sorted(r['detection_id'] for r in rows)))
    predictions=load_predictions(run,model_id,manifest['splits']['final'])
    original_quality=read(run/'summary.json')['models'][model_id]['splits']['final']['quality']
    for method, params in frozen['parameters'].items():
        correction = Correction(method,params['a'],params['b'])
        p = correction.apply([r['score'] for r in rows])
        corrected = dict(zip([r['detection_id'] for r in rows],map(float,p)))
        after = metrics(p,[r['correct'] for r in rows],cfg['bins'])
        interval = bootstrap(rows,manifest['splits']['final'],cfg['bins'],cfg['bootstrap_samples'],cfg['seed'],corrected)
        # Rankings are preserved mathematically by the positive affine logit map.
        # Store logits as well: sigmoid rounding can create ties at extremes.
        logits = correction.logits([r['score'] for r in rows])
        assert np.array_equal(np.argsort([r['score'] for r in rows],kind='stable'),np.argsort(logits,kind='stable'))
        assert before['count']==after['count'] and before['precision']==after['precision']
        # COCO accepts arbitrary scores for sorting. Corrected logits retain
        # all original detections and avoid numerical sigmoid saturation ties.
        ranked=[]
        for image in predictions:
            mapped=correction.logits([d['score'] for d in image['detections']])
            ranked.append(dict(image_id=image['image_id'],detections=[dict(d,score=float(z)) for d,z in zip(image['detections'],mapped)]))
        _,corrected_quality=evaluate_coco(annotations,ranked,manifest['splits']['final'])
        assert all(np.isclose(original_quality[k],corrected_quality[k],rtol=0,atol=1e-12) for k in original_quality)
        results['methods'][method] = dict(after=after,delta={k:after[k]-before[k] for k in ('ece','nll','brier')},
                                         paired_delta_ci=interval['paired_delta_ci'],parameters=params,
                                         fixed_precision_unchanged=True,rankings_unchanged=True,
                                         map_unchanged=True,quality_before=original_quality,quality_after=corrected_quality,
                                         map_check='COCO re-evaluated with corrected logits for sorting, on all original retained detections')
    write(run/'correction_results.json',results)
    return results


def export_tables(run):
    run = Path(run)
    summary = read(run/'summary.json')
    records, bins = [], []
    for model_id,m in summary['models'].items():
        if m['status']!='complete':
            records.append(dict(model_id=model_id,status=m['status']))
            continue
        for split,result in m['splits'].items():
            primary = result['primary']
            row = dict(model_id=model_id,status='complete',split=split,
                       **{k:v for k,v in primary.items() if k not in ('bins','ci','paired_delta_ci')},**result['quality'])
            for metric,interval in primary['ci'].items():
                row[metric+'_ci_lower'],row[metric+'_ci_upper'] = interval or [None,None]
            records.append(row)
            bins.extend(dict(model_id=model_id,split=split,**b) for b in primary['bins'])
    for name,rows in [('metrics',records),('bins',bins)]:
        fields = sorted(set(k for r in rows for k in r))
        with (run/f'{name}.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
