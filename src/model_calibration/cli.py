import argparse
import json
from pathlib import Path
from .io import read


def main():
    p=argparse.ArgumentParser(description='Reproducible detector confidence benchmark')
    p.add_argument('--run',default='results/runs/coco-v1')
    p.add_argument('--annotations',default='data/annotations/instances_val2017.json')
    p.add_argument('--images',default='data/val2017')
    p.add_argument('--config',default='configs/experiment.json')
    p.add_argument('--registry',default='configs/models.json')
    sub=p.add_subparsers(dest='command',required=True)
    download=sub.add_parser('download-coco');download.add_argument('--with-images',action='store_true')
    sub.add_parser('fetch-models')
    sub.add_parser('prepare')
    infer=sub.add_parser('infer');infer.add_argument('model');infer.add_argument('--limit',type=int);infer.add_argument('--device',default='cuda:0')
    sub.add_parser('evaluate');sub.add_parser('select');sub.add_parser('calibrate')
    report=sub.add_parser('report');report.add_argument('--figures',default='results/figures')
    sub.add_parser('analyze')
    run=sub.add_parser('run');run.add_argument('--interpreters',required=True,help='JSON mapping model ID to isolated Python executable')
    sub.add_parser('status')
    draft=sub.add_parser('draft');draft.add_argument('--publication');draft.add_argument('--methodology-url',required=True)
    args=p.parse_args()
    if args.command not in ('download-coco','fetch-models'):
        from . import pipeline
    if args.command=='download-coco':
        from .acquisition import coco
        coco('data',args.with_images)
    elif args.command=='fetch-models':
        from .acquisition import fetch_models
        fetch_models(read(args.registry),'configs/provenance.lock.json')
    elif args.command=='prepare':
        pipeline.prepare(args.annotations,args.run,read(args.config),read(args.registry))
    elif args.command=='infer':
        import fcntl
        with (Path(args.run)/f'.infer-{args.model}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            pipeline.infer(args.run,args.images,args.annotations,args.model,limit=args.limit,device=args.device)
    elif args.command in ('evaluate','select','calibrate'):
        result=getattr(pipeline,args.command)(args.run,args.annotations) if args.command in ('evaluate','calibrate') else getattr(pipeline,args.command)(args.run)
        print(json.dumps({'stage':args.command,'saved':args.run}))
    elif args.command in ('report','analyze','run'):
        if args.command=='run':
            import subprocess
            pipeline.prepare(args.annotations,args.run,read(args.config),read(args.registry))
            interpreters=read(args.interpreters)
            for model in read(args.registry):
                subprocess.run([interpreters[model],'-m','model_calibration','--run',args.run,'--annotations',args.annotations,
                                '--images',args.images,'infer',model],check=True)
        if args.command in ('analyze','run'):
            pipeline.evaluate(args.run,args.annotations)
            pipeline.select(args.run)
            pipeline.calibrate(args.run,args.annotations)
        from .reporting import report
        report(args.run,getattr(args,'figures','results/figures'))
    elif args.command=='status':
        m=read(Path(args.run)/'manifest.json')
        states={name:{k:v for k,v in state.items() if k!='provenance'} for name,state in m['models'].items()}
        print(json.dumps({'synthetic':m['synthetic'],'models':states,'exclusions':m.get('exclusions',{})},indent=2))
    elif args.command=='draft':
        from .hashnode import create_draft
        result=create_draft(args.run,args.publication,args.methodology_url)
        print(json.dumps({'draft_id':result['draft_id'],'verified':result['verified']}))
