import os
from pathlib import Path
import numpy as np
from .io import read, write, digest, sha256
from .pipeline import eligible, export_tables
from .fitting import Correction


def figures(run,output):
    os.environ.setdefault('MPLCONFIGDIR',str(Path('/tmp/model-calibration-mpl')))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run,output=Path(run),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    summary=read(run/'summary.json')
    correction=read(run/'correction_results.json')
    names=list(summary['models'])
    final=[summary['models'][k]['splits']['final'] for k in names]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(fig,name):
        if summary['synthetic']:
            fig.suptitle('SYNTHETIC PIPELINE TEST — NOT BENCHMARK EVIDENCE',fontsize=11,color='crimson')
        fig.tight_layout()
        for ext in ('png','svg'):
            fig.savefig(output/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.8))
    values=np.array([m['primary']['ece'] for m in final])
    cis=np.array([m['primary']['ci']['ece'] for m in final])
    # Percentile intervals need not contain the point estimate. Plot endpoints.
    x=np.arange(len(names))
    ax.bar(x,values*100,color='#377eb8')
    ax.vlines(x,cis[:,0]*100,cis[:,1]*100,color='black')
    ax.scatter(x,cis[:,0]*100,marker='_',color='black')
    ax.scatter(x,cis[:,1]*100,marker='_',color='black')
    ax.set(xticks=x,xticklabels=names,ylabel='ECE (percentage points)',title='Final COCO split: emitted scores ≥0.25, IoU ≥0.50')
    ax.tick_params(axis='x',rotation=25)
    save(fig,'calibration_comparison')
    fig,axes=plt.subplots((len(names)+1)//2,4,figsize=(16,3.5*((len(names)+1)//2)),squeeze=False)
    for i,(name,m) in enumerate(zip(names,final)):
        reliability(axes[i//2,(i%2)*2],m['primary']['bins'],name)
        rows=eligible(read(run/'matches'/name/'final.json')['0.5'],.25)
        ax=axes[i//2,(i%2)*2+1]
        ax.hist([r['score'] for r in rows],bins=np.linspace(0,1,16),color='#377eb8')
        ax.set(xlim=(0,1),xlabel='Original confidence',ylabel='Detection count',title=name)
    for ax in axes.flat[2*len(names):]:
        ax.set_visible(False)
    save(fig,'reliability_all')
    fig,ax=plt.subplots(figsize=(8,5))
    for name,m in zip(names,final):
        ax.scatter(m['quality']['map']*100,m['primary']['ece']*100)
        ax.annotate(name,(m['quality']['map']*100,m['primary']['ece']*100),xytext=(5,5),textcoords='offset points')
    ax.set(xlabel='COCO mAP (%, scores ≥0.001)',ylabel='ECE (percentage points, scores ≥0.25)',title='Detection quality and confidence calibration')
    save(fig,'quality_vs_calibration')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    reliability(axes[0],correction['before']['bins'],'Original: '+correction['model_id'])
    for ax,(method,result) in zip(axes[1:],correction['methods'].items()):
        label=method+(' (chosen by calibration CV)' if method==correction['selected_method'] else '')
        reliability(ax,result['after']['bins'],label)
    save(fig,'correction')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,metric in zip(axes,['ece','nll','brier']):
        for i,(method,result) in enumerate(correction['methods'].items()):
            lo,hi=result['paired_delta_ci'][metric]
            ax.plot([lo,hi],[i,i],color='#377eb8')
            ax.scatter(result['delta'][metric],i,color='#377eb8')
        ax.axvline(0,color='gray',linestyle='--')
        ax.set(yticks=[0,1],yticklabels=list(correction['methods']),xlabel='After minus before',title=metric.upper())
    save(fig,'metric_changes')
    fig,ax=plt.subplots(figsize=(10,4.8))
    y=np.arange(len(names))
    labels={'yolo26n':'YOLO26n','yolo11n':'YOLO11n','rfdetr_nano':'RF-DETR Nano','yolox_s':'YOLOX-S','yolov3':'YOLOv3'}
    confidence=ax.barh(y-.18,[m['primary']['mean_confidence']*100 for m in final],height=.34,color='#377eb8',label='Average confidence')
    precision=ax.barh(y+.18,[m['primary']['precision']*100 for m in final],height=.34,color='#e58a24',label='Correct detections')
    for bars in (confidence,precision):
        ax.bar_label(bars,fmt='%.1f%%',padding=4,fontsize=10)
    ax.set(yticks=y,yticklabels=[labels.get(name,name) for name in names],xlim=(0,100),xlabel='Percent',title='How confident is the detector — and how often is it right?')
    ax.invert_yaxis()
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,frameon=False)
    save(fig,'confidence_vs_precision')


def reliability(ax,bins,title):
    occupied=[b for b in bins if b['count']]
    ax.plot([0,1],[0,1],linestyle='--',color='gray')
    ax.plot([b['confidence'] for b in occupied],[b['precision'] for b in occupied],marker='o',color='#377eb8')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean confidence',ylabel='Observed precision',title=title)


def render_article(run,template='article/article.template.md'):
    run=Path(run)
    summary,correction=read(run/'summary.json'),read(run/'correction_results.json')
    cfg=read(run/'manifest.json')['settings']
    final={k:v['splits']['final']['primary'] for k,v in summary['models'].items()}
    best=min(final,key=lambda k:final[k]['ece'])
    chosen=correction['selected_method']
    result=correction['methods'][chosen]
    def pct(x):
        return f'{100*x:.2f}'
    uncertainty=read(run/'selection.json')['overlapping_ece_intervals']
    lo,hi=result['paired_delta_ci']['ece']
    conclusion=('The interval supports an improvement for these detections.' if hi<0 else
                'The interval supports worse calibration for these detections.' if lo>0 else
                'The result is uncertain because the interval includes zero.')
    older=[final['yolov3']['ece']]
    recent=[final[k]['ece'] for k in ('yolo26n','yolo11n','rfdetr_nano')]
    age_result=('YOLOv3 has lower estimated ECE than all three recent candidates in this run.' if max(older)<min(recent) else
                'YOLOv3 has higher estimated ECE than all three recent candidates in this run.' if min(older)>max(recent) else
                'YOLOv3 and the recent candidates do not show a consistent calibration ordering by age.')
    params=result['parameters']
    mapped_090=float(Correction(chosen,params['a'],params['b']).apply([.9])[0])
    values=dict(mean_confidence=pct(correction['before']['mean_confidence']),example_090=f'{mapped_090:.3f}',age_result=age_result,selected=correction['model_id'],chosen=chosen,best=best,best_ece=pct(final[best]['ece']),
                before_ece=pct(correction['before']['ece']),after_ece=pct(result['after']['ece']),
                delta_ece=pct(result['delta']['ece']),delta_lo=pct(lo),delta_hi=pct(hi),
                before_nll=f"{correction['before']['nll']:.4f}",after_nll=f"{result['after']['nll']:.4f}",
                count=str(correction['before']['count']),precision=pct(correction['before']['precision']),
                parameters=', '.join(f'{k}={v:.5f}' for k,v in result['parameters'].items() if isinstance(v,float)),
                conclusion=conclusion,selection_uncertainty=('Selection intervals overlap for '+', '.join(uncertainty)+'. Treat the ranking as uncertain.' if uncertainty else 'The selected checkpoint’s ECE interval does not overlap the other two candidate intervals.'),
                methodology='../README.md',figure_prefix='../results/figures')
    display={'yolo26n':'YOLO26n','yolo11n':'YOLO11n','rfdetr_nano':'RF-DETR Nano','yolox_s':'YOLOX-S','yolov3':'YOLOv3','platt':'Platt correction','temperature':'temperature correction'}
    for key in ('selected','chosen','best'):
        values[key]=display.get(values[key],values[key])
    article=Path(template).read_text()
    for key,value in values.items():
        article=article.replace('{{'+key+'}}',value)
    if '{{' in article:
        raise ValueError('Unresolved article placeholders')
    words=len(article.split())
    if not 800<=words<=1100:
        raise ValueError(f'Article must be 800–1100 words; got {words}')
    if summary['synthetic']:
        article='> SYNTHETIC PIPELINE TEST. No empirical detector claims.\n\n'+article
    write(run/'article_validation.json',dict(words=words,summary_hash=digest(summary),correction_hash=digest(correction),synthetic=summary['synthetic']))
    path=run/'article.md' if summary['synthetic'] else Path('article/article.md')
    path.write_text(article)
    validation=read(run/'article_validation.json')
    validation['article_sha256']=sha256(path)
    write(run/'article_validation.json',validation)
    return path


def report(run,output='results/figures'):
    run=Path(run)
    summary=read(run/'summary.json')
    if set(summary['models'])!=set(read(run/'manifest.json')['registry']) or any(m['status']!='complete' for m in summary['models'].values()):
        raise ValueError('Final reports require complete registered-model coverage')
    if any(m['splits']['final']['primary']['count']==0 for m in summary['models'].values()):
        raise ValueError('Cannot report a model with no eligible final detections')
    export_tables(run)
    figures(run,output)
    article=render_article(run)
    correction=read(run/'correction_results.json')
    grid=[.25,.5,.75,.9,.95,.99]
    examples={}
    for method,result in correction['methods'].items():
        params=result['parameters']
        mapped=Correction(method,params['a'],params['b']).apply(grid)
        examples[method]=[dict(original_score=p,corrected_score=float(q)) for p,q in zip(grid,mapped)]
    write(run/'score_examples.json',dict(model_id=correction['model_id'],selected_method=correction['selected_method'],examples=examples))
    decision=read(run/'selection.json')
    lines=['# Detector calibration results','',
           '> SYNTHETIC VERIFICATION ONLY' if summary['synthetic'] else 'COCO val2017 benchmark of specific checkpoints.',
           '', 'Excluded checkpoints: '+ '; '.join(f"{name}: {item['reason']}" for name,item in read(run/'manifest.json').get('exclusions',{}).items()),
           '', 'Primary metrics: original scores ≥0.25, correct class, COCO IoU ≥0.50, 15 equal-width bins.',
           'COCO mAP and AR100 use the full cached ≥0.001 prediction population. All values are fractions.',
           'Intervals are 95% percentile intervals from 1,000 image-level bootstrap samples (seed 42).',
           'Ignored crowd matches are excluded. Empty images stay in the bootstrap universe.', '',
           '| Checkpoint | ECE [95% CI] | NLL | Brier | Confidence | Precision | Count | mAP | AR100 |',
           '|---|---|---|---|---|---|---|---|---|']
    for name,m in summary['models'].items():
        p=m['splits']['final']['primary'];q=m['splits']['final']['quality'];lo,hi=p['ci']['ece']
        lines.append(f"| {name} | {p['ece']:.4f} [{lo:.4f}, {hi:.4f}] | {p['nll']:.4f} | {p['brier']:.4f} | {p['mean_confidence']:.4f} | {p['precision']:.4f} | {p['count']} | {q['map']:.4f} | {q['recall100']:.4f} |")
    lines+=['','Adapter diagnostics on final data (retained malformed boxes and excluded unused COCO slots):','',
            *[f"- {name}: {m['splits']['final']['diagnostics']}" for name,m in summary['models'].items()]]
    lines+=['',f"Selected using selection data: **{decision['model_id']}**.",
            f"Overlapping selection ECE intervals: {decision['overlapping_ece_intervals']}.",
            f"Chosen using grouped calibration CV: **{correction['selected_method']}**.",'',
            '| Method | ECE change [paired 95% CI] | NLL change | Brier change |','|---|---|---|---|']
    for method,r in correction['methods'].items():
        lo,hi=r['paired_delta_ci']['ece']
        lines.append(f"| {method} | {r['delta']['ece']:.4f} [{lo:.4f}, {hi:.4f}] | {r['delta']['nll']:.4f} | {r['delta']['brier']:.4f} |")
    lines+=['','Correction preserves the original population, classes, boxes, matching, and rankings. No corrected-score threshold is applied.',
            'Precision and COCO AP are checked numerically. AP is re-evaluated using corrected logits for sorting on all originally retained detections, avoiding sigmoid saturation ties.',
            'Sensitivity metrics and intervals are in summary.json. Failed models cannot silently disappear.',
            'These measurements do not establish a causal effect of architecture age or measure confidence for missed objects.',
            '',f'Article: {article}', 'Full provenance, partitions, native suppression settings, and timing: manifest.json.']
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    if not summary['synthetic']:
        Path('results/report.md').write_text('\n'.join(lines)+'\n')
    return article
