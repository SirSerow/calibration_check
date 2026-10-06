#!/usr/bin/env python3
"""Bounded project-owned on-demand pod lifecycle. Tokens stay in the environment."""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from model_calibration.io import read,write
from model_calibration.environment import load_env
load_env()

GPU_PREFERENCE=['NVIDIA GeForce RTX 4090','NVIDIA GeForce RTX 3090','NVIDIA RTX A5000']
GQL='https://api.runpod.io/graphql'
STATE=Path('results/runpod-state.json')


def request(query,variables=None):
    key=os.environ.get('RUNPOD_API_KEY')
    if not key:
        raise RuntimeError('RUNPOD_API_KEY is unset in this process')
    req=urllib.request.Request(GQL+'?'+urllib.parse.urlencode({'api_key':key}),data=json.dumps({'query':query,'variables':variables or {}}).encode(),
                               headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','User-Agent':'model-calibration/0.1'})
    try:
        with urllib.request.urlopen(req,timeout=45) as response:
            body=json.load(response)
    except urllib.error.HTTPError as error:
        message=error.read().decode()[:500].replace(key,'[REDACTED]')
        raise RuntimeError(f'RunPod HTTP {error.code}: {message}') from None
    except urllib.error.URLError:
        raise RuntimeError('RunPod network request failed') from None
    if body.get('errors'):
        raise RuntimeError(json.dumps([e['message'] for e in body['errors']]))
    return body['data']


def state():
    return read(STATE) if STATE.exists() else {'pods':[],'budget_usd':10,'reserve_usd':1}


def storage_quote(container_gb=80,volume_gb=0):
    url='https://docs.runpod.io/pods/pricing.md'
    with urllib.request.urlopen(url,timeout=45) as response:
        document=response.read().decode()
    rates={}
    for label,key in [('Container disk','container'),('Volume disk','volume')]:
        cleaned=document.replace('**','').replace('\\$', '$')
        match=re.search(re.escape(label)+r'\s*\|\s*\$(\d+\.\d+)\s*/GB/month',cleaned)
        if not match:
            raise RuntimeError('Cannot parse current storage prices; inspect official pricing before provisioning')
        rates[key]=float(match[1])
    return {'source':url,'retrieved_at':time.time(),'rates_usd_gb_month':rates,
            'container_gb':container_gb,'volume_gb':volume_gb,
            'hourly_usd':(rates['container']*container_gb+rates['volume']*volume_gb)/(30*24)}


def quotes():
    storage=storage_quote()
    data=request('''query { gpuTypes { id displayName memoryInGb lowestPrice(input:{gpuCount:1,secureCloud:false,minMemoryInGb:24,minVcpuCount:4,totalDisk:80,supportPublicIp:true}) { uninterruptablePrice stockStatus } } }''')
    output=[]
    for gpu_id in GPU_PREFERENCE:
        gpu=next((x for x in data['gpuTypes'] if x['id']==gpu_id),None)
        if not gpu or not gpu['lowestPrice'] or gpu['memoryInGb']<24:
            continue
        price=gpu['lowestPrice']['uninterruptablePrice']
        if price is None:
            continue
        output.append(dict(gpu_id=gpu_id,gpu_hourly_usd=float(price),storage=storage,
                           total_hourly_usd=float(price)+storage['hourly_usd'],
                           stock=gpu['lowestPrice']['stockStatus'],cloud='COMMUNITY',retrieved_at=time.time()))
    write('results/runpod-quotes.json',output)
    return output


def spent(s,now=None):
    now=now or time.time()
    return sum((p.get('terminated_at',now)-p['created_at'])/3600*p['hourly_usd'] for p in s['pods'])


def provision(image_name,ssh_public_key):
    s=state()
    if any(not p.get('terminated_at') for p in s['pods']):
        raise RuntimeError('Project already has a nonterminated pod')
    eligible=[q for q in quotes() if q['total_hourly_usd']<=1 and q['stock'] not in ('Unavailable','unavailable','OUT_OF_STOCK','None')]
    if not eligible:
        raise RuntimeError('No approved 24 GB on-demand GPU within the $1/hour including storage ceiling')
    q=eligible[0]
    balance=request('query { myself { clientBalance } }')['myself']['clientBalance']
    if balance<q['total_hourly_usd']:
        raise RuntimeError('Account credit is below the required one-hour deployment amount')
    remaining=10-1-spent(s)
    if remaining<q['total_hourly_usd']:
        raise RuntimeError('Insufficient remaining budget with retrieval reserve')
    # Server-side termination survives local controller disconnect. Reserve is
    # excluded from duration; cleanup and reconciliation remain explicit.
    max_seconds=int(remaining/q['total_hourly_usd']*3600)-60
    deadline=dt.datetime.fromtimestamp(time.time()+max_seconds,dt.timezone.utc).isoformat()
    public_key=Path(ssh_public_key).read_text().strip()
    if not public_key.startswith(('ssh-ed25519 ','ssh-rsa ')):
        raise ValueError('Expected an SSH public key file')
    input=dict(cloudType=q['cloud'],gpuCount=1,gpuTypeId=q['gpu_id'],name='model-calibration-coco',
               imageName=image_name,containerDiskInGb=80,volumeInGb=0,volumeMountPath='/workspace',
               minMemoryInGb=24,minVcpuCount=4,supportPublicIp=True,ports='22/tcp',startSsh=True,
               terminateAfter=deadline,env=[{'key':'PUBLIC_KEY','value':public_key}])
    attempts=[]
    pod=None
    for candidate in eligible:
        input['gpuTypeId']=candidate['gpu_id']
        input['deployCost']=candidate['gpu_hourly_usd']
        try:
            pod=request('''mutation ($input:PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input:$input) { id costPerHr desiredStatus } }''',{'input':input})['podFindAndDeployOnDemand']
            q=candidate
            break
        except RuntimeError as error:
            attempts.append({'gpu_id':candidate['gpu_id'],'error':str(error),'time':time.time()})
            write('results/runpod-attempts.json',attempts)
    if pod is None:
        raise RuntimeError('All allowed on-demand deployments failed: '+json.dumps(attempts))
    p=dict(id=pod['id'],created_at=time.time(),hourly_usd=q['total_hourly_usd'],quote=q,
           server_terminate_at=deadline,actual_compute_hourly_usd=pod['costPerHr'])
    s['pods'].append(p);write(STATE,s)
    if pod['costPerHr'] is None or pod['costPerHr']+q['storage']['hourly_usd']>1 or pod['costPerHr']>q['gpu_hourly_usd']+1e-9:
        terminate(p['id'])
        raise RuntimeError('Provisioned rate exceeded live quote; project pod immediately terminated')
    return p


def project_pod(pod_id=None):
    s=state()
    found=[p for p in s['pods'] if p['id']==pod_id] if pod_id else [p for p in s['pods'] if not p.get('terminated_at')]
    if len(found)!=1:
        raise ValueError('Expected exactly one recorded project-owned pod')
    return s,found[0]


def status(pod_id=None):
    s,p=project_pod(pod_id)
    data=request('''query ($input:PodFilter!) { pod(input:$input) { id desiredStatus costPerHr runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort } } } }''',{'input':{'podId':p['id']}})['pod']
    return {'pod':data,'estimated_total_spent_usd':spent(s),'remaining_usd':10-spent(s),'record':p}


def terminate(pod_id=None):
    s,p=project_pod(pod_id)
    if p.get('terminated_at'):
        return p
    request('mutation ($input:PodTerminateInput!) { podTerminate(input:$input) }',{'input':{'podId':p['id']}})
    p['terminated_at']=time.time();write(STATE,s)
    return p


def ssh_target(pod_id=None):
    data=status(pod_id)['pod']
    if not data or not data['runtime']:
        raise RuntimeError('Pod is not ready')
    ports=data['runtime']['ports']
    port=next((p for p in ports if p['privatePort']==22 and p['isIpPublic']),None)
    if not port:
        raise RuntimeError('No public SSH endpoint available')
    return port['ip'],port['publicPort']


def sync(key,run,pod_id=None,upload=False):
    host,port=ssh_target(pod_id)
    ssh=f'ssh -i {__import__("shlex").quote(key)} -p {port} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/model-calibration-known-hosts'
    if upload:
        subprocess.run(['rsync','-az','--no-owner','--no-group','-e',ssh,'--exclude=/.git','--exclude=.venv*','--exclude=.env*',
                        '--exclude=.agents','--exclude=.aws','--exclude=.codex','--exclude=data/downloads',
                        '--exclude=weights/*.partial','--exclude=results/runpod*','./',f'root@{host}:/workspace/model-calibration/'],check=True)
    else:
        local=Path(run);local.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-az','--no-owner','--no-group','-e',ssh,f'root@{host}:/workspace/model-calibration/{run}/',str(local)+'/' ],check=True)


def monitor(key,run,pod_id=None):
    while True:
        info=status(pod_id)
        if not info['pod'] or info['pod']['desiredStatus']=='TERMINATED':
            return
        try:
            sync(key,run,pod_id)
        except (RuntimeError,subprocess.CalledProcessError) as e:
            print(f'Synchronization pending: {type(e).__name__}',flush=True)
        print(json.dumps({'spent_usd':info['estimated_total_spent_usd'],'remaining_usd':info['remaining_usd']}),flush=True)
        if info['estimated_total_spent_usd']>=9:
            terminate(pod_id)
            raise RuntimeError('Retrieval reserve reached; pod terminated')
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['quotes','provision','status','sync','upload','monitor','terminate'])
    parser.add_argument('--pod');parser.add_argument('--key');parser.add_argument('--public-key')
    parser.add_argument('--run',default='results/runs/coco-v1')
    parser.add_argument('--image',default='runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04')
    args=parser.parse_args()
    if args.command=='quotes': result=quotes()
    elif args.command=='provision': result=provision(args.image,args.public_key)
    elif args.command=='status': result=status(args.pod)
    elif args.command=='terminate': result=terminate(args.pod)
    elif args.command=='monitor': result=monitor(args.key,args.run,args.pod)
    else: result=sync(args.key,args.run,args.pod,args.command=='upload')
    if result is not None: print(json.dumps(result,indent=2))

if __name__=='__main__': main()
