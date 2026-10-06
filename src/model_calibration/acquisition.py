"""Explicit download and provenance-freezing stages, never called by infer."""
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from .io import read, write, sha256


def download(url,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        return
    temporary=path.with_suffix(path.suffix+'.partial')
    with urllib.request.urlopen(url,timeout=120) as response,temporary.open('wb') as f:
        while chunk:=response.read(1024*1024):
            f.write(chunk)
    temporary.replace(path)


def coco(root,images=False):
    root=Path(root)
    archive=root/'downloads/annotations_trainval2017.zip'
    download('https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip',archive)
    with zipfile.ZipFile(archive) as z:
        z.extract('annotations/instances_val2017.json',root)
    if images:
        archive=root/'downloads/val2017.zip'
        download('https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip',archive)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target=(root/member.filename).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError('Unsafe COCO ZIP path')
                if not target.exists():
                    z.extract(member,root)
    write(root/'provenance.json',{'annotation_sha256':sha256(root/'annotations/instances_val2017.json'),
                                 'source':'https://cocodataset.org/#download'})


def fetch_models(registry,lock_path):
    lock_path=Path(lock_path)
    lock=read(lock_path) if lock_path.exists() else {'models':{}}
    for model_id,m in registry.items():
        weight=Path(m['weight'])
        if m.get('archive'):
            archive=Path('weights')/(model_id+'.tar')
            download(m['checkpoint_url'],archive)
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    if member.issym() or member.islnk() or not (Path('weights')/member.name).resolve().is_relative_to(Path('weights').resolve()):
                        raise ValueError('Unsafe checkpoint archive')
                tar.extractall('weights')
        else:
            download(m['checkpoint_url'],weight)
        revision=None
        if m.get('repo_dir'):
            repo=Path(m['repo_dir'])
            if not repo.exists():
                pinned=lock['models'].get(model_id,{}).get('repository_revision')
                if pinned:
                    subprocess.run(['git','init',str(repo)],check=True)
                    subprocess.run(['git','-C',str(repo),'remote','add','origin',m['repo_url']],check=True)
                    subprocess.run(['git','-C',str(repo),'fetch','--depth','1','origin',pinned],check=True)
                    subprocess.run(['git','-C',str(repo),'checkout','--detach','FETCH_HEAD'],check=True)
                else:
                    subprocess.run(['git','clone','--depth','1','--branch',m['repo_ref'],m['repo_url'],str(repo)],check=True)
            revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        artifacts={str(weight):sha256(weight)}
        if m.get('archive'):
            artifacts.update({str(p):sha256(p) for p in weight.parent.iterdir() if p.is_file()})
        if model_id=='yolov3':
            config=Path(m['config_file'])
            if not config.exists():
                source=Path(m['repo_dir'])/'cfg/yolov3.cfg'
                text=source.read_text().replace('width=608','width=416').replace('height=608','height=416')
                config.write_text(text)
            names=Path('weights/coco.names')
            names.write_text((Path(m['repo_dir'])/'data/coco.names').read_text())
            Path(m['data_file']).write_text('classes=80\nnames=weights/coco.names\n')
            artifacts.update({str(p):sha256(p) for p in [config,names,Path(m['data_file'])]})
        if model_id=='rtmdet_tiny':
            artifacts[m['config_file']]=sha256(m['config_file'])
        entry=dict(artifacts=artifacts,repository_revision=revision,checkpoint_url=m['checkpoint_url'])
        if model_id in lock['models'] and lock['models'][model_id]!=entry:
            raise ValueError(f'Frozen artifact changed for {model_id}; do not overwrite lock')
        lock['models'][model_id]=entry
        write(lock_path,lock)
    return lock
