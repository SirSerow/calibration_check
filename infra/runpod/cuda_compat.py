"""Paddle 2.6 dlopen expects unversioned names; NVIDIA wheels ship SONAMEs."""
from pathlib import Path
import argparse


def create_links(environment):
    environment=Path(environment)
    sites=list(environment.glob('lib/python*/site-packages'))
    if len(sites)!=1: raise ValueError('Expected one Python site-packages directory')
    destination=environment/'cuda-compat/lib'
    destination.mkdir(parents=True,exist_ok=True)
    for library in sites[0].glob('nvidia/*/lib/*.so.*'):
        name=library.name.split('.so.')[0]+'.so'
        link=destination/name
        if not link.exists(): link.symlink_to(library.resolve())
    return destination

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('environment');args=parser.parse_args()
    print(create_links(args.environment))
