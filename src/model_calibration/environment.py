import os
import shlex
from pathlib import Path


def load_env(path='.env'):
    """Load simple KEY=value entries without shell evaluation or token logging."""
    file=Path(path)
    if not file.exists():
        return
    for line in file.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '): line=line[7:]
        key,sep,value=line.partition('=')
        key=key.strip()
        if not sep or not key.replace('_','').isalnum():
            raise ValueError('Invalid .env entry; use KEY=value')
        parts=shlex.split(value,comments=True)
        os.environ.setdefault(key,' '.join(parts))
