"""Fresh native-volume write check inside a registered scientific reservation."""
import os
from pathlib import Path
import shutil

def before_write(path,peak_bytes=2*1024**2):
    value=os.environ.get('AIHCIR_NATIVE_MIN_FREE_BYTES')
    if value is None:return
    minimum=max(5*1024**3,int(value));p=Path(path)
    while not p.exists():p=p.parent
    free=shutil.disk_usage(p).free
    if free-peak_bytes<minimum:raise RuntimeError('Registered native output peak or5GiB storage floor exhausted')
