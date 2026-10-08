"""Resolve a diagnosed attachment while retaining the original failed guard."""
from pathlib import Path
from deploy_oct07_image_campaign import read_guard_json
def protection_folder(owner):
    owner=Path(owner);pointer=owner/'active_protection.json'
    if not pointer.is_file():return owner
    d=read_guard_json(pointer);target=owner/d['relative_folder']
    if target.resolve().parent!=owner.resolve():raise ValueError('Protection attachment must stay within original task')
    context=read_guard_json(target/'guard_context.json')
    if context['nonce']!=d['nonce'] or context['existing_job_identity']!=read_guard_json(owner/'job_identity.json'):
        raise ValueError('Attached protection identity differs')
    return target
