"""Absolute material windows; renewal requires explicit local authorization."""
import json
from pathlib import Path
import time

def deadline(root,attempt):
    root=Path(root)
    if attempt=='scene_material4_complete7':
        b=json.loads((root/'material_completion_budget.json').read_text())
        if (b.get('approved') is not True or b.get('frame_limit')!=7 or b.get('window_seconds')!=600
            or b['deadline']!=b['started_at']+600 or b.get('original_budget_reset') is not False):
            raise ValueError('Material completion authorization/clock differs')
        return b['deadline']
    detail=root/'material_detail_budget.json'
    if detail.exists():
        b=json.loads(detail.read_text())
        recovery=root/('detail_launch_recovery.json' if attempt=='scene_material4_detailfix1_config1' else 'detail_interface_recovery.json')
        r=json.loads(recovery.read_text()) if recovery.exists() else {}
        continuation=attempt==r.get('execution_id')
        if attempt==b['attempt'] or continuation:
            if (b['attempt']!='scene_material4_detailfix1' or b.get('approved') is not True
                or b.get('frame_limit')!=14 or b.get('window_seconds')!=1200
                or b.get('original_budget_reset') is not False
                or b['deadline']!=b['started_at']+1200):
                raise ValueError('Detail repair window identity/budget changed')
            if continuation and (attempt not in ('scene_material4_detailfix1_interface1','scene_material4_detailfix1_config1')
                or r.get('original_attempt')!=b['attempt']
                or r.get('previous_measurement_frames')!=0
                or r.get('previous_shutdown_confirmed') is not True
                or r.get('original_deadline')!=b['deadline']
                or r.get('maximum_measurement_frames')!=14
                or r.get('maximum_interface_recoveries')!=1):
                raise ValueError('Unverified interface continuation or reset')
            correction=json.loads((root/'native_detail_correction.json').read_text())
            if correction.get('attempt')!=attempt or correction.get('frozen') is not True:
                raise ValueError('Native detail correction has not been frozen')
            return b['deadline']
    sourcefix=root/'material_sourcefix_budget.json'
    if sourcefix.exists():
        b=json.loads(sourcefix.read_text())
        if attempt==b['attempt']:
            if b.get('approved') is not True or b.get('frame_limit')!=14 or b.get('window_seconds')!=3600:
                raise ValueError('Source repair identity/budget differs')
            if b['deadline']!=b['started_at']+3600 or b.get('original_budget_reset') is not False:
                raise ValueError('Source repair window was reset')
            correction=json.loads((root/'native_source_correction.json').read_text())
            if correction.get('attempt')!=attempt or correction.get('frozen') is not True:
                raise ValueError('Actual source/configuration correction is not frozen')
            return b['deadline']
    renewal=root/'material_renewal_budget.json'
    if renewal.exists():
        b=json.loads(renewal.read_text())
        if attempt==b['attempt']:
            if b.get('approved') is not True or b.get('frame_limit')!=14 or b.get('window_seconds')!=1800:
                raise ValueError('Material renewal identity/budget differs')
            if b['deadline']!=b['started_at']+1800:raise ValueError('Material window was reset')
            return b['deadline']
    b=json.loads((root/'material_budget.json').read_text())
    if attempt not in b['allowed_execution_ids'] or b['completed_frames_before_continuation']!=0:
        raise ValueError('Unapproved material continuation')
    return b['original_deadline']


def begin_detail(root,attempt):
    root=Path(root);auth=json.loads((root/'native_detail_repair_authorization.json').read_text())
    if (attempt!='scene_material4_detailfix1' or auth.get('attempt')!=attempt
        or auth.get('approved') is not True or auth.get('total_seconds_limit')!=1200
        or auth.get('native_frame_limit')!=14):
        raise ValueError('Explicit20-minute/14-frame detail authorization is required')
    now=time.time();value=dict(attempt=attempt,approved=True,started_at=now,deadline=now+1200,
        window_seconds=1200,frame_limit=14,original_budget_reset=False,
        authorization='native_detail_repair_authorization.json')
    with (root/'material_detail_budget.json').open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2)
    return value

def begin_completion(root):
    root=Path(root);auth=json.loads((root/'native_material_completion_authorization.json').read_text())
    if (auth.get('approved') is not True or auth.get('new_native_frame_limit')!=7
        or auth.get('total_seconds_limit')!=600 or auth.get('attempt')!='scene_material4_complete7'):
        raise ValueError('Explicit10-minute/seven-frame completion authorization required')
    now=time.time();b=dict(attempt=auth['attempt'],approved=True,started_at=now,deadline=now+600,
        window_seconds=600,frame_limit=7,original_budget_reset=False,
        authorization='native_material_completion_authorization.json')
    with (root/'material_completion_budget.json').open('x') as f:json.dump(b,f,indent=2)
    return b

def begin_renewal(root,attempt):
    root=Path(root);auth=json.loads((root/'implementation_authorization.json').read_text())
    assessment=json.loads((root/'material_recovery_assessment.json').read_text())
    if (attempt!=auth.get('material_attempt') or auth.get('new_material_frame_limit')!=14 or
        auth.get('new_material_window_seconds')!=1800 or assessment.get('new_window_required') is not True):
        raise ValueError('Conditional material renewal is not authorized or necessary')
    if (root/'native_material_selection.json').exists():raise ValueError('Material already qualified')
    now=time.time();value=dict(attempt=attempt,approved=True,started_at=now,deadline=now+1800,
        window_seconds=1800,frame_limit=14,original_budget_reset=False,
        authorization='implementation_authorization.json',recovery_assessment='material_recovery_assessment.json')
    with (root/'material_renewal_budget.json').open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2)
    return value
