"""Bind infrastructure continuations to the first TRAIN24 absolute deadline."""
import json
import hashlib
from pathlib import Path


def operational_window(root, batch):
    """A hash-bound user amendment changes a window, never the science/caps."""
    binding = batch.get('operational_authorization')
    if not binding:
        return None
    root = Path(root)
    name = binding.get('path', '')
    if not name or Path(name).name != name:
        raise ValueError('Unsafe native operational authorization path')
    raw = (root / name).read_bytes()
    value = json.loads(raw)
    parent = root / 'tonight_core_authorization_20261007.json'
    approval = json.loads(parent.read_bytes())
    from datetime import datetime
    cutoff = datetime.fromisoformat(approval['new_compute_stop']).timestamp()
    if (hashlib.sha256(raw).hexdigest() != binding.get('sha256')
            or value.get('approved') is not True
            or value.get('source_is_current_session') is not True
            or value.get('parent_authorization_sha256') != hashlib.sha256(parent.read_bytes()).hexdigest()
            or value.get('owner_thread') != approval.get('owner_thread')
            or value.get('owned_uuid') != approval.get('owned_uuid')
            or value.get('scientific_parameters_changed') is not False
            or value.get('prior_attempts_costs_and_elapsed_retained') is not True
            or value.get('parent60816_paused') is not True
            or value.get('TEST_sealed') is not True
            or value.get('native_engineering_powered_cap_seconds') != approval['native_repair_maximum_new_powered_seconds']
            or value.get('native_capture_powered_cap_seconds') != approval['new_native_capture_powered_limit_seconds']
            or value.get('engineering_task_RGB_cap') != approval['engineering_task_RGB_cap']
            or not 0 < value.get('admission_deadline', 0) <= cutoff):
        raise ValueError('Current native operational amendment differs from original science/caps')
    return value


def continuation_batch(root,attempt):
    """New explicit operational allocation, separate from expired old grants."""
    from datetime import datetime
    root=Path(root);path=root/'continuation48h_native_execution.json'
    if not path.is_file():return None
    policy=json.loads(path.read_text(encoding='utf-8'))
    if attempt not in policy.get('batches',{}):return None
    source=root/'tonight_core_authorization_20261007.json'
    raw=source.read_bytes();approval=json.loads(raw)
    batch=policy['batches'][attempt]
    if (hashlib.sha256(raw).hexdigest()!=policy.get('authorization_sha256')
            or approval.get('authorization_id')!='USER-AIHCIR-48H-CONTINUATION-20261007'
            or approval.get('approved') is not True or approval.get('TEST_sealed') is not True
            or approval.get('parent60816_paused') is not True
            or approval.get('prior_attempts_costs_and_elapsed_retained') is not True
            or batch.get('scientific_parameters_changed') is not False):
        raise ValueError('Current bounded continuation authority differs')
    mode=batch['mode'];ids=batch['scene_ids']
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Native batch IDs are empty or duplicated')
    if mode=='qualification':
        if not set(ids)<=set(approval['affected_scene_ids']):raise ValueError('Repair expanded beyond SAME3')
        amendment=operational_window(root,batch)
        cutoff=(amendment['admission_deadline'] if amendment is not None
                else datetime.fromisoformat(approval['native_repair_stop']).timestamp())
        limit=approval['native_repair_maximum_new_powered_seconds']
    elif mode=='capture':
        from oct07_capture_scope import selected_ids
        if not set(ids)<=selected_ids(root,'tonight') or len(ids)>24:
            raise ValueError('Capture batch differs from frozen original subset')
        cutoff=datetime.fromisoformat(approval['new_compute_stop']).timestamp()
        limit=approval['capture_batch_maximum_seconds']
    else:raise ValueError('Unsupported native continuation mode')
    if not 0<batch['maximum_powered_seconds']<=limit:raise ValueError('Native batch allowance expanded')
    if batch['absolute_deadline']>cutoff:raise ValueError('Native batch exceeds explicit cutoff')
    return dict(batch,authorization_sha256=policy['authorization_sha256'])


def verify_source_freeze(root,attempt,code_directory,read_code=None):
    root=Path(root);budget=json.loads((root/'qualification_execution_budget.json').read_text(encoding='utf-8'))
    current=continuation_batch(root,attempt)
    if current is not None:
        name=current.get('source_freeze_path','continuation48h_source_freeze.json')
        if Path(name).name!=name:raise ValueError('Unsafe continuation source freeze path')
        path=root/name;raw=path.read_bytes();frozen=json.loads(raw)
        if hashlib.sha256(raw).hexdigest()!=current['source_freeze_sha256'] or frozen.get('frozen') is not True:
            raise ValueError('Current source/provenance freeze changed')
        for name,digest in frozen['code_sha256'].items():
            raw_code=read_code(name) if read_code else (Path(code_directory)/name).read_bytes()
            if Path(name).name!=name or hashlib.sha256(raw_code).hexdigest()!=digest:
                raise ValueError('Frozen continuation source changed: '+name)
        return frozen
    if attempt not in budget.get('priority_recovery_allocation',{}).get('execution_ids',[]):
        return None
    # Validate the explicit user allocation before allowing a source version.
    deadline(root,attempt)
    auth=json.loads((root/'priority_recovery_authorization.json').read_text(encoding='utf-8'))
    raw=(root/'priority_capture_source_freeze.json').read_bytes();frozen=json.loads(raw)
    prior=root/'native_detail_correction.json'
    if (hashlib.sha256(raw).hexdigest()!=auth['batches'][attempt].get('source_freeze_sha256')
            or frozen.get('frozen') is not True or attempt not in frozen['execution_ids']
            or hashlib.sha256(prior.read_bytes()).hexdigest()!=frozen['inherited_detail_receipt_sha256']):
        raise ValueError('Priority capture source inheritance is not verified')
    if frozen.get('scientific_capture_parameters_changed') is not False:
        binding=frozen.get('physical_construction_repair',{})
        path=root/binding.get('path','missing_repair_receipt.json')
        raw_repair=path.read_bytes()
        repair=json.loads(raw_repair)
        raw_approval=(root/'qualification_recovery_cap_authorization_20261006.json').read_bytes()
        approval=json.loads(raw_approval)
        batch=auth['batches'][attempt]
        ids=set(batch['scene_ids'])
        engineering_only=(
            attempt in budget.get('native_component_execution_ids',[])
            and attempt in budget.get('diagnostic_only_execution_ids',[])
            and batch.get('run_class')=='qualification_component_metrology'
            and batch.get('maximum_engineering_rgb_frames')==4
            and batch.get('scientific_rgb_frames')==0
            and batch.get('algorithm_rows')==0
            and ids<=set(approval.get('affected_scene_ids',[])))
        if (binding.get('path')!='targeted_occlusion_pose_correction.json'
                or hashlib.sha256(raw_repair).hexdigest()!=binding.get('sha256')
                or hashlib.sha256(raw_approval).hexdigest()!=repair.get('user_authorization_sha256')
                or approval.get('approved') is not True
                or approval.get('authorization_id')!='USER-TRAIN24-CAP-AMENDMENT-20261006-START'
                or repair.get('affected_scene_ids')!=['train_100005','train_100010']
                or not ids or (not ids<=set(repair['affected_scene_ids']) and not engineering_only)
                or repair.get('original_bbox_overlap')!=.55 or repair.get('repaired_bbox_overlap')!=.65
                or repair.get('all_matching_candidates_share_same_pose_rule') is not True
                or repair.get('preserve19qualified') is not True
                or any(repair.get(k) is not False for k in ('scientific_thresholds_changed','input_resolution_changed','exposure_changed','scene_seed_or_split_changed','algorithm_or_baseline_changed','TEST_opened'))):
            raise ValueError('Necessary isolated physical construction repair is not verified')
    for name,digest in frozen['code_sha256'].items():
        raw_code=read_code(name) if read_code is not None else (Path(code_directory)/name).read_bytes()
        if Path(name).name!=name or hashlib.sha256(raw_code).hexdigest()!=digest:
            raise ValueError('Frozen priority capture source changed: '+name)
    return frozen


def deadline(root, attempt):
    current=continuation_batch(root,attempt)
    if current is not None:return current['absolute_deadline']
    receipt = json.loads((Path(root) / 'qualification_execution_budget.json').read_text(encoding='utf-8'))
    if (receipt.get('logical_attempt') != 'scene_qualification1'
            or receipt.get('maximum_seconds') != 10800
            or receipt.get('scientific_attempt_limit') != 1
            or receipt.get('original_budget_reset') is not False
            or attempt not in receipt['allowed_execution_ids']):
        raise ValueError('Qualification continuation is not bound to its original allowance')
    if receipt['deadline'] != receipt['started_at'] + receipt['maximum_seconds']:
        raise ValueError('Qualification deadline was reset')
    recovery=receipt.get('priority_recovery_allocation',{})
    if attempt in recovery.get('execution_ids',[]):
        raw=(Path(root)/'priority_recovery_authorization.json').read_bytes()
        approved=json.loads(raw)
        contract=json.loads((Path(root)/'contract.json').read_text(encoding='utf-8'))
        if (hashlib.sha256(raw).hexdigest()!=recovery.get('authorization_sha256')
                or approved.get('approved') is not True
                or approved.get('capture_stop_lifted') is not True
                or approved.get('expired_capture_window_replaced') is not True
                or approved.get('logical_attempt')!=receipt['logical_attempt']
                or approved.get('original_budget_reset') is not False
                or approved.get('planned_units')!=24
                or approved.get('task_ids')!=contract['scene_qualification']['qualification_scene_ids']
                or approved.get('compute_deadline')!=contract['compute_stop']):
            raise ValueError('Priority recovery authorization is missing or changed')
        from datetime import datetime
        return datetime.fromisoformat(approved['compute_deadline']).timestamp()
    extension = receipt.get('explicit_completion_allocation')
    if extension and (attempt == extension.get('execution_id') or attempt in extension.get('diagnosed_repair_execution_ids',[])):
        path = Path(root) / 'qualification_completion_authorization.json'
        raw = path.read_bytes()
        approved = json.loads(raw)
        contract = json.loads((Path(root) / 'contract.json').read_text(encoding='utf-8'))
        if (hashlib.sha256(raw).hexdigest() != extension.get('authorization_sha256')
                or approved.get('approved') is not True
                or approved.get('execution_id') != extension['execution_id']
                or approved.get('logical_attempt') != receipt['logical_attempt']
                or approved.get('original_deadline') != receipt['deadline']
                or approved.get('effective_deadline') != 1791082800.0
                or approved.get('planned_units') != 24
                or approved.get('task_ids') != contract['scene_qualification']['qualification_scene_ids']
                or approved.get('original_budget_reset') is not False):
            raise ValueError('Explicit completion authorization is missing or changed')
        return approved['effective_deadline']
    return receipt['deadline']
