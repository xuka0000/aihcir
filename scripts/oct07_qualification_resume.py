"""Replace only independently rejected captures, preserving their exact data."""
import json,shutil
from pathlib import Path
from oct07_capture_scope import sha,verify_existing_capture

def apply_plan(root,inventory,attempt):
    root=Path(root);phase=root/'oct07_scene_repair'
    path=phase/'qualification_completion_resume_plan.json'
    plan=json.loads(path.read_text(encoding='utf-8'))
    authorization=phase/'qualification_completion_authorization.json'
    if (plan.get('execution_id')!=attempt or plan.get('approved') is not True
            or sha(authorization)!=plan.get('authorization_sha256')
            or plan.get('prior_assets_preserved_outside_instance') is not True):
        raise ValueError('Qualified scene resume/replacement plan is not verified')
    replaced={}
    previous={}
    if plan.get('previous_replacement_attempt'):
        old=phase/(plan['previous_replacement_attempt']+'_replaced_captures.json')
        if sha(old)!=plan.get('previous_replacement_sha256'):raise ValueError('Prior diagnosed replacement receipt changed')
        previous=json.loads(old.read_text(encoding='utf-8'))['replaced']
    for eid,row in plan['rejected_complete_captures'].items():
        if not row.get('blockers'):raise ValueError('Capture replacement needs an actual rejection')
        item=inventory.get(eid) or previous[eid]
        if any(item[k]!=row[k] for k in ('source_sha256','observations_sha256')):
            raise ValueError('Rejected capture identity changed')
        verify_existing_capture(root,eid,item)
        source=root/'releases/oct07_scenes_rgb_v4'/item['split']/eid/'scene.json'
        archive=Path(item['probe']).parent/'rejected_sources'/eid/'scene.json'
        archive.parent.mkdir(parents=True,exist_ok=True)
        if archive.exists():
            if sha(archive)!=row['source_sha256']:raise ValueError('Rejected source archive differs')
        else:shutil.copy2(source,archive)
        replaced[eid]=dict(item,blockers=row['blockers'],preserved_source=str(archive),
            external_audit_sha256=plan['prior_effective_audit_sha256'])
    # Commit the identity-verified replacements only after every check above.
    for eid,row in replaced.items():
        source=root/'releases/oct07_scenes_rgb_v4'/row['split']/eid/'scene.json'
        if sha(Path(row['preserved_source']))!=row['source_sha256']:raise ValueError('Rejected source preservation did not verify')
        source.unlink()
        inventory.pop(eid,None)
    history=phase/(attempt+'_replaced_captures.json')
    if history.exists():raise ValueError('Replacement plan already executed')
    history.write_text(json.dumps(dict(replaced=replaced,raw_RGB_deleted=False,active_rejected_sources_removed_after_preservation=True),indent=2))
    return sorted(replaced)
