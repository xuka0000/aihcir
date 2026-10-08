"""Public static constraint baseline for an observation-dependent direction.

This is a transparent baseline, not learned TRACE. Uses detector appearance,
the existing flat-ground projection and fixed spatial margins. It never reads
scene truth, pixel masks or unobserved-view descriptors.
"""
import itertools
import math
from pilot_comparison import canonical
from pilot_spatial_control import ground_xy, cluster, holds

INSTRUCTION=('Locate the white car near the yellow van whose east or west position '
             'relative to the fire engine matches that of the blue car.')
ROLE_PHRASES={'target':'white car','van':'yellow van','engine':'fire engine','marker':'blue car'}


def public_candidates(detections,observations,allowed_views):
    views={o['image']:o for o in observations}
    allowed=set(allowed_views)
    if not allowed <= set(views):raise ValueError('Unknown view access')
    items=[]
    for row in detections:
        name=row['image']
        if name not in allowed:continue  # No feature access before view access.
        if row['image_sha256']!=views[name]['sha256']:raise ValueError('Detector/view hash mismatch')
        for index,d in enumerate(row['detections']):
            tokens=set(d['label'].lower().split())
            roles=[r for r,phrase in ROLE_PHRASES.items() if set(phrase.split())<=tokens]
            if len(roles)!=1:continue  # Ambiguous detector phrase is not ground truth.
            role=roles[0];box=d['bbox_xyxy'];obs=views[name]
            if len(box)!=4 or not all(math.isfinite(v) for v in box):raise ValueError('Invalid public box')
            if not 0<=box[0]<box[2]<=640 or not 0<=box[1]<box[3]<=480:raise ValueError('Box outside source')
            center=ground_xy([(box[0]+box[2])/2,(box[1]+box[3])/2],obs)
            corners=[ground_xy([u,v],obs) for u in (box[0],box[2]) for v in (box[1],box[3])]
            span=max(math.dist(a,b) for a in corners for b in corners)
            if not .5<=span<=(9 if role in ('target','marker') else 20):continue
            score=d['score']
            if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid detector score')
            items.append(dict(role=role,xy=center,score=score,span_m=span,
                              entity=dict(view_id=name,bbox_xyxy=box),sources=[name],
                              image_sha256=row['image_sha256']))
    return list({canonical((x['role'],x['entity'])):x for x in items}.values())


def infer(items,instruction,shortlist=6,association='legacy'):
    if instruction!=INSTRUCTION or type(shortlist)!=int or not 1<=shortlist<=12:
        raise ValueError('Unsupported latent task or search bound')
    if association=='legacy':items=cluster(items)
    elif association=='source_aware':
        from trace_evidence_memory import associate
        items=associate(items)
    else:raise ValueError('Unsupported association control')
    roles=tuple(ROLE_PHRASES)
    pools={r:sorted([c for c in items if c['role']==r],key=lambda c:(-c['score'],canonical(c['entity'])))[:shortlist] for r in roles}
    empty=dict(target=None,relations=[],confidence=0.,unresolved='No complete observed constraint assignment',next_view_ids=[])
    candidates=[]
    for values in itertools.product(*(pools[r] for r in roles)):
        a=dict(zip(roles,values));delta=a['marker']['xy'][0]-a['engine']['xy'][0]
        direction='east' if delta>=5 else 'west' if delta<=-5 else None
        if direction is None or not holds(a['van'],direction,a['engine']) or not holds(a['target'],'near',a['van']):continue
        utility=sum(math.log(max(c['score'],1e-6)) for c in values)/4
        candidates.append((utility,a,direction))
    diagnostic=dict(role_pool_sizes={r:len(pools[r]) for r in roles},supported_assignments=len(candidates),
                    latent_direction_domain=sorted({c[2] for c in candidates}) or ['east','west'],
                    posterior_calibrated=False,flat_ground_assumption=True,full_TRACE_implemented=False,
                    association=association)
    if not candidates:return empty,diagnostic
    utility,a,direction=max(candidates,key=lambda c:c[0])
    edges=[('target','near','van'),('van',direction,'engine'),('marker',direction,'engine')]
    prediction=dict(empty,target=a['target']['entity'],confidence=math.exp(utility),unresolved='',
                    relations=[dict(subject=a[s]['entity'],relation=r,object=a[o]['entity']) for s,r,o in edges])
    return prediction,diagnostic
