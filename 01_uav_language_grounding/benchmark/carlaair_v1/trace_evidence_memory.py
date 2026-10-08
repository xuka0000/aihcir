"""Source-aware, deterministic geometric evidence memory for development.

This component does not supply calibrated probabilities or complete TRACE.
The 4m association bound is inherited from the static control. Complete-link
membership prevents a chain of nearby proposals from drifting across objects.
Alternative detections from one image are retained as alternatives, not votes.
"""
import copy
import math
from pilot_comparison import canonical


def source_key(item):
    return item.get('image_sha256') or item['entity']['view_id']


def associate(items, radius=4.):
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Invalid association bound')
    unique={}
    for original in items:
        item=copy.deepcopy(original)
        if (len(item['xy'])!=2 or not all(math.isfinite(x) for x in item['xy'])
                or not math.isfinite(item['score']) or not 0<=item['score']<=1):
            raise ValueError('Invalid evidence')
        key=canonical((item['role'],source_key(item),item['entity']['bbox_xyxy']))
        # An identical pixel witness must have identical model output. A changed
        # output requires a separate model-version comparison, not extra support.
        if key in unique:
            prior=unique[key]
            if prior['xy']!=item['xy'] or prior['score']!=item['score']:
                raise ValueError('Conflicting replay of the same evidence')
            if canonical(item['entity']) < canonical(prior['entity']): unique[key]=item
        else: unique[key]=item
    ordered=sorted(unique.values(),key=lambda x:(-x['score'],canonical((x['role'],source_key(x),x['entity']))))
    groups=[]
    for item in ordered:
        group=next((g for g in groups if g[0]['role']==item['role']
                    and source_key(item) not in {source_key(x) for x in g}
                    and all(math.dist(x['xy'],item['xy'])<=radius for x in g)),None)
        if group is None: groups.append([item])
        else: group.append(item)
    output=[]
    for group in groups:
        scores=[x['score'] for x in group];total=sum(scores)
        xy=[sum(x['xy'][axis]*x['score'] for x in group)/total for axis in (0,1)] if total else list(group[0]['xy'])
        output.append(dict(group[0],xy=xy,score=total/len(group),members=group,
            sources=sorted({x['entity']['view_id'] for x in group}),
            source_keys=sorted({source_key(x) for x in group}),
            support_count=len(group),posterior_calibrated=False))
    return output
