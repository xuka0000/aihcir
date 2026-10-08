"""Public detector proposals plus flat-ground geometric controls, not full TRACE.

Pinhole ray-plane intersection is geometric modelling. Explicit task parsing,
physical-size screens and association radii are local engineering controls,
not a reproduction of any published learned scene-graph algorithm.
"""
import math
import re
from pilot_comparison import canonical


SPATIAL_METHODS=('candidate_unary','candidate_explicit_relations','candidate_cross_view_relations')


def task_relations(instruction):
    if 'east or west' in instruction:
        raise ValueError('Observation-dependent direction requires the versioned latent task interface')
    if not re.match(r'(Find|Locate) the [a-z]+ car near',instruction):
        raise ValueError('Unsupported development instruction')
    direction=re.search(r'\b(north|south|east|west)\b',instruction).group(1)
    if 'fire engine' in instruction:
        return [('target','near','van'),('van' if 'that is' in instruction else 'target',direction,'engine')]
    return [('target','near','van'),('target',direction,'van')]


def ground_xy(pixel,observation):
    w,x,y,z=observation['camera_orientation_ned_wxyz']
    norm=math.sqrt(w*w+x*x+y*y+z*z)
    if norm<1e-8:raise ValueError('Zero rotation')
    w,x,y,z=[v/norm for v in (w,x,y,z)]
    rotation=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
              [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
              [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    ray=[1,(pixel[0]-320)/320,(pixel[1]-240)/320]
    ray=[sum(a*b for a,b in zip(row,ray)) for row in rotation]
    if ray[2]<=1e-6:raise ValueError('Ray above ground horizon')
    p=observation['camera_world_position']
    scale=p[2]/ray[2]
    if not math.isfinite(scale) or scale<=0:raise ValueError('Invalid ground intersection')
    return [p[0]+scale*ray[0],-(p[1]+scale*ray[1])]


def candidates(detector_conditions,observations):
    by_id={o['image']:o for o in observations}
    result=[]
    for condition in detector_conditions:
        target_query=condition['method']=='grounding_dino_target_phrase'
        for call in condition['calls']:
            name=call['input_view_ids'][0];obs=by_id[name]
            ranked=sorted(zip(call['scores'],call['boxes'],call['labels']),reverse=True)[:16]
            for score,box,label in ranked:
                if not (0<=box[0]<box[2]<=640 and 0<=box[1]<box[3]<=480):continue
                try:
                    center=ground_xy([(box[0]+box[2])/2,(box[1]+box[3])/2],obs)
                    corners=[ground_xy([u,v],obs) for u in (box[0],box[2]) for v in (box[1],box[3])]
                except ValueError:continue
                span=max(math.dist(a,b) for a in corners for b in corners)
                # Reject building-sized proposals before relation reasoning.
                # Limits are fixed local priors, not measured vehicle truth.
                roles=['target'] if target_query else (['van'] if 'van' in label.lower() else
                         ['engine'] if 'engine' in label.lower() or 'truck' in label.lower() else [])
                for role in roles:
                    if not .5<=span<=(9 if role=='target' else 20):continue
                    result.append({'role':role,'xy':center,'score':float(score),'span_m':span,
                                   'entity':{'view_id':name,'bbox_xyxy':box},'sources':[name]})
    # Identical per-frame boxes can occur in repeated exposures; deduplicate
    # before any confidence aggregation or cross-view support accounting.
    unique={canonical((c['role'],c['entity'])):c for c in result}
    return list(unique.values())


def cluster(items):
    groups=[]
    for item in sorted(items,key=lambda c:(-c['score'],c['entity']['view_id'])):
        group=next((g for g in groups if g['role']==item['role'] and math.dist(g['xy'],item['xy'])<=4),None)
        if group is None:
            groups.append(dict(item,sources=list(item['sources']),members=[item]));continue
        if item['entity']['view_id'] in group['sources']:continue
        group['members'].append(item);group['sources'].extend(item['sources'])
        weights=[v['score'] for v in group['members']];total=sum(weights)
        group['xy']=[sum(v['xy'][axis]*v['score'] for v in group['members'])/total for axis in (0,1)]
        # Keep the best actual box as the output witness; confidence is an
        # average rather than repeated correlated evidence multiplication.
        group['score']=sum(weights)/len(weights)
    return groups


def holds(a,relation,b):
    dx,dy=a['xy'][0]-b['xy'][0],a['xy'][1]-b['xy'][1]
    if math.hypot(dx,dy)<3:return False
    return {'near':math.hypot(dx,dy)<=20,'east':dx>=5,'west':dx<=-5,
            'north':dy>=5,'south':dy<=-5}[relation]


def infer(items,instruction,method):
    import itertools
    if method not in SPATIAL_METHODS:raise ValueError('Unknown control')
    if method=='candidate_cross_view_relations':items=cluster(items)
    targets=[x for x in items if x['role']=='target']
    empty={'target':None,'relations':[],'confidence':0.,'unresolved':'No supported candidate assignment','next_view_ids':[]}
    if not targets:return empty
    if method=='candidate_unary':
        best=max(targets,key=lambda c:c['score'])
        return dict(empty,target=best['entity'],confidence=best['score'],unresolved='Relations not evaluated')
    edges=task_relations(instruction);roles=sorted({r for a,_,b in edges for r in (a,b)})
    pools=[[x for x in items if x['role']==role] for role in roles]
    best=None
    for values in itertools.product(*pools):
        assignment=dict(zip(roles,values))
        if not all(holds(assignment[a],r,assignment[b]) for a,r,b in edges):continue
        if method=='candidate_explicit_relations' and len({x['entity']['view_id'] for x in values})!=1:continue
        utility=sum(math.log(max(x['score'],1e-6)) for x in values)/len(values)
        if best is None or utility>best[0]:best=(utility,assignment)
    if best is None:return empty
    utility,assignment=best
    return {'target':assignment['target']['entity'],'confidence':math.exp(utility),'unresolved':'',
            'next_view_ids':[],'relations':[{'subject':assignment[a]['entity'],'relation':r,
                                           'object':assignment[b]['entity']} for a,r,b in edges]}
