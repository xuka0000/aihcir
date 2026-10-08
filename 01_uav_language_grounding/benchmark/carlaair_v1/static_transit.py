"""Bounded geometric acquisition routing around enclosing static obstacles.

No target/label selection or autonomous-policy claim. Uses the same 6m
horizontal clearance footprint as capture screening. Finite static geometry
cannot certify dynamic collision freedom; flight/collision gates stay active.
"""
import heapq
import math
import numpy as np


def segment_blocked(start,end,low,high):
    start,end=np.asarray(start,float),np.asarray(end,float)
    delta=end-start
    near=np.full(len(low),-np.inf);far=np.full(len(low),np.inf)
    possible=np.ones(len(low),bool)
    for axis in range(2):
        if abs(delta[axis])<1e-12:
            possible&=(low[:,axis]<=start[axis])&(high[:,axis]>=start[axis])
        else:
            a=(low[:,axis]-start[axis])/delta[axis];b=(high[:,axis]-start[axis])/delta[axis]
            near=np.maximum(near,np.minimum(a,b));far=np.minimum(far,np.maximum(a,b))
    return bool(np.any(possible&(np.maximum(near,0)<=np.minimum(far,1))))


def horizontal_route(start,end,boxes,step=10.,padding=60.,margin=6.,max_expansions=10000):
    start,end=np.asarray(start,float),np.asarray(end,float)
    if (start.shape!=(3,) or end.shape!=(3,) or not np.isfinite([start,end]).all()
            or abs(start[2]-end[2])>1e-6 or step<=0 or padding<step or margin<0
            or not np.isfinite([step,padding,margin]).all()
            or not 1<=max_expansions<=10000):raise ValueError('Invalid static route bound')
    level=start[2]
    relevant=(boxes.high[:,2]>=level-2)&(boxes.low[:,2]<=level+100.)
    low=boxes.low[relevant,:2]-margin;high=boxes.high[relevant,:2]+margin
    def blocked(a,b):return segment_blocked(a,b,low,high)
    if blocked(start[:2],start[:2]) or blocked(end[:2],end[:2]):
        raise ValueError('Cruise endpoint overlaps static clearance geometry')
    metadata=dict(step_m=step,padding_m=padding,margin_m=margin,max_expansions=max_expansions,
                  obstacle_boxes=len(low),altitude_m=float(level),uses_object_truth=False,
                  policy_input=False,collision_freedom_proven=False)
    if not blocked(start[:2],end[:2]):return [end.tolist()],dict(metadata,expanded=0,direct=True,
                                                             path_length_m=float(np.linalg.norm(end-start)))
    lower=np.minimum(start[:2],end[:2])-padding;upper=np.maximum(start[:2],end[:2])+padding
    def point(node):return start[:2]+step*np.asarray(node)
    queue=[(float(np.linalg.norm(end[:2]-start[:2])),0.,(0,0))]
    best={(0,0):0.};parent={};closed=set();finish=None
    moves=((-1,0),(0,-1),(0,1),(1,0),(-1,-1),(-1,1),(1,-1),(1,1))
    while queue and len(closed)<max_expansions:
        _,cost,node=heapq.heappop(queue)
        if node in closed:continue
        closed.add(node);p=point(node)
        if np.linalg.norm(p-end[:2])<=1.5*step and not blocked(p,end[:2]):finish=node;break
        for dx,dy in moves:
            nxt=(node[0]+dx,node[1]+dy);q=point(nxt)
            if (q<lower).any() or (q>upper).any() or nxt in closed or blocked(p,q):continue
            value=cost+step*math.hypot(dx,dy)
            if value>=best.get(nxt,float('inf')):continue
            best[nxt]=value;parent[nxt]=node
            heapq.heappush(queue,(value+float(np.linalg.norm(q-end[:2])),value,nxt))
    if finish is None:raise ValueError('No static path within fixed grid and expansion bound')
    nodes=[finish]
    while nodes[-1]!=(0,0):nodes.append(parent[nodes[-1]])
    path=[point(n) for n in reversed(nodes)]+[end[:2]]
    simplified=[path[0]];i=0
    while i<len(path)-1:
        j=len(path)-1
        while j>i+1 and blocked(path[i],path[j]):j-=1
        simplified.append(path[j]);i=j
    result=[[float(p[0]),float(p[1]),float(level)] for p in simplified[1:]]
    return result,dict(metadata,expanded=len(closed),direct=False,
                       path_length_m=sum(math.dist(a,b) for a,b in zip(simplified,simplified[1:])))


def route_cruise_phases(phases,boxes):
    if [p[0] for p in phases]!=['climb','cruise','approach']:raise ValueError('Unexpected transit phases')
    paths=[];audits=[]
    for start,end in zip(phases[0][1],phases[1][1]):
        path,audit=horizontal_route(start,end,boxes);paths.append(path);audits.append(audit)
    stages=[phases[0]]
    for i in range(max(map(len,paths))):
        stages.append((f'cruise_segment_{i}',[path[min(i,len(path)-1)] for path in paths]))
    stages.append(phases[-1])
    return stages,dict(agents=audits,paths=paths,collision_freedom_proven=False)
