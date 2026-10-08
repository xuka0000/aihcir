"""Private deterministic framing at the frozen height, FOV and two-round budget."""
import itertools
import math


def cover(scene,height,hide_marker):
    from audit_observation_coverage import projected_support
    from visual_condition_geometry import full_frame_projection
    from oct07_scene_geometry_v4 import box_vertices
    objects=[o for o in scene['objects'] if not o['key'].startswith('physical_occluder__')]
    # Spatial order and equal treatment of matched cars avoid an acquisition
    # index that identifies the hidden target. No model scores enter this search.
    objects.sort(key=lambda o:(*o['transform'][:2],o['blueprint'],o.get('color','')))
    centering=[o for o in objects if not (hide_marker and o['key']=='direction_marker')]
    # The frozen auditor requires both near hypotheses and the engine in the
    # initial team, then measured motion revealing the marker and engine.
    # The far negative must appear somewhere in the complete history. Do not
    # add a second-round requirement to reobserve every initial near endpoint.
    required=[o for o in centering if o['key']!='distractor_1'] if hide_marker else [
        o for o in centering if o['key'] in ('direction_marker','second_reference','distractor_1')]
    all_bits=(1<<len(required))-1
    centers=set()
    for size in (1,2,3):
        for group in itertools.combinations(centering,size):
            center=tuple(sum(o['transform'][j] for o in group)/size for j in (0,1))
            for dx,dy in ((0,0),(12,0),(-12,0),(0,12),(0,-12)):
                centers.add((round(center[0]+dx,8),round(center[1]+dy,8)))
    candidates=[]
    marker=next(o for o in objects if o['key']=='direction_marker')
    for x,y in sorted(centers):
        point=[x,y,height];obs=dict(camera_world_position=point,camera_orientation_ned_wxyz=[2**-.5,0,-2**-.5,0])
        if hide_marker and projected_support(obs,box_vertices(marker))['box_intersects_image']:continue
        bits=0
        for index,obj in enumerate(required):
            p=projected_support(obs,box_vertices(obj))
            if full_frame_projection(p) and p.get('projected_area_px2',0)>=16:bits|=1<<index
        if bits:
            optional=next((o for o in centering if o['key']=='distractor_1'),None)
            p=projected_support(obs,box_vertices(optional)) if optional else {}
            extra=bool(optional and full_frame_projection(p) and p.get('projected_area_px2',0)>=16)
            candidates.append((bits,point,extra))
    candidates.sort(key=lambda c:(-c[0].bit_count(),-int(c[2]),*c[1][:2]))
    team=scene['team_size'];visits=0
    def solve(chosen,seen):
        nonlocal visits
        visits+=1
        if visits>4096:return None
        if seen==all_bits:
            filled=list(chosen)
            for c in candidates:
                if len(filled)==team:break
                if all(math.dist(c[1][:2],p[:2])>=12 for p in filled):filled.append(c[1])
            return sorted(filled,key=lambda p:tuple(p[:2])) if len(filled)==team else None
        if len(chosen)==team:return None
        uncovered=[i for i in range(len(required)) if not seen&(1<<i)]
        feasible=[c for c in candidates if c[0]&~seen and all(math.dist(c[1][:2],p[:2])>=12 for p in chosen)]
        key=min(uncovered,key=lambda i:sum(bool(c[0]&(1<<i)) for c in feasible))
        for bits,p,extra in [c for c in feasible if c[0]&(1<<key)][:12]:
            result=solve(chosen+[p],seen|bits)
            if result is not None:return result
        return None
    result=solve([],0)
    if result is None:raise ValueError('No bounded private camera cover at frozen height and FOV')
    return result
