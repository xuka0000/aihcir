"""Finite full-segment geometry sampling. Never authorizes flight or removes boxes."""
import itertools
import math
import time

def corridor_segments(current, phases):
    previous=current;rows=[]
    for phase,points in phases:
        if len(points)!=len(previous):raise ValueError('Agent count changed')
        for agent,(start,end) in enumerate(zip(previous,points)):
            if len(start)!=3 or len(end)!=3 or not all(math.isfinite(x) for x in [*start,*end]):
                raise ValueError('Invalid corridor endpoints')
            rows.append(dict(phase=phase,agent=agent,start=list(start),end=list(end)))
        previous=points
    return rows

def sample_corridors(segments, cast, *, controls, max_rays=20000, max_seconds=60., clock=time.monotonic):
    # Sample the unchanged 6m horizontal arrival/body margin and2m vertical
    # allowance along the entire segment, rather than checking endpoints only.
    if not 1<=max_rays<=50000 or not 0<max_seconds<=120:raise ValueError('Invalid query bound')
    if controls.get('ground_hits',0)<1 or controls.get('air_hits')!=0:
        raise ValueError('Ray controls failed or absent')
    offsets=list(itertools.product(range(-6,7),range(-6,7),range(-2,3)))
    required=len(segments)*len(offsets)
    base=dict(required_rays=required,max_rays=max_rays,controls=controls,
              spacing_m=1.,horizontal_margin_m=6.,vertical_margin_m=2.,
              flight_authorized=False,swept_volume_safety_proven=False,coarse_gate_overridden=False)
    if required>max_rays:return dict(base,stage='budget_rejected',rays_completed=0,sampled_clear=None,segments=[])
    start_time=clock();rows=[];completed=0
    for segment in segments:
        if len(segment['start'])!=3 or len(segment['end'])!=3 or not all(math.isfinite(x) for x in segment['start']+segment['end']):
            raise ValueError('Invalid segment')
        hits=[]
        for offset in offsets:
            if clock()-start_time>=max_seconds:
                return dict(base,stage='bounded_partial',rays_completed=completed,sampled_clear=None,segments=rows,partial_segment=dict(segment,hits=hits))
            a=[v+d for v,d in zip(segment['start'],offset)];b=[v+d for v,d in zip(segment['end'],offset)]
            try:
                found=cast(a,b)
                if not isinstance(found,list):raise ValueError('Malformed ray response')
            except Exception as error:
                return dict(base,stage='query_failed',rays_completed=completed,sampled_clear=None,segments=rows,error=repr(error),failed_segment=segment)
            completed+=1
            if found:hits.append(dict(offset=offset,hits=found))
        rows.append(dict(segment,rays=len(offsets),hits=hits))
    return dict(base,stage='completed',rays_completed=completed,sampled_clear=not any(r['hits'] for r in rows),segments=rows)


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    p=argparse.ArgumentParser()
    p.add_argument('--record',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--expected-map',required=True)
    p.add_argument('--max-rays',type=int,default=20000)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior diagnostic attempt')
    raw=a.record.read_bytes();record=json.loads(raw)
    result=dict(observed_at=time.time(),source_sha256=hashlib.sha256(raw).hexdigest(),flight_authorized=False)
    try:
        segments=record['private_static_transit_failure']['segments']
        import carla
        client=carla.Client('127.0.0.1',a.port);client.set_timeout(2.)
        world=client.get_world();actual=world.get_map().name.rsplit('/',1)[-1]
        if actual!=a.expected_map:raise ValueError('Map mismatch; diagnostic never loads a map')
        def cast(start,end):
            hits=world.cast_ray(carla.Location(x=start[0],y=start[1],z=start[2]),carla.Location(x=end[0],y=end[1],z=end[2]))
            return [dict(xyz=[h.location.x,h.location.y,h.location.z],label=str(h.label)) for h in hits]
        origin=world.get_map().get_spawn_points()[0].location
        controls=dict(ground_hits=len(cast([origin.x,origin.y,origin.z+5],[origin.x,origin.y,origin.z-5])),air_hits=len(cast([origin.x,origin.y,150],[origin.x,origin.y,140])))
        from static_box_clearance import StaticBoxScreen
        boxes=StaticBoxScreen.from_world(world)
        endpoints=[]
        for segment in segments:
            if segment['phase']!='cruise':continue
            for side in ('start','end'):
                point=segment[side];level=point[2]
                indices=[i for i,(low,high) in enumerate(zip(boxes.low,boxes.high))
                         if high[2]>=level-2 and low[2]<=level+100
                         and all(low[d]-6<=point[d]<=high[d]+6 for d in (0,1))]
                endpoints.append(dict(agent=segment['agent'],side=side,point=point,
                    blockers=[dict(boxes.records[i],aabb_min=boxes.low[i].tolist(),
                                   aabb_max=boxes.high[i].tolist()) for i in indices]))
        result.update(map=actual,coarse_endpoint_blockers=endpoints,
                      **sample_corridors(segments,cast,controls=controls,max_rays=a.max_rays))
    except Exception as error:
        result.update(stage='failed',error=repr(error),sampled_clear=None)
        raise
    finally:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
