"""Conservative static AABB rejection in addition to the existing ray screen.

Used only for development sensor acquisition, never a policy input. Enclosing
boxes can over-reject empty space and the inventory may omit geometry; this
cannot certify a collision-free path. Flight feedback remains mandatory.
"""
import numpy as np


class StaticBoxScreen:
    def __init__(self, records):
        if not records:raise ValueError('Static geometry inventory is empty')
        self.records=records
        vertices=np.asarray([r['vertices'] for r in records],dtype=float)
        if vertices.shape!=(len(records),8,3) or not np.isfinite(vertices).all():
            raise ValueError('Invalid static bounding-box vertices')
        self.low,self.high=vertices.min(axis=1),vertices.max(axis=1)

    @classmethod
    def from_world(cls,world):
        import carla
        records=[]
        for obj in world.get_environment_objects(carla.CityObjectLabel.Any):
            records.append(dict(id=int(obj.id),name=obj.name,label=str(obj.type),
                vertices=[[v.x,v.y,v.z] for v in obj.bounding_box.get_world_vertices(carla.Transform())]))
        return cls(records)

    def __call__(self,targets,cruise_top=120.,margin=6.):
        points=np.asarray(targets,dtype=float)
        if (points.ndim!=3 or points.shape[-1]!=3 or not points.size
                or not np.isfinite(points).all() or not np.isfinite([cruise_top,margin]).all()
                or margin<0 or np.max(points[:,:,2])>cruise_top):
            raise ValueError('Invalid static descent columns')
        tests=[]
        for t,batch in enumerate(points):
            for i,p in enumerate(batch):
                # Same 6m horizontal arrival/body footprint and 2m vertical
                # endpoint allowance used by the established ray rejection.
                overlap=((self.low[:,:2]<=p[:2]+margin).all(1)
                         &(self.high[:,:2]>=p[:2]-margin).all(1)
                         &(self.high[:,2]>=p[2]-2.)&(self.low[:,2]<=cruise_top+2.))
                hits=[dict(id=self.records[j]['id'],name=self.records[j]['name'],
                           label=self.records[j]['label'],aabb_min=self.low[j].tolist(),
                           aabb_max=self.high[j].tolist()) for j in np.flatnonzero(overlap)]
                tests.append(dict(round=t,agent_index=i,point=p.tolist(),hits=hits,
                                  channel='environment_bounding_box'))
        return dict(kind='conservative_environment_box_rejection',rays=tests,
                    passed=all(not r['hits'] for r in tests),inventory_count=len(self.records),
                    margin_m=margin,policy_input=False,swept_volume_safety_proven=False)


def combined_screen(world,targets,boxes,cruise_top):
    from survey_geometry import screen_descent_columns
    box=boxes(targets,cruise_top)
    if not box['passed']:
        return dict(box,kind='ray_and_static_box_rejection',box_passed=False,
                    ray_passed=None,ray_not_run_reason='Already rejected by enclosing static geometry')
    ray=screen_descent_columns(world,targets,cruise_top)
    return dict(kind='ray_and_static_box_rejection',passed=ray['passed'] and box['passed'],
                rays=ray['rays']+box['rays'],ray_passed=ray['passed'],box_passed=box['passed'],
                inventory_count=box['inventory_count'],policy_input=False,swept_volume_safety_proven=False)


def independent_clearance_waypoints(targets,screen):
    """Choose among the existing eight 20m offsets independently per viewpoint.

    Same 70m height ceiling, 10m height increments and 12m endpoint separation.
    At most nine offsets per viewpoint, five heights per offset, and 9**4
    assignments per round. Selection uses displacement only, no scene targets.
    """
    import itertools
    import math
    from survey_geometry import clearance_waypoints
    original=np.asarray(targets,dtype=float)
    if (original.ndim!=3 or original.shape[1] not in (2,4) or original.shape[2]!=3
            or len(original)>4 or not np.isfinite(original).all()):
        raise ValueError('Invalid bounded independent acquisition route')
    offsets=[(0,0),(20,0),(-20,0),(0,20),(0,-20),(20,20),(-20,20),(20,-20),(-20,-20)]
    record=dict(kind='independent_static_clearance_acquisition',requested=original.tolist(),
                offsets_m=offsets,ceiling_m=70.,attempts=[],uses_object_truth=False,
                policy_input=False,swept_volume_safety_proven=False)
    # A globally feasible zero-displacement route cannot be improved under
    # the existing nonnegative displacement objective. Test every original
    # column and pair before skipping alternative offsets; no gate is removed.
    initial=screen(original.tolist())
    record['initial_screen']=initial
    separated=all(math.dist(batch[i][:2],batch[j][:2])>=12
                  for batch in original for i in range(len(batch)) for j in range(i))
    if initial['passed'] and separated:
        record.update(passed=True,actual=original.tolist(),final_screen=initial,
                      selection='feasible_zero_displacement',alternative_search_skipped=True)
        return original.tolist(),record
    chosen=[]
    for t,batch in enumerate(original):
        options=[]
        for i,point in enumerate(batch):
            valid=[]
            for index,(dx,dy) in enumerate(offsets):
                shifted=(point+[dx,dy,0]).tolist()
                revised,audit=clearance_waypoints([[shifted]],screen)
                record['attempts'].append(dict(round=t,agent_index=i,offset_index=index,audit=audit))
                if audit['passed']:
                    candidate=revised[0][0]
                    valid.append((math.dist(point,candidate),index,candidate))
            options.append(valid)
        feasible=[]
        for combination in itertools.product(*options):
            points=[v[2] for v in combination]
            if all(math.dist(points[i][:2],points[j][:2])>=12 for i in range(len(points)) for j in range(i)):
                feasible.append((sum(v[0] for v in combination),tuple(v[1] for v in combination),points))
        if not feasible:
            record.update(passed=False,failed_round=t,actual=original.tolist())
            return original.tolist(),record
        chosen.append(min(feasible,key=lambda v:(v[0],v[1]))[2])
    final=screen(chosen)
    record.update(passed=final['passed'],actual=chosen,final_screen=final)
    return chosen,record


class MeasuredRayRefinement:
    """Opt-in construction diagnostic, never a swept-volume clearance proof.

    Preserve coarse AABB rejections; query the unchanged 6m footprint at 1m
    spacing. Ground/air controls are mandatory. Actual flight collision and
    arrival checks still determine whether a captured episode can be admitted.
    """
    def __init__(self,world,boxes,on_progress=None):
        import carla
        self.world,self.boxes=world,boxes;self.cache={};self.carla=carla
        self.on_progress=on_progress;self.queries_completed=0
        p=world.get_map().get_spawn_points()[0].location
        self.ground=self.cast(p.x,p.y,p.z+5,p.z-5)
        self.air=self.cast(p.x,p.y,150,140)
        if not self.ground or self.air:raise RuntimeError('Static ray controls failed')

    def cast(self,x,y,top,bottom):
        c=self.carla
        result=[dict(xyz=[h.location.x,h.location.y,h.location.z],label=str(h.label))
                for h in self.world.cast_ray(c.Location(x=x,y=y,z=top),c.Location(x=x,y=y,z=bottom))]
        self.queries_completed+=1
        if self.on_progress is not None and (self.queries_completed==2 or self.queries_completed%169==0):
            self.on_progress(dict(queries_completed=self.queries_completed,
                                  cached_columns=len(self.cache),last_point=[x,y,top,bottom]))
        return result

    def __call__(self,targets,cruise_top):
        box=self.boxes(targets,cruise_top);tests=[]
        for t,batch in enumerate(targets):
            for i,p in enumerate(batch):
                key=tuple(p)+(cruise_top,)
                if key not in self.cache:
                    self.cache[key]=[dict(offset_xy_m=[dx,dy],hits=self.cast(p[0]+dx,p[1]+dy,cruise_top+2.,p[2]-2.))
                                     for dx in range(-6,7) for dy in range(-6,7)]
                tests.extend(dict(row,round=t,agent_index=i) for row in self.cache[key])
        return dict(kind='explicit_dense_ray_construction_diagnostic',passed=all(not r['hits'] for r in tests),
                    rays=tests,coarse_box_screen=box,box_passed=box['passed'],ground_control=self.ground,air_control=self.air,
                    static_gate_mode='coarse_AABB_recorded_then_dense_ray_refinement',sample_spacing_m=1,margin_m=6,
                    policy_input=False,swept_volume_safety_proven=False,
                    requires_actual_collision_and_arrival_checks=True)

