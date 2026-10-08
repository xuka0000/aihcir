"""Opt-in offline dataset construction routes; excluded from model input."""
import hashlib
import json
import math
from pathlib import Path


def load_profiles(path=None):
    path=Path(path) if path is not None else Path(__file__).with_name('construction_profiles.json')
    if not path.exists():return {}
    data=json.loads(path.read_text())
    for name,value in data.items():
        ep=value['episode_id'];points=value['offsets']
        if (name!='constructed_'+ep+'_independent_clearance' or not ep.startswith('train_')
                or value.get('construction_uses_private_geometry') is not True or value.get('policy_input') is not False
                or len(value['scene_sha256'])!=64 or len(points) not in (2,4)):raise ValueError('Invalid construction profile identity')
        for phase in [points,value.get('reveal_offsets',points)]:
            if len(phase)!=len(points):raise ValueError('Construction phases have different team sizes')
            for p in phase:
                if len(p)!=3 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in p):raise ValueError('Invalid construction coordinates')
                if max(abs(p[0]),abs(p[1]))>120 or not 30<=p[2]<=70:raise ValueError('Construction coordinates exceed bounds')
            if any(math.dist(phase[i][:2],phase[j][:2])<12 for i in range(len(phase)) for j in range(i)):
                raise ValueError('Insufficient viewpoint separation')
    return data


def waypoints(profile,center,team,rounds):
    if team not in (2,4) or team!=len(profile['offsets']) or rounds!=4:
        raise ValueError('Construction routes require matching two/four agents and four rounds')
    return [[[float(center[0]+x+4*(t%2)),float(center[1]+y),float(z)]
             for x,y,z in (profile.get('reveal_offsets',profile['offsets']) if t>=2 else profile['offsets'])] for t in range(4)]


def validate_binding(profile,episode_ids,scene_file):
    if episode_ids!=profile['episode_id'] or hashlib.sha256(Path(scene_file).read_bytes()).hexdigest()!=profile['scene_sha256']:
        raise ValueError('Construction route does not match immutable scene')
