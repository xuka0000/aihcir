"""Explicit local controls for a frozen-model development pilot, not full TRACE.

Single/late/central fusion are experimental controls, not Where2comm replicas.
The relation-guided condition is a two-call prompting prototype, without the
manuscript's trained graph, association, voxel, or utility heads.
"""
import json
import math
import re


VLM_METHODS = ('single_uav_history', 'repeat_view', 'team_latest',
               'team_uniform', 'team_pose_diverse', 'team_late_fusion',
               'relation_guided_prompt_prototype')
DETECTOR_METHODS = ('grounding_dino_full_instruction', 'grounding_dino_target_phrase')
METHODS = VLM_METHODS + DETECTOR_METHODS
COMMUNICATION_LIMIT = 262144


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def round_number(row):
    match = re.fullmatch(r'round(\d+)_UAV_(\d+)\.png', row['image'])
    if not match or row['agent'] != 'UAV_'+str(int(match[2])):
        raise ValueError('Invalid observation identity')
    return int(match[1])


def prefix(observations, cutoff):
    result = sorted((o for o in observations if round_number(o) <= cutoff),
                    key=lambda o: (round_number(o), o['agent']))
    if not result or len({o['image'] for o in result}) != len(result):
        raise ValueError('Empty or duplicate observation prefix')
    return result


def select_views(pool, method):
    if method == 'single_uav_history':
        pool = [o for o in pool if o['agent'] == 'UAV_0']
    if not pool:
        raise ValueError('Empty view pool')
    if method == 'repeat_view':
        return [next(o for o in pool if o['agent'] == 'UAV_0')]*4
    if method == 'team_latest':
        result = pool[-4:]
        return (result*4)[:4] if len(result)<4 else result
    if method == 'team_pose_diverse':
        selected = [pool[0]]
        while len(selected) < min(4, len(pool)):
            remaining = [o for o in pool if o not in selected]
            selected.append(max(remaining, key=lambda o: min(
                math.dist(o['camera_world_position'], s['camera_world_position']) for s in selected)))
        return (selected*4)[:4]
    if method in METHODS:
        return [pool[round(i*(len(pool)-1)/3)] for i in range(4)]
    raise ValueError('Unknown method')


def parse_prediction(text, allowed_views, selection_pool=None):
    value = text.strip()
    if value.startswith('```'):
        lines = value.splitlines()
        if not lines[-1].strip() == '```':
            raise ValueError('Unclosed JSON')
        value = '\n'.join(lines[1:-1])
    data = json.loads(value)
    if not isinstance(data, dict) or set(data)-{'target','relations','confidence','unresolved','next_view_ids'}:
        raise ValueError('Invalid prediction fields')
    def entity(obj):
        if not isinstance(obj,dict) or set(obj)!={'view_id','bbox_xyxy'} or obj['view_id'] not in allowed_views:
            raise ValueError('Entity is not grounded in an observed image')
        b = obj['bbox_xyxy']
        if (not isinstance(b,list) or len(b)!=4
                or not all(type(v) in (int,float) and math.isfinite(v) for v in b)
                or not (0<=b[0]<b[2]<=640 and 0<=b[1]<b[3]<=480)):
            raise ValueError('Invalid original-pixel box')
    if 'target' not in data:
        raise ValueError('Missing target')
    if data['target'] is not None:
        entity(data['target'])
    confidence = data.get('confidence')
    if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1:
        raise ValueError('Invalid confidence')
    relations = data.get('relations')
    if not isinstance(relations,list) or len(relations)>6:
        raise ValueError('Invalid relation list')
    for rel in relations:
        if (not isinstance(rel,dict) or set(rel)!={'subject','relation','object'}
                or rel['relation'] not in ('near','east','west','north','south')):
            raise ValueError('Invalid spatial predicate')
        entity(rel['subject']); entity(rel['object'])
    if not isinstance(data.get('unresolved'),str):
        raise ValueError('Missing uncertainty')
    chosen = data.get('next_view_ids', [])
    if not isinstance(chosen,list) or len(chosen)>2 or any(not isinstance(x,str) for x in chosen):
        raise ValueError('Invalid next view selection')
    if selection_pool is not None and any(x not in selection_pool for x in chosen):
        raise ValueError('Selection references an unavailable/future view')
    return data


def target_phrase(instruction):
    match = re.match(r'(?:Locate|Find) the (.+?) (?:near|north|south|east|west)\b', instruction)
    if not match:
        raise ValueError('Unsupported public instruction grammar')
    return match[1].strip()+'.'


def box_iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


class PacketLedger:
    """Actual serialized payload bytes; no claimed physical network latency."""
    def __init__(self, limit=COMMUNICATION_LIMIT):
        self.limit=limit; self.bytes=0; self.events=[]; self.sources=set()

    def send(self, sender, kind, payload, binary=b'', source=None):
        if sender=='UAV_0':
            return True
        if source is not None and source in self.sources:
            return True
        envelope={'sender':sender,'recipient':'UAV_0','kind':kind,'payload':payload}
        size=len(canonical(envelope).encode('utf-8'))+len(binary)
        accepted=self.bytes+size<=self.limit
        self.events.append({'sender':sender,'kind':kind,'bytes':size,'accepted':accepted,
                            'source':source})
        if accepted:
            self.bytes+=size
            if source is not None:
                self.sources.add(source)
        return accepted


def image_axes(q):
    if len(q)!=4 or not all(math.isfinite(v) for v in q):
        raise ValueError('Invalid camera quaternion')
    norm=math.sqrt(sum(v*v for v in q))
    if norm<1e-8: raise ValueError('Zero quaternion')
    w,x,y,z=[v/norm for v in q]
    # Quaternion columns for camera right/down expressed in global NED.
    return {'image_right_east_north':[2*(x*y-z*w), -(1-2*(x*x+z*z))],
            'image_down_east_north':[2*(x*z+y*w), -2*(y*z-x*w)]}


def prompt(instruction, current, memory=None, available=None, relation_guided=False):
    mapping=[{'image_index':i,'view_id':o['image'], 'agent':o['agent'],
              'camera_world_position':o['camera_world_position'],
              'camera_orientation_ned_wxyz':o['camera_orientation_ned_wxyz'],
              **image_axes(o['camera_orientation_ned_wxyz'])} for i,o in enumerate(current)]
    text = '''Solve the UAV visual localization task using only supplied images and prior observed evidence.
All boxes MUST use ORIGINAL 640x480 pixel coordinates of the named source image.
Return one target instance, not all similar vehicles; abstain with target=null if unresolved.
World east is increasing CARLA x; north is decreasing CARLA y. Image left is not world west.
The camera quaternion is wxyz in NED; its local axes are forward, right, down. Horizontal FOV is 90 degrees.
The supplied image-axis east/north vectors are derived from that camera rotation, not from object annotations.
World positions and camera rotations are observations, not target locations. No object depth is given.
Do not claim a geographic relation if you cannot establish its endpoints and orientation.
Return ONLY JSON with target, relations, confidence, unresolved, next_view_ids.
target is null or {"view_id":"...","bbox_xyxy":[x0,y0,x1,y1]}.
relations is a list of {"subject":<same entity schema>,"relation":"near/east/west/north/south","object":<same entity schema>}.
confidence is between 0 and 1. unresolved is a short string. next_view_ids is a list of at most two image names.
Do not confuse an instruction requirement or a hypothesis with visually verified evidence.
'''
    text+='\nTASK: '+instruction+'\nCURRENT IMAGE MAPPING: '+canonical(mapping)
    if memory is not None:
        text+='\nPRIOR OBSERVED EVIDENCE (fallible; correct it when contradicted): '+canonical(memory)
    if relation_guided:
        text+='\nIdentify the missing task relation. Use new evidence to verify or reject intermediate hypotheses; do not merely repeat the instruction.'
    if available is not None:
        text+='\nSelect two helpful retained views using only the metadata below. Their image contents are not yet supplied.\nAVAILABLE VIEW METADATA: '+canonical(available)
    else:
        text+='\nThis is the final decision; return next_view_ids=[].'
    return text
