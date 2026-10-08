"""One global, predeclared exposure-unit correction for the native UE4.26 build.

The shipped project enables ExtendDefaultLuminanceRange. These bounds are EV100,
not the old0.03/0.64 luminance example. The physical camera (-1) must receive
the same profile because upstream PIPCamera copies its postprocess to Scene.
No image pixels, vehicle assets or frozen color thresholds are modified.
"""
import json

PROFILE_ID='ue426_extended_EV100_camera_and_scene_v1'


def apply(config):
    # JSON round-trip intentionally breaks make_settings' shared-list aliases.
    config=json.loads(json.dumps(config))
    cameras=[config['CameraDefaults']]
    cameras.extend(c for vehicle in config['Vehicles'].values() for c in vehicle['Cameras'].values())
    for camera in cameras:
        captures=camera['CaptureSettings']
        if any(c['ImageType']==-1 for c in captures):raise ValueError('Existing physical-camera override requires review')
        scene=next(c for c in captures if c['ImageType']==0)
        values=dict(AutoExposureMinBrightness=-10.,AutoExposureMaxBrightness=20.,AutoExposureBias=0.,
                    AutoExposureSpeed=100.,MotionBlurAmount=0.)
        scene.update(values,TargetGamma=1.4)
        captures.append(dict(ImageType=-1,Width=scene['Width'],Height=scene['Height'],
            FOV_Degrees=scene['FOV_Degrees'],**values))
    return config


def verify_loaded(config):
    cameras=[config['CameraDefaults']]
    cameras.extend(c for vehicle in config['Vehicles'].values() for c in vehicle['Cameras'].values())
    for camera in cameras:
        for kind in (-1,0):
            rows=[c for c in camera['CaptureSettings'] if c['ImageType']==kind]
            if len(rows)!=1 or any(rows[0].get(k)!=v for k,v in dict(AutoExposureMinBrightness=-10.,
                    AutoExposureMaxBrightness=20.,AutoExposureBias=0.,FOV_Degrees=90).items()):
                raise ValueError('Actually loaded native profile differs')
    return dict(profile=PROFILE_ID,camera_count=len(cameras),loaded_settings_verified=True,
                actual_render_postprocess_readback=False,rendered_color_gate_still_required=True)
