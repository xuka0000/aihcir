"""One prospective native rendering correction; never edits RGB or cue gates."""
COMMAND='r.PostProcessAAQuality 0'
PROFILE='ue426_native_no_postprocess_AA_v1'
SOURCE_URL='https://dev.epicgames.com/documentation/en-us/unreal-engine/scalability-reference?application_version=4.27'


def launch_argument():
    # Epic documents zero for both FXAA and temporal AA. Applied globally at
    # native launch, not to an actor, split, method, crop or exported image.
    return '-ExecCmds='+COMMAND+',r.PostProcessAAQuality'


def verify_startup_log(text):
    import re
    values=re.findall(r'r\.PostProcessAAQuality\s*=\s*[\"\']?([0-9]+)(?![\d.])',text)
    if not values or values[-1]!='0':
        raise ValueError('Native console has not verified postprocess AA zero')
    return dict(profile=PROFILE,console_variable='r.PostProcessAAQuality',
        effective_value=0,readback_verified=True,measurement_frames_before_readback=0,
        readback_source='current simulator console response',observed_values=values)


def set_native_console(client):
    # The pinned native API acknowledges transport, not the resulting value.
    # The caller must still verify the fresh engine response before any frame.
    commands=(COMMAND,'r.PostProcessAAQuality')
    acknowledgements=[bool(client.simRunConsoleCommand(c)) for c in commands]
    if not all(acknowledgements):
        raise RuntimeError('Native console transport unavailable')
    return dict(commands=list(commands),transport_ack=acknowledgements,
                readback_verified=False,measurement_frames=0)


if __name__=='__main__':
    import argparse,json
    import airsim
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True)
    a=p.parse_args();client=airsim.MultirotorClient(port=a.port,timeout_value=10)
    names=sorted(client.listVehicles())
    expected=['UAV_0','UAV_1'] if a.port==41451 else ['UAV_0','UAV_1','UAV_2','UAV_3']
    if names!=expected:raise ValueError('Actually loaded UAV identities differ')
    from oct07_native_exposure_profile import verify_loaded
    settings=json.loads(client.getSettingsString());loaded=verify_loaded(settings)
    receipt=set_native_console(client);receipt.update(loaded_settings=loaded,native_vehicles=names)
    print(json.dumps(receipt))


def proposal(raw_audit_sha256):
    return dict(attempt='scene_material4_detailfix1',approved=None,
        previous_attempt='scene_material3_sourcefix1',previous_frames_used=14,
        previous_allowance_reopened=False,requested_total_seconds=1200,
        requested_frame_limit=14,profile=PROFILE,
        native_command=COMMAND,native_launch_argument=launch_argument(),
        correction='Disable native postprocess AA globally;retain the same640x480 '
                   'camera/FOV/gamma/exposure/colors/poses/HSV masks.',
        prerequisite='Verify startup console receipt before the original14-frame '
                     'Ford/Lincoln/red-blue-white/two-height and yellow-reference run.',
        evidence_sha256=raw_audit_sha256,
        diagnosis='Ford far-blue176 pixels contains76 accepted cues and20 additional '
                  'pixels that meet hue/brightness but lack saturation;brightness-only '
                  'failures0. Pixel mixing is a source-supported hypothesis,not proven '
                  'causality. Lincoln requested paint does not match observed appearance.',
        source=dict(url=SOURCE_URL,section='Anti-Aliasing',engine_reference='UE4.27',
                    actual_binary='UE4.26',version_difference_requires_live_readback=True),
        predicted_result='Ford all6 frozen cue checks pass with unchanged masks; '
                         'native raw RGB remains recognizable.',
        falsification='Missing live command receipt,failed cue check or failed visual '
                      'review stops this attempt;no parameter sweep or extra frames.',
        after_pass='Freeze native renderer version;one TRAIN24 qualification then '
                   '192-scene first capture and prepared12-baseline pipeline.',
        compute_stop='2026-10-07T12:00:00+08:00',
        delivery_stop='2026-10-07T23:59:59+08:00',new_instance_required=False)
