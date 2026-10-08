"""Explicit visual-weather configuration, not AirSim flight-disturbance wind.

CARLA 0.9.13 WeatherParameters.wind_intensity affects leaf animation. In the
compiled CarlaAir v0.1.7 renderer, paused camera poses and clocks did not freeze
its occlusion changes. A zero-wind rendering profile is an explicit limitation,
not a general fix for dynamic-weather annotation. All RGB and masks must share
the same profile; never switch it between an RGB image and its mask.
"""
import math
import time

FROZEN_FIELDS=('cloudiness','precipitation','precipitation_deposits','wind_intensity',
    'sun_azimuth_angle','sun_altitude_angle','fog_density','fog_distance','wetness','fog_falloff',
    'scattering_intensity','mie_scattering_scale','rayleigh_scattering_scale')
_FROZEN=None

def set_frozen_visual_profile(world,timeout=5.):
    """Freeze every supported ClearNoon field, before RGB or masks are read."""
    import carla
    global _FROZEN
    requested=carla.WeatherParameters.ClearNoon
    requested.wind_intensity=0.
    expected={k:float(getattr(requested,k)) for k in FROZEN_FIELDS if hasattr(requested,k)}
    world.set_weather(requested);deadline=time.monotonic()+timeout
    while True:
        actual=world.get_weather()
        if all(abs(float(getattr(actual,k))-v)<=1e-5 for k,v in expected.items()):break
        if time.monotonic()>=deadline:raise TimeoutError('Full visual-weather readback failed')
        time.sleep(.05)
    _FROZEN=expected
    return verify_static_visual_wind(world)


def verify_static_visual_wind(world):
    value = float(world.get_weather().wind_intensity)
    if not math.isfinite(value) or abs(value) > 1e-6:
        raise RuntimeError(f'Static visual weather changed before capture: {value}')
    result={'wind_intensity': value, 'scope': 'visual_weather_not_flight_wind',
            'dynamic_weather_annotation_validated': False}
    if _FROZEN is not None:
        actual=world.get_weather()
        fields={k:float(getattr(actual,k)) for k in _FROZEN}
        if any(not math.isfinite(v) or abs(v-_FROZEN[k])>1e-5 for k,v in fields.items()):raise RuntimeError('Frozen visual weather changed')
        result.update(profile='ClearNoon_static_v4',fields=fields,all_supported_fields_readback_verified=True)
    return result


def set_visual_wind(world, value, timeout=5.):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 100 or timeout <= 0:
        raise ValueError('Invalid visual wind or timeout')
    before = world.get_weather()
    previous = float(before.wind_intensity)
    before.wind_intensity = value
    world.set_weather(before)
    deadline = time.monotonic()+timeout
    attempts = 0
    while True:
        readback = float(world.get_weather().wind_intensity)
        attempts += 1
        if math.isfinite(readback) and abs(readback-value) <= 1e-6:
            return {'previous': previous, 'requested': value, 'readback': readback,
                    'readback_attempts': attempts, 'scope': 'visual_weather_not_flight_wind'}
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Visual wind readback failed: expected {value}, got {readback}')
        time.sleep(.05)
