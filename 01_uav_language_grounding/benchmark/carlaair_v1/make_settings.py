"""Write process-local AirSim settings, without replacing user global settings."""
import argparse
import json
from pathlib import Path


def settings(agents, port):
    if agents not in (2, 4):
        raise ValueError("Only registered team sizes are supported")
    captures = [{"ImageType": kind, "Width": 640, "Height": 480,
                 "FOV_Degrees": 90, "MotionBlurAmount": 0} for kind in (0, 1, 5)]
    return {
        "SettingsVersion": 1.2, "SimMode": "Multirotor", "ClockSpeed": 1,
        "ApiServerPort": port, "ViewMode": "NoDisplay",
        "CameraDefaults": {"CaptureSettings": captures},
        "Vehicles": {f"UAV_{i}": {
            "VehicleType": "SimpleFlight", "AutoCreate": True,
            "X": 8 * i, "Y": 0, "Z": -5, "Yaw": 0,
            "Cameras": {"0": {"X": 0.5, "Y": 0, "Z": 0.1, "Pitch": -35,
                              "CaptureSettings": captures},
                        "front_center": {"X": 0.5, "Y": 0, "Z": 0.1, "Pitch": -35,
                                         "CaptureSettings": captures}}}
            for i in range(agents)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, required=True)
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(settings(args.agents, args.port), indent=2), encoding="utf-8")
