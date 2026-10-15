# VSO-Z VisionX

A computer-vision toolkit powered by YOLO. Object detection, tracking, line and zone counting, motion analysis and annotated video export, all from one command line tool.

Python, OpenCV and YOLO11. Runs on CPU. Has a tiny preset for low-power and mobile devices.

![python](https://img.shields.io/badge/python-3.9%2B-3776ab) ![license](https://img.shields.io/badge/code-Apache--2.0-blue) ![yolo](https://img.shields.io/badge/model-YOLO11n-00c2a8) ![tests](https://img.shields.io/badge/tests-31%20passing-brightgreen)

## Install

    git clone https://github.com/SomSokHeng/VSO-Z-VisionX
    cd VSO-Z-VisionX
    pip install ".[ort]"
    visionx get yolo11n

`[ort]` installs onnxruntime, the recommended backend. Without it only the OpenCV backend is available (see [Mobile and tiny mode](#mobile-and-tiny-mode)). `visionx get` downloads the official YOLO11n model and verifies its SHA-256 before use.

## Usage

    visionx detect photo.jpg -o out.jpg
    visionx track street.mp4 -o tracked.mp4 --csv tracks.csv
    visionx count street.mp4 --line 0.5,0,0.5,1 -o counted.mp4 --summary counts.json
    visionx count lobby.mp4 --zone "0.2,0.2;0.8,0.2;0.8,0.9;0.2,0.9" --anchor bottom
    visionx motion cam.mp4 --heatmap heat.png --events events.json -o motion.mp4
    visionx info video.mp4

| Command | What it does | Outputs |
|---------|--------------|---------|
| `detect` | YOLO object detection on an image, video or camera | annotated image or video, `--jsonl`, `--summary` |
| `track` | Detection with persistent IDs and trails | annotated video, `--csv`, `--jsonl`, `--summary` |
| `count` | Line crossing (in/out per class) and zone occupancy with dwell time | annotated video, `--summary` |
| `motion` | Background-subtraction motion analysis, events and heatmap. No model needed | annotated video, `--csv`, `--events`, `--heatmap`, `--summary` |
| `get` | Download and verify an official model | |
| `info` | Show backends, cached models and source properties | |

### Counting

`--line x1,y1,x2,y2` and `--zone x,y;x,y;...` take fractions of the frame (0 to 1), so they work at any resolution.

"in" means crossing toward the right-hand side when facing from point 1 to point 2. For a line drawn top to bottom, that is right to left on screen. Use `--flip` to swap. `--anchor bottom` counts the foot point of a box, which suits people. `--cool N` ignores repeat crossings by the same ID for N frames.

### Common options

| Option | Meaning |
|--------|---------|
| `-o, --out` | Annotated output: `.mp4 .avi .mkv .mov` for video, `.jpg .png` for images |
| `--stride N` | Process every Nth frame |
| `--max-frames N` | Stop after N processed frames (required for camera and network sources) |
| `-w, --weights` | `yolo11n` or a path to your own `.onnx` file |
| `--conf`, `--iou`, `--imgsz` | Detection thresholds and input size (multiple of 32) |
| `--classes person,car` | Keep only these classes |
| `--backend auto\|ort\|cv` | onnxruntime or OpenCV DNN |
| `--tiny` (or `--mobile`) | Low-power preset |
| `--force` | Overwrite existing output files |
| `--allow-net` | Allow `rtsp://` and `http(s)://` sources |

Exit codes: `0` success, `1` runtime error, `2` invalid input, `130` interrupted.

## Mobile and tiny mode

YOLO11n is the smallest YOLO11 model (about 2.6M parameters, an 11 MB file). `--tiny` adds a low-power preset on top of it:

- 320 px inference instead of 640 px
- every 2nd frame processed
- threads capped at 4
- motion analysis at 320 px wide

Any flag you pass explicitly overrides the preset.

Measured on a desktop-class x86_64 CPU with a test clip I made by panning the standard Ultralytics `bus.jpg` sample photo across the frame (540x720, 90 frames, 4 people and a bus crossing a counting line). The clip is not included in this repository:

| Mode | ms per frame | Frames processed | People counted | Bus counted |
|------|-------------:|-----------------:|---------------:|------------:|
| default (640 px) | ~115 | 90 | 4 of 4 | yes |
| `--tiny` (320 px, stride 2) | ~30 | 45 | 3 of 4 | yes |

That is roughly 7x less compute for the tiny preset, at the cost of missing one of the four people in this clip. Expect lower recall on small or distant objects. These are not phone numbers: this tool has only been tested on Linux x86_64.

### Devices without onnxruntime

If pip cannot install onnxruntime on your device, use the OpenCV backend. It needs only numpy and OpenCV:

    visionx detect photo.jpg -w yolov8n-320.onnx --backend cv --tiny

The OpenCV backend only loads fixed-size models exported with ONNX opset 17 or lower, for example:

    yolo export model=yolov8n.pt format=onnx imgsz=320 opset=12 dynamic=False

It was verified with a static opset-17 YOLOv8n at 640 px. It refuses the official `yolo11n.onnx` (opset 22, dynamic shapes) because OpenCV 4.13 crashes while loading it. VisionX checks the model header first and tells you, instead of crashing. Because a fixed-size model has a fixed input size, `--tiny` only helps through frame skipping and thread limits on this backend.

## Security

VisionX treats every input as untrusted.

- Sources: files must exist, be regular files and have a known video or image extension (checked after resolving symlinks). Camera indexes are limited to 0-9. Network sources are off unless you pass `--allow-net`, only `rtsp`, `http` and `https` are accepted, and FFmpeg protocols are whitelisted.
- Outputs: refuses to overwrite without `--force`, refuses to write through symlinks, and only writes known extensions into existing directories.
- Models: no pickle files. `.pt` weights are rejected, so loading a model cannot run code. The official model is downloaded from a pinned URL, only from GitHub hosts (redirects to other hosts are blocked), capped at the expected size, verified against a pinned SHA-256, stored in a private directory (`~/.cache/visionx/models`, mode 0700) and re-verified every time it is loaded.
- Model files: a small bounds-checked ONNX header reader inspects any `.onnx` file before it is loaded. Malformed or oversized files raise an error, never a crash.
- Model output: boxes are checked for NaN and infinity, clipped to the frame, and capped at 1000 candidates before NMS.
- Text: class names come from the model file. They are stripped of control characters before printing or drawing, and prefixed with `'` in CSV cells that would start a spreadsheet formula.
- Resource limits: frame size is capped at 8192 px per side, and counts, zone points, string lengths and per-ID state are bounded.

## Models and licensing

VisionX code is licensed under the Apache License 2.0 (see `LICENSE`). It does not import or bundle any Ultralytics code.

The official YOLO11 weights that `visionx get` downloads are published by Ultralytics under AGPL-3.0 (https://ultralytics.com/license). They are not part of this repository and stay under their own license. If you distribute those weights or run them as a network service, AGPL-3.0 terms apply to you. If that does not suit your use, pass your own ONNX model with `-w`. This is not legal advice.

Supported model heads: YOLOv8 and YOLO11 style detection exports (one output of shape `1 x (4+classes) x anchors`). Not supported: YOLOv5-style outputs with an objectness column, NMS-baked outputs such as YOLOv10, segmentation and pose models.

## Notes

- Annotated video is written with OpenCV's `mp4v` codec (`XVID` for `.avi`). Some browsers and phone galleries will not play it. Re-encode with ffmpeg if needed.
- Tracking is an IoU tracker with constant-velocity prediction and a second pass for low-confidence detections. It does not use appearance features, so IDs can swap when people cross closely or are occluded for long.
- CUDA is used automatically when your onnxruntime build provides it (`--device cuda` to require it). This path is untested.

## Development

    pip install -e ".[ort]"
    python -m unittest discover -s tests

The 31 tests cover input validation, the ONNX header reader (including random corruption), box decoding, tracking, counting and motion analysis. They do not need a model download.

## License

Apache-2.0 for the code. Model weights are licensed separately, see above.
