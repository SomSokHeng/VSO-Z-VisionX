import math
import os
import re

vso_ver = "0.1.0"

vso_lim = {
    "dim": 8192,
    "frm": 10000000,
    "pts": 64,
    "cls": 80,
    "str": 512,
}

VID_EXT = (".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
OUT_VID = (".mp4", ".avi", ".mkv", ".mov")
OFF_MDL = ("yolo11n",)


class VxErr(Exception):
    pass


def san(val, cap=64):
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in str(val))[:cap]


def csv_safe(val):
    out = san(val, 128)
    if out and out[0] in "=+-@":
        return "'" + out
    return out


def chk_int(val, lo, hi, nm):
    try:
        num = int(val)
    except (TypeError, ValueError):
        raise VxErr(nm + " must be an integer")
    if num < lo or num > hi:
        raise VxErr("%s must be between %d and %d" % (nm, lo, hi))
    return num


def chk_flt(val, lo, hi, nm):
    try:
        num = float(val)
    except (TypeError, ValueError):
        raise VxErr(nm + " must be a number")
    if not math.isfinite(num) or num < lo or num > hi:
        raise VxErr("%s must be between %g and %g" % (nm, lo, hi))
    return num


def bad_str(raw):
    return not isinstance(raw, str) or not raw or len(raw) > vso_lim["str"] or "\x00" in raw


def chk_src(raw, allow_net=False):
    if bad_str(raw):
        raise VxErr("bad source")
    if re.fullmatch(r"[0-9]", raw):
        return ("cam", int(raw))
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", raw):
        sch = raw.split("://", 1)[0].lower()
        if sch not in ("rtsp", "http", "https"):
            raise VxErr("unsupported url scheme")
        if not allow_net:
            raise VxErr("network sources need --allow-net")
        return ("net", raw)
    pth = os.path.realpath(raw)
    if not os.path.isfile(pth):
        raise VxErr("file not found: " + san(raw))
    ext = os.path.splitext(pth)[1].lower()
    if ext in VID_EXT:
        return ("vid", pth)
    if ext in IMG_EXT:
        return ("img", pth)
    raise VxErr("unsupported file type: " + san(ext))


def chk_out(raw, exts, force=False):
    if bad_str(raw):
        raise VxErr("bad output path")
    pth = os.path.abspath(raw)
    ext = os.path.splitext(pth)[1].lower()
    if ext not in exts:
        raise VxErr("output must end with one of: " + " ".join(exts))
    if not os.path.isdir(os.path.dirname(pth)):
        raise VxErr("output directory does not exist")
    if os.path.islink(pth):
        raise VxErr("refusing to write through a symlink")
    if os.path.exists(pth):
        if not os.path.isfile(pth):
            raise VxErr("output is not a regular file")
        if not force:
            raise VxErr("output exists, use --force to overwrite: " + san(raw))
    return pth


def chk_wts(raw):
    if bad_str(raw):
        raise VxErr("bad weights")
    base = raw[:-5] if raw.lower().endswith(".onnx") else raw
    if base in OFF_MDL and os.sep not in raw and (os.altsep is None or os.altsep not in raw):
        return ("off", base)
    ext = os.path.splitext(raw)[1].lower()
    if ext == ".onnx":
        pth = os.path.realpath(raw)
        if not os.path.isfile(pth):
            raise VxErr("weights file not found: " + san(raw))
        if os.path.getsize(pth) > 1 << 30:
            raise VxErr("weights file too large")
        return ("onx", pth)
    if ext in (".pt", ".pth"):
        raise VxErr("pickle weights are not loaded; export to onnx first (yolo export model=your.pt format=onnx)")
    raise VxErr("weights must be an official name (%s) or an .onnx file" % ", ".join(OFF_MDL))


def p_line(raw):
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise VxErr("line needs 4 values: x1,y1,x2,y2")
    val = tuple(chk_flt(p, 0.0, 1.0, "line") for p in parts)
    if val[0] == val[2] and val[1] == val[3]:
        raise VxErr("line has zero length")
    return val


def p_zone(raw):
    pts = [p for p in raw.split(";") if p.strip()]
    if len(pts) < 3 or len(pts) > vso_lim["pts"]:
        raise VxErr("zone needs 3 to %d points: x,y;x,y;..." % vso_lim["pts"])
    out = []
    for p in pts:
        xy = [v.strip() for v in p.split(",")]
        if len(xy) != 2:
            raise VxErr("zone point must be x,y")
        out.append((chk_flt(xy[0], 0.0, 1.0, "zone"), chk_flt(xy[1], 0.0, 1.0, "zone")))
    ar = 0.0
    for i in range(len(out)):
        x1, y1 = out[i]
        x2, y2 = out[(i + 1) % len(out)]
        ar += x1 * y2 - x2 * y1
    if abs(ar) / 2.0 < 1e-6:
        raise VxErr("zone has no area")
    return tuple(out)


def p_cls(raw):
    out = []
    for p in raw.split(","):
        nm = p.strip().lower()
        if not re.fullmatch(r"[a-z0-9 _-]{1,40}", nm):
            raise VxErr("bad class name: " + san(p))
        out.append(nm)
    if not out or len(out) > vso_lim["cls"]:
        raise VxErr("bad class list")
    return out
