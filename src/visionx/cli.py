import argparse
import csv
import json
import os
import sys
import time

import cv2

from . import __version__, drw, mdl
from .cnt import InfCnt
from .det import has_ort, mk_det
from .media import Src, Wtr
from .mot import InfMot
from .sec import IMG_EXT, OFF_MDL, OUT_VID, VxErr, chk_flt, chk_int, chk_out, chk_src, csv_safe, p_cls, p_line, p_zone, san, vso_lim
from .trk import InfTrk


def adp(fn):
    def wrp(val):
        try:
            return fn(val)
        except VxErr as er:
            raise argparse.ArgumentTypeError(str(er))
    return wrp


def t_int(lo, hi, nm):
    return adp(lambda v: chk_int(v, lo, hi, nm))


def t_flt(lo, hi, nm):
    return adp(lambda v: chk_flt(v, lo, hi, nm))


def t_sz(val):
    num = chk_int(val, 32, 2048, "imgsz")
    if num % 32:
        raise VxErr("imgsz must be a multiple of 32")
    return num


def bld_com():
    com = argparse.ArgumentParser(add_help=False)
    com.add_argument("src", help="image or video file, camera index 0-9, or rtsp/http(s) url with --allow-net")
    com.add_argument("-o", "--out", help="annotated output (.mp4 .avi .mkv .mov for video, .jpg .png for image)")
    com.add_argument("--summary", help="write a json summary to this path")
    com.add_argument("--force", action="store_true", help="overwrite existing output files")
    com.add_argument("--stride", type=t_int(1, 1000, "stride"), default=None, help="process every Nth frame")
    com.add_argument("--max-frames", type=t_int(1, vso_lim["frm"], "max-frames"), default=0, help="stop after N processed frames (required for camera and network sources)")
    com.add_argument("--allow-net", action="store_true", help="allow rtsp/http(s) sources")
    com.add_argument("--tiny", "--mobile", dest="tiny", action="store_true", help="low-power preset: 320px input, every 2nd frame, capped threads, smaller motion analysis")
    com.add_argument("-q", "--quiet", action="store_true")
    return com


def bld_det():
    dtc = argparse.ArgumentParser(add_help=False)
    dtc.add_argument("-w", "--weights", default="yolo11n", help="official model name (%s) or path to a .onnx file" % ", ".join(OFF_MDL))
    dtc.add_argument("--conf", type=t_flt(0.01, 1.0, "conf"), default=0.25)
    dtc.add_argument("--iou", type=t_flt(0.05, 1.0, "iou"), default=0.45)
    dtc.add_argument("--imgsz", type=adp(t_sz), default=None, help="inference size, multiple of 32 (default 640, tiny 320)")
    dtc.add_argument("--classes", type=adp(p_cls), default=None, help="comma separated class names to keep")
    dtc.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    dtc.add_argument("--backend", choices=("auto", "ort", "cv"), default="auto", help="ort = onnxruntime, cv = opencv dnn (fixed-size opset<=17 models only)")
    dtc.add_argument("--threads", type=t_int(0, 64, "threads"), default=None, help="cpu threads, 0 = library default")
    return dtc


def bld_trk():
    trc = argparse.ArgumentParser(add_help=False)
    trc.add_argument("--trail", type=t_int(0, 200, "trail"), default=30, help="trail length in frames, 0 to hide")
    trc.add_argument("--max-age", type=t_int(0, 1000, "max-age"), default=30, help="frames to keep a lost track")
    trc.add_argument("--min-hits", type=t_int(1, 20, "min-hits"), default=2, help="detections before a track gets an id")
    return trc


def bld():
    prs = argparse.ArgumentParser(prog="visionx", description="Computer-vision toolkit: YOLO detection, tracking, counting, motion analysis, annotated video export.")
    prs.add_argument("-V", "--version", action="version", version="visionx " + __version__)
    sub = prs.add_subparsers(dest="cmd", required=True)
    com = bld_com()
    dtc = bld_det()
    trc = bld_trk()

    p = sub.add_parser("info", help="show backends, cached models and optional source properties")
    p.add_argument("src", nargs="?")
    p.add_argument("--allow-net", action="store_true")

    p = sub.add_parser("get", help="download and verify an official model")
    p.add_argument("name", choices=OFF_MDL)
    p.add_argument("--force", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")

    p = sub.add_parser("detect", parents=[com, dtc], help="object detection")
    p.add_argument("--jsonl", help="stream per-frame detections to a .jsonl file")

    p = sub.add_parser("track", parents=[com, dtc, trc], help="detection with persistent track ids and trails")
    p.add_argument("--jsonl", help="stream per-frame tracks to a .jsonl file")
    p.add_argument("--csv", help="write tracks to a .csv file")

    p = sub.add_parser("count", parents=[com, dtc, trc], help="line-crossing and zone counting")
    p.add_argument("--line", type=adp(p_line), help="x1,y1,x2,y2 as fractions of the frame (0-1)")
    p.add_argument("--zone", type=adp(p_zone), help="x,y;x,y;... polygon as fractions of the frame (0-1)")
    p.add_argument("--flip", action="store_true", help="swap in and out")
    p.add_argument("--anchor", choices=("center", "bottom"), default="center", help="box point used for counting")
    p.add_argument("--cool", type=t_int(0, 1000, "cool"), default=5, help="frames before the same id can count again")

    p = sub.add_parser("motion", parents=[com], help="motion analysis, events and heatmap (no model needed)")
    p.add_argument("--thr", type=t_flt(0.01, 100.0, "thr"), default=0.5, help="motion threshold, percent of frame")
    p.add_argument("--min-area", type=t_int(1, 10000000, "min-area"), default=400, help="smallest moving region in pixels")
    p.add_argument("--warm", type=t_int(0, 1000, "warm"), default=15, help="frames used to learn the background")
    p.add_argument("--heatmap", help="write a motion heatmap image (.png .jpg)")
    p.add_argument("--csv", help="write the per-frame motion timeline to a .csv file")
    p.add_argument("--events", help="write detected motion events to a .json file")
    return prs


def out_path(raw, exts, force):
    return chk_out(raw, exts, force) if raw else None


def prep(a):
    kind, ref = chk_src(a.src, a.allow_net)
    if kind in ("cam", "net") and not a.max_frames:
        raise VxErr("camera and network sources need --max-frames")
    stride = a.stride or (2 if a.tiny else 1)
    paths = {
        "out": out_path(a.out, IMG_EXT if kind == "img" else OUT_VID, a.force),
        "summary": out_path(a.summary, (".json",), a.force),
        "jsonl": out_path(getattr(a, "jsonl", None), (".jsonl",), a.force),
        "csv": out_path(getattr(a, "csv", None), (".csv",), a.force),
        "events": out_path(getattr(a, "events", None), (".json",), a.force),
        "heatmap": out_path(getattr(a, "heatmap", None), IMG_EXT, a.force),
    }
    src = Src(kind, ref, stride, a.max_frames)
    return src, stride, paths


def mkd(a):
    sz = a.imgsz or (320 if a.tiny else 640)
    thr = a.threads if a.threads is not None else (min(4, os.cpu_count() or 2) if a.tiny else 0)
    det = mk_det(a.weights, a.conf, a.iou, sz, a.device, a.classes, a.backend, thr)
    if not a.quiet:
        note = "" if not det.bk.static or det.bk.sh == sz else " (fixed by model)"
        eprint("model %s | backend %s | input %dx%d%s%s" % (san(a.weights, 80), det.bk.name, det.bk.sw, det.bk.sh, note, " | tiny preset" if a.tiny else ""))
        if det.bk.lic:
            eprint("model license: " + det.bk.lic)
    return det


def eprint(msg):
    print(msg, file=sys.stderr)


def prog(a, src, k):
    if a.quiet or k % 30:
        return
    tot = "/" + str(max(1, src.n // src.stride)) if src.n else ""
    sys.stderr.write("\rframes %d%s" % (k, tot))
    sys.stderr.flush()


def done(a):
    if not a.quiet:
        sys.stderr.write("\r" + " " * 30 + "\r")


def wr_json(pth, obj):
    if pth:
        with open(pth, "w") as fp:
            json.dump(obj, fp, indent=2)


def bx(box):
    return [round(float(v), 1) for v in box]


def show_sum(a, sm):
    if not a.quiet:
        print(json.dumps(sm, indent=2))


def run_dt(a, use_trk):
    src, stride, ph = prep(a)
    wtr = None
    jl = None
    cf = None
    try:
        det = mkd(a)
        trk = InfTrk(hi=max(0.5, a.conf), lo=a.conf, max_age=a.max_age, min_hits=a.min_hits, trl=max(a.trail, 1)) if use_trk else None
        if ph["out"]:
            wtr = Wtr(ph["out"], src.w, src.h, max(1.0, src.fps / stride))
        if ph["jsonl"]:
            jl = open(ph["jsonl"], "w")
        cw = None
        if ph["csv"]:
            cf = open(ph["csv"], "w", newline="")
            cw = csv.writer(cf)
            cw.writerow(["frame", "t", "id", "cls", "conf", "x1", "y1", "x2", "y2"])
        by = {}
        nfr = 0
        t0 = time.time()
        for idx, frm in src.frames():
            tm = idx / src.fps
            ds = det(frm)
            nfr += 1
            if use_trk:
                ts = trk.step(ds)
                if jl:
                    jl.write(json.dumps({"frame": idx, "t": round(tm, 3), "tracks": [{"id": t.uid, "cls": det.nm(t.cid), "conf": round(t.conf, 3), "box": bx(t.box)} for t in ts]}) + "\n")
                if cw:
                    for t in ts:
                        cw.writerow([idx, round(tm, 3), t.uid, csv_safe(det.nm(t.cid)), round(t.conf, 3)] + bx(t.box))
                if wtr:
                    img = frm.copy()
                    drw.trks(img, ts, det.nm, a.trail)
                    drw.hud(img, ["tracks %d | ids %d" % (len(ts), len(trk.seen))])
                    wtr.put(img)
            else:
                for d in ds:
                    by[det.nm(d.cid)] = by.get(det.nm(d.cid), 0) + 1
                if jl:
                    jl.write(json.dumps({"frame": idx, "t": round(tm, 3), "dets": [{"cls": det.nm(d.cid), "conf": round(d.conf, 3), "box": bx(d.box)} for d in ds]}) + "\n")
                if wtr:
                    img = frm.copy()
                    drw.dets(img, ds, det.nm)
                    drw.hud(img, ["detections %d" % len(ds)])
                    wtr.put(img)
            prog(a, src, nfr)
        done(a)
        el = max(time.time() - t0, 1e-9)
        if use_trk:
            for uid, cid in trk.seen.items():
                by[det.nm(cid)] = by.get(det.nm(cid), 0) + 1
        sm = {"frames": nfr, "ms_per_frame": round(el / max(nfr, 1) * 1000, 1), "unique_tracks" if use_trk else "detections": sum(by.values()), "by_class": dict(sorted(by.items()))}
        wr_json(ph["summary"], sm)
        show_sum(a, sm)
    finally:
        src.close()
        if wtr:
            wtr.close()
        if jl:
            jl.close()
        if cf:
            cf.close()
    return 0


def run_cnt(a):
    if not a.line and not a.zone:
        raise VxErr("count needs --line and/or --zone")
    src, stride, ph = prep(a)
    wtr = None
    try:
        det = mkd(a)
        trk = InfTrk(hi=max(0.5, a.conf), lo=a.conf, max_age=a.max_age, min_hits=a.min_hits, trl=max(a.trail, 1))
        cnt = InfCnt(src.w, src.h, a.line, a.zone, a.anchor, a.flip, a.cool, stride / src.fps, det.nm)
        if ph["out"]:
            wtr = Wtr(ph["out"], src.w, src.h, max(1.0, src.fps / stride))
        nfr = 0
        t0 = time.time()
        for idx, frm in src.frames():
            ds = det(frm)
            ts = trk.step(ds)
            cnt.upd(ts, idx)
            nfr += 1
            if wtr:
                img = frm.copy()
                if cnt.zp is not None:
                    drw.zone(img, cnt.zp)
                if cnt.ln is not None:
                    drw.line(img, cnt.ln)
                drw.trks(img, ts, det.nm, a.trail)
                drw.hud(img, cnt.hud())
                wtr.put(img)
            prog(a, src, nfr)
        done(a)
        sm = {"frames": nfr, "ms_per_frame": round((time.time() - t0) / max(nfr, 1) * 1000, 1)}
        sm.update(cnt.summary())
        wr_json(ph["summary"], sm)
        show_sum(a, sm)
    finally:
        src.close()
        if wtr:
            wtr.close()
    return 0


def run_mot(a):
    src, stride, ph = prep(a)
    wtr = None
    try:
        mot = InfMot(src.w, src.h, thr=a.thr / 100.0, min_area=a.min_area, warm=a.warm, mw=320 if a.tiny else 640)
        if ph["out"]:
            wtr = Wtr(ph["out"], src.w, src.h, max(1.0, src.fps / stride))
        nfr = 0
        for idx, frm in src.frames():
            boxes, ratio, act = mot.step(frm, idx, idx / src.fps)
            nfr += 1
            if wtr:
                img = frm.copy()
                drw.mot(img, boxes, act)
                drw.hud(img, ["motion %.2f%%" % (ratio * 100), "EVENT" if act else "idle"])
                wtr.put(img)
            prog(a, src, nfr)
        done(a)
        evs = mot.fin()
        sm = mot.summary(src.fps, stride)
        if ph["csv"]:
            with open(ph["csv"], "w", newline="") as fp:
                cw = csv.writer(fp)
                cw.writerow(["frame", "t", "motion_pct", "regions", "active"])
                for r in mot.rows:
                    cw.writerow([r[0], round(r[1], 3), round(r[2] * 100, 3), r[3], r[4]])
        if ph["events"]:
            wr_json(ph["events"], evs)
        if ph["heatmap"]:
            if not cv2.imwrite(ph["heatmap"], mot.heat_img()):
                raise VxErr("cannot write heatmap")
        wr_json(ph["summary"], sm)
        show_sum(a, sm)
    finally:
        src.close()
        if wtr:
            wtr.close()
    return 0


def run_info(a):
    print("visionx " + __version__)
    print("opencv " + cv2.__version__)
    if has_ort():
        import onnxruntime as ort
        print("onnxruntime %s (%s)" % (ort.__version__, ", ".join(p.replace("ExecutionProvider", "") for p in ort.get_available_providers())))
    else:
        print("onnxruntime not installed (pip install onnxruntime)")
    for nm in OFF_MDL:
        try:
            ok = os.path.isfile(mdl.path_of(nm))
        except VxErr:
            ok = False
        print("model %s: %s" % (nm, "downloaded" if ok else "not downloaded, run: visionx get " + nm))
    if a.src:
        kind, ref = chk_src(a.src, a.allow_net)
        src = Src(kind, ref, 1, 1)
        try:
            print("source %s: %dx%d, %.2f fps, %s frames" % (kind, src.w, src.h, src.fps, src.n or "unknown"))
        finally:
            src.close()
    return 0


def run_get(a):
    def lg(cur, tot):
        if not a.quiet:
            sys.stderr.write("\rdownloading %d%%" % (100 * cur // tot))
            sys.stderr.flush()
    pth = mdl.fetch(a.name, a.force, lg)
    if not a.quiet:
        sys.stderr.write("\r" + " " * 30 + "\r")
    print(pth)
    print("sha256 verified: " + mdl.vso_mdl[a.name]["sha"])
    print("license: AGPL-3.0 (Ultralytics weights), see https://ultralytics.com/license")
    return 0


def main(argv=None):
    a = bld().parse_args(argv)
    try:
        if a.cmd == "info":
            return run_info(a)
        if a.cmd == "get":
            return run_get(a)
        if a.cmd == "detect":
            return run_dt(a, False)
        if a.cmd == "track":
            return run_dt(a, True)
        if a.cmd == "count":
            return run_cnt(a)
        return run_mot(a)
    except VxErr as er:
        eprint("error: " + str(er))
        return 2
    except KeyboardInterrupt:
        return 130
    except (cv2.error, OSError) as er:
        eprint("error: " + san(er, 200))
        return 1
