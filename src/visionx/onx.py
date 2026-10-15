import mmap
import os

from .sec import VxErr

MAX_FLD = 5000000
MAX_META = 20000
WANT = ("names", "license", "stride", "imgsz", "task")


def varint(buf, pos, end):
    val = 0
    sft = 0
    while True:
        if pos >= end or sft > 63:
            raise VxErr("malformed onnx file")
        bte = buf[pos]
        pos += 1
        val |= (bte & 0x7F) << sft
        if not bte & 0x80:
            return val, pos
        sft += 7


def scan(buf, pos, end, cnt):
    while pos < end:
        key, pos = varint(buf, pos, end)
        fnm = key >> 3
        wtp = key & 7
        if wtp == 0:
            val, pos = varint(buf, pos, end)
            yield fnm, wtp, val
        elif wtp == 2:
            ln, pos = varint(buf, pos, end)
            if pos + ln > end:
                raise VxErr("malformed onnx file")
            yield fnm, wtp, (pos, pos + ln)
            pos += ln
        elif wtp == 1:
            if pos + 8 > end:
                raise VxErr("malformed onnx file")
            pos += 8
        elif wtp == 5:
            if pos + 4 > end:
                raise VxErr("malformed onnx file")
            pos += 4
        else:
            raise VxErr("malformed onnx file")
        cnt[0] += 1
        if cnt[0] > MAX_FLD:
            raise VxErr("onnx file too complex")


def dims(buf, rng, cnt):
    for f1, w1, v1 in scan(buf, rng[0], rng[1], cnt):
        if f1 != 2 or w1 != 2:
            continue
        for f2, w2, v2 in scan(buf, v1[0], v1[1], cnt):
            if f2 != 1 or w2 != 2:
                continue
            for f3, w3, v3 in scan(buf, v2[0], v2[1], cnt):
                if f3 != 2 or w3 != 2:
                    continue
                out = []
                for f4, w4, v4 in scan(buf, v3[0], v3[1], cnt):
                    if f4 != 1 or w4 != 2:
                        continue
                    dim = None
                    for f5, w5, v5 in scan(buf, v4[0], v4[1], cnt):
                        if f5 == 1 and w5 == 0 and 0 < v5 < 1 << 31:
                            dim = v5
                    out.append(dim)
                return out
    return []


def txt(buf, rng, cap):
    return bytes(buf[rng[0]:min(rng[1], rng[0] + cap)]).decode("utf-8", "replace")


def peek(pth):
    cnt = [0]
    ops = 0
    inp = []
    out = []
    meta = {}
    with open(pth, "rb") as fp:
        if os.fstat(fp.fileno()).st_size == 0:
            raise VxErr("empty onnx file")
        with mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            for fnm, wtp, val in scan(buf, 0, len(buf), cnt):
                if wtp != 2:
                    continue
                if fnm == 7:
                    for gf, gw, gv in scan(buf, val[0], val[1], cnt):
                        if gw != 2:
                            continue
                        if gf == 11 and not inp:
                            inp = dims(buf, gv, cnt)
                        elif gf == 12 and not out:
                            out = dims(buf, gv, cnt)
                elif fnm == 8:
                    dom = ""
                    ver = 0
                    for of, ow, ov in scan(buf, val[0], val[1], cnt):
                        if of == 1 and ow == 2:
                            dom = txt(buf, ov, 64)
                        elif of == 2 and ow == 0:
                            ver = ov
                    if dom in ("", "ai.onnx"):
                        ops = max(ops, ver)
                elif fnm == 14:
                    key = ""
                    vrng = None
                    for mf, mw, mv in scan(buf, val[0], val[1], cnt):
                        if mf == 1 and mw == 2:
                            key = txt(buf, mv, 64)
                        elif mf == 2 and mw == 2:
                            vrng = mv
                    if key in WANT and vrng is not None and vrng[1] - vrng[0] <= MAX_META:
                        meta[key] = txt(buf, vrng, MAX_META)
    return {"opset": ops, "inp": inp, "out": out, "meta": meta}
