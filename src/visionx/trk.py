from collections import deque

import numpy as np


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    itr = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return itr / (aa[:, None] + ab[None, :] - itr + 1e-9)


def match(iou, thr, same):
    nr, nc = iou.shape
    if nr == 0 or nc == 0:
        return [], list(range(nr)), list(range(nc))
    mt = np.where(same, iou, 0.0)
    ur = set()
    uc = set()
    prs = []
    for flt in np.argsort(-mt, axis=None):
        r, c = divmod(int(flt), nc)
        if mt[r, c] < thr:
            break
        if r in ur or c in uc:
            continue
        ur.add(r)
        uc.add(c)
        prs.append((r, c))
    return prs, [i for i in range(nr) if i not in ur], [i for i in range(nc) if i not in uc]


class Trk:
    __slots__ = ("uid", "box", "cid", "conf", "hits", "miss", "vel", "trail")

    def __init__(self, box, cid, conf, trl):
        self.uid = 0
        self.box = np.asarray(box, dtype=np.float32).copy()
        self.cid = int(cid)
        self.conf = float(conf)
        self.hits = 1
        self.miss = 0
        self.vel = np.zeros(4, dtype=np.float32)
        self.trail = deque(maxlen=max(1, trl))
        self.trail.append(self.ctr())

    def ctr(self):
        return (float((self.box[0] + self.box[2]) / 2), float((self.box[1] + self.box[3]) / 2))

    def pred(self):
        return self.box + self.vel

    def upd(self, box, conf):
        nb = np.asarray(box, dtype=np.float32)
        self.vel = 0.6 * self.vel + 0.4 * (nb - self.box)
        self.box = nb.copy()
        self.conf = float(conf)
        self.hits += 1
        self.miss = 0
        self.trail.append(self.ctr())

    def lost(self):
        self.box = self.box + self.vel
        self.vel = self.vel * 0.9
        self.miss += 1


class InfTrk:
    def __init__(self, hi=0.5, lo=0.1, thr=0.3, max_age=30, min_hits=2, trl=40):
        if not (0.0 < lo <= hi <= 1.0) or not (0.01 <= thr <= 1.0) or max_age < 0 or min_hits < 1:
            raise ValueError("bad tracker parameters")
        self.hi = hi
        self.lo = lo
        self.thr = thr
        self.max_age = max_age
        self.min_hits = min_hits
        self.trl = trl
        self.trks = []
        self.nid = 1
        self.frm = 0
        self.seen = {}

    def step(self, dets):
        self.frm += 1
        hid = [d for d in dets if d.conf >= self.hi]
        lod = [d for d in dets if self.lo <= d.conf < self.hi]
        trs = self.trks
        pb = np.array([t.pred() for t in trs], dtype=np.float32).reshape(-1, 4)
        ob = np.array([t.box for t in trs], dtype=np.float32).reshape(-1, 4)
        tc = np.array([t.cid for t in trs], dtype=np.int64)
        hb = np.array([d.box for d in hid], dtype=np.float32).reshape(-1, 4)
        hc = np.array([d.cid for d in hid], dtype=np.int64)
        prs, ut, uh = match(np.maximum(iou_mat(pb, hb), iou_mat(ob, hb)), self.thr, tc[:, None] == hc[None, :])
        for r, c in prs:
            trs[r].upd(hid[c].box, hid[c].conf)
        if ut and lod:
            lb = np.array([d.box for d in lod], dtype=np.float32).reshape(-1, 4)
            lc = np.array([d.cid for d in lod], dtype=np.int64)
            sb = pb[ut]
            so = ob[ut]
            sc = tc[ut]
            p2, u2, _ = match(np.maximum(iou_mat(sb, lb), iou_mat(so, lb)), self.thr, sc[:, None] == lc[None, :])
            for r, c in p2:
                trs[ut[r]].upd(lod[c].box, lod[c].conf)
            ut = [ut[i] for i in u2]
        for i in ut:
            trs[i].lost()
        fresh = [Trk(hid[c].box, hid[c].cid, hid[c].conf, self.trl) for c in uh]
        keep = []
        for t in trs:
            if t.miss > self.max_age:
                continue
            if t.uid == 0 and t.hits < self.min_hits and t.miss > 0:
                continue
            keep.append(t)
        self.trks = keep + fresh
        out = []
        for t in self.trks:
            if t.miss == 0 and t.hits >= self.min_hits:
                if t.uid == 0:
                    t.uid = self.nid
                    self.nid += 1
                    self.seen[t.uid] = t.cid
                out.append(t)
        return out
