import cv2
import numpy as np

PRUNE = 300


def crs(ax, ay, bx, by):
    return ax * by - ay * bx


class InfCnt:
    def __init__(self, w, h, line=None, zone=None, anchor="center", flip=False, cool=5, dt=1.0, nm=None):
        self.w = w
        self.h = h
        self.ln = None
        if line:
            self.ln = (line[0] * w, line[1] * h, line[2] * w, line[3] * h)
        self.zp = None
        if zone:
            self.zp = np.array([[x * w, y * h] for x, y in zone], dtype=np.float32).reshape(-1, 1, 2)
        self.anc = anchor
        self.flip = flip
        self.cool = cool
        self.dt = dt
        self.nm = nm or (lambda c: "cls%d" % c)
        self.cnt = {}
        self.ref = {}
        self.cd = {}
        self.sn = {}
        self.cls_of = {}
        self.occ = {}
        self.peak = 0
        self.ent = {}
        self.dw = {}
        self.evs = []
        self.n = 0

    def pt(self, t):
        b = t.box
        return (float((b[0] + b[2]) / 2.0), float(b[3] if self.anc == "bottom" else (b[1] + b[3]) / 2.0))

    def side(self, x, y):
        ax, ay, bx, by = self.ln
        val = crs(bx - ax, by - ay, x - ax, y - ay)
        if abs(val) < 1e-9:
            return 0
        return 1 if val > 0 else -1

    def hit(self, x0, y0, x1, y1):
        ax, ay, bx, by = self.ln
        d1 = crs(x1 - x0, y1 - y0, ax - x0, ay - y0)
        d2 = crs(x1 - x0, y1 - y0, bx - x0, by - y0)
        return d1 * d2 <= 0

    def upd(self, trks, idx):
        self.n += 1
        evs = []
        for t in trks:
            x, y = self.pt(t)
            self.cls_of[t.uid] = t.cid
            self.sn[t.uid] = self.n
            if self.ln is not None:
                s = self.side(x, y)
                if s != 0:
                    rf = self.ref.get(t.uid)
                    if rf is not None and rf[2] != s and self.hit(rf[0], rf[1], x, y):
                        if self.n - self.cd.get(t.uid, -10 ** 9) > self.cool:
                            din = (s > 0) != self.flip
                            rec = self.cnt.setdefault(t.cid, [0, 0])
                            rec[0 if din else 1] += 1
                            self.cd[t.uid] = self.n
                            evs.append({"frame": idx, "id": t.uid, "cls": self.nm(t.cid), "dir": "in" if din else "out"})
                    self.ref[t.uid] = (x, y, s)
        if self.zp is not None:
            occ = {}
            for t in trks:
                x, y = self.pt(t)
                if cv2.pointPolygonTest(self.zp, (x, y), False) >= 0:
                    occ[t.cid] = occ.get(t.cid, 0) + 1
                    self.ent.setdefault(t.cid, set()).add(t.uid)
                    self.dw[t.uid] = self.dw.get(t.uid, 0) + 1
            self.occ = occ
            self.peak = max(self.peak, sum(occ.values()))
        if self.n % PRUNE == 0:
            old = [u for u, v in self.sn.items() if self.n - v > PRUNE]
            for u in old:
                self.ref.pop(u, None)
                self.cd.pop(u, None)
                self.sn.pop(u, None)
        self.evs.extend(evs)
        return evs

    def hud(self):
        out = []
        if self.ln is not None:
            for c, v in sorted(self.cnt.items()):
                out.append("%s in %d out %d" % (self.nm(c), v[0], v[1]))
            if not self.cnt:
                out.append("line: no crossings yet")
        if self.zp is not None:
            out.append("zone now %d peak %d" % (sum(self.occ.values()), self.peak))
        return out

    def summary(self):
        res = {}
        if self.ln is not None:
            by = {self.nm(c): {"in": v[0], "out": v[1]} for c, v in sorted(self.cnt.items())}
            res["line"] = {
                "by_class": by,
                "total_in": sum(v[0] for v in self.cnt.values()),
                "total_out": sum(v[1] for v in self.cnt.values()),
            }
        if self.zp is not None:
            dwl = {}
            for u, f in self.dw.items():
                dwl.setdefault(self.cls_of.get(u, -1), []).append(f * self.dt)
            res["zone"] = {
                "peak": self.peak,
                "unique_by_class": {self.nm(c): len(s) for c, s in sorted(self.ent.items())},
                "mean_dwell_s": {self.nm(c): round(float(np.mean(v)), 2) for c, v in sorted(dwl.items())},
            }
        return res
