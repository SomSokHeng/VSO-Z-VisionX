import cv2
import numpy as np


class InfMot:
    def __init__(self, w, h, thr=0.005, min_area=400, warm=15, min_frm=3, gap=10, var=25, hist=300, mw=640):
        self.w = w
        self.h = h
        self.sc = min(1.0, float(mw) / w)
        self.sw = max(16, int(round(w * self.sc)))
        self.sh = max(16, int(round(h * self.sc)))
        self.thr = thr
        self.min_area = min_area
        self.warm = warm
        self.min_frm = min_frm
        self.gap = gap
        self.bs = cv2.createBackgroundSubtractorMOG2(history=hist, varThreshold=var, detectShadows=False)
        self.ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.heat = np.zeros((self.sh, self.sw), dtype=np.float32)
        self.base = None
        self.n = 0
        self.rows = []
        self.evs = []
        self.cur = None

    def step(self, frm, idx, tm):
        if self.base is None:
            self.base = frm.copy()
        sm = cv2.resize(frm, (self.sw, self.sh), interpolation=cv2.INTER_AREA) if self.sc < 1.0 else frm
        sm = cv2.GaussianBlur(sm, (7, 7), 0)
        msk = self.bs.apply(sm)
        _, msk = cv2.threshold(msk, 127, 255, cv2.THRESH_BINARY)
        msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, self.ker)
        msk = cv2.dilate(msk, self.ker, iterations=2)
        self.n += 1
        boxes = []
        ratio = 0.0
        if self.n > self.warm:
            cts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            big = [c for c in cts if cv2.contourArea(c) >= self.min_area * self.sc * self.sc]
            cln = np.zeros_like(msk)
            if big:
                cv2.drawContours(cln, big, -1, 255, -1)
            for c in big:
                x, y, bw, bh = cv2.boundingRect(c)
                boxes.append((int(x / self.sc), int(y / self.sc), int((x + bw) / self.sc), int((y + bh) / self.sc)))
            ratio = cv2.countNonZero(cln) / float(cln.size)
            self.heat += (cln > 0)
        act = self.n > self.warm and ratio >= self.thr
        self.rows.append((idx, tm, ratio, len(boxes), 1 if act else 0))
        self.evt(idx, tm, ratio, act)
        return boxes, ratio, act

    def evt(self, idx, tm, ratio, act):
        if act:
            if self.cur is None:
                self.cur = {"start_frame": idx, "start_s": tm, "end_frame": idx, "end_s": tm, "peak": ratio, "n": 1, "gap": 0}
            else:
                c = self.cur
                c["end_frame"] = idx
                c["end_s"] = tm
                c["peak"] = max(c["peak"], ratio)
                c["n"] += 1
                c["gap"] = 0
        elif self.cur is not None:
            self.cur["gap"] += 1
            if self.cur["gap"] > self.gap:
                self.close()

    def close(self):
        c = self.cur
        self.cur = None
        if c is not None and c["n"] >= self.min_frm:
            self.evs.append({
                "start_frame": c["start_frame"],
                "start_s": round(c["start_s"], 3),
                "end_frame": c["end_frame"],
                "end_s": round(c["end_s"], 3),
                "active_frames": c["n"],
                "peak_motion_pct": round(c["peak"] * 100, 2),
            })

    def fin(self):
        self.close()
        return self.evs

    def heat_img(self):
        base = self.base if self.base is not None else np.zeros((self.h, self.w, 3), dtype=np.uint8)
        mx = float(self.heat.max())
        gry = np.zeros((self.sh, self.sw), dtype=np.uint8) if mx <= 0 else (self.heat / mx * 255).astype(np.uint8)
        gry = cv2.resize(gry, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        col = cv2.applyColorMap(gry, cv2.COLORMAP_JET)
        bg = cv2.cvtColor(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float32)
        alp = (gry.astype(np.float32) / 255.0)[..., None] * 0.7
        return (bg * (1 - alp) + col.astype(np.float32) * alp).astype(np.uint8)

    def summary(self, fps, stride):
        tot = len(self.rows)
        act = sum(r[4] for r in self.rows)
        pk = max((r[2] for r in self.rows), default=0.0)
        return {
            "frames": tot,
            "active_frames": act,
            "active_pct": round(100.0 * act / tot, 2) if tot else 0.0,
            "peak_motion_pct": round(pk * 100, 2),
            "events": len(self.evs),
            "duration_s": round(tot * stride / fps, 2),
        }
