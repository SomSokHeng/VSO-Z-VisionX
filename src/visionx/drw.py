import cv2
import numpy as np

vso_pal = [
    (233, 180, 86), (115, 158, 0), (66, 228, 240), (178, 114, 0), (0, 94, 213), (167, 121, 204),
    (0, 159, 230), (189, 103, 148), (44, 160, 44), (14, 127, 255), (207, 190, 23), (194, 119, 227),
]

FNT = cv2.FONT_HERSHEY_SIMPLEX


def col(idx):
    return vso_pal[int(idx) % len(vso_pal)]


def fsz(img):
    return max(0.45, min(1.2, img.shape[1] / 1500.0))


def lab(img, x, y, txt, clr):
    fs = fsz(img)
    th = max(1, int(fs * 2))
    (tw, tl), bl = cv2.getTextSize(txt, FNT, fs, th)
    y0 = max(tl + bl + 2, y)
    x0 = max(0, min(x, img.shape[1] - tw - 4))
    cv2.rectangle(img, (x0, y0 - tl - bl - 2), (x0 + tw + 4, y0), clr, -1)
    lum = 0.114 * clr[0] + 0.587 * clr[1] + 0.299 * clr[2]
    cv2.putText(img, txt, (x0 + 2, y0 - bl - 1), FNT, fs, (0, 0, 0) if lum > 140 else (255, 255, 255), th, cv2.LINE_AA)


def dets(img, ds, nm):
    for d in ds:
        x1, y1, x2, y2 = [int(v) for v in d.box]
        clr = col(d.cid)
        cv2.rectangle(img, (x1, y1), (x2, y2), clr, max(2, int(fsz(img) * 3)))
        lab(img, x1, y1, "%s %.2f" % (nm(d.cid), d.conf), clr)


def trks(img, ts, nm, trl):
    for t in ts:
        x1, y1, x2, y2 = [int(v) for v in t.box]
        clr = col(t.uid)
        cv2.rectangle(img, (x1, y1), (x2, y2), clr, max(2, int(fsz(img) * 3)))
        lab(img, x1, y1, "#%d %s" % (t.uid, nm(t.cid)), clr)
        pts = list(t.trail)[-trl:] if trl else []
        if len(pts) > 1:
            arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [arr], False, clr, 2, cv2.LINE_AA)


def line(img, ln):
    p1 = (int(ln[0]), int(ln[1]))
    p2 = (int(ln[2]), int(ln[3]))
    cv2.line(img, p1, p2, (0, 255, 255), max(2, int(fsz(img) * 3)), cv2.LINE_AA)
    cv2.circle(img, p1, 6, (0, 255, 0), -1)
    cv2.circle(img, p2, 6, (0, 0, 255), -1)


def zone(img, zp):
    ovl = img.copy()
    pts = zp.astype(np.int32)
    cv2.fillPoly(ovl, [pts], (255, 160, 0))
    cv2.addWeighted(ovl, 0.2, img, 0.8, 0, img)
    cv2.polylines(img, [pts], True, (255, 160, 0), 2, cv2.LINE_AA)


def hud(img, lines):
    if not lines:
        return
    fs = fsz(img)
    th = max(1, int(fs * 2))
    sz = [cv2.getTextSize(s, FNT, fs, th) for s in lines]
    lh = max(s[0][1] + s[1] for s in sz) + 8
    pw = max(s[0][0] for s in sz) + 16
    ph = lh * len(lines) + 8
    ovl = img.copy()
    cv2.rectangle(ovl, (8, 8), (8 + pw, 8 + ph), (0, 0, 0), -1)
    cv2.addWeighted(ovl, 0.55, img, 0.45, 0, img)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (16, 8 + lh * (i + 1)), FNT, fs, (255, 255, 255), th, cv2.LINE_AA)


def mot(img, boxes, act):
    clr = (0, 0, 255) if act else (0, 200, 0)
    for b in boxes:
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), clr, 2)
