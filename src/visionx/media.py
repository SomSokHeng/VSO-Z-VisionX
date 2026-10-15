import os

import cv2

from .sec import IMG_EXT, VxErr, vso_lim


class Src:
    def __init__(self, kind, ref, stride=1, max_frm=0):
        self.kind = kind
        self.ref = ref
        self.stride = stride
        self.max_frm = max_frm
        self.cap = None
        self.first = None
        if kind == "img":
            img = cv2.imread(ref, cv2.IMREAD_COLOR)
            if img is None:
                raise VxErr("cannot read image")
            self.first = img
            self.fps = 1.0
            self.n = 1
        else:
            if kind == "net":
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;http,https,tcp,tls,crypto,rtsp,rtp,udp"
            self.cap = cv2.VideoCapture(ref)
            if not self.cap.isOpened():
                raise VxErr("cannot open source")
            ok, frm = self.cap.read()
            if not ok or frm is None:
                raise VxErr("source has no readable frames")
            self.first = frm
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if 1.0 <= fps <= 240.0 else 30.0
            num = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.n = num if 0 < num < vso_lim["frm"] else 0
        hh, ww = self.first.shape[:2]
        if ww > vso_lim["dim"] or hh > vso_lim["dim"] or ww < 16 or hh < 16:
            raise VxErr("frame size out of range")
        self.w = ww
        self.h = hh

    def frames(self):
        idx = 0
        done = 0
        frm = self.first
        self.first = None
        while frm is not None:
            yield idx, frm
            done += 1
            if self.max_frm and done >= self.max_frm:
                break
            if self.cap is None:
                break
            frm = None
            while True:
                if idx + 1 >= vso_lim["frm"]:
                    return
                if not self.cap.grab():
                    return
                idx += 1
                if idx % self.stride:
                    continue
                ok, frm = self.cap.retrieve()
                if not ok or frm is None:
                    return
                if frm.shape[:2] != (self.h, self.w):
                    frm = cv2.resize(frm, (self.w, self.h))
                break

    def close(self):
        if self.cap is not None:
            self.cap.release()


class Wtr:
    def __init__(self, pth, w, h, fps):
        self.pth = pth
        ext = os.path.splitext(pth)[1].lower()
        self.img = ext in IMG_EXT
        self.vw = None
        if not self.img:
            cod = "XVID" if ext == ".avi" else "mp4v"
            self.vw = cv2.VideoWriter(pth, cv2.VideoWriter_fourcc(*cod), fps, (w, h))
            if not self.vw.isOpened():
                raise VxErr("cannot open output video")

    def put(self, frm):
        if self.img:
            if not cv2.imwrite(self.pth, frm):
                raise VxErr("cannot write output image")
        else:
            self.vw.write(frm)

    def close(self):
        if self.vw is not None:
            self.vw.release()
