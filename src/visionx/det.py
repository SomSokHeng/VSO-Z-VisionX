import ast

import cv2
import numpy as np

import importlib.util

from . import mdl, onx
from .sec import VxErr, chk_wts, san

COCO = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

TOP_N = 1000


class Det:
    __slots__ = ("box", "conf", "cid")

    def __init__(self, box, conf, cid):
        self.box = np.asarray(box, dtype=np.float32)
        self.conf = float(conf)
        self.cid = int(cid)


def nms(bx, sc, thr):
    odr = np.argsort(-sc)
    keep = []
    area = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
    while len(odr):
        i = odr[0]
        keep.append(int(i))
        if len(odr) == 1:
            break
        rst = odr[1:]
        xx1 = np.maximum(bx[i, 0], bx[rst, 0])
        yy1 = np.maximum(bx[i, 1], bx[rst, 1])
        xx2 = np.minimum(bx[i, 2], bx[rst, 2])
        yy2 = np.minimum(bx[i, 3], bx[rst, 3])
        itr = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = itr / (area[i] + area[rst] - itr + 1e-9)
        odr = rst[iou <= thr]
    return keep


def letterbox(img, sh, sw):
    hh, ww = img.shape[:2]
    rat = min(sh / hh, sw / ww)
    nh = max(1, int(round(hh * rat)))
    nw = max(1, int(round(ww * rat)))
    rsz = img if (nh, nw) == (hh, ww) else cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (sh - nh) // 2
    left = (sw - nw) // 2
    can = np.full((sh, sw, 3), 114, dtype=np.uint8)
    can[top:top + nh, left:left + nw] = rsz
    return can, rat, left, top


def dec_yolo(out, rat, left, top, ww, hh, conf, iou, lay="fn"):
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise VxErr("batched model output not supported")
        arr = arr[0]
    if arr.ndim != 2:
        raise VxErr("unexpected model output shape")
    if lay == "fn":
        arr = arr.T
    if arr.shape[1] < 5 or arr.shape[0] == 0:
        raise VxErr("unexpected model output shape")
    bx = arr[:, :4]
    sc = arr[:, 4:]
    cid = sc.argmax(axis=1)
    cf = sc[np.arange(len(sc)), cid]
    ok = (cf >= conf) & np.isfinite(cf) & np.isfinite(bx).all(axis=1)
    bx = bx[ok]
    cf = cf[ok]
    cid = cid[ok]
    if len(cf) == 0:
        return []
    if len(cf) > TOP_N:
        top_i = np.argsort(-cf)[:TOP_N]
        bx = bx[top_i]
        cf = cf[top_i]
        cid = cid[top_i]
    x1 = np.clip((bx[:, 0] - bx[:, 2] / 2 - left) / rat, 0, ww)
    y1 = np.clip((bx[:, 1] - bx[:, 3] / 2 - top) / rat, 0, hh)
    x2 = np.clip((bx[:, 0] + bx[:, 2] / 2 - left) / rat, 0, ww)
    y2 = np.clip((bx[:, 1] + bx[:, 3] / 2 - top) / rat, 0, hh)
    xyxy = np.stack([x1, y1, x2, y2], axis=1)
    keep_sz = ((x2 - x1) >= 1) & ((y2 - y1) >= 1)
    xyxy = xyxy[keep_sz]
    cf = cf[keep_sz]
    cid = cid[keep_sz]
    if len(cf) == 0:
        return []
    off = cid[:, None].astype(np.float32) * (max(ww, hh) + 1)
    kp = nms(xyxy + off, cf, iou)
    return [Det(xyxy[i], cf[i], cid[i]) for i in kp]


def lay_of(osh):
    if len(osh) != 3:
        raise VxErr("model output must be 3-dimensional (YOLOv8/YOLO11 style)")
    one = osh[1]
    two = osh[2]
    if one is not None and two is None:
        return "fn", one
    if two is not None and one is None:
        return "nf", two
    if one is not None and two is not None:
        return ("fn", one) if one < two else ("nf", two)
    raise VxErr("cannot determine model output layout")


def parse_names(raw, nc):
    if raw and len(raw) < 20000:
        try:
            dic = ast.literal_eval(raw)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            dic = None
        if isinstance(dic, dict):
            if len(dic) != nc:
                raise VxErr("model output does not match a YOLOv8/YOLO11 detection head")
            return {int(k): san(v, 40) for k, v in dic.items()}
    if nc == 80:
        return dict(enumerate(COCO))
    return {i: "cls%d" % i for i in range(nc)}


class Yolo:
    def setup(self, info, conf, iou, sz):
        if len(info["inp"]) != 4:
            raise VxErr("model input must be NCHW")
        self.sh = info["inp"][2] or sz
        self.sw = info["inp"][3] or sz
        self.static = info["inp"][2] is not None and info["inp"][3] is not None
        self.lay, nf = lay_of(info["out"])
        if nf < 5:
            raise VxErr("unexpected model output shape")
        self.names = parse_names(info["meta"].get("names"), nf - 4)
        self.lic = san(info["meta"].get("license", ""), 120)
        self.conf = conf
        self.iou = iou

    def infer(self, frm):
        hh, ww = frm.shape[:2]
        can, rat, left, top = letterbox(frm, self.sh, self.sw)
        blob = cv2.dnn.blobFromImage(can, 1 / 255.0, (self.sw, self.sh), swapRB=True)
        return dec_yolo(self.run(blob), rat, left, top, ww, hh, self.conf, self.iou, self.lay)


class OnnxYolo(Yolo):
    name = "ort"

    def __init__(self, pth, info, conf=0.25, iou=0.45, sz=640, dev="auto", thr=0):
        import onnxruntime as ort
        av = ort.get_available_providers()
        prv = ["CPUExecutionProvider"]
        if dev in ("auto", "cuda") and "CUDAExecutionProvider" in av:
            prv = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif dev == "cuda":
            raise VxErr("cuda provider not available in this onnxruntime build")
        opt = ort.SessionOptions()
        opt.log_severity_level = 3
        if thr > 0:
            opt.intra_op_num_threads = thr
            opt.inter_op_num_threads = 1
        try:
            self.ses = ort.InferenceSession(pth, sess_options=opt, providers=prv)
        except Exception as er:
            raise VxErr("cannot load model: " + san(er, 200))
        ins = self.ses.get_inputs()
        if len(ins) != 1:
            raise VxErr("model must have exactly one input")
        self.inm = ins[0].name
        self.setup(info, conf, iou, sz)

    def run(self, blob):
        return self.ses.run(None, {self.inm: blob})[0]


class CvYolo(Yolo):
    name = "cv"

    def __init__(self, pth, info, conf=0.25, iou=0.45, sz=640, dev="auto", thr=0):
        if len(info["inp"]) != 4 or any(d is None for d in info["inp"]):
            raise VxErr("opencv backend needs a fixed-size model; export one with: yolo export model=yolov8n.pt format=onnx imgsz=320 opset=12 dynamic=False")
        if info["opset"] > 17:
            raise VxErr("opencv backend supports onnx opset 17 or lower (this model is opset %d); install onnxruntime or re-export with a lower opset" % info["opset"])
        if thr > 0:
            cv2.setNumThreads(thr)
        try:
            self.net = cv2.dnn.readNetFromONNX(pth)
        except cv2.error as er:
            raise VxErr("opencv cannot load this model: " + san(er, 200))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.setup(info, conf, iou, sz)

    def run(self, blob):
        self.net.setInput(blob)
        return self.net.forward()


class Detr:
    def __init__(self, bk, cls_ok=None):
        self.bk = bk
        self.names = bk.names
        self.cls_ok = cls_ok

    def __call__(self, frm):
        out = self.bk.infer(frm)
        if self.cls_ok is not None:
            out = [d for d in out if d.cid in self.cls_ok]
        return out

    def nm(self, cid):
        return san(self.names.get(cid, "cls%d" % cid), 40)


def res_cls(names, want):
    low = {v.lower(): k for k, v in names.items()}
    ids = set()
    for w in want:
        if w not in low:
            raise VxErr("unknown class: %s (model has %d classes)" % (san(w), len(names)))
        ids.add(low[w])
    return ids


def has_ort():
    return importlib.util.find_spec("onnxruntime") is not None


def mk_det(wts="yolo11n", conf=0.25, iou=0.45, sz=640, dev="auto", cls=None, backend="auto", thr=0):
    kind, ref = chk_wts(wts)
    pth = mdl.resolve(ref) if kind == "off" else ref
    info = onx.peek(pth)
    use = backend
    if use == "auto":
        use = "ort" if has_ort() else "cv"
    if use == "ort":
        if not has_ort():
            raise VxErr("onnxruntime is not installed: pip install onnxruntime (or use --backend cv with a fixed-size opset<=17 model)")
        bk = OnnxYolo(pth, info, conf, iou, sz, dev, thr)
    else:
        bk = CvYolo(pth, info, conf, iou, sz, dev, thr)
    return Detr(bk, res_cls(bk.names, cls) if cls else None)
