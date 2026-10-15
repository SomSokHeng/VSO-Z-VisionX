import os
import random
import struct
import tempfile
import unittest

import numpy as np

from visionx import det, mdl, onx, sec
from visionx.cnt import InfCnt
from visionx.mot import InfMot
from visionx.trk import InfTrk, iou_mat


def vi(num):
    out = bytearray()
    while True:
        bte = num & 0x7F
        num >>= 7
        if num:
            out.append(bte | 0x80)
        else:
            out.append(bte)
            return bytes(out)


def fld(fnm, wtp, val):
    key = vi((fnm << 3) | wtp)
    if wtp == 0:
        return key + vi(val)
    return key + vi(len(val)) + val


def shp(dm):
    dd = b"".join(fld(1, 2, fld(1, 0, d) if d else fld(2, 2, b"x")) for d in dm)
    return fld(2, 2, fld(1, 2, fld(2, 2, dd)))


def mk_onx(opset, inp, out, nms="{0: 'a', 1: 'b'}"):
    gr = fld(11, 2, shp(inp)) + fld(12, 2, shp(out))
    ops = fld(8, 2, fld(2, 0, opset))
    mt = fld(14, 2, fld(1, 2, b"names") + fld(2, 2, nms.encode()))
    return fld(1, 0, 10) + fld(7, 2, gr) + ops + mt


class Fk:
    def __init__(self, box, conf=0.9, cid=0):
        self.box = np.array(box, dtype=np.float32)
        self.conf = conf
        self.cid = cid


class TSec(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_src(self):
        for bad in ("", "x" * 600, "a\x00b", "file:///etc/passwd", "ftp://h/x.mp4"):
            with self.assertRaises(sec.VxErr):
                sec.chk_src(bad)
        with self.assertRaises(sec.VxErr):
            sec.chk_src("rtsp://h/x")
        self.assertEqual(sec.chk_src("rtsp://h/x", True)[0], "net")
        self.assertEqual(sec.chk_src("3"), ("cam", 3))
        with self.assertRaises(sec.VxErr):
            sec.chk_src("/etc/passwd")

    def test_symlink_ext(self):
        tgt = os.path.join(self.tmp, "secret.txt")
        with open(tgt, "w") as fp:
            fp.write("x")
        lnk = os.path.join(self.tmp, "v.mp4")
        os.symlink(tgt, lnk)
        with self.assertRaises(sec.VxErr):
            sec.chk_src(lnk)

    def test_out(self):
        pth = os.path.join(self.tmp, "o.mp4")
        self.assertEqual(sec.chk_out(pth, sec.OUT_VID), pth)
        with open(pth, "w") as fp:
            fp.write("x")
        with self.assertRaises(sec.VxErr):
            sec.chk_out(pth, sec.OUT_VID)
        self.assertEqual(sec.chk_out(pth, sec.OUT_VID, True), pth)
        with self.assertRaises(sec.VxErr):
            sec.chk_out(os.path.join(self.tmp, "o.exe"), sec.OUT_VID)
        with self.assertRaises(sec.VxErr):
            sec.chk_out(os.path.join(self.tmp, "nodir", "o.mp4"), sec.OUT_VID)
        lnk = os.path.join(self.tmp, "l.mp4")
        os.symlink(pth, lnk)
        with self.assertRaises(sec.VxErr):
            sec.chk_out(lnk, sec.OUT_VID, True)

    def test_wts(self):
        self.assertEqual(sec.chk_wts("yolo11n"), ("off", "yolo11n"))
        self.assertEqual(sec.chk_wts("yolo11n.onnx"), ("off", "yolo11n"))
        for bad in ("m.pt", "m.pth", "m.bin", "../x/yolo11n.pt"):
            with self.assertRaises(sec.VxErr):
                sec.chk_wts(bad)
        pth = os.path.join(self.tmp, "yolo11n.onnx")
        with open(pth, "wb") as fp:
            fp.write(b"x")
        self.assertEqual(sec.chk_wts(pth)[0], "onx")

    def test_geo(self):
        self.assertEqual(sec.p_line("0,0.5,1,0.5"), (0.0, 0.5, 1.0, 0.5))
        for bad in ("1,1,1,1", "0,0,2,1", "a,b,c,d", "0,0,1", "nan,0,1,1", "inf,0,1,1"):
            with self.assertRaises(sec.VxErr):
                sec.p_line(bad)
        self.assertEqual(len(sec.p_zone("0,0;1,0;1,1")), 3)
        for bad in ("0,0;1,1", "0,0;1,1;2,2", "0,0;0.5,0.5;1,1", "0,0;1,0"):
            with self.assertRaises(sec.VxErr):
                sec.p_zone(bad)

    def test_cls_csv(self):
        self.assertEqual(sec.p_cls("Person, car"), ["person", "car"])
        with self.assertRaises(sec.VxErr):
            sec.p_cls("person;rm")
        self.assertEqual(sec.csv_safe("=cmd|x"), "'=cmd|x")
        self.assertEqual(sec.csv_safe("@x"), "'@x")
        self.assertEqual(sec.san("a\x1b[31mb"), "a?[31mb")


class TOnx(unittest.TestCase):
    def wr(self, raw):
        fd, pth = tempfile.mkstemp()
        os.write(fd, raw)
        os.close(fd)
        self.addCleanup(os.unlink, pth)
        return pth

    def test_peek(self):
        i = onx.peek(self.wr(mk_onx(22, [None, 3, None, None], [None, 6, None])))
        self.assertEqual(i["opset"], 22)
        self.assertEqual(i["inp"], [None, 3, None, None])
        self.assertEqual(i["out"], [None, 6, None])
        self.assertIn("names", i["meta"])
        j = onx.peek(self.wr(mk_onx(17, [1, 3, 320, 320], [1, 84, 2100])))
        self.assertEqual(j["inp"], [1, 3, 320, 320])

    def test_hostile(self):
        base = mk_onx(17, [1, 3, 320, 320], [1, 84, 2100])
        rnd = random.Random(7)
        for _ in range(600):
            raw = bytearray(base)
            for _ in range(rnd.randint(1, 8)):
                raw[rnd.randrange(len(raw))] = rnd.getrandbits(8)
            if rnd.random() < 0.4:
                raw = raw[:rnd.randrange(1, len(raw))]
            try:
                onx.peek(self.wr(bytes(raw)))
            except sec.VxErr:
                pass
        for raw in (b"", b"\xff" * 64, os.urandom(2000), b"\x0a\xff\xff\xff\xff\x0f"):
            with self.assertRaises(sec.VxErr):
                onx.peek(self.wr(raw))


class TDet(unittest.TestCase):
    def test_coco(self):
        self.assertEqual(len(det.COCO), 80)

    def test_names(self):
        self.assertEqual(det.parse_names("{0: 'a', 1: 'b'}", 2), {0: "a", 1: "b"})
        self.assertEqual(det.parse_names(None, 80)[0], "person")
        self.assertEqual(det.parse_names(None, 3), {0: "cls0", 1: "cls1", 2: "cls2"})
        with self.assertRaises(sec.VxErr):
            det.parse_names("{0: 'a'}", 2)
        self.assertEqual(det.parse_names("__import__('os')", 2), {0: "cls0", 1: "cls1"})

    def test_lay(self):
        self.assertEqual(det.lay_of([1, 84, 8400]), ("fn", 84))
        self.assertEqual(det.lay_of([None, 84, None]), ("fn", 84))
        self.assertEqual(det.lay_of([1, None, 6]), ("nf", 6))
        with self.assertRaises(sec.VxErr):
            det.lay_of([1, 2])

    def test_dec(self):
        out = np.zeros((1, 6, 4), dtype=np.float32)
        out[0, :4, 0] = [100, 100, 50, 50]
        out[0, 4, 0] = 0.9
        out[0, :4, 1] = [102, 101, 50, 50]
        out[0, 4, 1] = 0.8
        out[0, :4, 2] = [300, 300, 40, 40]
        out[0, 5, 2] = 0.7
        out[0, :4, 3] = [100, 100, 50, 50]
        out[0, 5, 3] = 0.6
        res = det.dec_yolo(out, 1.0, 0, 0, 640, 640, 0.25, 0.45, "fn")
        self.assertEqual(sorted((d.cid, round(d.conf, 1)) for d in res), [(0, 0.9), (1, 0.6), (1, 0.7)])
        res2 = det.dec_yolo(np.transpose(out, (0, 2, 1)), 1.0, 0, 0, 640, 640, 0.25, 0.45, "nf")
        self.assertEqual(len(res2), 3)
        box = [d for d in res if d.cid == 0][0].box
        self.assertEqual([int(v) for v in box], [75, 75, 125, 125])

    def test_dec_hostile(self):
        out = np.full((1, 6, 50), np.nan, dtype=np.float32)
        self.assertEqual(det.dec_yolo(out, 1.0, 0, 0, 64, 64, 0.25, 0.45, "fn"), [])
        out = np.random.RandomState(1).rand(1, 6, 5000).astype(np.float32) * 1e9
        res = det.dec_yolo(out, 1.0, 0, 0, 64, 64, 0.25, 0.45, "fn")
        self.assertLessEqual(len(res), det.TOP_N)
        for d in res:
            self.assertTrue(np.isfinite(d.box).all())
            self.assertTrue((d.box >= 0).all() and (d.box <= 64).all())
        with self.assertRaises(sec.VxErr):
            det.dec_yolo(np.zeros((2, 6, 5)), 1.0, 0, 0, 64, 64, 0.25, 0.45, "fn")

    def test_letterbox(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        can, rat, left, top = det.letterbox(img, 320, 320)
        self.assertEqual(can.shape, (320, 320, 3))
        self.assertEqual((rat, left, top), (1.6, 0, 80))


class TMdl(unittest.TestCase):
    def test_host(self):
        self.assertTrue(mdl.ok_host("github.com"))
        self.assertTrue(mdl.ok_host("release-assets.githubusercontent.com"))
        for bad in ("evilgithub.com", "github.com.evil.io", "", None, "githubusercontent.com.evil"):
            self.assertFalse(mdl.ok_host(bad))

    def test_names(self):
        self.assertEqual(tuple(mdl.vso_mdl), sec.OFF_MDL)
        with self.assertRaises(sec.VxErr):
            mdl.path_of("../../etc/passwd")

    def test_verify(self):
        d = tempfile.mkdtemp()
        os.environ["VISIONX_HOME"] = d
        try:
            os.makedirs(os.path.join(d, "models"))
            pth = os.path.join(d, "models", "yolo11n.onnx")
            with open(pth, "wb") as fp:
                fp.write(b"tampered")
            with self.assertRaises(sec.VxErr):
                mdl.resolve("yolo11n")
            os.environ["VISIONX_HOME"] = "relative/path"
            with self.assertRaises(sec.VxErr):
                mdl.home()
        finally:
            del os.environ["VISIONX_HOME"]


class TTrk(unittest.TestCase):
    def test_iou(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=np.float32)
        self.assertEqual([round(float(v), 2) for v in iou_mat(a, b)[0]], [1.0, 0.33, 0.0])

    def test_ids_stable(self):
        trk = InfTrk()
        ids = []
        for i in range(30):
            ds = [Fk([10 + i * 3, 10, 50 + i * 3, 90]), Fk([300 - i * 3, 100, 340 - i * 3, 180], 0.8, 2)]
            ts = trk.step(ds)
            ids.append(sorted((t.uid, t.cid) for t in ts))
        self.assertEqual(ids[-1], [(1, 0), (2, 2)])
        self.assertEqual(len(trk.seen), 2)

    def test_occlusion_and_expiry(self):
        trk = InfTrk(max_age=5)
        for i in range(6):
            trk.step([Fk([10 + i * 4, 10, 50 + i * 4, 90])])
        for _ in range(3):
            self.assertEqual(trk.step([]), [])
        ts = trk.step([Fk([10 + 9 * 4, 10, 50 + 9 * 4, 90])])
        self.assertEqual([t.uid for t in ts], [1])
        for _ in range(8):
            trk.step([])
        self.assertEqual(trk.trks, [])

    def test_class_not_mixed(self):
        trk = InfTrk()
        trk.step([Fk([10, 10, 50, 90], 0.9, 0)])
        ts = trk.step([Fk([10, 10, 50, 90], 0.9, 1)])
        self.assertEqual(ts, [])

    def test_low_conf_rescue(self):
        trk = InfTrk()
        for i in range(4):
            trk.step([Fk([10 + i * 3, 10, 50 + i * 3, 90], 0.9)])
        ts = trk.step([Fk([22, 10, 62, 90], 0.3)])
        self.assertEqual([t.uid for t in ts], [1])


class TCnt(unittest.TestCase):
    def run_path(self, pts, hs=5, **kw):
        cnt = InfCnt(100, 100, line=(0.5, 0.0, 0.5, 1.0), **kw)
        trk = InfTrk(min_hits=1)
        for i, (x, y) in enumerate(pts):
            cnt.upd(trk.step([Fk([x - hs, y - hs, x + hs, y + hs])]), i)
        return cnt.cnt

    def test_dirs(self):
        self.assertEqual(self.run_path([(30 + i * 5, 50) for i in range(12)]), {0: [0, 1]})
        self.assertEqual(self.run_path([(70 - i * 5, 50) for i in range(12)]), {0: [1, 0]})

    def test_flip(self):
        self.assertEqual(self.run_path([(30 + i * 5, 50) for i in range(12)], flip=True), {0: [1, 0]})

    def test_no_cross_outside_segment(self):
        cnt = InfCnt(100, 100, line=(0.5, 0.0, 0.5, 0.4))
        trk = InfTrk(min_hits=1)
        for i in range(12):
            x = 30 + i * 5
            cnt.upd(trk.step([Fk([x - 2, 78, x + 2, 82])]), i)
        self.assertEqual(cnt.cnt, {})

    def test_jitter_cooldown(self):
        pts = [(40, 50), (44, 50), (48, 50), (52, 50), (48, 50), (52, 50), (48, 50), (52, 50)]
        for hs in (5, 20):
            res = self.run_path(pts, hs=hs, cool=5)
            self.assertEqual(sum(res[0]), 1)

    def test_exact_touch(self):
        self.assertEqual(self.run_path([(40, 50), (45, 50), (50, 50), (55, 50), (60, 50)]), {0: [0, 1]})

    def test_zone(self):
        cnt = InfCnt(100, 100, zone=((0.4, 0.0), (0.6, 0.0), (0.6, 1.0), (0.4, 1.0)), dt=0.5)
        trk = InfTrk(min_hits=1)
        for i in range(20):
            x = 20 + i * 3
            cnt.upd(trk.step([Fk([x - 4, 40, x + 4, 60])]), i)
        sm = cnt.summary()["zone"]
        self.assertEqual(sm["peak"], 1)
        self.assertEqual(sm["unique_by_class"], {"cls0": 1})
        self.assertGreater(sm["mean_dwell_s"]["cls0"], 0)

    def test_prune(self):
        cnt = InfCnt(100, 100, line=(0.5, 0.0, 0.5, 1.0))
        for i in range(700):
            cnt.upd([type("T", (), {"uid": i + 1, "cid": 0, "box": np.array([10, 10, 20, 20], dtype=np.float32)})()], i)
        self.assertLess(len(cnt.ref), 700)
        self.assertNotIn(1, cnt.ref)
        self.assertIn(700, cnt.ref)


class TMot(unittest.TestCase):
    def test_event(self):
        mot = InfMot(160, 120, warm=5, min_area=50)
        for i in range(60):
            frm = np.full((120, 160, 3), 60, dtype=np.uint8)
            if 20 <= i < 40:
                frm[40:80, 10 + (i - 20) * 5:50 + (i - 20) * 5] = 255
            mot.step(frm, i, i / 10.0)
        evs = mot.fin()
        self.assertEqual(len(evs), 1)
        self.assertLessEqual(abs(evs[0]["start_frame"] - 20), 1)
        self.assertGreaterEqual(evs[0]["end_frame"], 38)
        img = mot.heat_img()
        self.assertEqual(img.shape, (120, 160, 3))

    def test_static_scene(self):
        mot = InfMot(160, 120, warm=5)
        for i in range(40):
            mot.step(np.full((120, 160, 3), 90, dtype=np.uint8), i, i / 10.0)
        self.assertEqual(mot.fin(), [])
        self.assertEqual(mot.summary(10.0, 1)["active_frames"], 0)


if __name__ == "__main__":
    unittest.main()
