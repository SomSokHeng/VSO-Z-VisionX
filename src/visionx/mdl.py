import hashlib
import os
import urllib.error
import urllib.parse
import urllib.request

from .sec import OFF_MDL, VxErr, san

vso_mdl = {
    "yolo11n": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx",
        "sha": "634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b",
        "siz": 10930182,
    },
}

OK_HST = ("github.com", "githubusercontent.com")
CHUNK = 1 << 20


def ok_host(hst):
    if not hst:
        return False
    hst = hst.lower()
    return any(hst == h or hst.endswith("." + h) for h in OK_HST)


class Rdr(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, hdrs, newurl):
        usr = urllib.parse.urlparse(newurl)
        if usr.scheme != "https" or not ok_host(usr.hostname):
            raise VxErr("redirect to untrusted host blocked")
        return super().redirect_request(req, fp, code, msg, hdrs, newurl)


def home():
    base = os.environ.get("VISIONX_HOME")
    if base:
        if not os.path.isabs(base):
            raise VxErr("VISIONX_HOME must be an absolute path")
    else:
        base = os.path.join(os.path.expanduser("~"), ".cache", "visionx")
    return os.path.join(base, "models")


def path_of(nm):
    if nm not in vso_mdl:
        raise VxErr("unknown model: " + san(nm))
    return os.path.join(home(), nm + ".onnx")


def sha_file(pth):
    dig = hashlib.sha256()
    with open(pth, "rb") as fp:
        while True:
            buf = fp.read(CHUNK)
            if not buf:
                break
            dig.update(buf)
    return dig.hexdigest()


def verify(nm, pth):
    spec = vso_mdl[nm]
    if os.path.islink(pth) or not os.path.isfile(pth):
        raise VxErr("cached model is not a regular file")
    if os.path.getsize(pth) != spec["siz"] or sha_file(pth) != spec["sha"]:
        raise VxErr("cached model failed integrity check, re-run: visionx get %s --force" % nm)


def resolve(nm):
    pth = path_of(nm)
    if not os.path.isfile(pth):
        raise VxErr("model not downloaded, run: visionx get " + nm)
    verify(nm, pth)
    return pth


def fetch(nm, force=False, log=None):
    spec = vso_mdl.get(nm)
    if spec is None:
        raise VxErr("unknown model: " + san(nm))
    dst = path_of(nm)
    if os.path.isfile(dst) and not force:
        verify(nm, dst)
        return dst
    os.makedirs(os.path.dirname(dst), mode=0o700, exist_ok=True)
    tmp = dst + ".part"
    if os.path.lexists(tmp):
        os.unlink(tmp)
    dig = hashlib.sha256()
    tot = 0
    opn = urllib.request.build_opener(Rdr)
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "visionx"})
    try:
        with opn.open(req, timeout=30) as res:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as fp:
                while True:
                    buf = res.read(CHUNK)
                    if not buf:
                        break
                    tot += len(buf)
                    if tot > spec["siz"]:
                        raise VxErr("download larger than expected")
                    dig.update(buf)
                    fp.write(buf)
                    if log:
                        log(tot, spec["siz"])
    except (urllib.error.URLError, OSError) as er:
        if os.path.lexists(tmp):
            os.unlink(tmp)
        raise VxErr("download failed: " + san(getattr(er, "reason", er)))
    except VxErr:
        if os.path.lexists(tmp):
            os.unlink(tmp)
        raise
    if tot != spec["siz"] or dig.hexdigest() != spec["sha"]:
        os.unlink(tmp)
        raise VxErr("download failed integrity check")
    os.replace(tmp, dst)
    return dst


assert tuple(vso_mdl) == OFF_MDL
