"""Standard image codecs for the baselines: JPEG, JPEG2000, WebP (Pillow) and BPG (if installed).
Rate = full file size in bytes (headers included)."""
import io
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image


def _to_pil(x):  # x: (3,H,W) float [0,1]
    return Image.fromarray((x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8))


def _from_pil(im):
    import torch
    return torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0


def encode_decode(x, codec, q):
    """Returns (reconstruction (3,H,W), n_bytes)."""
    im = _to_pil(x)
    if codec == "raw":
        return _from_pil(im), x.shape[1] * x.shape[2] * 3
    if codec == "bpg":
        return _bpg(im, q)
    buf = io.BytesIO()
    if codec == "jpeg":
        im.save(buf, format="JPEG", quality=int(q), subsampling=0 if q >= 90 else 2, optimize=True)
    elif codec == "jp2":
        im.save(buf, format="JPEG2000", quality_mode="rates", quality_layers=[float(q)], irreversible=True)
    elif codec == "webp":
        im.save(buf, format="WEBP", quality=int(q), method=6)
    else:
        raise ValueError(codec)
    n = buf.tell()
    buf.seek(0)
    return _from_pil(Image.open(buf)), n


def available_codecs():
    out = ["raw", "jpeg", "webp"]
    try:
        from PIL import features
        if features.check("jpg_2000"):
            out.append("jp2")
    except Exception:
        pass
    if shutil.which("bpgenc") and shutil.which("bpgdec"):
        out.append("bpg")
    return out


def _bpg(im, qp):
    with tempfile.TemporaryDirectory() as d:
        src, enc, dec = (os.path.join(d, f) for f in ("a.png", "a.bpg", "b.png"))
        im.save(src)
        subprocess.run(["bpgenc", "-q", str(int(qp)), "-o", enc, src], check=True, capture_output=True)
        subprocess.run(["bpgdec", "-o", dec, enc], check=True, capture_output=True)
        return _from_pil(Image.open(dec)), os.path.getsize(enc)
