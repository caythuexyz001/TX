# vision_road.py — auto strip + ROI tương đối + debug overlay
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

Label = str  # 'B','P','T'

DEFAULT_CFG = {
    # ROI TƯƠNG ĐỐI (0..1) trên dải trung tâm (strip)
    "bead_rel":   {"x_rel": 0.03, "y_rel": 0.22, "w_rel": 0.34, "h_rel": 0.56, "rows": 6, "cols": 20},
    "big_rel":    {"x_rel": 0.63, "y_rel": 0.20, "w_rel": 0.34, "h_rel": 0.58, "rows": 6, "cols": 30},
    "bigeye_rel": {"x_rel": 0.63, "y_rel": 0.80, "w_rel": 0.17, "h_rel": 0.18, "rows": 6, "cols": 30},
    "small_rel":  {"x_rel": 0.80, "y_rel": 0.80, "w_rel": 0.17, "h_rel": 0.18, "rows": 6, "cols": 30},

    # HSV thresholds (OpenCV: H 0..179)
    "hsv": {
        "banker": {"low":[0,80,60], "high":[12,255,255], "alt_low":[170,80,60], "alt_high":[179,255,255]},
        "player": {"low":[95,70,60], "high":[130,255,255]},
        "tie":    {"low":[35,60,60], "high":[85,255,255]},
    },
    "min_area_ratio": 0.08,
    "cell_margin": 0.15,

    # auto-strip
    "strip_min_height_ratio": 0.12,
    "strip_smooth": 21,
    "strip_thresh": 10
}

def _mask2(hsv, low, high):
    return cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))

def _color_score(mask) -> float:
    area = float((mask>0).sum())
    h,w = mask.shape[:2]
    return area / max(1.0, h*w)

def classify_cell(cell_bgr, cfg_hsv)->Optional[Label]:
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    # đỏ có 2 khoảng
    m1  = _mask2(hsv, cfg_hsv["banker"]["low"], cfg_hsv["banker"]["high"])
    if "alt_low" in cfg_hsv["banker"]:
        m1b = _mask2(hsv, cfg_hsv["banker"]["alt_low"], cfg_hsv["banker"]["alt_high"])
        m_red = cv2.bitwise_or(m1, m1b)
    else:
        m_red = m1
    m_blue  = _mask2(hsv, cfg_hsv["player"]["low"], cfg_hsv["player"]["high"])
    m_green = _mask2(hsv, cfg_hsv["tie"]["low"],    cfg_hsv["tie"]["high"])

    scores = {"B": _color_score(m_red), "P": _color_score(m_blue), "T": _color_score(m_green)}
    lab, val = max(scores.items(), key=lambda kv: kv[1])
    return lab if val >= 0.02 else None

def _auto_center_strip(img_bgr, cfg) -> Tuple[int,int,int,int]:
    """Tìm dải giữa có nội dung, bỏ nền đen trên/dưới. Trả về (x,y,w,h)."""
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sob = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    prof = np.mean(np.abs(sob), axis=1)
    k = max(3, int(cfg.get("strip_smooth",21)))
    if k % 2 == 0: k += 1
    prof_s = cv2.GaussianBlur(prof.reshape(-1,1), (1,k), 0).ravel()

    thresh = np.percentile(prof_s, 60) + cfg.get("strip_thresh",10)
    mask = prof_s > thresh
    if mask.sum()==0:
        y = int(H*0.33); h = int(H*0.34)
        return (0, y, W, h)
    ys = np.where(mask)[0]
    best = (0,(0,0))
    cur0 = ys[0]; y1 = ys[0]
    for i in range(1,len(ys)):
        if ys[i] == ys[i-1] + 1:
            y1 = ys[i]
        else:
            ln = y1 - cur0 + 1
            if ln > best[0]: best = (ln,(cur0,y1))
            cur0 = ys[i]; y1 = ys[i]
    ln = y1 - cur0 + 1
    if ln > best[0]: best = (ln,(cur0,y1))
    y0,y1 = best[1]
    min_h = int(H * cfg.get("strip_min_height_ratio",0.12))
    if (y1 - y0 + 1) < min_h:
        cy = (y0 + y1)//2
        y0 = max(0, cy - min_h//2)
        y1 = min(H-1, y0 + min_h)
    return (0, y0, W, y1 - y0 + 1)

def _rel_to_abs(roi_rel:Dict, strip_xywh:Tuple[int,int,int,int]):
    sx, sy, sw, sh = strip_xywh
    x = int(sx + roi_rel["x_rel"] * sw)
    y = int(sy + roi_rel["y_rel"] * sh)
    w = int(roi_rel["w_rel"] * sw)
    h = int(roi_rel["h_rel"] * sh)
    return {"x":x, "y":y, "w":w, "h":h, "rows":roi_rel["rows"], "cols":roi_rel["cols"]}

def extract_grid_labels(img_bgr, roi_abs:Dict, cfg)->List[Label]:
    x,y,w,h = roi_abs["x"], roi_abs["y"], roi_abs["w"], roi_abs["h"]
    rows, cols = roi_abs["rows"], roi_abs["cols"]
    crop = img_bgr[y:y+h, x:x+w]
    if crop.size == 0: return []
    margin = cfg.get("cell_margin",0.15)
    ch = h / rows; cw = w / cols
    out: List[Label] = []
    for r in range(rows):
        for c in range(cols):
            x0 = int(c*cw + cw*margin); y0 = int(r*ch + ch*margin)
            x1 = int((c+1)*cw - cw*margin); y1 = int((r+1)*ch - ch*margin)
            if x1<=x0 or y1<=y0: continue
            lab = classify_cell(crop[y0:y1, x0:x1], cfg["hsv"])
            if lab: out.append(lab)
    return out

# === DEBUG DRAWING ===
def _draw_rect(img, xywh, color, thick=2):
    x, y, w, h = xywh
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thick)

def render_debug_png(img_bgr, strip_xywh, roi_abs_dict):
    """Vẽ strip + 4 ROI vào ảnh và trả về bytes PNG."""
    vis = img_bgr.copy()
    _draw_rect(vis, strip_xywh, (0,255,255), 3)  # strip: vàng
    colors = {"bead": (80,220,80), "big": (40,40,255), "bigeye": (255,120,20), "small": (220,80,220)}
    for k,roi in roi_abs_dict.items():
        _draw_rect(vis, (roi["x"],roi["y"],roi["w"],roi["h"]), colors.get(k,(255,255,255)), 2)
    ok, buf = cv2.imencode(".png", vis)
    return buf.tobytes() if ok else b""

class RoadExtractor:
    def __init__(self, cfg:Dict=None):
        self.cfg = (cfg or DEFAULT_CFG).copy()

    def set_config(self, cfg:Dict):
        self.cfg.update(cfg)

    def example_config(self)->Dict:
        return DEFAULT_CFG

    def process(self, img_bgr)->Dict:
        strip = _auto_center_strip(img_bgr, self.cfg)

        bead_abs   = _rel_to_abs(self.cfg["bead_rel"],   strip)
        big_abs    = _rel_to_abs(self.cfg["big_rel"],    strip)
        bigeye_abs = _rel_to_abs(self.cfg["bigeye_rel"], strip)
        small_abs  = _rel_to_abs(self.cfg["small_rel"],  strip)

        bead   = extract_grid_labels(img_bgr, bead_abs,   self.cfg)
        big    = extract_grid_labels(img_bgr, big_abs,    self.cfg)
        bigeye = extract_grid_labels(img_bgr, bigeye_abs, self.cfg)
        small  = extract_grid_labels(img_bgr, small_abs,  self.cfg)

        return {
            "bead": bead, "big": big, "bigeye": bigeye, "small": small,
            "strip": strip, "combined": bead,
            "roi_abs": {"bead": bead_abs, "big": big_abs, "bigeye": bigeye_abs, "small": small_abs}
  }
