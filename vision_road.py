# vision_road.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2

# -------------------- CONFIG MẶC ĐỊNH --------------------
# Tất cả toạ độ ROI là tỉ lệ theo chiều rộng/chiều cao ảnh gốc.
DEFAULT_CFG: Dict[str, Any] = {
    "resize_width": 1400,                # chuẩn hoá bề ngang
    "roi": {
        "bead":   [0.02, 0.60, 0.25, 0.20],   # x,y,w,h  (tỉ lệ)
        "big":    [0.73, 0.60, 0.25, 0.20],
        "bigeye": [0.73, 0.82, 0.25, 0.13],
        "small":  [0.73, 0.95, 0.25, 0.12],
    },
    # HSV cho đỏ & xanh (tuỳ giao diện, có thể cần chỉnh)
    "hsv_red1":  [0, 120, 80],   "hsv_red2":  [10, 255, 255],
    "hsv_red3":  [170, 120, 80], "hsv_red4": [180, 255, 255],
    "hsv_blue1": [95, 120, 80],  "hsv_blue2": [125, 255, 255],
    # Cấu hình lưới/ô (ước lượng phổ thông, cần tinh chỉnh nếu UI khác)
    "cell": {
        "bead":   {"cols": 30, "rows": 6,  "min_area": 35},
        "big":    {"cols": 40, "rows": 6,  "min_area": 35},
        "bigeye": {"cols": 40, "rows": 6,  "min_area": 30},
        "small":  {"cols": 40, "rows": 6,  "min_area": 25},
    }
}

@dataclass
class RoadOut:
    bead:   List[str]
    big:    List[str]
    bigeye: List[str]
    small:  List[str]
    combined: List[str]
    roi_abs: Dict[str, Tuple[int,int,int,int]]
    strip: Dict[str, List[Tuple[int,int,str]]]

def _resize_keep_w(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width: return img
    r = width / float(w)
    return cv2.resize(img, (width, int(h * r)), interpolation=cv2.INTER_AREA)

def _rel2abs(xywh_rel, w, h):
    x,y,ww,hh = xywh_rel
    X = max(0, int(x * w)); Y = max(0, int(y * h))
    W = max(1, int(ww * w)); H = max(1, int(hh * h))
    return X, Y, W, H

def _mask_red_blue(hsv, cfg):
    r1 = cv2.inRange(hsv, np.array(cfg["hsv_red1"]),  np.array(cfg["hsv_red2"]))
    r2 = cv2.inRange(hsv, np.array(cfg["hsv_red3"]),  np.array(cfg["hsv_red4"]))
    red = cv2.bitwise_or(r1, r2)
    blue = cv2.inRange(hsv, np.array(cfg["hsv_blue1"]), np.array(cfg["hsv_blue2"]))
    return red, blue

def _grid_detect(roi, cfg_cell, cfg):
    h, w = roi.shape[:2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red, blue = _mask_red_blue(hsv, cfg)
    cols, rows = cfg_cell["cols"], cfg_cell["rows"]
    cw, ch = w / cols, h / rows
    min_area = cfg_cell["min_area"]

    points = []
    seq = []

    for c in range(cols):
        for r in range(rows):
            x0 = int(c * cw); y0 = int(r * ch)
            x1 = int(min(w, (c+1)*cw)); y1 = int(min(h, (r+1)*ch))
            cell_r = red[y0:y1, x0:x1]
            cell_b = blue[y0:y1, x0:x1]
            ar = int(cv2.countNonZero(cell_r))
            ab = int(cv2.countNonZero(cell_b))
            lab = None
            if ar > min_area or ab > min_area:
                lab = "B" if ar < ab else "P"  # xanh -> B (banker), đỏ -> P (player) tùy giao diện
                # Ngược nếu màu map khác, đổi dòng trên thành lab = "B" if ar > ab else "P"
                points.append((c, r, lab))
                seq.append(lab)

    # tie là khó từ road phụ – ở đây bỏ qua, P/B là chính
    return seq, points

class RoadExtractor:
    def __init__(self, cfg: Dict[str, Any]=None) -> None:
        self.cfg = cfg.copy() if cfg else DEFAULT_CFG.copy()

    def set_config(self, cfg: Dict[str, Any]) -> None:
        self.cfg = DEFAULT_CFG.copy()
        self.cfg.update(cfg or {})

    def process(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        img = _resize_keep_w(img_bgr, self.cfg["resize_width"])
        H, W = img.shape[:2]

        roi_abs = {}
        strips = {}
        out = {}

        for key in ("bead","big","bigeye","small"):
            x,y,w,h = _rel2abs(self.cfg["roi"][key], W, H)
            roi_abs[key] = (x,y,w,h)
            roi = img[y:y+h, x:x+w].copy()
            seq, pts = _grid_detect(roi, self.cfg["cell"][key], self.cfg)
            out[key] = seq
            # lưu điểm tuyệt đối để debug overlay
            strips[key] = [(x+p[0], y+p[1], p[2]) for (p) in pts]  # chỉ cột/hàng tương đối (đủ dựng ảnh debug)

        # ghép chuỗi: ưu tiên bead → big → bigeye → small, bỏ trùng liền kề
        combined = []
        for key in ("bead","big","bigeye","small"):
            for v in out[key]:
                if not combined or combined[-1] != v:
                    combined.append(v)

        return {
            "bead":   out["bead"],
            "big":    out["big"],
            "bigeye": out
