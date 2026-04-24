"""
Trailer Surface Analyzer
========================
Detects the printable panel surfaces of a trailer and reports:
  - Surface area in pixels and square inches
  - Number of distinct print elements on the surface
  - Area of each element in pixels and square inches
  - Graphic layout type: STANDALONE (single cohesive graphic) vs.
    SET (multiple independent graphics spread across the surface)

Surface detection method
------------------------
1. GrabCut seeds foreground from a centre rectangle
2. Convex hull of the largest foreground blob fills wheel arches
   and concave gaps → gives the full panel silhouette
3. Combined color-gradient + Canny edge detects the bottom rail
   (validated by "darker below" brightness test) and trims
   undercarriage / wheel skirts below the panel
4. A vertical crease line (found via Hough on edges inside the mask)
   splits the body into two faces when two are visible.
   Crease is only accepted when both halves ≥ 20% of total and
   area ratio < 4:1, preventing false splits on single-face images.
5. Primary face = the larger half (or the whole mask if no crease).

Element detection
-----------------
Distinct print elements are found by:
  a. Color quantisation (K-means, k=8) → color regions
  b. Each significant color region that is NOT the background color
     is a candidate element
  c. MSER + adaptive-threshold contours find fine structure
  d. All candidates are deduped via NMS
  e. Elements covering < 0.3% or > 90% of the surface are discarded

Graphic layout classification
------------------------------
After element detection, classify_graphic_layout() analyses the
spatial distribution of elements to determine:

  STANDALONE — all detected elements form a single cohesive zone
    (e.g. one large logo, one continuous mural/wrap)

  SET — elements are spatially separated into two or more distinct
    independent graphic zones
    (e.g. a logo on the left + phone number on the right + tagline
     in the lower-right corner)

  Method:
    1. 1-D horizontal gap analysis: gaps > 22% of surface width
       between sorted element centres signal zone breaks
    2. 2-D DBSCAN clustering (eps=0.25 in normalised space)
    3. Element fill-ratio inside bounding box (low → sparse/set)
    4. Weighted scoring → 'set' if score ≥ 0.45, else 'standalone'
  Output: GraphicLayout(layout_type, confidence, num_clusters, reasoning)

Inch conversion
---------------
Uses the visible panel dimension that is most reliably estimated:
  - The pixel height of the panel corresponds to trailer body height
    (standard: 110 inches for a dry-van box, 162 in including frame)
  - We use 110 in for the box face height as the anchor
  - px_per_in = face_height_px / 110
  - All areas are then area_px / px_per_in²

Usage
-----
  python trailer_pipeline.py img1.jpg img2.jpg ...

Dependencies
------------
  pip install opencv-python pytesseract pillow matplotlib numpy scikit-learn
  apt install tesseract-ocr
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING
import warnings
warnings.filterwarnings('ignore')

# Standard trailer box height (inside the printable panel, excluding chassis)
PANEL_HEIGHT_IN = 110.0   # inches  — used for px→inch calibration


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Element:
    eid:      int
    box:      np.ndarray   # [x1,y1,x2,y2] in crop coords
    area_px:  float
    area_in2: float = 0.0
    pct:      float = 0.0  # % of surface

@dataclass
class Face:
    name:        str           # 'primary' | 'secondary'
    mask:        np.ndarray    # H×W uint8 full-image mask
    corners:     np.ndarray    # 4×2 bounding quad corners
    area_px:     float
    area_in2:    float   = 0.0
    px_per_in:   float   = 1.0
    height_in:   float   = 0.0
    width_in:    float   = 0.0
    crop_bgr:    Optional[np.ndarray] = None
    crop_mask:   Optional[np.ndarray] = None
    crop_offset: Optional[Tuple[int,int]] = None
    elements:    List[Element] = field(default_factory=list)
    graphic_layout: Optional['GraphicLayout'] = None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Surface mask: GrabCut → convex hull
# ══════════════════════════════════════════════════════════════════════════════

def surface_mask_grabcut_hull(img: np.ndarray) -> np.ndarray:
    """
    GrabCut with a generous centre rect, then convex hull of the
    largest blob.  The hull fills wheel arches and logo-induced gaps,
    giving the complete panel silhouette.
    """
    H, W = img.shape[:2]
    # Generous rect: 3% margin on sides, 5% top, 18% bottom (cuts chassis)
    rect = (int(W*.03), int(H*.05), int(W*.94), int(H*.77))

    bgd = np.zeros((1,65), np.float64)
    fgd = np.zeros((1,65), np.float64)
    gc  = np.zeros((H,W),  np.uint8)
    cv2.grabCut(img, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    raw = np.where((gc==2)|(gc==0), 0, 255).astype(np.uint8)

    # Morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k, iterations=3)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN,  k, iterations=2)

    # Largest blob → convex hull → filled
    cnts, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # fallback: centre rectangle
        m = np.zeros((H,W), np.uint8)
        m[int(H*.08):int(H*.85), int(W*.02):int(W*.98)] = 255
        return m

    largest = max(cnts, key=cv2.contourArea)
    hull    = cv2.convexHull(largest)
    hm = np.zeros((H,W), np.uint8)
    cv2.drawContours(hm, [hull], -1, 255, -1)
    return hm


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Bottom rail detection: cut undercarriage
# ══════════════════════════════════════════════════════════════════════════════

def cut_undercarriage(img: np.ndarray, hull_mask: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
    """
    Find the bottom edge of the painted panel using:
      - Structural Canny edge (7×7 blur)
      - LAB vertical color gradient
      - 'Darker below' brightness validation
    Everything below this line is undercarriage → zeroed.
    Returns refined mask and the detected bottom y coordinate.
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)

    # Edges only inside hull
    edges = cv2.Canny(cv2.GaussianBlur(gray,(7,7),0), 20, 70)
    edges[hull_mask == 0] = 0

    # LAB vertical (y) gradient
    lb = cv2.GaussianBlur(lab, (5,5), 0)
    gy = np.zeros((H,W,3))
    for c in range(3):
        gy[:,:,c] = cv2.Sobel(lb[:,:,c], cv2.CV_64F, 0, 1, ksize=3)
    grad_y = np.sqrt((gy**2).sum(axis=2))
    grad_y[hull_mask == 0] = 0

    # Combined signal
    vals = grad_y[hull_mask > 0]
    thr  = np.percentile(vals, 75) if len(vals) > 100 else 30
    comb = ((grad_y > thr) & (edges > 0)).astype(np.uint8) * 255
    comb = cv2.dilate(comb, cv2.getStructuringElement(cv2.MORPH_RECT,(15,1)), iterations=1)

    lines = cv2.HoughLinesP(comb, 1, np.pi/180, threshold=25,
                             minLineLength=W//10, maxLineGap=25)
    if lines is None:
        return hull_mask.copy(), None

    best = None; best_score = -1
    for seg in lines:
        x1,y1,x2,y2 = seg[0]
        angle = abs(np.degrees(np.arctan2(abs(y2-y1), abs(x2-x1)+1e-9)))
        if angle > 22: continue
        my = (y1+y2)/2
        if not (H*0.25 < my < H*0.93): continue
        xlo,xhi = max(0,min(x1,x2)), min(W-1,max(x1,x2))
        if xhi <= xlo: continue
        band = 15
        above = gray[max(0,int(my)-band):int(my), xlo:xhi].mean() if int(my)>0 else 200
        below = gray[int(my):min(H,int(my)+band), xlo:xhi].mean() if int(my)<H else 0
        if below < above - 8:                         # undercarriage is darker
            score = my * 0.65 + abs(x2-x1)/W * 0.35
            if score > best_score:
                best_score = score
                best = (my, x1, y1, x2, y2)

    if best is None:
        return hull_mask.copy(), None

    my, x1b, y1b, x2b, y2b = best
    above_mask = np.zeros((H,W), np.uint8)
    if abs(x2b-x1b) < 1:
        above_mask[:int(my), :] = 255
    else:
        s = (y2b-y1b)/(x2b-x1b+1e-9)
        for x in range(W):
            yb = int(y1b + s*(x-x1b))
            above_mask[:np.clip(yb,0,H), x] = 255

    return cv2.bitwise_and(hull_mask, above_mask), my


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Crease detection and face split
# ══════════════════════════════════════════════════════════════════════════════

def find_crease_and_split(img: np.ndarray, panel_mask: np.ndarray) -> List[dict]:
    """
    Detect a vertical crease separating two panel faces.
    Validated: both halves ≥ 20% of total, ratio < 4:1.
    Returns list of {'name', 'mask'} dicts; primary = larger.
    """
    H, W = panel_mask.shape
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(3,3),0), 35, 100)
    edges[panel_mask == 0] = 0

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                             minLineLength=H//5, maxLineGap=30)

    crease_x = None
    if lines is not None:
        candidates = []
        for seg in lines:
            x1,y1,x2,y2 = seg[0]
            if y1>y2: x1,y1,x2,y2 = x2,y2,x1,y1
            angle = abs(np.degrees(np.arctan2(abs(y2-y1), abs(x2-x1)+1e-9)))
            mx = (x1+x2)/2
            if angle > 70 and W*.08 < mx < W*.92 and abs(y2-y1) > H*.20:
                candidates.append((abs(y2-y1), int(mx)))
        if candidates:
            cx = max(candidates, key=lambda t: t[0])[1]
            tot = float((panel_mask>0).sum())
            if tot > 0:
                left  = (panel_mask[:,:cx]>0).sum()
                right = (panel_mask[:,cx:]>0).sum()
                lf, rf = left/tot, right/tot
                ratio  = max(left,right)/(min(left,right)+1e-6)
                if lf >= 0.20 and rf >= 0.20 and ratio < 4.0:
                    crease_x = cx

    if crease_x is not None:
        lm = panel_mask.copy(); lm[:,crease_x:] = 0
        rm = panel_mask.copy(); rm[:,:crease_x] = 0
        la, ra = (lm>0).sum(), (rm>0).sum()
        if la >= ra:
            return [{'name':'primary','mask':lm}, {'name':'secondary','mask':rm}]
        else:
            return [{'name':'primary','mask':rm}, {'name':'secondary','mask':lm}]

    return [{'name':'primary','mask':panel_mask.copy()}]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Real-world dimension calibration
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_dimensions(face_mask: np.ndarray) -> Tuple[float,float,float,float]:
    """
    Anchor on the VISIBLE HEIGHT of the panel (most reliable dimension
    since trailers all have the same box height: ~110 inches printable).
    Returns (px_per_in, height_in, width_in, area_in2).
    """
    ys, xs = np.where(face_mask > 0)
    if not len(ys):
        return 1.0, 0.0, 0.0, 0.0

    height_px = float(ys.max() - ys.min())
    width_px  = float(xs.max() - xs.min())
    area_px   = float((face_mask > 0).sum())

    # Use panel height as the anchor
    px_per_in = height_px / PANEL_HEIGHT_IN
    height_in = PANEL_HEIGHT_IN
    width_in  = width_px / px_per_in
    area_in2  = area_px  / (px_per_in**2)

    return px_per_in, height_in, width_in, area_in2


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Element detection on the primary surface
# ══════════════════════════════════════════════════════════════════════════════

def _nms(boxes: list, thr: float = 0.35) -> list:
    if not boxes: return []
    b = np.array(boxes, dtype=float)
    areas = (b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    order = areas.argsort()[::-1]; keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order)==1: break
        rest = order[1:]
        ix1=np.maximum(b[i,0],b[rest,0]); iy1=np.maximum(b[i,1],b[rest,1])
        ix2=np.minimum(b[i,2],b[rest,2]); iy2=np.minimum(b[i,3],b[rest,3])
        inter=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
        order=rest[inter/(areas[i]+areas[rest]-inter+1e-6)<thr]
    return [b[i].tolist() for i in keep]


def detect_elements(face: Face) -> List[Element]:
    """
    Find distinct print elements on the surface crop.

    Method:
      A. Color quantisation (k=8 K-means) → color region masks
         Each non-background color region = a candidate element
      B. MSER + adaptive-threshold contours → fine structure
      C. NMS deduplication across all candidates
      D. Filter: keep elements between 0.3% and 88% of surface area

    All coordinates are in crop space.
    """
    surf = face.crop_bgr
    smask = face.crop_mask
    if surf is None or smask is None:
        return []

    ch, cw = surf.shape[:2]
    mask_area = float((smask > 0).sum())
    if mask_area < 400:
        return []

    # Upscale for better detection
    scale   = max(1.0, 1000/max(cw, ch))
    surf_up = cv2.resize(surf,  (int(cw*scale), int(ch*scale)), interpolation=cv2.INTER_CUBIC)
    mask_up = cv2.resize(smask, (int(cw*scale), int(ch*scale)), interpolation=cv2.INTER_NEAREST)
    uh, uw  = surf_up.shape[:2]
    surf_up[mask_up == 0] = 255   # white outside mask

    all_boxes = []

    # ── A. Color quantisation elements ───────────────────────────────────────
    pixels = surf_up.reshape(-1, 3).astype(np.float32)
    mask_flat = mask_up.flatten()
    valid_px   = pixels[mask_flat > 0]
    if len(valid_px) > 200:
        k = min(8, max(3, len(valid_px)//500))
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        sample_n  = min(8000, len(valid_px))
        sample    = valid_px[np.random.choice(len(valid_px), sample_n, replace=False)]
        _, labels, centers = cv2.kmeans(sample, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

        # Background color = most common cluster
        counts    = np.array([(labels==i).sum() for i in range(k)])
        bg_idx    = np.argmax(counts)
        bg_color  = centers[bg_idx]

        # Build a full-image label map
        all_dists = np.array([np.linalg.norm(pixels - c, axis=1) for c in centers])
        label_map = all_dists.argmin(axis=0).reshape(uh, uw)
        label_map[mask_up == 0] = bg_idx   # force background outside mask

        for ci in range(k):
            if ci == bg_idx: continue
            if counts[ci] < sample_n * 0.005: continue   # too rare

            region = ((label_map == ci) & (mask_up > 0)).astype(np.uint8) * 255
            k_cls  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, k_cls, iterations=2)

            cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                a = cv2.contourArea(cnt)
                if a < uw*uh*0.002: continue
                x,y,w,h = cv2.boundingRect(cnt)
                # scale back to crop coords
                all_boxes.append([int(x/scale),int(y/scale),
                                  int((x+w)/scale),int((y+h)/scale)])

    # ── B. MSER + contour fine structure ─────────────────────────────────────
    gray_up = cv2.cvtColor(surf_up, cv2.COLOR_BGR2GRAY)
    gray_up[mask_up == 0] = 255

    min_a_up = max(150, int(uw*uh*0.004))
    max_a_up = int(uw*uh*0.40)

    mser = cv2.MSER_create(5, min_a_up, max_a_up, 0.25)
    for reg in mser.detectRegions(gray_up)[0]:
        rx,ry,rw,rh = cv2.boundingRect(reg.reshape(-1,1,2))
        if rw > 15 and rh > 15:
            all_boxes.append([int(rx/scale),int(ry/scale),
                              int((rx+rw)/scale),int((ry+rh)/scale)])

    thresh = cv2.adaptiveThreshold(gray_up, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 4)
    thresh[mask_up == 0] = 0
    for cnt in cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        a = cv2.contourArea(cnt)
        if a < min_a_up or a > max_a_up: continue
        rx,ry,rw,rh = cv2.boundingRect(cnt)
        if rw > 12 and rh > 12:
            all_boxes.append([int(rx/scale),int(ry/scale),
                              int((rx+rw)/scale),int((ry+rh)/scale)])

    # ── C. NMS ───────────────────────────────────────────────────────────────
    deduped = _nms(all_boxes, thr=0.35)

    # ── D. Filter by area fraction ────────────────────────────────────────────
    elements = []
    min_frac, max_frac = 0.003, 0.88
    for eid, box in enumerate(deduped):
        bx1,by1,bx2,by2 = [int(v) for v in box]
        bx1,by1 = max(0,bx1), max(0,by1)
        bx2,by2 = min(cw,bx2), min(ch,by2)
        if bx2<=bx1 or by2<=by1: continue

        # Centre must be inside mask
        cx_m = min((bx1+bx2)//2, cw-1)
        cy_m = min((by1+by2)//2, ch-1)
        if smask[cy_m, cx_m] == 0: continue

        # Clipped area (intersection with mask)
        roi = smask[by1:by2, bx1:bx2]
        clipped = float((roi > 0).sum())
        frac = clipped / mask_area

        if not (min_frac <= frac <= max_frac): continue

        ppi2  = face.px_per_in**2
        el = Element(eid=eid,
                     box=np.array([bx1,by1,bx2,by2]),
                     area_px=clipped,
                     area_in2=clipped/ppi2 if ppi2>0 else 0,
                     pct=frac*100)
        elements.append(el)

    return elements


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5b — Graphic layout classification: standalone vs. set
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GraphicLayout:
    layout_type:   str    # 'standalone' | 'set'
    confidence:    float  # 0.0 – 1.0
    num_clusters:  int    # spatial clusters found
    reasoning:     str    # human-readable explanation


def classify_graphic_layout(face: Face, orig_img: np.ndarray = None) -> GraphicLayout:
    """
    Classify the primary face graphic as 'standalone' (Single Item) or
    'set' (Set of Items).

    Runs the COMPLETE notebook pipeline on the original image:
      1. detect_side()           — aspect ratio → side / back
      2. find_trailer_roi_back() or find_trailer_roi_side()  — ROI bbox
      3. detect_graphics()       — adaptive threshold + dilation grouping
      4. classify()              — gap-based: any gap > 5% → Set of Items

    This matches exactly how the notebook was designed to work.
    Falls back to the face crop if orig_img is not supplied.
    """

    # ── Choose image to run on ────────────────────────────────────────────────
    img = orig_img if orig_img is not None else face.crop_bgr
    if img is None or img.size == 0:
        return GraphicLayout('standalone', 0.5, 0, "No image available.")

    H, W = img.shape[:2]

    # ── Step 1: detect_side (notebook logic) ─────────────────────────────────
    aspect = W / H
    side   = "side" if aspect > 2.0 else "back"

    # ── Step 2: find_trailer_roi (notebook logic) ─────────────────────────────
    def find_trailer_roi_back(img):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        light   = gray > 110
        not_sky = ~cv2.inRange(hsv, np.array([90,20,100]),  np.array([140,255,255])).astype(bool)
        not_veg = ~cv2.inRange(hsv, np.array([30,30,30]),   np.array([90,255,200])).astype(bool)
        mask = (light & not_sky & not_veg).astype(np.uint8) * 255
        mask[:, :int(w*0.3)] = 0
        mask[int(h*0.75):, :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((10,10), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest)
            if cw > w*0.1 and ch > h*0.15:
                return (int(x), int(y), int(cw), int(ch))
        return (int(w*0.3), 0, int(w*0.45), int(h*0.65))

    def find_trailer_roi_side(img):
        h, w = img.shape[:2]
        band    = img[int(h*0.15):int(h*0.5), int(w*0.05):int(w*0.6)]
        pixels  = band.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 4, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        u, counts = np.unique(labels, return_counts=True)
        dominant  = centers[u[np.argmax(counts)]].astype(np.uint8)
        lo = np.clip(dominant.astype(int) - 25, 0, 255).astype(np.uint8)
        hi = np.clip(dominant.astype(int) + 25, 0, 255).astype(np.uint8)
        mask = cv2.inRange(img, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest)
            if cw > w*0.2 and ch > h*0.15:
                return (int(x), int(y), int(cw), int(ch))
        return (int(w*0.05), int(h*0.05), int(w*0.6), int(h*0.5))

    trailer_bbox = find_trailer_roi_back(img) if side == "back" else find_trailer_roi_side(img)
    tx, ty, tw, th = trailer_bbox

    # ── Step 3: detect_graphics (notebook logic, verbatim) ────────────────────
    roi      = img[ty:ty+th, tx:tx+tw].copy()
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    mask_det = cv2.adaptiveThreshold(
        roi_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51, C=10
    )
    mask_det = cv2.morphologyEx(mask_det, cv2.MORPH_OPEN,  np.ones((2,2), np.uint8))
    mask_det = cv2.morphologyEx(mask_det, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

    if side == "back":
        mx, my = max(8, tw//10), max(8, th//10)
    else:
        mx, my = max(5, tw//20), max(5, th//20)
    mask_det[:my, :] = 0;  mask_det[-my:, :] = 0
    mask_det[:, :mx] = 0;  mask_det[:, -mx:] = 0

    # Remove conspicuity tape
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_roi = (cv2.inRange(hsv_roi, np.array([0,   80, 70]), np.array([12,  255, 255])) |
               cv2.inRange(hsv_roi, np.array([160, 80, 70]), np.array([180, 255, 255])))
    for cnt in cv2.findContours(red_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, cw2, ch2 = cv2.boundingRect(cnt)
        if cw2 / (ch2 + 1e-6) > 2 and ch2 < 20:
            mask_det[y:y+ch2, x:x+cw2] = 0

    # Remove structural lines
    for c in cv2.findContours(mask_det.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, cw2, ch2 = cv2.boundingRect(c)
        mn, mx2 = min(cw2, ch2), max(cw2, ch2)
        if mn < 4 and mx2 / (mn + 1e-6) > 10:
            mask_det[y:y+ch2, x:x+cw2] = 0

    # Group nearby pixels into elements
    group_k = np.ones((8, 10), np.uint8) if side == "back" else np.ones((15, 25), np.uint8)
    grouped = cv2.dilate(mask_det, group_k)
    cnts, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = tw * th * 0.003
    elements = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, ew, eh = cv2.boundingRect(c)
        filled     = int(np.sum(mask_det[y:y+eh, x:x+ew] > 0))
        fill_ratio = filled / (ew * eh + 1e-6)
        if filled < min_area * 0.3 or fill_ratio < 0.05:
            continue
        elements.append({'bbox': (int(tx+x), int(ty+y), int(ew), int(eh))})

    # Remove false merges (containment check — notebook's _remove_contained)
    to_remove = set()
    for i, ei in enumerate(elements):
        xi, yi, wi_, hi_ = ei['bbox']
        for j, ej in enumerate(elements):
            if i == j: continue
            xj, yj, wj_, hj_ = ej['bbox']
            if xi<=xj and yi<=yj and xi+wi_>=xj+wj_ and yi+hi_>=yj+hj_:
                if wj_*hj_ > wi_*hi_*0.08:
                    to_remove.add(i)
    elements = [e for k, e in enumerate(elements) if k not in to_remove]

    n = len(elements)

    # ── Step 4: classify (notebook logic, verbatim) ───────────────────────────
    if n <= 1:
        return GraphicLayout(
            layout_type  = 'standalone',
            confidence   = 1.0,
            num_clusters = n,
            reasoning    = (
                f"{side} view | ROI={tw}×{th}px | "
                f"detect_graphics found {n} element(s) → Single Item (standalone)"
            ),
        )

    layout_type = 'standalone'
    triggering  = None
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, wi_, hi_ = elements[i]['bbox']
            xj, yj, wj_, hj_ = elements[j]['bbox']
            hgap = max(0, max(xj - (xi+wi_), xi - (xj+wj_)))
            vgap = max(0, max(yj - (yi+hi_), yi - (yj+hj_)))
            if hgap > tw * 0.05 or vgap > th * 0.05:
                triggering  = (i+1, j+1, hgap, vgap)
                layout_type = 'set'
                break
        if triggering:
            break

    if triggering:
        ei, ej, hg, vg = triggering
        reasoning = (
            f"{side} view | ROI={tw}×{th}px | {n} elements detected; "
            f"gap #{ei}↔#{ej}: hgap={hg:.0f}px ({hg/tw*100:.1f}%), "
            f"vgap={vg:.0f}px ({vg/th*100:.1f}%) exceeds 5% → Set of Items"
        )
    else:
        reasoning = (
            f"{side} view | ROI={tw}×{th}px | {n} elements; "
            f"no gap exceeds 5% threshold → Single Item (standalone)"
        )

    return GraphicLayout(
        layout_type  = layout_type,
        confidence   = 1.0,
        num_clusters = n,
        reasoning    = reasoning,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Crop extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_crop(face: Face, img: np.ndarray) -> None:
    ys, xs = np.where(face.mask > 0)
    if not len(xs): return
    x1,y1,x2,y2 = int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())
    crop_bgr  = img[y1:y2, x1:x2].copy()
    crop_mask = face.mask[y1:y2, x1:x2].copy()
    result    = np.full_like(crop_bgr, 255)
    result[crop_mask > 0] = crop_bgr[crop_mask > 0]
    face.crop_bgr    = result
    face.crop_mask   = crop_mask
    face.crop_offset = (x1, y1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Visualisation
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = [
    '#FF3366','#33CCFF','#FFCC00','#66FF66','#FF9933',
    '#CC66FF','#00FFCC','#FF6699','#99FF33','#3366FF',
    '#FF6633','#33FFCC','#FFFF33','#FF33CC','#33FF66',
]
FACE_STYLE = {
    'primary':   {'edge':'#00D4FF','fill':(0.00,0.83,1.00),'bg':'#002A3A'},
    'secondary': {'edge':'#FF8C00','fill':(1.00,0.55,0.00),'bg':'#3A1A00'},
}


def make_output(img_bgr, hull_mask, faces, crease_x,
                bottom_y, title, out_path):

    H, W    = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    primary = next((f for f in faces if f.name=='primary'), faces[0])

    fig = plt.figure(figsize=(30, 12), facecolor='#060606')
    ax_full = fig.add_axes([0.00, 0.00, 0.36, 1.00])
    ax_crop = fig.add_axes([0.37, 0.07, 0.27, 0.86])
    ax_stat = fig.add_axes([0.65, 0.00, 0.35, 1.00])

    # ── Left: full annotated image ────────────────────────────────────────────
    # Darken everything outside the hull
    dark = (img_rgb * 0.15).astype(np.uint8)
    for face in faces:
        dark[face.mask > 0] = img_rgb[face.mask > 0]
    ax_full.imshow(dark); ax_full.axis('off')
    ax_full.set_title(title, color='white', fontsize=13, fontweight='bold', pad=8)

    # Face polygons
    for face in faces:
        st = FACE_STYLE[face.name]
        r,g,b = st['fill']
        corners = np.clip(face.corners.astype(int), 0, [W-1,H-1])
        ax_full.add_patch(MplPolygon(corners, closed=True,
            facecolor=(r,g,b,.12), edgecolor=st['edge'],
            lw=3 if face.name=='primary' else 1.8,
            ls='-' if face.name=='primary' else '--', zorder=5))
        for pt in corners:
            ax_full.plot(pt[0],pt[1], 'o', color=st['edge'],
                markersize=9 if face.name=='primary' else 5,
                markeredgecolor='white', markeredgewidth=1.3, zorder=6)
        cx,cy = corners.mean(axis=0)
        lbl = '★ PRIMARY' if face.name=='primary' else 'secondary'
        ax_full.text(cx,cy, lbl, color=st['edge'],
            fontsize=10, fontweight='bold', ha='center', va='center', zorder=7,
            bbox=dict(facecolor=st['bg'], alpha=.85, pad=3, edgecolor=st['edge'], lw=1))

    if crease_x:
        ax_full.axvline(crease_x, color='#FF8C00', lw=2, ls=':', alpha=.7)
    if bottom_y:
        ax_full.axhline(bottom_y, color='#FFE84D', lw=1.5, ls='--', alpha=.5)

    from matplotlib.lines import Line2D
    leg = [
        Line2D([0],[0],color=FACE_STYLE['primary']['edge'],   lw=2.5,label='Primary face'),
        Line2D([0],[0],color=FACE_STYLE['secondary']['edge'], lw=2,ls='--',label='Secondary'),
        Line2D([0],[0],color='#FFE84D',lw=1.5,ls='--',label='Bottom rail'),
        Line2D([0],[0],color='#FF8C00',lw=2,ls=':',label='Face crease'),
    ]
    ax_full.legend(handles=leg, loc='lower left', fontsize=8,
                   facecolor='#141414', labelcolor='white', edgecolor='#333')

    # ── Centre: primary surface crop with element boxes ───────────────────────
    ax_crop.set_facecolor('#111'); ax_crop.axis('off')
    if primary.crop_bgr is not None:
        ax_crop.imshow(cv2.cvtColor(primary.crop_bgr, cv2.COLOR_BGR2RGB), aspect='auto')
        for eid, el in enumerate(primary.elements):
            color = PALETTE[eid % len(PALETTE)]
            x1,y1,x2,y2 = [int(v) for v in el.box]
            ax_crop.add_patch(mpatches.Rectangle((x1,y1),x2-x1,y2-y1,
                lw=1.8, edgecolor=color, facecolor=(*[int(color[j:j+2],16)/255 for j in (1,3,5)], .08),
                zorder=5))
            ax_crop.text(x1+2, y1+2, str(eid+1), color=color,
                fontsize=6, fontweight='bold', va='top', zorder=6)
        ch_c, cw_c = primary.crop_bgr.shape[:2]
        ax_crop.set_title(
            f"Primary surface  {cw_c}×{ch_c}px\n"
            f"{primary.width_in:.0f}\" × {primary.height_in:.0f}\"  "
            f"({primary.area_in2/144:.1f} ft²)",
            color='#AAA', fontsize=9, pad=5)

    # ── Right: report panel ───────────────────────────────────────────────────
    ax_stat.set_facecolor('#0C0C0C'); ax_stat.axis('off')
    ax_stat.set_title('Print Area Report', color='white',
                       fontsize=13, fontweight='bold', pad=10)

    img_area_px = H * W
    y = 0.97

    def tx(yy, s, **kw):
        ax_stat.text(.03, yy, s, transform=ax_stat.transAxes, **kw)
    def hl(yy, c='#222', lw=.7):
        ax_stat.plot([.02,.98],[yy,yy], color=c, lw=lw, transform=ax_stat.transAxes)

    for face in faces:
        st   = FACE_STYLE[face.name]
        star = '★ PRIMARY' if face.name=='primary' else 'secondary'
        tx(y, star, color=st['edge'], fontsize=11, fontweight='bold'); y -= .038
        tx(y, f"  Area    : {face.area_px:>10,.0f} px²",
           color='#CCC', fontsize=9, fontfamily='monospace'); y -= .026
        tx(y, f"  Area    : {face.area_in2:>10,.0f} in²  ({face.area_in2/144:.1f} ft²)",
           color='#CCC', fontsize=9, fontfamily='monospace'); y -= .026
        tx(y, f"  Dims    :  {face.width_in:.0f}\" × {face.height_in:.0f}\"",
           color='#BBB', fontsize=9, fontfamily='monospace'); y -= .026
        tx(y, f"  Scale   :  1\" = {face.px_per_in:.2f} px",
           color='#888', fontsize=9, fontfamily='monospace'); y -= .026

        if face.name == 'primary' and face.elements:
            total_el_px  = sum(e.area_px  for e in face.elements)
            total_el_in2 = sum(e.area_in2 for e in face.elements)
            cov = total_el_px/face.area_px*100 if face.area_px else 0
            tx(y, f"  Elements:  {len(face.elements)}  |  coverage {cov:.1f}%  ({total_el_in2:.0f} in²)",
               color='#CCC', fontsize=9, fontfamily='monospace'); y -= .030

            # ── Graphic layout classification ─────────────────────────────────
            if face.graphic_layout is not None:
                gl = face.graphic_layout
                is_standalone = gl.layout_type == 'standalone'
                gl_color  = '#00FF99' if is_standalone else '#FF9933'
                gl_icon   = '◉' if is_standalone else '⊞'
                gl_label  = 'STANDALONE GRAPHIC' if is_standalone else 'SET OF GRAPHICS'
                tx(y, f"  {gl_icon} Layout : {gl_label}",
                   color=gl_color, fontsize=9.5, fontfamily='monospace',
                   fontweight='bold'); y -= .028
                tx(y, f"    Clusters: {gl.num_clusters}",
                   color='#888', fontsize=8, fontfamily='monospace'); y -= .022
                # Wrap reasoning text
                reasoning = gl.reasoning
                max_chars = 52
                while reasoning:
                    chunk = reasoning[:max_chars]
                    cut   = chunk.rfind(' ') if len(reasoning) > max_chars else len(reasoning)
                    line  = reasoning[:cut].strip()
                    tx(y, f"    {line}", color='#666', fontsize=7.5,
                       fontfamily='monospace'); y -= .020
                    reasoning = reasoning[cut:].strip()
                    if y < 0.18: break
                y -= .008
            # ─────────────────────────────────────────────────────────────────

            hl(y); y -= .014
            tx(y, f"  {'#':>3}  {'AREA (px²)':>12}  {'AREA (in²)':>10}  {'% surf':>7}",
               color='#3A3A3A', fontsize=8, fontfamily='monospace'); y -= .004
            hl(y,'#1C1C1C',.5); y -= .026

            for eid, el in enumerate(sorted(face.elements, key=lambda e: -e.area_px)):
                color = PALETTE[eid % len(PALETTE)]
                tx(y, f"  {eid+1:>3}  {el.area_px:>12,.0f}  {el.area_in2:>10.1f}  {el.pct:>6.1f}%",
                   color=color, fontsize=8, fontfamily='monospace'); y -= .026
                if y < .07:
                    tx(y, f"  … {len(face.elements)} elements total",
                       color='#444', fontsize=7.5); y -= .026; break

        elif face.name == 'secondary':
            tx(y, "  (secondary — not analysed)", color='#444', fontsize=8.5); y -= .026

        hl(y,'#1A1A1A'); y -= .028

    total_px  = sum(f.area_px  for f in faces)
    total_in2 = sum(f.area_in2 for f in faces)
    tx(.04, f"Image: {W}×{H} = {img_area_px:,} px²",
       color='#2E2E2E', fontsize=8, fontfamily='monospace')
    tx(.02, f"All visible surface: {total_px:,.0f} px²  ·  {total_in2:,.0f} in²  ({total_in2/144:.1f} ft²)",
       color='#383838', fontsize=8, fontfamily='monospace')

    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='#060606')
    plt.close(fig)
    print(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def analyze(image_path: str, out_path: str, verbose=True) -> List[Face]:
    name = Path(image_path).stem
    img  = cv2.imread(image_path)
    if img is None: raise FileNotFoundError(image_path)
    H, W = img.shape[:2]

    if verbose:
        print(f"\n{'═'*58}")
        print(f"  {Path(image_path).name}  ({W}×{H})")
        print(f"{'═'*58}")

    # 1. GrabCut → convex hull
    if verbose: print("  [1] GrabCut + convex hull…")
    hull_mask = surface_mask_grabcut_hull(img)
    if verbose: print(f"      hull coverage: {(hull_mask>0).mean()*100:.1f}%")

    # 2. Cut undercarriage with color+edge bottom rail
    if verbose: print("  [2] Cutting undercarriage (color+edge bottom rail)…")
    panel_mask, bottom_y = cut_undercarriage(img, hull_mask)
    removed = ((hull_mask>0).sum()-(panel_mask>0).sum()) / max((hull_mask>0).sum(),1) * 100
    bot_str = f"{bottom_y:.0f}" if bottom_y is not None else "none"
    if verbose: print(f"      bottom_y={bot_str}  "
                      f"removed {removed:.1f}% undercarriage")

    # 3. Crease → face split
    if verbose: print("  [3] Checking for face crease…")
    face_dicts = find_crease_and_split(img, panel_mask)
    crease_x   = None
    if len(face_dicts) == 2:
        # Recover crease_x for visualisation
        for cx in range(W):
            lf = (face_dicts[0]['mask'][:,:cx]>0).sum()
            if lf > 0: crease_x = cx; break
    if verbose: print(f"      {'two faces' if len(face_dicts)==2 else 'single face'}")

    # 4. Build Face objects
    faces = []
    for fd in face_dicts:
        ys, xs = np.where(fd['mask'] > 0)
        if not len(xs): continue
        area_px = float((fd['mask']>0).sum())
        corners = np.array([[xs.min(),ys.min()],[xs.max(),ys.min()],
                             [xs.max(),ys.max()],[xs.min(),ys.max()]], dtype=float)
        ppi, h_in, w_in, a_in2 = calibrate_dimensions(fd['mask'])
        face = Face(name=fd['name'], mask=fd['mask'], corners=corners,
                    area_px=area_px, area_in2=a_in2,
                    px_per_in=ppi, height_in=h_in, width_in=w_in)
        extract_crop(face, img)
        faces.append(face)
        if verbose:
            star='★' if fd['name']=='primary' else ' '
            print(f"      {star} {fd['name']:<10}  {area_px:>8,.0f}px²  "
                  f"{a_in2:>7,.0f}in²  {w_in:.0f}\"×{h_in:.0f}\"  1\"={ppi:.2f}px")

    # 5. Detect elements on primary
    if verbose: print("  [4] Detecting print elements…")
    primary = next((f for f in faces if f.name=='primary'), None)
    if primary:
        primary.elements = detect_elements(primary)
        if verbose: print(f"      {len(primary.elements)} elements found")

    # 5b. Classify graphic layout: standalone vs. set
    if primary and primary.elements:
        if verbose: print("  [5] Classifying graphic layout…")
        primary.graphic_layout = classify_graphic_layout(primary, orig_img=img)
        if verbose:
            gl = primary.graphic_layout
            print(f"      layout : {gl.layout_type.upper()}  ({gl.num_clusters} element(s))")
            print(f"      reason : {gl.reasoning}")

    # 6. Output
    label = Path(image_path).stem.replace('_',' ')
    make_output(img, hull_mask, faces, crease_x, bottom_y,
                title=label, out_path=out_path)

    # Console report
    print(f"\n  PRINT AREA REPORT")
    print(f"  {'─'*56}")
    for face in sorted(faces, key=lambda f: f.name!='primary'):
        star = '★' if face.name=='primary' else ' '
        print(f"\n  {star} {face.name.upper()}")
        print(f"    Surface area : {face.area_px:>10,.0f} px²")
        print(f"    Surface area : {face.area_in2:>10,.0f} in²  ({face.area_in2/144:.1f} ft²)")
        print(f"    Dimensions   : {face.width_in:.0f}\" wide × {face.height_in:.0f}\" tall")
        print(f"    Scale        : 1\" = {face.px_per_in:.2f} px")
        if face.name == 'primary' and face.elements:
            tel = sum(e.area_px for e in face.elements)
            cov = tel/face.area_px*100 if face.area_px else 0
            print(f"    Elements     : {len(face.elements)}   coverage {cov:.1f}%")
            if face.graphic_layout is not None:
                gl = face.graphic_layout
                icon = '◉' if gl.layout_type == 'standalone' else '⊞'
                print(f"    Graphic type : {icon} {gl.layout_type.upper()}")
                print(f"    Reasoning    : {gl.reasoning}")
            print(f"    {'─'*52}")
            print(f"    {'#':>3}  {'AREA (px²)':>12}  {'AREA (in²)':>10}  {'% surface':>9}")
            print(f"    {'─'*52}")
            for eid, el in enumerate(sorted(face.elements, key=lambda e: -e.area_px)):
                print(f"    {eid+1:>3}  {el.area_px:>12,.0f}  {el.area_in2:>10.1f}  {el.pct:>8.1f}%")
    return faces


# ══════════════════════════════════════════════════════════════════════════════
# WEB APP WRAPPER — used by Flask app.py
# ══════════════════════════════════════════════════════════════════════════════

def process_image(input_path: str, output_path: str) -> dict:
    """
    Flask-compatible wrapper around the existing trailer analysis pipeline.

    Important:
      - This does NOT change the detection, surface-area, element-area, or
        layout-classification logic.
      - It only calls analyze() and converts the returned Face / Element /
        GraphicLayout objects into dictionaries that the HTML page can render.
    """
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    faces = analyze(input_path, output_path, verbose=False)
    primary = next((f for f in faces if f.name == 'primary'), None)

    face_summaries = []
    for face in faces:
        face_summaries.append({
            "name": face.name,
            "area_px": round(float(face.area_px), 2),
            "area_in2": round(float(face.area_in2), 2),
            "area_ft2": round(float(face.area_in2) / 144.0, 2),
            "width_in": round(float(face.width_in), 2),
            "height_in": round(float(face.height_in), 2),
            "px_per_in": round(float(face.px_per_in), 4),
        })

    elements = []
    if primary is not None:
        for idx, el in enumerate(primary.elements):
            elements.append({
                "id": idx + 1,
                "eid": int(el.eid),
                "area_px": round(float(el.area_px), 2),
                "area_in2": round(float(el.area_in2), 2),
                "area_ft2": round(float(el.area_in2) / 144.0, 2),
                "pct": round(float(el.pct), 2),
                "box": [int(v) for v in el.box.tolist()] if hasattr(el.box, "tolist") else list(el.box),
            })

    layout = {
        "layout_type": "unknown",
        "display_label": "Unknown",
        "confidence": 0.0,
        "num_clusters": 0,
        "reasoning": "Layout classification was not available.",
    }
    if primary is not None and primary.graphic_layout is not None:
        gl = primary.graphic_layout
        layout_type = gl.layout_type
        layout = {
            "layout_type": layout_type,
            "display_label": "Standalone Graphic" if layout_type == "standalone" else "Set of Graphics",
            "confidence": round(float(gl.confidence), 2),
            "num_clusters": int(gl.num_clusters),
            "reasoning": gl.reasoning,
        }

    primary_area_in2 = float(primary.area_in2) if primary is not None else 0.0
    total_element_area_in2 = sum(float(el["area_in2"]) for el in elements)

    return {
        "output_path": output_path,
        "surface_area_in2": round(primary_area_in2, 2),
        "surface_area_ft2": round(primary_area_in2 / 144.0, 2),
        "element_count": len(elements),
        "total_element_area_in2": round(total_element_area_in2, 2),
        "total_element_area_ft2": round(total_element_area_in2 / 144.0, 2),
        "layout": layout,
        "faces": face_summaries,
        "elements": elements,
    }


if __name__ == '__main__':
    import sys
    paths = sys.argv[1:] if len(sys.argv)>1 else \
            sorted(str(p) for p in Path('/home/claude').glob('*.jpeg'))
    out_dir = Path('/home/claude/outputs/final')
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in paths:
        out = str(out_dir / f"{Path(img_path).stem}_out.png")
        try:
            analyze(img_path, out)
        except Exception as e:
            import traceback
            print(f"\n  ERROR — {img_path}: {e}")
            traceback.print_exc()
    print("\n✅  All done.")
