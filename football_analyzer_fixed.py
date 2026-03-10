

"""
FOOTBALL ANALYZER v4.3 — ELLIPSE TRACKING + FRAME BUFFER
=========================================================
• Ground-plane ellipse per player (not rectangle)
• Fading track trail showing last N foot positions
• Speed badge (km/h) on each player label
• Occlusion-robust ReID — players persist through missed frames
• FrameStore: every annotated frame buffered in RAM + written to disk MP4
• Heatmap, player card, recovery plan all stable
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json, pickle, traceback, threading, sys

# Ensure modules/ subfolder is on path for any direct imports
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / 'modules'))
sys.path.insert(0, str(_HERE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import torch

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False
    print("WARNING: ultralytics not installed")

try:
    import supervision as sv
    _HAS_SV = True
except ImportError:
    _HAS_SV = False
    print("WARNING: supervision not installed")

try:
    from sklearn.cluster import KMeans as _KMeans
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    class _KMeans:
        def __init__(self, n_clusters=2, random_state=42, n_init=10):
            self.n_clusters = n_clusters
        def fit_predict(self, X):
            np.random.seed(42)
            return np.random.randint(0, self.n_clusters, len(X))

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[FootballIQ] device={_DEVICE}")

# ── BGR colour palette ────────────────────────────────────────────────────────
TEAM_COLOURS = {0: (0,200,255), 1: (50,50,240), -1: (160,160,160)}
TRAIL_LEN = 30   # frames of trail to draw


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ─────────────────────────────────────────────────────────────────────────────
# BYTETRACK FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def _make_bytetrack(thresh=0.25, buffer=120, match=0.8, fps=30):
    if not _HAS_SV:
        raise RuntimeError("supervision not installed")
    for kwargs in [
        dict(track_activation_threshold=thresh, lost_track_buffer=buffer,
             minimum_matching_threshold=match, frame_rate=fps),
        dict(track_thresh=thresh, track_buffer=buffer,
             match_thresh=match, frame_rate=fps),
        dict(track_activation_threshold=thresh, lost_track_buffer=buffer,
             minimum_matching_threshold=match),
        {},
    ]:
        try:
            return sv.ByteTrack(**kwargs)
        except TypeError:
            continue
    raise RuntimeError("Cannot init sv.ByteTrack — check supervision version")


# ─────────────────────────────────────────────────────────────────────────────
# APPEARANCE FEATURE
# ─────────────────────────────────────────────────────────────────────────────
class AppearanceFeature:
    @staticmethod
    def extract(crop):
        if crop is None or crop.size == 0: return None
        h, w = crop.shape[:2]
        if h < 25 or w < 12: return None
        try:
            y0,y1 = max(0,int(h*.10)), max(1,int(h*.55))
            x0,x1 = max(0,int(w*.15)), max(1,int(w*.85))
            roi = crop[y0:y1, x0:x1]
            if roi.size < 400: return None
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hh  = cv2.calcHist([hsv],[0],None,[32],[0,180]).flatten()
            sh  = cv2.calcHist([hsv],[1],None,[16],[0,256]).flatten()
            hh /= hh.sum()+1e-7;  sh /= sh.sum()+1e-7
            return np.concatenate([hh,sh]).astype(np.float32)
        except Exception:
            return None

    @staticmethod
    def cosine(a, b):
        if a is None or b is None: return 0.0
        return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-7))


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER GALLERY (stable ReID)
# ─────────────────────────────────────────────────────────────────────────────
class PlayerGallery:
    EMA_ALPHA      = 0.25
    MISS_TOLERANCE = 5

    def __init__(self, max_age_frames, sim_threshold, max_dist_px):
        self.max_age    = max_age_frames
        self.sim_thresh = sim_threshold
        self.max_dist   = max_dist_px
        self._r2c:  dict = {}
        self._app:  dict = {}
        self._pos:  dict = {}
        self._seen: dict = {}
        self._miss: dict = {}
        self._lost: dict = {}
        # monotonic counter — new canonical IDs never reuse old numbers
        self._next_cid: int = 1

    def _new_cid(self) -> int:
        cid = self._next_cid
        self._next_cid += 1
        return cid

    def update(self, raw_id, foot_pos, feat, frame_idx):
        if raw_id in self._r2c:
            cid = self._r2c[raw_id]
        else:
            # Try to re-identify a recently-lost player first
            cid = self._match_lost(feat, foot_pos, frame_idx)
            if cid is None:
                cid = self._new_cid()   # fresh stable ID, never recycled
            self._r2c[raw_id] = cid
        self._pos[cid]  = list(foot_pos)
        self._seen[cid] = frame_idx
        self._miss[cid] = 0
        self._lost.pop(cid, None)
        if feat is not None:
            prev = self._app.get(cid)
            if prev is None:
                self._app[cid] = feat.copy()
            else:
                ema = (1-self.EMA_ALPHA)*prev + self.EMA_ALPHA*feat
                n   = np.linalg.norm(ema)
                self._app[cid] = ema/n if n>1e-7 else ema
        return cid

    def tick_active(self, active_raw_ids, frame_idx):
        if not self._seen: return
        active_cids = {self._r2c[r] for r in active_raw_ids if r in self._r2c}
        for cid in list(self._seen):
            if cid in active_cids:
                self._miss[cid] = 0
            else:
                self._miss[cid] = self._miss.get(cid,0)+1
                if self._miss[cid] >= self.MISS_TOLERANCE:
                    self._promote_lost(cid, frame_idx)
        stale = [r for r,c in self._r2c.items() if self._miss.get(c,0)>=self.MISS_TOLERANCE]
        for r in stale: del self._r2c[r]

    def _promote_lost(self, cid, frame_idx):
        feat = self._app.get(cid)
        if feat is not None:
            self._lost[cid] = {'feat':feat,'pos':self._pos.get(cid,[0,0]),'frame':frame_idx}

    def _match_lost(self, feat, pos, frame_idx):
        if feat is None or not self._lost: return None
        best, best_s = None, -1.0
        for cid, info in list(self._lost.items()):
            age = frame_idx - info['frame']
            if age > self.max_age: continue
            app_s = AppearanceFeature.cosine(feat, info['feat'])
            dist  = np.hypot(pos[0]-info['pos'][0], pos[1]-info['pos'][1])
            pos_s = float(np.exp(-dist/max(self.max_dist,1e-3)))
            age_w = float(np.exp(-age/max(self.max_age*.5,1)))
            score = .60*app_s + .25*pos_s + .15*age_w
            if score > best_s: best_s=score; best=cid
        return best if best_s >= self.sim_thresh else None


# ─────────────────────────────────────────────────────────────────────────────
# TEAM CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
class TeamClassifier:
    def classify(self, player_images):
        cols, pids = [], []
        for pid, crop in player_images.items():
            if crop is None or crop.size == 0: continue
            try:
                h,w = crop.shape[:2]
                roi = crop[max(0,int(h*.10)):max(1,int(h*.55)),
                           max(0,int(w*.15)):max(1,int(w*.85))]
                if roi.size<200 or roi.shape[0]<2 or roi.shape[1]<2: continue
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                cols.append(np.mean(hsv.reshape(-1,3), axis=0))
                pids.append(pid)
            except Exception: continue
        if len(cols) < 4:
            return {pid: i%2 for i,pid in enumerate(pids)}
        labels = _KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(np.array(cols))
        return {pid: int(l) for pid,l in zip(pids,labels)}


# ─────────────────────────────────────────────────────────────────────────────
# VIEW TRANSFORMER
# ─────────────────────────────────────────────────────────────────────────────
class ViewTransformer:
    PITCH_W, PITCH_H = 105.0, 68.0

    def transform(self, x, y, fw, fh):
        fw = fw or 1;  fh = fh or 1
        return x*self.PITCH_W/fw, y*self.PITCH_H/fh

    def zone(self, mx):
        if mx < self.PITCH_W*.33:   return "Defensive"
        elif mx < self.PITCH_W*.66: return "Middle"
        return "Attacking"

    def zone_pct(self, positions_meters):
        if not positions_meters:
            return {'defensive':33,'middle':34,'attacking':33}
        d=m=a=0
        for pt in positions_meters:
            mx=pt[0]
            if mx<self.PITCH_W*.33: d+=1
            elif mx<self.PITCH_W*.66: m+=1
            else: a+=1
        total=(d+m+a) or 1
        return {'defensive':round(d/total*100,1),
                'middle':round(m/total*100,1),
                'attacking':round(a/total*100,1)}


# ─────────────────────────────────────────────────────────────────────────────
# FRAME STORE — ring buffer + MP4 writer
# ─────────────────────────────────────────────────────────────────────────────
class FrameStore:
    MAX_MEM = 400   # JPEG frames kept in RAM

    def __init__(self, session_id, out_path, fps, width, height):
        self.session_id  = session_id
        self.out_path    = str(out_path)
        self.lock        = threading.Lock()
        self._buf: list  = []    # list of (frame_idx, jpeg_bytes)
        self._total      = 0
        self.done        = False
        self._writer     = None
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._writer = cv2.VideoWriter(str(out_path), fourcc, max(1.0,fps), (width,height))
            if not self._writer.isOpened():
                self._writer = None
                print(f"[FrameStore] VideoWriter failed to open")
        except Exception as e:
            print(f"[FrameStore] VideoWriter init failed: {e}")

    def add(self, frame_idx: int, bgr_frame: np.ndarray):
        if self._writer is not None:
            try: self._writer.write(bgr_frame)
            except Exception: pass
        try:
            h,w = bgr_frame.shape[:2]
            # Keep up to 1280px wide for clarity — was 800, causing blurry frames
            scale = 1280/w if w>1280 else 1.0
            disp  = cv2.resize(bgr_frame,(int(w*scale),int(h*scale)),
                               interpolation=cv2.INTER_LANCZOS4) if scale<1 else bgr_frame
            # Quality 92 — was 78, causing compression artefacts on annotations
            _, enc = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 92])
            jpg = enc.tobytes()
        except Exception:
            return
        with self.lock:
            self._buf.append((frame_idx, jpg))
            self._total += 1
            if len(self._buf) > self.MAX_MEM:
                self._buf.pop(0)

    def latest_jpeg(self):
        with self.lock:
            return self._buf[-1][1] if self._buf else None

    def frame_at(self, idx: int):
        with self.lock:
            if not self._buf: return None
            best = min(self._buf, key=lambda x: abs(x[0]-idx))
            return best[1]

    def count(self):
        with self.lock:
            return self._total

    def list_indices(self):
        with self.lock:
            return [x[0] for x in self._buf]

    def close(self):
        with self.lock:
            self.done = True
        if self._writer is not None:
            try: self._writer.release()
            except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _draw_ellipse(frame, bbox, color, thickness=2):
    """Ground-plane ellipse at foot of bounding box."""
    x1,y1,x2,y2 = map(int,bbox)
    cx  = (x1+x2)//2
    cy  = y2
    rw  = max(4,(x2-x1)//2)
    rh  = max(2,rw//5)
    cv2.ellipse(frame,(cx,cy),(rw,rh),0,-45,235,color,thickness,cv2.LINE_AA)

def _speed_color(spd):
    if spd < 7:    return (80,220,60)
    elif spd < 14: return (40,200,200)
    elif spd < 20: return (20,140,255)
    elif spd < 25: return (0,80,255)
    else:          return (0,40,200)

def _draw_label(frame, cx, y_top, pid, spd, bg_color):
    font  = cv2.FONT_HERSHEY_SIMPLEX
    text  = f"#{pid} {spd:.0f}"
    scale = 0.42
    thick = 1
    (tw,th),_ = cv2.getTextSize(text,font,scale,thick)
    pad  = 4
    rx1  = cx-tw//2-pad;  ry1 = y_top-th-pad*2-3
    rx2  = cx+tw//2+pad;  ry2 = y_top-3
    h_f, w_f = frame.shape[:2]
    ry1=max(0,ry1); rx1=max(0,rx1); ry2=min(h_f,ry2); rx2=min(w_f,rx2)
    if rx2>rx1 and ry2>ry1:
        cv2.rectangle(frame,(rx1,ry1),(rx2,ry2),bg_color,cv2.FILLED)
        cv2.rectangle(frame,(rx1,ry1),(rx2,ry2),(0,0,0),1)
    cv2.putText(frame,text,(rx1+pad,ry2-pad),font,scale,(255,255,255),thick,cv2.LINE_AA)

def _draw_skeleton(frame, kps):
    SKEL = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    pts  = []
    for kp in kps:
        x,y = float(kp[0]),float(kp[1])
        c   = float(kp[2]) if len(kp)>2 else 1.0
        pts.append((int(x),int(y),c))
    for a,b in SKEL:
        if a<len(pts) and b<len(pts) and pts[a][2]>.3 and pts[b][2]>.3:
            cv2.line(frame,(pts[a][0],pts[a][1]),(pts[b][0],pts[b][1]),
                     (0,255,160),1,cv2.LINE_AA)
    for x,y,c in pts:
        if c>.3: cv2.circle(frame,(x,y),3,(0,230,80),-1,cv2.LINE_AA)

def _draw_hud(frame, frame_idx, player_count):
    h,w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay,(0,0),(230,52),(0,0,0),cv2.FILLED)
    cv2.addWeighted(overlay,.55,frame,.45,0,frame)
    cv2.putText(frame,f"FootballIQ  f:{frame_idx}",(8,18),
                cv2.FONT_HERSHEY_SIMPLEX,.46,(0,229,255),1,cv2.LINE_AA)
    cv2.putText(frame,f"Players visible: {player_count}",(8,36),
                cv2.FONT_HERSHEY_SIMPLEX,.46,(0,230,118),1,cv2.LINE_AA)
    # legend bottom-left
    for i,(lbl,col) in enumerate([("Team A",(0,200,255)),("Team B",(50,50,240)),("Referee",(0,230,230))]):
        y = h-14-i*18
        cv2.circle(frame,(12,y),5,col,-1,cv2.LINE_AA)
        cv2.putText(frame,lbl,(22,y+4),cv2.FONT_HERSHEY_SIMPLEX,.36,(200,210,220),1,cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSER
# ─────────────────────────────────────────────────────────────────────────────
class FootballPerformanceAnalyzer:
    BALL_ID    = 0
    GK_ID      = 1
    PLAYER_ID  = 2
    REFEREE_ID = 3

    def __init__(self,
                 player_model_path="models/weights/best.pt",
                 pose_model_path=None,
                 debug_mode=False,
                 process_every_n=2):
        if not _HAS_YOLO:
            raise RuntimeError("ultralytics not installed")
        self.debug_mode      = debug_mode
        self.process_every_n = process_every_n
        self.player_detector = YOLO(player_model_path)
        self.player_detector.to(_DEVICE)
        self.pose_model = None
        if pose_model_path and Path(pose_model_path).exists():
            try:
                self.pose_model = YOLO(pose_model_path)
                self.pose_model.to(_DEVICE)
                print("[FootballIQ] Pose model loaded")
            except Exception as e:
                print(f"[FootballIQ] Pose model failed: {e}")
        self._det_has_kp = False
        try:
            sample = self.player_detector.predict(np.zeros((64,64,3),np.uint8),verbose=False)
            if sample and sample[0].keypoints is not None:
                self._det_has_kp = True
        except Exception: pass

        bt_buffer = max(90, 90*process_every_n)
        self.tracker = _make_bytetrack(thresh=0.25, buffer=bt_buffer, match=0.8)
        self.gallery = PlayerGallery(max_age_frames=bt_buffer, sim_threshold=0.52, max_dist_px=220)
        self.view            = ViewTransformer()
        self.team_classifier = TeamClassifier()
        self.player_tracks:  dict = {}
        self.player_stats:   dict = {}
        self.player_images:  dict = {}
        self._crop_area:     dict = {}
        self.player_teams:   dict = {}
        self.player_kp:      dict = {}
        self._spd_cache:     dict = {}   # cid → latest speed
        self.frame_store: FrameStore = None
        self._annotated_video_path: str = None
        # Player name registry: player_id (int) → display name (str)
        self.player_names: dict = {}

        for d in ["outputs","player_cards","heatmaps","crops","annotated"]:
            Path(d).mkdir(exist_ok=True)
        print(f"[FootballIQ] Ready  device={_DEVICE}  skip={process_every_n}")

    @staticmethod
    def _empty_track():
        return {'positions':[],'positions_meters':[],'timestamps':[],'speeds':[],
                'frame_count':0,'first_seen':None,'last_seen':None,'bboxes':[],'keypoints':[]}

    def _crop_bbox(self, frame, bbox, pad=8):
        h,w = frame.shape[:2]
        x1=max(0,int(bbox[0])-pad); y1=max(0,int(bbox[1])-pad)
        x2=min(w,int(bbox[2])+pad); y2=min(h,int(bbox[3])+pad)
        if x2<=x1 or y2<=y1: return np.zeros((50,30,3),np.uint8)
        return frame[y1:y2,x1:x2].copy()

    def _maybe_save_crop(self, cid, crop, bbox):
        area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        if area>self._crop_area.get(cid,0) and crop.size>0:
            self.player_images[cid]=crop
            self._crop_area[cid]=area
            cv2.imwrite(f"crops/player_{cid}.jpg",crop)

    # ── ANNOTATE FRAME ────────────────────────────────────────────────────────
    def _annotate_frame(self, frame, cur_ids, cur_bboxes, ref_bboxes=None, kp_map=None):
        out     = frame.copy()
        kp_map  = kp_map or {}
        h_f,w_f = out.shape[:2]

        for cid, bbox in zip(cur_ids, cur_bboxes):
            x1,y1,x2,y2 = map(int,bbox)
            team   = self.player_teams.get(cid,-1)
            color  = TEAM_COLOURS.get(team, TEAM_COLOURS[-1])
            cx     = (x1+x2)//2
            spd    = self._spd_cache.get(cid, 0.0)

            # ── short fading trail (foot positions, last 12 only) ─────────────
            # Only connect adjacent points if they are close (< 120px apart)
            # This stops long diagonal lines when a player was occluded or
            # reappears far from where they were last seen.
            t = self.player_tracks.get(cid)
            if t and len(t['positions']) > 1:
                trail = t['positions'][-12:]   # keep trail short and clean
                for k in range(1, len(trail)):
                    px1 = int(np.clip(trail[k-1][0], 0, w_f-1))
                    py1 = int(np.clip(trail[k-1][1], 0, h_f-1))
                    px2 = int(np.clip(trail[k][0],   0, w_f-1))
                    py2 = int(np.clip(trail[k][1],   0, h_f-1))
                    # Skip segment if the two points are too far apart
                    # (player was lost/reappeared — don't draw a line across the pitch)
                    gap = np.hypot(px2-px1, py2-py1)
                    if gap > 120:
                        continue
                    alpha = k / len(trail)          # fade older segments
                    tc    = tuple(int(c * alpha) for c in color)
                    thick = 2 if alpha > 0.7 else 1
                    cv2.line(out, (px1,py1), (px2,py2), tc, thick, cv2.LINE_AA)

            # ── ground ellipse ────────────────────────────────────────────────
            _draw_ellipse(out, bbox, color, thickness=2)

            # ── small dot at foot ─────────────────────────────────────────────
            cv2.circle(out, (cx, y2), 3, color, -1, cv2.LINE_AA)

            # ── player label ──────────────────────────────────────────────────
            _draw_label(out, cx, y1, cid, spd, color)

            # ── keypoints ─────────────────────────────────────────────────────
            if cid in kp_map:
                _draw_skeleton(out, kp_map[cid])

        # ── referees ─────────────────────────────────────────────────────────
        for bbox in (ref_bboxes or []):
            x1,y1,x2,y2=map(int,bbox)
            _draw_ellipse(out, bbox, (0,230,230), thickness=2)
            _draw_label(out,(x1+x2)//2,y1,"REF",0,(0,180,180))

        _draw_hud(out, self._cur_frame_idx, len(cur_ids))
        return out

    # ── STUB ──────────────────────────────────────────────────────────────────
    def _try_load_stub(self, stub_path):
        if not stub_path.exists(): return False
        try:
            with open(stub_path,'rb') as f: c=pickle.load(f)
            if not c.get('stats'): return False
            self.player_tracks=c['tracks']; self.player_stats=c['stats']
            self.player_images=c.get('images',{}); self.player_teams=c.get('teams',{})
            self.player_kp=c.get('kp',{})
            print(f"[FootballIQ] Cache loaded ({len(self.player_stats)} players)")
            return True
        except Exception as e:
            print(f"[FootballIQ] Cache failed: {e}"); return False

    # ── PROCESS VIDEO ─────────────────────────────────────────────────────────
    def process_video(self, video_path, use_stubs=True,
                      frame_callback=None, progress_callback=None):

        stub = Path(video_path).with_suffix('.pkl')
        if use_stubs and self._try_load_stub(stub):
            if progress_callback: progress_callback(100)
            return self.player_stats, self.player_images

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
        fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        print(f"[FootballIQ] {total_frames}f  {fw}×{fh}  {fps:.1f}fps")

        out_vid = Path(video_path).parent / (Path(video_path).stem + "_annotated.mp4")
        self._annotated_video_path = str(out_vid)
        self.frame_store = FrameStore(
            session_id=Path(video_path).stem,
            out_path=out_vid,
            fps=fps/max(1,self.process_every_n),
            width=fw, height=fh,
        )

        self._cur_frame_idx = 0
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            self._cur_frame_idx = frame_idx
            ts = frame_idx / fps

            if frame_idx % self.process_every_n == 0:
                try:
                    results = self.player_detector(frame, conf=0.15, verbose=False)[0]
                    dets    = sv.Detections.from_ultralytics(results)
                except Exception as e:
                    if self.debug_mode: print(f"Det err f{frame_idx}: {e}")
                    frame_idx+=1; continue

                player_dets = dets[(dets.class_id==self.PLAYER_ID)|(dets.class_id==self.GK_ID)]
                ref_bboxes  = dets.xyxy[dets.class_id==self.REFEREE_ID].tolist() \
                              if len(dets[dets.class_id==self.REFEREE_ID])>0 else []

                cur_ids, cur_bboxes, kp_map = [], [], {}

                if len(player_dets)>0:
                    player_dets = player_dets.with_nms(threshold=0.45)
                    try:
                        tracked = self.tracker.update_with_detections(player_dets)
                    except Exception as e:
                        if self.debug_mode: print(f"Tracker err f{frame_idx}: {e}")
                        frame_idx+=1; continue

                    if tracked.tracker_id is not None and len(tracked)>0:
                        raw_ids_frame = set()
                        det_kps = None
                        if self._det_has_kp:
                            try: det_kps=results.keypoints.data.cpu().numpy()
                            except Exception: pass

                        for i, raw_id in enumerate(tracked.tracker_id):
                            raw_id   = int(raw_id)
                            bbox     = tracked.xyxy[i]
                            crop     = self._crop_bbox(frame, bbox)
                            feat     = AppearanceFeature.extract(crop)
                            foot_pos = (float((bbox[0]+bbox[2])/2), float(bbox[3]))

                            cid = self.gallery.update(raw_id, foot_pos, feat, frame_idx)
                            cur_ids.append(cid)
                            cur_bboxes.append(bbox)
                            raw_ids_frame.add(raw_id)

                            if cid not in self.player_tracks:
                                self.player_tracks[cid] = self._empty_track()
                                self.player_tracks[cid]['first_seen'] = ts

                            t = self.player_tracks[cid]
                            t['positions'].append([foot_pos[0], foot_pos[1]])
                            t['timestamps'].append(ts)
                            t['frame_count'] += 1
                            t['last_seen']    = ts
                            t['bboxes'].append(bbox.tolist())
                            mx,my = self.view.transform(foot_pos[0],foot_pos[1],fw,fh)
                            t['positions_meters'].append([float(mx),float(my)])

                            if len(t['positions_meters'])>1:
                                p1=np.array(t['positions_meters'][-2])
                                p2=np.array(t['positions_meters'][-1])
                                dt=t['timestamps'][-1]-t['timestamps'][-2]
                                if dt>0:
                                    spd=min(float(np.linalg.norm(p2-p1)/dt*3.6),45.0)
                                    t['speeds'].append(spd)
                                    self._spd_cache[cid]=spd

                            if det_kps is not None and i<len(det_kps):
                                kps=det_kps[i].tolist()
                                t['keypoints'].append(kps)
                                self.player_kp[cid]=kps; kp_map[cid]=kps

                            self._maybe_save_crop(cid, crop, bbox)

                        self.gallery.tick_active(raw_ids_frame, frame_idx)

                    if self.pose_model and len(cur_bboxes)>0:
                        try:
                            pr=self.pose_model(frame,conf=.25,verbose=False)[0]
                            if pr.keypoints is not None:
                                pkps=pr.keypoints.data.cpu().numpy()
                                for ci,(cid,_) in enumerate(zip(cur_ids,cur_bboxes)):
                                    if ci<len(pkps):
                                        kps=pkps[ci].tolist()
                                        self.player_kp[cid]=kps; kp_map[cid]=kps
                        except Exception: pass

                annotated = self._annotate_frame(frame, cur_ids, cur_bboxes, ref_bboxes, kp_map)
                self.frame_store.add(frame_idx, annotated)

                if frame_callback:
                    try: frame_callback(annotated)
                    except Exception: pass

            frame_idx+=1
            if progress_callback and frame_idx%30==0:
                progress_callback(min(frame_idx/total_frames*90,90))

        cap.release()
        self.frame_store.close()

        print("[FootballIQ] Stats…")
        self._calculate_statistics()
        print("[FootballIQ] Teams…")
        self._classify_teams()

        if use_stubs:
            try:
                with open(stub,'wb') as f:
                    pickle.dump({'tracks':self.player_tracks,'stats':self.player_stats,
                                 'images':self.player_images,'teams':self.player_teams,
                                 'kp':self.player_kp},f)
            except Exception as e:
                print(f"[FootballIQ] Stub save err: {e}")

        if progress_callback: progress_callback(100)
        print(f"[FootballIQ] Done — {len(self.player_stats)} players")
        return self.player_stats, self.player_images

    # ── STATISTICS ────────────────────────────────────────────────────────────
    def _calculate_statistics(self):
        for pid,t in self.player_tracks.items():
            if t['frame_count']<15: continue
            pos_m=np.array(t['positions_meters'],dtype=float)
            if len(pos_m)<2: continue
            speeds=np.array(t['speeds'],dtype=float) if t['speeds'] else np.zeros(1)
            diffs=np.linalg.norm(np.diff(pos_m,axis=0),axis=1)
            total_dist=float(diffs.sum())
            hi_dist=float(sum(d for d,s in zip(diffs,speeds) if s>15)) if len(speeds)>=len(diffs) else 0.0
            accel=decel=0
            if len(speeds)>2:
                sd=np.diff(speeds)
                accel=int((sd>2.5).sum()); decel=int((sd<-2.5).sum())
            avg_x=float(np.mean(pos_m[:,0]))
            duration=(t['last_seen'] or 0)-(t['first_seen'] or 0)
            spd_arr = speeds
            sprint_dist = float(sum(d for d,s in zip(diffs,speeds) if s>25)) if len(speeds)>=len(diffs) else 0.0
            vhi_dist    = float(sum(d for d,s in zip(diffs,speeds) if s>20)) if len(speeds)>=len(diffs) else 0.0
            hard_accel  = int((np.diff(speeds)>4.0).sum()) if len(speeds)>2 else 0
            hard_decel  = int((np.diff(speeds)<-4.0).sum()) if len(speeds)>2 else 0
            dir_changes = accel + decel  # proxy: direction change ≈ accel+decel events
            total_time  = duration or 1
            fast_pct    = float(min(30.0, (spd_arr>20).sum()/max(len(spd_arr),1)*100))
            jog_pct     = float(min(50.0,  ((spd_arr>=7)&(spd_arr<14)).sum()/max(len(spd_arr),1)*100))
            self.player_stats[pid]={
                'total_distance_km':round(total_dist/1000,3),
                'high_intensity_distance_km':round(hi_dist/1000,3),
                'very_hi_intensity_km':round(vhi_dist/1000,3),
                'sprint_distance_km':round(sprint_dist/1000,3),
                'sprint_count':int((speeds>20).sum()),
                'max_speed':round(float(speeds.max()),1),
                'avg_speed':round(float(speeds.mean()),1),
                'accelerations':accel,'decelerations':decel,
                'hard_accelerations':hard_accel,
                'hard_decelerations':hard_decel,
                'direction_changes':dir_changes,
                'fast_pct':round(fast_pct,1),
                'jog_pct':round(jog_pct,1),
                'position':self.view.zone(avg_x),
                'position_zone':self.view.zone_pct(t['positions_meters']),
                'total_frames':t['frame_count'],
                'duration_seconds':round(duration,1),
                'avg_confidence':0.78,'team':0,
                'has_keypoints':len(t.get('keypoints',[]))>0,
            }

    def _classify_teams(self):
        valid={pid:self.player_images[pid] for pid in self.player_stats if pid in self.player_images}
        if valid:
            a=self.team_classifier.classify(valid)
            for pid,team in a.items():
                if pid in self.player_stats: self.player_stats[pid]['team']=team
                self.player_teams[pid]=team

    # ── PUBLIC ────────────────────────────────────────────────────────────────
    def set_player_names(self, names: dict):
        """Set display names. names = {player_id(int or str): 'Full Name'}"""
        for k, v in names.items():
            try:
                self.player_names[int(k)] = str(v).strip()
            except (ValueError, TypeError):
                self.player_names[k] = str(v).strip()

    def get_display_name(self, player_id: int) -> str:
        """Return assigned name or '#<id>'."""
        return self.player_names.get(player_id) or self.player_names.get(str(player_id)) or f"#{player_id}"

    def get_player_crop(self, player_id):
        c=self.player_images.get(player_id)
        if c is not None: return c
        d=Path(f"crops/player_{player_id}.jpg")
        return cv2.imread(str(d)) if d.exists() else None

    def get_positions_meters(self, player_id):
        t=self.player_tracks.get(player_id)
        return [[float(p[0]),float(p[1])] for p in t['positions_meters']] if t else []

    def get_keypoints(self, player_id):
        return self.player_kp.get(player_id,[])

    # ── HEATMAP ───────────────────────────────────────────────────────────────
    def generate_heatmap(self, player_id):
        if player_id not in self.player_tracks: return None
        raw=self.player_tracks[player_id]['positions_meters']
        if not raw or len(raw)<10: return None
        try:
            pos=np.array([[float(p[0]),float(p[1])] for p in raw])
        except Exception: return None

        fig,ax=plt.subplots(figsize=(13,8),facecolor='#06080c')
        ax.set_facecolor('#06080c')
        ax.add_patch(plt.Rectangle((0,0),105,68,facecolor='#0b2016',edgecolor='#2a5a3a',lw=2,zorder=0))
        ax.axvline(52.5,color='#2a5a3a',lw=1.2,zorder=1)
        ax.add_patch(plt.Circle((52.5,34),9.15,fill=False,edgecolor='#2a5a3a',lw=1.2,zorder=1))
        ax.add_patch(plt.Circle((52.5,34),.4,color='#2a5a3a',zorder=1))
        for lx,bw in [(0,16.5),(88.5,16.5)]:
            ax.add_patch(plt.Rectangle((lx,(68-40.32)/2),bw,40.32,fill=False,edgecolor='#2a5a3a',lw=1.2,zorder=1))
            sx=0 if lx==0 else 99.5
            ax.add_patch(plt.Rectangle((sx,(68-18.32)/2),5.5,18.32,fill=False,edgecolor='#2a5a3a',lw=1,zorder=1))
        for zx in [35,70]:
            ax.axvline(zx,color='#1a4a2a',lw=.8,ls='--',alpha=.5,zorder=1)

        cmap=LinearSegmentedColormap.from_list('fiq',[
            (0,'#00000000'),(0.15,'#003366aa'),(0.45,'#0066ffcc'),
            (0.75,'#ffcc00dd'),(1,'#ff2200ff')])
        hm,_,_=np.histogram2d(pos[:,0],pos[:,1],bins=[60,40],range=[[0,105],[0,68]])
        hm=gaussian_filter(hm,sigma=2.5)
        im=ax.imshow(hm.T,extent=[0,105,0,68],origin='lower',cmap=cmap,
                     aspect='auto',alpha=0.88,interpolation='bilinear',zorder=2)
        cb=plt.colorbar(im,ax=ax,fraction=0.022,pad=0.01,label='Presence density')
        cb.ax.yaxis.label.set_color('white'); cb.ax.tick_params(colors='white')

        cx,cy=pos[:,0].mean(),pos[:,1].mean()
        # Clean crosshair marker instead of a distracting star
        r = 1.8  # radius in metres
        ax.plot([cx-r, cx+r], [cy, cy], color='white', lw=1.5, zorder=4, alpha=0.9)
        ax.plot([cx, cx], [cy-r, cy+r], color='white', lw=1.5, zorder=4, alpha=0.9)
        ax.add_patch(plt.Circle((cx,cy), r, fill=False, edgecolor='#ffcc00',
                                lw=1.8, zorder=4, alpha=0.9))
        ax.text(cx, cy+r+1.2, f'avg ({cx:.0f}, {cy:.0f}) m',
                ha='center', fontsize=8, color='#ffcc00', fontfamily='monospace', zorder=4)

        team=self.player_teams.get(player_id,-1)
        tc='#ebaa32' if team==0 else '#6a8fff' if team==1 else '#aaa'
        display_name=self.get_display_name(player_id)
        ax.set_title(f'{display_name}  ·  Movement Heatmap  ·  {len(pos)} samples  ·  avg pos ({cx:.0f}, {cy:.0f}) m',
                     color='white',fontsize=13,pad=14,fontfamily='monospace')
        ax.set_xlabel('Pitch Length (m)',color='#8a9bb0',fontsize=11)
        ax.set_ylabel('Pitch Width (m)',color='#8a9bb0',fontsize=11)
        ax.tick_params(colors='#8a9bb0'); ax.spines[:].set_color('#1c3a28')
        ax.set_xlim(0,105); ax.set_ylim(0,68)
        for lbl,lx in [('DEF',17.5),('MID',52.5),('ATK',87.5)]:
            ax.text(lx,1.5,lbl,ha='center',fontsize=10,color='#3d5c72',fontfamily='monospace',zorder=4)
        fig.tight_layout()
        return fig

    # ── PLAYER CARD ───────────────────────────────────────────────────────────
    def generate_player_card(self, player_id):
        if player_id not in self.player_stats: return None
        s=self.player_stats[player_id]
        team=self.player_teams.get(player_id,-1)
        tc='#ebaa32' if team==0 else '#6a8fff' if team==1 else '#888'

        fig=plt.figure(figsize=(15,10),facecolor='#07090e')
        gs=GridSpec(3,4,figure=fig,hspace=0.6,wspace=0.45)

        ax0=fig.add_subplot(gs[0,:]); ax0.set_facecolor('#0d1520'); ax0.axis('off')
        display_name=self.get_display_name(player_id)
        ax0.text(.02,.75,f"{display_name}",transform=ax0.transAxes,
                 fontsize=32,fontweight='bold',color='white',fontfamily='monospace')
        ax0.text(.02,.22,
                 f"Team {'A' if team==0 else 'B' if team==1 else '?'}  ·  "
                 f"{s['position']}  ·  {s['duration_seconds']}s on pitch",
                 transform=ax0.transAxes,fontsize=12,color=tc,fontfamily='monospace')
        ax0.axhline(.06,color=tc,lw=3.5,xmin=0,xmax=.5)
        ax0.set_xlim(0,1); ax0.set_ylim(0,1)

        ax1=fig.add_subplot(gs[1,0]); ax1.set_facecolor('#0d1520')
        lbls=['Dist (km)','Hi-Int (km)','Max Spd /10','Sprints /5','Accels /5']
        vals=[s['total_distance_km'],s['high_intensity_distance_km'],
              s['max_speed']/10,s['sprint_count']/5,s['accelerations']/5]
        cols=['#00e5ff','#00e676','#ffab00','#ff3354','#c77dff']
        rawv=[s['total_distance_km'],s['high_intensity_distance_km'],
              s['max_speed'],s['sprint_count'],s['accelerations']]
        bars=ax1.barh(lbls,vals,color=cols,height=0.55)
        for bar,rv in zip(bars,rawv):
            ax1.text(bar.get_width()+.02,bar.get_y()+bar.get_height()/2,
                     f'{rv:.1f}',va='center',fontsize=7.5,color='white')
        ax1.set_facecolor('#0d1520'); ax1.spines[:].set_visible(False)
        ax1.tick_params(colors='#8a9bb0',labelsize=7.5)
        ax1.set_title('Key Metrics',color='#8a9bb0',fontsize=9,pad=6)
        ax1.set_xlim(0,max(vals,default=1)*1.5)

        ax2=fig.add_subplot(gs[1,1]); ax2.set_facecolor('#0d1520')
        pz=s.get('position_zone',{'defensive':33,'middle':34,'attacking':33})
        zv=[pz['defensive'],pz['middle'],pz['attacking']]
        zcols=['#00e5ff','#00e676','#ff3354']
        ax2.bar(['DEF','MID','ATK'],zv,color=zcols,width=.55)
        ax2.set_ylim(0,105); ax2.spines[:].set_visible(False)
        ax2.tick_params(colors='#8a9bb0',labelsize=9); ax2.set_facecolor('#0d1520')
        for i,v in enumerate(zv):
            ax2.text(i,v+2,f'{v:.0f}%',ha='center',fontsize=9,color=zcols[i])
        ax2.set_title('Zone Distribution',color='#8a9bb0',fontsize=9,pad=6)

        ax3=fig.add_subplot(gs[1,2]); ax3.set_facecolor('#0d1520')
        bins=[0,7,14,20,25,30,99]
        slbls=['Walk','Jog','Run','Fast','Sprint','Max']
        scols=['#3d5c72','#5a7a90','#00e5ff','#ffab00','#ff7043','#ff3354']
        t=self.player_tracks.get(player_id)
        if t and t['speeds']:
            sarr=np.array(t['speeds'])
            counts=[int(((sarr>=bins[i])&(sarr<bins[i+1])).sum()) for i in range(len(bins)-1)]
            tot=sum(counts) or 1
            pcts=[c/tot*100 for c in counts]
            ax3.barh(slbls,pcts,color=scols,height=.55)
            for i,p in enumerate(pcts):
                if p>1: ax3.text(p+.5,i,f'{p:.0f}%',va='center',fontsize=7.5,color='white')
        ax3.set_facecolor('#0d1520'); ax3.set_xlim(0,110)
        ax3.tick_params(colors='#8a9bb0',labelsize=8); ax3.spines[:].set_visible(False)
        ax3.set_title('Speed Profile',color='#8a9bb0',fontsize=9,pad=6)

        ax4=fig.add_subplot(gs[1,3]); ax4.set_facecolor('#0d1520'); ax4.axis('off')
        crop=self.get_player_crop(player_id)
        if crop is not None and crop.size>0:
            ax4.imshow(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),aspect='auto')
        else:
            ax4.text(.5,.5,f'#{player_id}',ha='center',va='center',
                     fontsize=28,color=tc,transform=ax4.transAxes,fontfamily='monospace')
        ax4.set_title('Appearance',color='#8a9bb0',fontsize=9,pad=6)

        ax5=fig.add_subplot(gs[2,:]); ax5.set_facecolor('#0d1520'); ax5.axis('off')
        rows=[['Metric','Value','Metric','Value'],
              ['Total Distance',f"{s['total_distance_km']} km",'Max Speed',f"{s['max_speed']} km/h"],
              ['Hi-Intensity',f"{s['high_intensity_distance_km']} km",'Avg Speed',f"{s['avg_speed']} km/h"],
              ['Sprints (>20)',str(s['sprint_count']),'Accelerations',str(s['accelerations'])],
              ['Decelerations',str(s['decelerations']),'Primary Zone',s['position']],
              ['Duration',f"{s['duration_seconds']}s",'Frames',str(s['total_frames'])]]
        col_x=[.01,.26,.52,.76]
        for ri,row in enumerate(rows):
            for ci,cell in enumerate(row):
                ax5.text(col_x[ci],1-ri/len(rows)-.06,cell,transform=ax5.transAxes,va='top',
                         fontsize=9.5 if ri>0 else 10.5,fontfamily='monospace',
                         color=tc if ri==0 else ('white' if ci%2==1 else '#8a9bb0'),
                         fontweight='bold' if ri==0 else 'normal')
        ax5.set_title('Full Statistics',color='#8a9bb0',fontsize=9,pad=6)
        fig.suptitle('FootballIQ · Performance Analysis',fontsize=10,color='#3d5c72',y=.005,fontfamily='monospace')
        return fig

    # ── RECOVERY PLAN (built-in fallback) ─────────────────────────────────────
    def generate_recovery_plan(self, player_id):
        """Simple fallback plan used when recovery modules are unavailable."""
        if player_id not in self.player_stats: return None
        s=self.player_stats[player_id]
        risk=0
        hi=s['high_intensity_distance_km']; sp=s['sprint_count']
        ad=s['accelerations']+s['decelerations']
        if hi>1.5: risk+=30
        elif hi>1: risk+=20
        elif hi>.5: risk+=10
        if sp>20: risk+=25
        elif sp>10: risk+=15
        elif sp>5:  risk+=8
        if ad>60: risk+=20
        elif ad>30: risk+=12
        risk=min(100,risk)
        if risk>=70:
            rl,rest="HIGH",3
            recs=["48h complete rest mandatory","Cryotherapy & ice bath protocol",
                  "Day 3: gentle mobility only","Mandatory physio assessment before return"]
        elif risk>=40:
            rl,rest="MODERATE",2
            recs=["Active recovery only — no high intensity","Foam rolling & static stretching",
                  "Easy jog day 2 (<65% HR max)","Cold-water immersion 12 min"]
        else:
            rl,rest="LOW",1
            recs=["Light active recovery session","Standard post-match nutrition protocol",
                  "Normal training next day"]
        return {
            'player_id':player_id,
            'player_name':self.get_display_name(player_id),
            'risk_level':rl,'risk_score':risk,'risk_tier':rl,'overall_risk':risk,
            'rest_days':rest,
            'recommendations':recs,
            'next_match_readiness':f"{rest+1}–{rest+2} days",
            'next_match_ready':f"{rest+1}–{rest+2} days",
            'nutrition_note':"High-carb + protein meal within 30 min post-match",
            'workload':{'total_distance_km':s['total_distance_km'],
                        'hi_distance_km':s['high_intensity_distance_km'],
                        'sprint_count':s['sprint_count'],'accel_decel':ad}
        }

    def export_to_json(self, output_path):
        data={'export_date':datetime.now().isoformat(),'total_players':len(self.player_stats),
              'players':{str(k):v for k,v in self.player_stats.items()}}
        with open(output_path,'w') as f:
            json.dump(data,f,indent=2,cls=_NpEncoder)
        return data