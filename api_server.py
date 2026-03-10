
"""
FootballIQ API Server v5.0
==========================
Run:  python api_server.py            → http://localhost:8000
      python api_server.py --port 9000

File layout (flat — all in same directory):
    api_server.py
    football_analyzer_fixed.py
    injury_predictor.py
    recovery_planner.py
    recovery_card_generator.py
    index.html
    models/weights/best.pt

Endpoints
─────────
POST /api/upload                          Upload video → session_id
POST /api/process                         Start analysis
GET  /api/status/{session_id}             Poll progress
GET  /api/session/{session_id}/players    List all players (with names)
GET  /api/session/{session_id}/teams      Team-level aggregated stats
POST /api/session/{session_id}/names      Assign player names {"names":{"3":"John"}}
GET  /api/session/{session_id}/names      Read name registry
GET  /api/player/{id}/stats?session_id=  Player stats JSON
GET  /api/player/{id}/heatmap?session_id= Heatmap PNG (mirror-fixed)
GET  /api/player/{id}/card?session_id=   Player card PNG
GET  /api/player/{id}/recovery?session_id=  Full injury + recovery JSON
GET  /api/player/{id}/recovery/card?session_id= Recovery card PNG
GET  /api/player/{id}/crop?session_id=   Player crop JPEG
GET  /api/frames/{session_id}/latest     Latest annotated frame (poll ~10 fps)
GET  /api/frames/{session_id}/count      How many frames buffered
GET  /api/video/{session_id}             Download annotated MP4
GET  /api/stream/{session_id}            SSE push stream
GET  /api/export/{session_id}            Download full JSON export
GET  /api/health                         Health + module status
"""

import os, sys, uuid, threading, time, json, base64, traceback, shutil, argparse
from pathlib import Path
from datetime import datetime

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException, Request
    from fastapi.responses import (JSONResponse, StreamingResponse,
                                   FileResponse, HTMLResponse, Response)
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    sys.exit("pip install fastapi uvicorn python-multipart")

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Module imports ────────────────────────────────────────────────────────────
# Folder layout:
#   footballv2/
#     api_server.py
#     football_analyzer_fixed.py
#     index.html
#     models/weights/best.pt
#     modules/
#       injury_predictor.py
#       recovery_planner.py
#       recovery_card_generator.py
#       heatmap_generator.py
#       ...
HERE = Path(__file__).parent
MODULES_DIR = HERE / 'modules'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MODULES_DIR))   # makes "from injury_predictor import ..." work

try:
    from modules.injury_predictor import InjuryPredictor
    _injury_predictor = InjuryPredictor()
    print("[API] ✓ InjuryPredictor loaded")
except Exception as _e:
    _injury_predictor = None
    print(f"[API] ✗ InjuryPredictor: {_e}")

try:
    from modules.recovery_planner import RecoveryPlanner
    _recovery_planner = RecoveryPlanner()
    print("[API] ✓ RecoveryPlanner loaded")
except Exception as _e:
    _recovery_planner = None
    print(f"[API] ✗ RecoveryPlanner: {_e}")

try:
    from modules.recovery_card_generator import RecoveryCardGenerator
    _recovery_card = RecoveryCardGenerator()
    print("[API] ✓ RecoveryCardGenerator loaded")
except Exception as _e:
    _recovery_card = None
    print(f"[API] ✗ RecoveryCardGenerator: {_e}")

_HAS_MODULES = bool(_injury_predictor and _recovery_planner)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="FootballIQ API", version="5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

for _d in ['uploads', 'outputs', 'player_cards', 'heatmaps',
           'crops', 'annotated', 'recovery_cards']:
    (HERE / _d).mkdir(exist_ok=True)

UPLOAD_FOLDER   = HERE / 'uploads'
OUTPUT_FOLDER   = HERE / 'outputs'
PLAYER_CARDS    = HERE / 'player_cards'
HEATMAP_FOLDER  = HERE / 'heatmaps'
RECOVERY_FOLDER = HERE / 'recovery_cards'

sessions:       dict = {}
analyzers:      dict = {}
stream_buffers: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# SSE STREAM BUFFER
# ─────────────────────────────────────────────────────────────────────────────
class StreamBuffer:
    def __init__(self):
        self.frames = []
        self.lock   = threading.Lock()
        self.done   = False

    def add(self, b64):
        with self.lock:
            self.frames.append(b64)
            if len(self.frames) > 8:
                self.frames.pop(0)

    def latest(self):
        with self.lock:
            return self.frames[-1] if self.frames else None

    def stop(self):
        with self.lock:
            self.done = True

    def is_done(self):
        with self.lock:
            return self.done


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def process_background(session_id, video_path, enable_streaming, process_every_n, show_trails):
    session = sessions[session_id]
    session['status'] = 'processing'

    if enable_streaming:
        stream_buffers[session_id] = StreamBuffer()

    try:
        from football_analyzer_fixed import FootballPerformanceAnalyzer

        analyzer = FootballPerformanceAnalyzer(
            player_model_path=str(HERE / 'models/weights/best.pt'),
            process_every_n=process_every_n,
        )
        analyzer._show_trails = show_trails

        # Push any names set before processing started
        pending = session.get('pending_names', {})
        if pending:
            analyzer.set_player_names(pending)

        analyzers[session_id] = analyzer

        def on_progress(p):
            session['progress'] = p

        def on_frame(bgr_frame):
            if enable_streaming and session_id in stream_buffers:
                try:
                    h, w = bgr_frame.shape[:2]
                    if w > 1280:
                        bgr_frame = cv2.resize(bgr_frame, (1280, int(h * 1280 / w)),
                                               interpolation=cv2.INTER_LANCZOS4)
                    _, buf = cv2.imencode('.jpg', bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    stream_buffers[session_id].add(base64.b64encode(buf).decode())
                except Exception:
                    pass

        stats, images = analyzer.process_video(
            video_path=video_path,
            use_stubs=True,
            frame_callback=on_frame,
            progress_callback=on_progress,
        )

        session['status']          = 'processed'
        session['stats']           = stats
        session['total_players']   = len(stats)
        session['progress']        = 100
        session['annotated_video'] = analyzer._annotated_video_path
        print(f"[server] Done {session_id[:8]}  {len(stats)} players")

    except Exception as e:
        traceback.print_exc()
        session['status'] = 'error'
        session['error']  = str(e)
    finally:
        if enable_streaming and session_id in stream_buffers:
            stream_buffers[session_id].stop()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _get_az(session_id: str):
    az = analyzers.get(session_id)
    if az is None:
        raise HTTPException(404, detail="Analyzer not ready — processing may not be complete")
    return az


def _display_name(session_id: str, player_id: int) -> str:
    az = analyzers.get(session_id)
    if az:
        return az.get_display_name(player_id)
    names = sessions.get(session_id, {}).get('pending_names', {})
    return names.get(player_id) or names.get(str(player_id)) or f"#{player_id}"


def _build_recovery_data(session_id: str, player_id: int) -> dict:
    az = _get_az(session_id)
    if player_id not in az.player_stats:
        raise HTTPException(404, detail=f"Player {player_id} not found")

    stats = dict(az.player_stats[player_id])
    stats['player_id'] = player_id
    stats['team']      = az.player_teams.get(player_id, -1)
    stats['name']      = az.get_display_name(player_id)

    if _HAS_MODULES:
        try:
            injury = _injury_predictor.predict(stats)
            plan   = _recovery_planner.generate_recovery_plan(stats, injury)
            plan['injury_prediction']    = injury
            plan['likely_injuries']      = injury.get('likely_injuries', [])
            plan['tissue_scores']        = injury.get('tissue_scores', {})
            plan['tissue_tiers']         = injury.get('tissue_tiers', {})
            plan['tier_colors']          = injury.get('tier_colors', {})
            plan['overall_risk']         = injury.get('overall_risk', 0)
            plan['risk_tier']            = injury.get('overall_tier', 'LOW')
            plan['risk_level']           = plan['risk_tier']
            plan['risk_score']           = round(injury.get('overall_risk', 0), 1)
            plan['next_match_readiness'] = plan.get('next_match_ready', '—')
            plan['player_name']          = stats['name']
            return plan
        except Exception as e:
            print(f"[API] Recovery module error: {e} — falling back to simple plan")

    return az.generate_recovery_plan(player_id)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    for c in [HERE / 'index.html', Path.cwd() / 'index.html']:
        if c.exists():
            return FileResponse(str(c), media_type='text/html')
    return HTMLResponse("<pre>index.html not found</pre>", status_code=404)


@app.get("/api/health")
async def health():
    gpu = "CPU"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return JSONResponse({
        "status":    "healthy",
        "version":   "5.0",
        "gpu":       gpu,
        "modules": {
            "injury_predictor": _injury_predictor is not None,
            "recovery_planner": _recovery_planner is not None,
            "recovery_card":    _recovery_card is not None,
        },
        "timestamp": datetime.now().isoformat(),
    })


@app.post("/api/upload")
async def upload_video(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(400, detail="No file provided")
    ext = video.filename.rsplit('.', 1)[-1].lower() if '.' in video.filename else ''
    if ext not in {'mp4', 'avi', 'mov', 'mkv'}:
        raise HTTPException(400, detail=f"Unsupported format: .{ext}")

    session_id = str(uuid.uuid4())
    dest = UPLOAD_FOLDER / f"{session_id}_{video.filename}"
    with open(dest, 'wb') as f:
        shutil.copyfileobj(video.file, f)

    info = {}
    if _HAS_CV2:
        try:
            cap   = cv2.VideoCapture(str(dest))
            fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info  = {
                'fps':              round(fps, 2),
                'width':            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height':           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'total_frames':     total,
                'duration_seconds': round(total / fps, 2),
            }
            cap.release()
        except Exception:
            pass

    sessions[session_id] = {
        'video_path':    str(dest),
        'filename':      video.filename,
        'status':        'uploaded',
        'progress':      0,
        'video_info':    info,
        'pending_names': {},
    }
    return JSONResponse({
        'success':    True,
        'session_id': session_id,
        'video_path': str(dest),
        'video_info': info,
        'filename':   video.filename,
    })


@app.post("/api/process")
async def process_video(request: Request):
    ct = request.headers.get('content-type', '')
    try:
        if 'multipart' in ct or 'form' in ct:
            form = await request.form()
            data = dict(form)
        else:
            body = await request.body()
            data = json.loads(body) if body else {}
    except Exception as e:
        raise HTTPException(400, detail=str(e))

    session_id       = data.get('session_id', '').strip()
    enable_streaming = str(data.get('enable_streaming', 'false')).lower() == 'true'
    show_trails      = str(data.get('show_trails', 'true')).lower() != 'false'
    try:
        process_every_n = int(data.get('process_every_n', 2) or 2)
    except (TypeError, ValueError):
        process_every_n = 2

    if not session_id:
        raise HTTPException(400, detail="session_id required")
    if session_id not in sessions:
        raise HTTPException(404, detail=f"Session not found: {session_id}")
    if sessions[session_id].get('status') == 'processing':
        raise HTTPException(409, detail="Already processing")

    video_path = data.get('video_path') or sessions[session_id]['video_path']
    threading.Thread(
        target=process_background,
        args=(session_id, video_path, enable_streaming, process_every_n, show_trails),
        daemon=True,
    ).start()

    return JSONResponse({
        'status':     'processing_started',
        'session_id': session_id,
        'stream_url': f'/api/stream/{session_id}' if enable_streaming else None,
    })


@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, detail="Session not found")
    s  = sessions[session_id]
    az = analyzers.get(session_id)
    frame_count = az.frame_store.count() if az and az.frame_store else 0
    return JSONResponse({
        'status':        s.get('status'),
        'progress':      s.get('progress', 0),
        'total_players': s.get('total_players', 0),
        'error':         s.get('error'),
        'video_info':    s.get('video_info', {}),
        'frames_ready':  frame_count,
    })


# ── Player Name Registry ──────────────────────────────────────────────────────

@app.post("/api/session/{session_id}/names")
async def set_player_names(session_id: str, request: Request):
    """
    Assign human-readable names to player IDs. Call once after upload
    (before or after processing — both work).

    Body JSON:
        {"names": {"1": "James Walker", "7": "Mike Brown", "9": "Erik Johansson"}}
    """
    if session_id not in sessions:
        raise HTTPException(404, detail="Session not found")

    body = await request.json()
    raw  = body.get('names', {})
    if not isinstance(raw, dict):
        raise HTTPException(400, detail="'names' must be a JSON object")

    pending = sessions[session_id].setdefault('pending_names', {})
    for k, v in raw.items():
        try:
            pending[int(k)] = str(v).strip()
        except (ValueError, TypeError):
            pending[k] = str(v).strip()

    az = analyzers.get(session_id)
    if az:
        az.set_player_names(pending)

    return JSONResponse({
        'success':  True,
        'names_set': len(raw),
        'registry': {str(k): v for k, v in pending.items()},
    })


@app.get("/api/session/{session_id}/names")
async def get_player_names(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, detail="Session not found")
    az  = analyzers.get(session_id)
    reg = az.player_names if az else sessions[session_id].get('pending_names', {})
    return JSONResponse({
        'session_id': session_id,
        'names':      {str(k): v for k, v in reg.items()},
    })


# ── Players List ──────────────────────────────────────────────────────────────

@app.get("/api/session/{session_id}/players")
async def get_players(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, detail="Session not found")
    stats = sessions[session_id].get('stats', {})
    players = [
        {
            'player_id': int(pid),
            'name':      _display_name(session_id, int(pid)),
            'team':      s.get('team', 0),
            'stats':     s,
        }
        for pid, s in stats.items()
    ]
    return JSONResponse({'total_players': len(players), 'players': players})


# ── Team Analysis ─────────────────────────────────────────────────────────────

@app.get("/api/session/{session_id}/teams")
async def get_team_analysis(session_id: str):
    """Per-team totals, averages and player rosters."""
    if session_id not in sessions:
        raise HTTPException(404, detail="Session not found")
    stats = sessions[session_id].get('stats', {})
    if not stats:
        raise HTTPException(404, detail="No stats yet — processing not complete")

    teams: dict = {}
    for pid, s in stats.items():
        tid = s.get('team', -1)
        if tid not in teams:
            teams[tid] = {
                'team_id':                    tid,
                'team_label':                 f"Team {'A' if tid==0 else 'B' if tid==1 else '?'}",
                'players':                    [],
                'count':                      0,
                'total_distance_km':          0.0,
                'high_intensity_distance_km': 0.0,
                'sprint_count':               0,
                'max_speed':                  0.0,
                'accelerations':              0,
                'decelerations':              0,
            }
        t = teams[tid]
        t['players'].append({
            'player_id': int(pid),
            'name':      _display_name(session_id, int(pid)),
            'position':  s.get('position', '?'),
        })
        t['count']                      += 1
        t['total_distance_km']          += s.get('total_distance_km', 0)
        t['high_intensity_distance_km'] += s.get('high_intensity_distance_km', 0)
        t['sprint_count']               += s.get('sprint_count', 0)
        t['max_speed']                   = max(t['max_speed'], s.get('max_speed', 0))
        t['accelerations']              += s.get('accelerations', 0)
        t['decelerations']              += s.get('decelerations', 0)

    for t in teams.values():
        n = t['count'] or 1
        t['total_distance_km']          = round(t['total_distance_km'], 3)
        t['high_intensity_distance_km'] = round(t['high_intensity_distance_km'], 3)
        t['max_speed']                  = round(t['max_speed'], 1)
        t['avg_distance_km']            = round(t['total_distance_km'] / n, 3)
        t['avg_hi_distance_km']         = round(t['high_intensity_distance_km'] / n, 3)
        t['avg_sprint_count']           = round(t['sprint_count'] / n, 1)
        t['avg_accelerations']          = round(t['accelerations'] / n, 1)

    return JSONResponse({'session_id': session_id, 'teams': {str(k): v for k, v in teams.items()}})


# ── Individual Player ─────────────────────────────────────────────────────────

@app.get("/api/player/{player_id}/stats")
async def get_player_stats(player_id: int, session_id: str):
    az = _get_az(session_id)
    if player_id not in az.player_stats:
        raise HTTPException(404, detail=f"Player {player_id} not found")
    return JSONResponse({
        'player_id': player_id,
        'name':      az.get_display_name(player_id),
        'team':      az.player_teams.get(player_id, -1),
        'stats':     az.player_stats[player_id],
        'positions': az.get_positions_meters(player_id),
    })


@app.get("/api/player/{player_id}/heatmap")
async def get_heatmap(player_id: int, session_id: str):
    az  = _get_az(session_id)
    fig = az.generate_heatmap(player_id)
    if fig is None:
        raise HTTPException(404, detail="Not enough position data (need ≥10 samples)")
    out = HEATMAP_FOLDER / f'player_{player_id}_heatmap.png'
    fig.savefig(str(out), dpi=110, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return FileResponse(str(out), media_type='image/png')


@app.get("/api/player/{player_id}/card")
async def get_card(player_id: int, session_id: str):
    az  = _get_az(session_id)
    fig = az.generate_player_card(player_id)
    if fig is None:
        raise HTTPException(404, detail="Player not found")
    out = PLAYER_CARDS / f'player_{player_id}_card.png'
    fig.savefig(str(out), dpi=110, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return FileResponse(str(out), media_type='image/png')


@app.get("/api/player/{player_id}/recovery")
async def get_recovery(player_id: int, session_id: str):
    plan = _build_recovery_data(session_id, player_id)
    return JSONResponse(plan)


@app.get("/api/player/{player_id}/recovery/card")
async def get_recovery_card(player_id: int, session_id: str):
    if not _recovery_card:
        raise HTTPException(501, detail="RecoveryCardGenerator not available")
    plan = _build_recovery_data(session_id, player_id)
    out  = RECOVERY_FOLDER / f'player_{player_id}_recovery.png'
    try:
        _recovery_card.generate_card(plan, output_path=str(out))
    except Exception as e:
        raise HTTPException(500, detail=f"Card generation failed: {e}")
    if not out.exists():
        raise HTTPException(500, detail="Recovery card file was not created")
    return FileResponse(str(out), media_type='image/png')


@app.get("/api/player/{player_id}/crop")
async def get_crop(player_id: int, session_id: str):
    az   = _get_az(session_id)
    crop = az.get_player_crop(player_id)
    if crop is None:
        raise HTTPException(404, detail="Crop not available")
    if not _HAS_CV2:
        raise HTTPException(500, detail="opencv-python not installed")
    _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(buf.tobytes(), media_type='image/jpeg')


@app.get("/api/player/{player_id}/keypoints")
async def get_keypoints(player_id: int, session_id: str):
    az  = _get_az(session_id)
    kps = az.get_keypoints(player_id)
    KP_NAMES = ["nose","left_eye","right_eye","left_ear","right_ear",
                "left_shoulder","right_shoulder","left_elbow","right_elbow",
                "left_wrist","right_wrist","left_hip","right_hip",
                "left_knee","right_knee","left_ankle","right_ankle"]
    result = [{'name': KP_NAMES[i] if i < len(KP_NAMES) else f'kp_{i}',
               'x': float(kp[0]), 'y': float(kp[1]),
               'confidence': float(kp[2]) if len(kp) > 2 else 1.0}
              for i, kp in enumerate(kps)]
    return JSONResponse({'player_id': player_id, 'keypoints': result})


# ── Frames / Streaming ────────────────────────────────────────────────────────

@app.get("/api/frames/{session_id}/latest")
async def frames_latest(session_id: str):
    az = analyzers.get(session_id)
    if az is None or az.frame_store is None:
        raise HTTPException(404, detail="No frames yet")
    jpg = az.frame_store.latest_jpeg()
    if jpg is None:
        raise HTTPException(404, detail="No frames yet")
    return Response(jpg, media_type='image/jpeg',
                    headers={'Cache-Control': 'no-cache, no-store',
                             'Access-Control-Allow-Origin': '*'})


@app.get("/api/frames/{session_id}/at/{frame_idx}")
async def frames_at(session_id: str, frame_idx: int):
    az = analyzers.get(session_id)
    if az is None or az.frame_store is None:
        raise HTTPException(404, detail="No frames yet")
    jpg = az.frame_store.frame_at(frame_idx)
    if jpg is None:
        raise HTTPException(404, detail=f"Frame {frame_idx} not available")
    return Response(jpg, media_type='image/jpeg',
                    headers={'Cache-Control': 'no-cache',
                             'Access-Control-Allow-Origin': '*'})


@app.get("/api/frames/{session_id}/count")
async def frames_count(session_id: str):
    az    = analyzers.get(session_id)
    count = az.frame_store.count() if az and az.frame_store else 0
    return JSONResponse({'count': count, 'session_id': session_id})


@app.get("/api/video/{session_id}")
async def get_annotated_video(session_id: str):
    s   = sessions.get(session_id, {})
    vid = s.get('annotated_video')
    az  = analyzers.get(session_id)
    if az and az._annotated_video_path:
        vid = az._annotated_video_path
    if not vid or not Path(vid).exists():
        raise HTTPException(404, detail="Annotated video not ready yet")
    return FileResponse(vid, media_type='video/mp4',
                        filename=f'footballiq_{session_id[:8]}_annotated.mp4')


@app.get("/api/stream/{session_id}")
async def stream_video(session_id: str):
    if session_id not in stream_buffers:
        raise HTTPException(404, detail="No active stream for this session")

    def generate():
        buf  = stream_buffers[session_id]
        last = None
        yield f'data: {json.dumps({"type": "connected"})}\n\n'
        while True:
            frame = buf.latest()
            if frame and frame != last:
                last = frame
                yield f'data: {json.dumps({"type": "frame", "data": frame})}\n\n'
            if sessions.get(session_id, {}).get('status') in ('processed', 'error') and buf.is_done():
                yield f'data: {json.dumps({"type": "end"})}\n\n'
                break
            time.sleep(0.04)

    return StreamingResponse(generate(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ── Export ────────────────────────────────────────────────────────────────────

@app.get("/api/export/{session_id}")
async def export_data(session_id: str):
    az  = _get_az(session_id)
    out = OUTPUT_FOLDER / f'session_{session_id}.json'
    az.export_to_json(str(out))
    return FileResponse(str(out), media_type='application/json',
                        filename=f'footballiq_{session_id[:8]}.json')


# ── Static Fallback ───────────────────────────────────────────────────────────

@app.get("/{filepath:path}")
async def serve_static(filepath: str):
    for base in [HERE, Path.cwd()]:
        f = base / filepath
        if f.exists() and f.is_file():
            return FileResponse(str(f))
    raise HTTPException(404)


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP CHECK
# ─────────────────────────────────────────────────────────────────────────────
def _startup_check(port):
    print("\n" + "=" * 62)
    print("  FootballIQ API v5.0")
    print("=" * 62)
    for label, path in [
        ("index.html",                          HERE / 'index.html'),
        ("models/weights/best.pt",              HERE / 'models/weights/best.pt'),
        ("football_analyzer_fixed.py",          HERE / 'football_analyzer_fixed.py'),
        ("modules/injury_predictor.py",         HERE / 'modules/injury_predictor.py'),
        ("modules/recovery_planner.py",         HERE / 'modules/recovery_planner.py'),
        ("modules/recovery_card_generator.py",  HERE / 'modules/recovery_card_generator.py'),
        ("modules/heatmap_generator.py",        HERE / 'modules/heatmap_generator.py'),
    ]:
        print(f"  {'✓' if path.exists() else '✗'} {label}")
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        print(f"  ✓ torch  ({gpu})")
    except ImportError:
        print("  ✗ torch  (not installed)")
    print(f"  {'✓' if _HAS_MODULES else '✗'} recovery pipeline")
    print(f"  {'✓' if _recovery_card else '✗'} recovery card generator")
    print(f"\n  → http://localhost:{port}\n" + "=" * 62 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FootballIQ API Server")
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()
    _startup_check(args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')