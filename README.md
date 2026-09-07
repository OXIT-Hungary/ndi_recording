# NDI Recording — Technical Documentation

Technical documentation for `ndi_recording/`.

## What this is

An on-site water polo match capture system for OXIT's camera rig at a venue: one fixed **panorama
camera** (RTSP, fisheye/wide-FOV, undistorted+cropped) does real-time player detection to drive
auto-tracking, and two **PTZ cameras** (NDI capture + VISCA pan/tilt control) follow the play,
recording locally to MP4 and/or live-streaming to YouTube via RTMP. It is a separate git repository
(`git@github.com:OXIT-Hungary/ndi_recording.git`) nested inside `Documents/wp/`, unrelated to and not
imported by `wp_core`/`wp_gsr`/`tracknet` — a sibling capture-and-control project, not an analysis
pipeline. (`wp_core`'s own multi-camera match pipeline consumes *recordings* this system could
produce, but there is no code-level dependency between them.)

To run/build it:
- "make build"
- "make run"

## Repo layout

```
ndi_recording/
  main.py                       # CLI entrypoint (Config -> CameraSystem, run-for-duration or scheduled)
  shared_manager.py              # unused multiprocessing.Manager singleton (StreamStatus enum) — dead code, see Gotchas
  Dockerfile                      # CUDA/NVENC-enabled FFmpeg build + uvicorn CMD
  Makefile                         # lint/format/build/run (docker run with --gpus, --network host, avahi/dbus mounts)
  setup.sh                          # one-time host bootstrap: NVIDIA driver + Docker + nvidia-container-toolkit
  pyproject.toml                     # black/isort/flake8 config only, no [project] section
  requirements/{prod,dev}.{in,txt}    # pip-tools compiled deps
  metadata.txt                         # tab-separated log of every match started via the API (append-only)
  database.db, stream_schedules.db      # SQLite files (see Persistence below)

  configs/
    default_config.yaml           # generic/example, used by main.py's --config default
    bvsc_config.yaml               # venue: BVSC pool — used by ManualControlRouter (hardcoded)
    fradi_config.yaml               # venue: Fradi pool — used by streaming.py's YouTube router (hardcoded, dormant)
    komjadi_config.yaml              # venue: Komjádi pool
    schedule.yaml                     # daily cron times for main.py --scheduled ("17:00", "18:15", "19:30")

  camera_params/{bvsc,fradi,komjadi}/  # per-venue panorama calibration
    pano_params_bvsc.npy               # (K, dist) pair for cv2 undistortion (legacy path, camera_params key)
    remap_4096x1484_fov*.npz            # precomputed cv2.remap map_x/map_y (current path, distort key)

  models/rtdetrv2_pano.onnx        # RT-DETRv2 ONNX detector run on panorama frames (git-lfs tracked)

  src/
    config.py                      # YAML -> Config/PTZConfig/PanoramaConfig/CameraSystemConfig/BEVConfig
    game.py                        # Game: homography BEV projection, Kalman-filter player tracking, court drawing
    kalman_filter.py                # a second, unused Track/Tracker implementation — dead code, see Gotchas
    camera/
      camera.py                     # Camera base class (near-empty)
      camera_system.py               # CameraSystem: owns all camera processes + the detect/track/pan loop
      pano_camera.py                  # PanoCamrera [sic]: RTSP/mp4 -> ffmpeg raw-video pipe -> frame queue (+ optional local save)
      ptz_camera.py                    # PTZCamera / Avonic_CM93_NDI: NDI frame capture -> ffmpeg (record + RTMP stream)
      visca.py                         # Sony VISCA-over-UDP protocol: power, absolute move, pan/tilt inquiry
    utils/
      tmp.py                          # VISCA<->Euler angle conversion, DBSCAN cluster centroid, pan-shift math
      visualize.py                     # matplotlib/cv2 debug-drawing helpers (detections, tracks, homography)
      logger.py                        # setup_logger() for the CLI path

  app/
    main.py                        # FastAPI() app, mounts api_router, serves templates/streaming/redirect.html at "/"
    api/
      __init__.py                   # /api -> v1 router
      dependencies.py                 # get_schedule/get_scheduler/get_api_logger (unused outside dormant schedule.py)
      services/youtube_service.py      # Google OAuth + YouTube Data API stream scheduling — dormant, see Gotchas
      v1/
        __init__.py                    # mounts version_router, manual_control_router, database_router ONLY
        version.py                      # GET /api/v1/version
        manual_control_router.py         # POST start-stream/stop-stream, GET get-stream-status — THE live control surface
        database_router.py               # GET /api/v1/database/get-all-matches + Database (sqlite match log)
        authenticator.py                  # X-API-Key header auth (ADMIN_API_KEY / USER_API_KEY from .env.production)
        configuration.py                  # pydantic Settings for the (unused) YouTube integration
        schedule.py                       # generic Schedule CRUD router — written, never mounted, see Gotchas
        streaming.py                       # YouTube live-stream scheduling router — written, never mounted, see Gotchas
    core/
      record_manager.py               # RecordManager(Schedulable) — every method is `pass`, unfinished stub
      scheduler.py                     # generic Scheduler/ScheduledTask singleton (thread polling loop) — unused outside dormant schedule.py
      schedulable.py                   # Schedulable ABC (start/stop/is_running)
      exceptions/http_exceptions.py     # HTTPException subclasses for the schedule API (has a typo bug, see Gotchas)
      utils/{logger,dir_creator,custom_unique_id,timezone,remaining_time}.py
    schemas/                        # pydantic request/response models (StreamStartRequest, Match, Schedule, ...)
    templates/streaming/             # Jinja2 HTML for the YouTube-auth flow (redirect/success/error/streams)

  utils/                            # ad-hoc one-off scripts, NOT the `app/core/utils` or `src/utils` packages
    logging.py                       # configure_logging() — RotatingFileHandler + console, called once from app/api/v1/__init__.py
    standbyme.py                      # hardcoded-IP script: sends one raw VISCA "standby" command and exits
    import_camera_visca_parameter.py   # hardcoded-IP script: queries + prints one camera's pan/tilt in degrees
```

Note there are **three same-purpose-but-distinct `utils` locations** — `src/utils/` (used by the
CLI/camera-system path), `app/core/utils/` (used by the FastAPI scheduling subsystem), and the
top-level `utils/` (one-off scripts plus `configure_logging`) — they are not variants of each other,
each is imported only from its own subtree except `utils/logging.py`, which `app/api/v1/__init__.py`
imports as a bare top-level module (relies on the process being launched with the repo root on
`sys.path`, e.g. via `uvicorn app.main:app` from the repo root — not `pip install`able as a package,
there is no `[project]` table in `pyproject.toml`).

## Runtime architecture

### `CameraSystem` (`src/camera/camera_system.py`)

The orchestrator, constructed once per recording/stream session (`CameraSystem(config, stream_token=None)`).
On `__init__`:

- Creates an output directory `{config.out_path}/{YYYYMMDD_HHMMSS}` for this session.
- Builds a `Game(config.bev)` (see below) for BEV projection/tracking.
- If `camera_system.pano_camera.enable`, constructs a `PanoCamrera` — a `multiprocessing.Process` — wired
  to a bounded (`maxsize=1`) shared `Queue` and a shared stop `Event`.
- For every enabled entry under `camera_system.cameras.ptz*`, looks up a class by
  `cfg.name` in `src.camera.ptz_camera` (currently only `Avonic_CM93_NDI` exists) and constructs it —
  also a `multiprocessing.Process` — each with its own move-command queue/event.
- Loads the RT-DETRv2 ONNX model (`camera_system.pano_onnx`) via `onnxruntime.InferenceSession`
  (CUDA, falling back to CPU) **in the parent process** — every PTZ/pano camera runs in its own OS
  process, but detection runs in the main process's `_detect_and_track` thread, not a subprocess.

`start()` starts every camera's process and spawns `_detect_and_track` as a background thread in the
parent process — it requires a `pano` camera (`raise RuntimeError("No Panorama Camera.")` otherwise;
there is no player-detection path without the panorama feed). `stop()` sets the shared stop event and
joins every camera process (5s timeout each) plus the detect thread.

`is_running()` / `are_cameras_healthy()` — polled by `manual_control_router`'s
`GET /get-stream-status` — check the stop event and, for every camera, whether its process is alive
and whether its `ffmpeg`/`ffmpeg_stream` subprocess (if any) is still running; any dead FFmpeg process
flips health to `False` even if the Python process itself is still alive.

### `_detect_and_track` loop (runs at ~5 Hz, `sleep_time = 1/5`)

Per iteration:

1. Pull the newest panorama frame from `pano_queue` (non-blocking with a 0.5s timeout; `None` if
   nothing arrived — the loop tolerates a starved queue rather than blocking indefinitely).
2. Run the ONNX detector (`_transform`: resize to 640×640, normalize, HWC→CHW, batch dim). Output is
   `(labels, boxes, scores)` — an RT-DETRv2-style postprocessed head, 3 classes: `1=Goalkeeper,
   2=Player, 3=Referee` (see `src/utils/visualize.py`'s `class2str`).
3. `game.update(labels, boxes, scores)` — projects detected box bottom-centers to BEV court
   coordinates via a fixed homography, associates them against existing Kalman tracks (Hungarian
   assignment, Euclidean gating), and manages track lifetime (see `Game`/`Track` below). Only
   `label == 2` (Player) detections above `score > 0.4` become BEV points — goalkeepers and referees
   are detected but never tracked/drawn.
4. Computes a per-track *direction* vote: any confirmed track moving faster than `0.35` court-units/s
   contributes +1 (moving right) or −1 (moving left) to a `direction` tally. If the tally exceeds ±2, the
   average position of the majority-direction tracks becomes a `cluster_center`; if that center is
   within 3 units of one of two fixed "border" x-positions (`±court_width/6`), the target camera pan
   position snaps to a fixed preset slot (`cam_pos ∈ {-9, 0, 9}`) — **this is a simple 3-position
   camera-blocking heuristic** (left/center/right), not continuous tracking of the ball or a centroid.
5. Pushes `(pan_pos, 0)` (tilt fixed at 0) onto each PTZ camera's move queue — `ptz1` uses `cam_pos`
   directly, `ptz2` uses `-cam_pos` (the two PTZ cameras face opposite directions across the pool, so
   their pan targets mirror).
6. Composites a debug frame (annotated panorama on top, BEV minimap below) — this is only ever held in
   memory / written to a (currently commented-out) `cv2.VideoWriter`; nothing consumes it in the FastAPI
   path today.

### `PanoCamrera` (`src/camera/pano_camera.py`, class name misspelled — not `PanoCamera`)

A `multiprocessing.Process`. Two possible input modes selected by substring match on `config.src`:
`rtmp`/`rtsp` → spawns an `ffmpeg` subprocess (`-rtsp_transport tcp`, low-latency flags, downscaled/FPS'd
via `-vf`) and reads raw BGR frames off its stdout; `mp4` → a plain `cv2.VideoCapture` (used for
config-driven playback/testing against a recorded file). Every frame is optionally lens-undistorted
(`cv2.remap` using precomputed `map_x`/`map_y` from the venue's `remap_*.npz`), optionally cropped to
the configured court ROI, optionally piped to a **second** `ffmpeg` process (NVENC h264, `save=True`)
for local MP4 recording, and always pushed (non-blocking, overwrite-oldest) onto the shared
single-slot `Queue` the detect/track loop reads from.

Self-healing: if 5 consecutive empty reads occur (`empty_frame_count >= efc_threshold`), it kills and
restarts the ingest `ffmpeg` process (`ffmpeg_restart`) — handles the RTSP source dropping without
requiring the whole `CameraSystem` to be restarted.

### `PTZCamera` / `Avonic_CM93_NDI` (`src/camera/ptz_camera.py`)

Also a `multiprocessing.Process`. Two concerns running concurrently:

- **Video path**: connects to the camera's NDI source (`ndi.recv_create_v3`, matched to the configured
  IP), pulls frames via `get_frame_with_timeout`, and depending on config writes them to a local NVENC
  `ffmpeg` recorder (`config.save`) and/or an RTMP `ffmpeg` push to
  `rtmp://a.rtmp.youtube.com/live2/{stream_token}` (`config.stream`) with a synthetic silent audio
  track (`anullsrc`) muxed in. If frames stop arriving, it repeats the last frame to the RTMP stream
  (up to whatever cadence keeps `target_frame_time`) rather than stalling the stream — a deliberate
  choice to avoid the YouTube ingest dropping the broadcast on a brief NDI hiccup. Both FFmpeg
  subprocesses' exit codes are polled every loop iteration; a dead process sets `is_healthy = False`
  and (for local recording) breaks the loop entirely.
- **Pan path**: a separate daemon thread (`_move_thread`) polls `queue_move` for new pan targets and
  issues VISCA "continuous pan" commands (direction + eased speed) toward the target, using
  `Avonic_CM93_NDI.PAN_SPEEDS`/`TILT_SPEEDS` — lookup tables converting the 24 discrete VISCA speed
  codes to measured degrees/second for this specific camera model (`PAN_TIMES`/`TILT_TIMES` are
  empirically measured full-sweep durations, not computed). `_build_pan_command` snaps to a "stop"
  command once within 20 units of the destination and re-latches `start_pan` whenever the direction of
  travel reverses (so easing distance is measured from the most recent direction change, not the
  session start).
- `move_with_easing`/`move` (cubic ease-in/ease-out **absolute**-position VISCA moves) exist for preset
  recall (`power_on` calls `move_to_preset` — though note `move_to_preset` itself is referenced but not
  defined anywhere in this file; calling `power_on()` would raise `AttributeError`) but the live
  auto-tracking path (`_move_thread`) only ever issues *continuous*-pan commands, never absolute moves.
- `restart_and_stream` is a Vivotek-camera-specific (not Avonic) HTTP `/ajaxcom` control sequence
  (enable audio, reboot, wait, then configure `StreamPublish` to push RTMP) — it is defined on the base
  `PTZCamera` class but only makes sense for the panorama camera hardware; nothing currently calls it.

### `visca.py` — VISCA-over-UDP

Minimal implementation of the subset of the Sony VISCA protocol these cameras need:
`send_command`/`send_inquiry` (raw UDP datagrams, ACK/Completion byte matching), `power_on`/`power_off`
(checks current power state via inquiry before sending the toggle, to avoid re-sending to an
already-correct state), `get_camera_pan_tilt` (parses the 4-nibble-per-axis inquiry reply, with the
`+65536 if < 3000` wraparound correction used throughout the codebase to normalize VISCA's signed
16-bit position value into an unsigned range consumers expect).

### `Game` / `Track` (`src/game.py`)

- **Homography**: `Game.__init__` computes `H` once via `cv2.findHomography(image_points, world_points)`
  from the venue config's `bev.points` — a set of manually-surveyed image-pixel ↔ court-meter
  correspondences (each venue config has a different point set, matching that pool's panorama camera
  mounting position/FOV). `_project_boxes` maps a detection box's **bottom-center** (assumed foot
  position) through `H` to court (x, y) meters; `project_to_bev` additionally masks out projections
  falling outside `court_size/2` (rejects spurious detections that homography-project to nonsense
  locations, e.g. stands/ceiling).
- **Tracking**: `Track` wraps a 4-state (`x, y, vx, vy`) constant-velocity `filterpy.kalman.KalmanFilter`
  (`dt=1/5` matching the 5 Hz loop) plus a `confidence` counter (0–10, +2 per matched update, −1 per
  missed predict, `TENTATIVE→CONFIRMED` once confidence exceeds 4, removed once confidence hits 0 — no
  `DEAD` state is ever actually assigned in `Game.update`, tracks are just spliced out of the list once
  confidence bottoms out). `Game.associate` does per-frame Hungarian assignment (`linear_sum_assignment`
  on a Euclidean cost matrix, gated: costs `>2` are set to `1e9` before solving, and a solved pair is
  only accepted if its cost is `≤1.5` — the `>2`/`1e9` masking makes the `2` threshold effectively the
  real gate, `1.5` is a slightly tighter final filter on top of it). Unmatched detections spawn new
  tracks; only `tracks` filtered to `CONFIRMED` status are exposed to the panning heuristic and BEV
  draw calls — a track needs several consecutive hits before it can influence camera movement, damping
  single-frame false positives.
- `get_pan_from_bev` converts a BEV x-coordinate target into an absolute VISCA pan (and fixed tilt)
  position, using the *camera's own* preset `left`/`right`/`center` VISCA values (from the venue config)
  converted to degrees (`visca_to_euler`) to establish that camera's actual pan range, then
  interpolating (`calc_pan_shift`) proportionally to how far across the court width the target is.
- `draw_court`/`draw`/`draw_detections`/`coord_to_px` — an OpenCV-rendered scale drawing of a water polo
  court (2m/5m lines, goal areas, corner boxes) used only for the debug composite frame in
  `_detect_and_track`; not exposed via any API endpoint.

### FastAPI service (`app/`)

`app/main.py` builds the `FastAPI` app, mounts `app.api.router` (prefix `/api`), and serves a single
HTML page at `/` (`redirect.html` — a leftover from the YouTube-auth flow; the YouTube router that would
receive the redirect is not mounted, see below). `app.api.v1.__init__` is the **actual routing surface
of record** and mounts exactly three routers:

| Router | Prefix | Auth | Purpose |
|---|---|---|---|
| `version_router` | `/api/v1/version` | none | `GET` returns the FastAPI app's `version` string |
| `manual_control_router` | `/api/v1/manual_control` | `user_or_admin_auth` | `POST /start-stream`, `POST /stop-stream`, `GET /get-stream-status` — the live control surface |
| `database_router` | `/api/v1/database` | `admin_only_auth` | `GET /get-all-matches` — reads the SQLite match log |

`schedule_router` and `youtube_router` are **imported and defined but explicitly commented out** in
`app/api/v1/__init__.py` — see [Gotchas](#gotchas--known-issues) for the two entire dormant subsystems
this leaves in the tree.

**`ManualControlRouter`** (`app/api/v1/manual_control_router.py`) is a stateful singleton instantiated
once at import time, holding a single `Optional[CameraSystem]` (so the process supports exactly one
active stream at a time — starting a second while one is running raises `RuntimeError`, surfaced as
HTTP 500). `POST /start-stream` accepts a `StreamStartRequest` (match metadata + a required
`stream_token`), always loads `./configs/bvsc_config.yaml` (hardcoded — **this router is
venue-pinned to BVSC**, not parameterized by the request), constructs and starts a `CameraSystem`, and
persists the match to both `metadata.txt` (append-only tab-separated log) and a SQLite `matches` table
(`Database.insert_match`, `metadata.db`) — errors from either persistence call are only logged
(`logging.error`), never surfaced to the caller, so a start-stream response of `200` does not guarantee
the metadata was actually recorded.

**Auth** (`authenticator.py`): a single `X-API-Key` header checked against `ADMIN_API_KEY`/`USER_API_KEY`
environment variables loaded once from `app/api/v1/.env.production` (not present in this checkout — must
be provisioned per-deployment; the `Dockerfile` explicitly `COPY`s it in). `admin_only_auth` accepts only
the admin key; `user_or_admin_auth` accepts either. There is no per-request rate limiting, token
expiry, or user identity beyond "matches one of two shared secrets."

## Configuration (`src/config.py`)

Plain-YAML, hand-parsed into typed wrapper classes (no pydantic here, unlike the FastAPI schemas) —
`load_config(path) -> Config`:

- **`ScheduleConfig`** — `start_time`/`end_time`/`duration` with a fairly permissive parser:
  `start_time`/`end_time` accept either `"HH:MM"` (assumed today) or `"YYYY.MM.DD_HH:MM"`; if only
  `duration` (`"HH:MM"`) is given, `end_time = start_time + duration`; if none are given, defaults to a
  1h45m session starting now (`"01:45"` — a full water-polo match length). Only actually consumed by
  the CLI (`main.py`) path — the FastAPI path has no notion of session duration, a stream runs until
  explicitly stopped via `/stop-stream`.
- **`PTZConfig`** — per-PTZ-camera: `ip`, `visca_port`, `presets` (VISCA `left`/`center`/`right`
  pan/tilt pairs, hex or decimal in YAML), `speed` (VISCA speed code), `codec`/`bitrate`/`fps` for local
  recording, `stream`/`ptz_stream`/`save` flags. `camera_params` (K/dist `.npy` pair) is loaded but
  **`PTZCamera` never actually uses `self.config.camera_params`** — it's parsed for symmetry with
  `PanoramaConfig` but there is no PTZ-side undistortion in this codebase.
- **`PanoramaConfig`** — `src` (RTSP URL or `.mp4` path), `crop` (`[top_y, top_x, bottom_y, bottom_x]`),
  `frame_size`, `fps`, and **two different, redundant calibration mechanisms**: `camera_params` (a
  legacy `.npy` K/dist pair, with a hardcoded `K[:2] *= 0.92753623188` scale correction baked into the
  loader — presumably a resolution mismatch fixup between calibration-time and capture-time frame size)
  and `distort` (an `.npz` with precomputed `map_x`/`map_y` for direct `cv2.remap`, the path actually
  wired into `PanoCamrera.undist_image`). In practice only one of the two is ever set per venue config
  (see table below) — `camera_params` and `distort` are not meant to be combined.
- **`CameraSystemConfig`** — dispatches each `cameras.*` YAML entry to `PanoramaConfig` (if the key is
  literally `"pano"`) or `PTZConfig` (if `"ptz"` is a substring of the key — i.e. `ptz1`, `ptz2`, ...,
  arbitrarily many), plus `pano_onnx` (detector weights path) and `track_threshold` (loaded but not
  read anywhere in `Game`/`CameraSystem` today — dead config field).
- **`BEVConfig`** — `points.world`/`points.image` (paired arrays for `cv2.findHomography`), `court_size`
  (default `[25, 20]` meters — FINA-regulation water polo court), `court_padding`.

### Venue configs compared

| Config | Pano calibration | Pano `fps` | PTZ1 `stream` | Notable |
|---|---|---|---|---|
| `default_config.yaml` | `camera_params` (legacy `.npy`) | 20 | `True` | generic/example, also the one `main.py --config` defaults to |
| `bvsc_config.yaml` | `distort` (`.npz` remap) | 5 | `True` | used by `manual_control_router` (hardcoded) |
| `fradi_config.yaml` | `distort` (`.npz` remap) | 5 | `False` | used by the dormant `streaming.py` YouTube router (hardcoded) |
| `komjadi_config.yaml` | none set | — | `True` | shorter/incomplete BEV point set in the copy read during this review |

Every venue has its own `bev.points` (image↔world correspondences) and often a different
`camera_params`/`crop`/`fps`, because each pool has a physically different panorama camera mounting
position and lens — **configs are not interchangeable between venues**; running `bvsc_config.yaml`
against the Fradi camera would produce a nonsensical BEV homography.

## Persistence

Two independent SQLite databases plus one text log, all opened relative to the process's **current
working directory** (not the repo root explicitly) — must run `uvicorn` from the repo root for the
paths in code to resolve where the `Makefile`'s docker `run` target expects:

| File | Written by | Schema |
|---|---|---|
| `metadata.db` (referenced as `Database._database_path` in `database_router.py`; the tracked `database.db` in this checkout appears to be a stale/differently-named artifact from a prior run) | `Database.insert_match` on every `/start-stream` call | `matches(id, division, league, home_team, away_team, playing_field, scheduled_match_time)`, `UNIQUE` on the 6 content columns (so re-submitting the same match is a silent no-op via `INSERT OR IGNORE`) |
| `stream_schedules.db` | `app/api/v1/streaming.py` (dormant router, but its module-level `init_db()`/`restore_scheduled_streams()` still run on import if that module is ever imported) | `stream_schedules(stream_id, stream_token, start_time, end_time, status)` |
| `metadata.txt` | `Database.save_to_txt` on every `/start-stream` call | tab-separated, header row written once, one line per match started |

## Deployment

`Dockerfile` builds on `nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04` and **compiles FFmpeg 4.1.11 from
source** with `--enable-nvenc --enable-cuvid --enable-libnpp` (NVIDIA hardware encode/decode) — the
prebuilt Ubuntu FFmpeg package does not have NVENC support, which every camera process's local-recording
and RTMP-streaming path depends on (`h264_nvenc` codec). Application code itself is **not** baked into
the image (the `COPY main.py` / `COPY app` / etc. lines are commented out) — it's bind-mounted at
`docker run` time (`-v ./:/app/`), so the image only needs rebuilding when dependencies or the OS layer
change, not on every code change.

`make run` / `make run-xhost` (`Makefile`) run the container with `--network host` (required for NDI
mDNS/Avahi discovery of the PTZ cameras, and for RTSP to the panorama camera) plus bind-mounts for
`/var/run/dbus` and `/run/avahi-daemon/socket` (host's Avahi daemon, since NDI discovery needs
`avahi-daemon` running and reachable — the container does *not* run its own, despite installing the
package) and `/srv/sftp/RECORDINGS:/app/output/` (recordings land on a host path presumably served over
SFTP for offload). `--gpus all --runtime=nvidia` for the CUDA/NVENC dependency chain above.
`setup.sh` is the one-time host bootstrap for a fresh Ubuntu box: NVIDIA driver 560, Docker CE, and
`nvidia-container-toolkit`, ending in a reboot.

## Commands

```bash
# CLI mode: run for the configured schedule duration then stop
python main.py --config configs/bvsc_config.yaml

# CLI mode: run daily at the times in configs/schedule.yaml
python main.py --config configs/bvsc_config.yaml --scheduled

# FastAPI mode (what Docker actually runs)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload   # from repo root

# Start/stop a stream via the API (BVSC venue only, see manual_control_router above)
curl -X POST 'http://localhost:8000/api/v1/manual_control/start-stream' \
  -H 'X-API-Key: <ADMIN_or_USER_key>' -H 'Content-Type: application/json' \
  -d '{"home_team": "BVSC", "away_team": "FTC", "stream_token": "<youtube-rtmp-key>"}'
curl -X POST 'http://localhost:8000/api/v1/manual_control/stop-stream' -H 'X-API-Key: <key>'

# Lint / format
make lint      # flake8 .
make format    # black . && isort .

# Docker build/run
make build
make run       # or run-xhost for an X11-forwarded debug window
```
