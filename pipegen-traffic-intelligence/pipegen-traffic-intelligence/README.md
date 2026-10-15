# PipeGen Traffic Intelligence
Vehicle detection → tracking → line counting → region congestion alerts → live dashboard.
Edge pipeline: **NVIDIA DeepStream + TensorRT**. A CPU/PyTorch prototype and a simulator let you develop on Windows first.

## Honest status
| Part | Status |
|---|---|
| Analytics core (`core/`): counting, ROI, congestion, config, store | **Tested** – `python -m unittest discover -s tests` (28 tests passing when written) |
| Simulator → metrics/alerts files | **Run** end-to-end; produced Normal→Critical→Normal alerts |
| Streamlit dashboard, YOLO prototype | Written, syntax-checked only. **Not run yet** – you will run them first on Windows |
| DeepStream/TensorRT pipeline | Written, **not executed on NVIDIA hardware**. Needs your GPU machine; expect small fixes for your DeepStream version |
| Accuracy / FPS / latency | **Not measured.** Fill in `docs/deployment.md` after real runs |

Never claim TensorRT/GPU results unless the dashboard badge says *NVIDIA DeepStream + TensorRT*.

## A. Windows 11 (development)  – all commands from the project root
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-prototype.txt
python -m unittest discover -s tests          # expect: OK
```
(If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process Bypass`.)

**1. Dashboard + simulator** (no model needed; badge shows SIMULATED):
```powershell
# terminal 1
streamlit run dashboard/app.py
# terminal 2
python prototype/simulate.py --speed 5
```
**2. Real detection on CPU** – put a clip in `videos/` (see `videos/README.md`):
```powershell
python prototype/detect_track.py --source videos/traffic.mp4 --show
```
First run downloads `yolov8n.pt`. Expect ~3–15 FPS on CPU. Badge reads *prototype-python (cpu) – not TensorRT*.
Errors you may see: `Video file not found` (check path), `Unsupported video format`, `Invalid JSON in config`.

**Tune the lines/zones**: edit `config/analytics_config.json` (coordinates are 0–1 fractions of the frame; `warning`/`critical` are vehicle counts per region). Rerun.

## B. NVIDIA deployment
See **docs/deployment.md** (GPU check, DeepStream, ONNX export, engine build, smoke test, our pipeline, troubleshooting, results table).

## Layout
`core/` shared analytics · `prototype/` Windows prototype + simulator · `deepstream/` TensorRT/DeepStream · `dashboard/` Streamlit · `config/` · `tests/` · `docs/`

## Limitations
Count-based congestion estimate (not speed or forecast); occlusion/overlap, camera angle, night/rain and ID switches affect counts; tracking IDs may switch and cause double counts; one camera per run; vehicles only, no people or faces are identified.
Use only authorized footage and record sources in `videos/README.md`.
