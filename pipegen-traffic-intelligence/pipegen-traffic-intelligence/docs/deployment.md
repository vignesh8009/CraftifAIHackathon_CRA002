# NVIDIA deployment (DeepStream + TensorRT)

Your Windows laptop (no known NVIDIA GPU) **cannot** run this stage. Installing WSL/CUDA/TensorRT does not make integrated graphics DeepStream-compatible. Pick one:
- **Cloud GPU VM** (Ubuntu + NVIDIA driver, e.g. a T4 instance) – quickest for a hackathon; rent by the hour.
- A friend's/lab PC with an NVIDIA GPU + Ubuntu, or a **Jetson** (Orin Nano/NX/AGX with JetPack).

Choose a DeepStream release first, then match driver/CUDA/TensorRT to its compatibility table on NVIDIA's DeepStream docs (versions are coupled – do not mix). Record the versions you end up with in the table at the bottom.

## 1. Verify the GPU  (on the NVIDIA machine)
```bash
nvidia-smi            # dGPU: must list a GPU + driver.   Jetson: use `sudo tegrastats`
```
## 2. Get DeepStream (container is easiest on a dGPU server)
```bash
docker run --gpus all -it --rm --net=host -v $PWD:/workspace -w /workspace \
  nvcr.io/nvidia/deepstream:<TAG>-triton-multiarch      # pick <TAG> for your release on NGC
```
Needs Docker + NVIDIA Container Toolkit (`docker run --gpus all ubuntu nvidia-smi` must work). On Jetson install via NVIDIA's JetPack/DeepStream instructions instead.
## 3. Copy the project there
`git clone` your repo (or `scp -r`) and `cd pipegen-traffic-intelligence`. Put a video in `videos/traffic.mp4`.
## 4. Python bindings (pyds)
Install the `pyds` wheel that matches your DeepStream version from NVIDIA's *deepstream_python_apps* releases, then:
```bash
python3 -c "import pyds, gi; print('pyds ok')"      # expect: pyds ok
gst-inspect-1.0 nvinfer | head -3                    # expect plugin details, not "No such element"
```
## 5. Export the model and build the YOLO parser
```bash
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git       # in project root
pip install ultralytics onnx onnxslim
cp DeepStream-Yolo/utils/export_yoloV8.py . && python3 export_yoloV8.py -w yolov8n.pt --dynamic   # or follow that repo's current README
mv yolov8n.onnx models/
export CUDA_VER=<your CUDA major.minor>     # e.g. 12.2 ; Jetson: use the JetPack CUDA version
make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
```
The repo's instructions change between DeepStream versions – follow its README for yours. `config_infer_primary.txt` expects `../DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`.
## 6. Smoke test with NVIDIA's own app  (from `deepstream/`)
Edit the `uri` in `deepstream_app_config.txt`, then:
```bash
cd deepstream && deepstream-app -c deepstream_app_config.txt
```
First run builds the TensorRT engine (minutes; the `.engine` file appears in `models/`). Expected: periodic `**PERF: xx.xx` FPS lines. This proves GPU + TensorRT + tracker work.
## 7. Run our pipeline  (from the project root)
```bash
python3 deepstream/run_pipeline.py --source videos/traffic.mp4
```
Expected: `ALERT ...` lines when congestion is triggered, then `Done: N frames ...`. Output: `outputs/annotated.mp4`, `outputs/metrics.json`, `outputs/events.db`.
## 8. Dashboard
```bash
pip install -r requirements.txt && streamlit run dashboard/app.py --server.address 0.0.0.0
```
Open `http://<machine-ip>:8501` (open the port on a cloud VM, or SSH tunnel `ssh -L 8501:localhost:8501 user@host`). The badge must read **NVIDIA DeepStream + TensorRT**.

## Measuring performance (do not invent numbers)
- FPS: the `**PERF` lines from step 6 (steady state) and the dashboard FPS.
- Utilisation: `nvidia-smi dmon -s u` (dGPU) or `tegrastats` (Jetson) during the run.
- Write results + test conditions (video, resolution, FP16/FP32, GPU) in the table below.

| Item | Value |
|---|---|
| GPU / Jetson model | |
| Driver / JetPack | |
| CUDA | |
| TensorRT | |
| DeepStream | |
| Model + precision | |
| Video + resolution | |
| Steady-state FPS | |
| GPU util / memory | |

## Troubleshooting
| Symptom | Likely cause / fix |
|---|---|
| `nvidia-smi` fails / Docker can't see GPU | Driver or NVIDIA Container Toolkit missing; reboot after install. |
| `No such element nvinfer` / missing plugins | Not inside the DeepStream container/install; `rm -rf ~/.cache/gstreamer-1.0` and retry; check `gst-inspect-1.0`. |
| `ImportError: pyds` | Wheel doesn't match DeepStream/Python version (step 4). |
| `Failed to build CUDA engine` / parser errors | ONNX export not from DeepStream-Yolo script, or `CUDA_VER` wrong when running `make`; TensorRT too old for the model. |
| Engine fails to load on another machine | Engines are GPU/TensorRT-specific – delete the `.engine` and let it rebuild. |
| Boxes but no/odd track IDs | Tracker config path wrong (`--tracker-config`); look under `$DS/samples/configs/deepstream-app/`. |
| Boxes at wrong positions in analytics | Mux size must equal `MUX_W/MUX_H` in `run_pipeline.py`. |
| Zero detections | `pre-cluster-threshold` too high, wrong `num-detected-classes`, or `filter-out-class-ids` mismatched with `labels.txt`. |
| `Error: gst-resource-error` / can't open video | Bad path/codec; test with `ffprobe`; use absolute path; H.264 MP4 is safest. |
| Dashboard says "stalled" | Pipeline stopped or `--out` dir differs from the dashboard's (`PIPEGEN_OUT` env var). |
