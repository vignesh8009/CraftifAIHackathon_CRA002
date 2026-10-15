# Models
Model files are NOT committed. Two different files are used:
- `yolov8n.pt` – PyTorch weights for the Windows prototype (ultralytics downloads it automatically on first run).
- `yolov8n.onnx` – ONNX export for DeepStream, created on the NVIDIA machine (see `docs/deployment.md` step 5).
- `*.engine` – TensorRT engine, built by DeepStream on first run **on the same GPU/TensorRT version that will run it**. Engines are not portable between GPUs or TensorRT versions.
- `labels.txt` – the 80 COCO class names (committed). Vehicle IDs: car=2, motorcycle=3, bus=5, truck=7.
