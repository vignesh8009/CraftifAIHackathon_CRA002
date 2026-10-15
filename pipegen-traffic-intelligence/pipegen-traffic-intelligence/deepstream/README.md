# DeepStream files
- `config_infer_primary.txt` – nvinfer (TensorRT) config for YOLOv8 + DeepStream-Yolo parser; filters to car/motorcycle/bus/truck.
- `deepstream_app_config.txt` – NVIDIA `deepstream-app` smoke test (validates GPU, engine, tracker, FPS) before using our Python app.
- `run_pipeline.py` – our pipeline: decode → mux → nvinfer → nvtracker → analytics probe → annotated MP4 + metrics for the dashboard.
Full steps and troubleshooting: `../docs/deployment.md`.
