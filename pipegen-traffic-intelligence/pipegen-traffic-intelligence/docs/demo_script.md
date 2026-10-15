# Demo script (≈3 min)
Prepare: DeepStream run finished once so the engine is cached; dashboard open; busy clip chosen.
1. Show the dashboard badge and Environment expander: GPU, driver, DeepStream version (real, reported by the pipeline).
2. `python3 deepstream/run_pipeline.py --source videos/traffic.mp4` – narrate the architecture diagram.
3. Open the annotated video: boxes, class labels, track IDs, red counting line, yellow region.
4. Point to a vehicle crossing the line; show the counter rising and "by class" chart.
5. As the busy section arrives, show region occupancy hit the warning threshold, then **Warning** after the persistence window.
6. Open the alert row: it states the rule (`count 9 >= warning threshold 8 for 3s`) and timestamp.
7. Show FPS and the measured table from `docs/deployment.md`. State limitations honestly (occlusion, camera angle, night, ID switches).
Fallback if the GPU box is down: run `prototype/simulate.py` – the badge will say **SIMULATED**; say so out loud.
