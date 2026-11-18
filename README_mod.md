# YOLO object detection + object memory on the Go2 (Jetson, Docker)

A ROS 2 Humble container that runs on the Go2's extension board (Jetson Orin,
L4T R35.3.1 / JetPack 5.1.1). It subscribes to the RealSense topics that the
robot already publishes, runs a fine-tuned YOLO model (TensorRT FP16 `.engine`),
and publishes:

| Topic | Type | What |
|---|---|---|
| `/go2/yolo/image_annotated/compressed` | `sensor_msgs/CompressedImage` | Camera frame with boxes drawn (JPEG) |
| `/go2/yolo/detections` | `vision_msgs/Detection2DArray` | All boxes, classes, scores |
| `/yolo/target_pose` | `geometry_msgs/PoseStamped` | 3D target position in `TARGET_FRAME` (see below) |
| `/yolo/static_objects` | `visualization_msgs/MarkerArray` | Remembered **static** objects (cube + `id (n=…)` label), 1 Hz |
| `/yolo/dynamic_objects` | `visualization_msgs/MarkerArray` | **Dynamic** detections of the current frame (short-lived spheres) |

Services: `/yolo_subscriber_node/clear_static_memory`, `/yolo_subscriber_node/save_static_memory` (`std_srvs/Trigger`).

The container does not open the camera itself. It needs the RealSense driver
running on the robot with depth aligned to color
(`enable_depth:=true align_depth.enable:=true`).

## Repository layout

| Path | Purpose |
|---|---|
| `yolo_subscriber_node.py` | The ROS 2 node (inference, 3D back-projection, TF, markers) |
| `object_memory.py` | Static object memory + dynamic placeholder (pure Python, no ROS) |
| `config/object_classes.yaml` | Which classes are static / dynamic, and tuning per class |
| `run_yolo.sh` | **Main launcher** on the robot (all settings via env vars) |
| `Dockerfile.subscriber`, `entrypoint.sh` | Image build |
| `models/` | Weights (`*.pt`, `*.engine`) + the onnxruntime-gpu Jetson wheel used by the build |
| `memory/` | Saved static objects (created at runtime, git-ignored) |
| `tools/capture_dataset.py` | Capture a RealSense image dataset for training |
| `notebooks/` | `01` annotate/augment/prepare, `02` train/evaluate/export |
| `launch.sh` | Legacy launcher (COCO `yolov8n.pt`, no depth/memory); use `run_yolo.sh` |

## Build (on the Go2)

```bash
cd ~/go2_yolo_docker
docker build -f Dockerfile.subscriber -t go2-yolo:latest .
```

- The first build compiles torchvision from source (20–40 min). Later builds reuse that cached layer.
- BuildKit is required (Docker ≥ 23 has it on by default; otherwise prefix with `DOCKER_BUILDKIT=1`).
- `models/onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl` must exist. It is not in git (39 MB).
- The image pins `numpy==1.23.5`. TensorRT 8.5 (JetPack 5) breaks on numpy ≥ 1.24 (`np.bool`),
  which shows up as a misleading `'str' object has no attribute 'names'` when loading an `.engine`.
- `yolo_subscriber_node.py`, `object_memory.py` and `config/` are copied into the image.
  **Rebuild after changing the Python files** (it takes seconds; only the last layers rebuild).
  `config/` is also mounted at runtime by `run_yolo.sh`, so config edits only need a container restart.

## Export the model to TensorRT FP16 (on the Go2, once per model)

An engine only works with the GPU and TensorRT version it was built with.
**Build it on the robot, inside this image**, not on a PC:

```bash
docker run -it --rm --runtime nvidia \
  --entrypoint yolo \
  -v "$(pwd)/models:/workspace/models" \
  go2-yolo:latest \
  export model=/workspace/models/go2_ext_bluebox_best.pt \
         format=engine half=True imgsz=640 device=0 workspace=4
```

`imgsz` must match the training size. The export writes `models/go2_ext_bluebox_best.engine`.

## Run

```bash
./run_yolo.sh                              # defaults below
TARGET_FRAME=odom ./run_yolo.sh            # no SLAM/map running yet
TARGET_CLASS=blue_box ./run_yolo.sh        # stable target pose from memory
HOST_MODEL_PATH=$(pwd)/models/other.engine ./run_yolo.sh
```

Main settings (environment variables of `run_yolo.sh`):

| Variable | Default | Meaning |
|---|---|---|
| `HOST_MODEL_PATH` | `models/go2_ext_bluebox_best.engine` | Model file on the robot (`.engine`, `.pt`, `.onnx`) |
| `CONF_THRES` | `0.5` | Detection confidence threshold |
| `CLASSES_FILTER` | all | Comma-separated class **ids** to keep |
| `INPUT_TOPIC` | `/camera/color/image_raw` | Color image |
| `DEPTH_TOPIC` | `/camera/aligned_depth_to_color/image_raw` | Aligned depth (16UC1 mm or 32FC1 m) |
| `CAMERA_INFO_TOPIC` | `/camera/color/camera_info` | Intrinsics |
| `TARGET_FRAME` | `map` | Frame for poses **and** for the static memory |
| `FIXED_FRAME` | `odom` | Odometry frame for tf2 time-travel lookups; empty = off |
| `CAMERA_FRAME` | empty | Empty = use the image's `header.frame_id` |
| `PUBLISH_CAM_ON_FAIL` | `false` | If TF fails, publish the target pose in the camera frame |
| `OBJECT_CONFIG` | `/workspace/config/object_classes.yaml` | Static/dynamic class config |
| `MEMORY_FILE` | `/workspace/memory/static_objects_<TARGET_FRAME>.yaml` | Saved memory; `MEMORY_FILE=` disables |
| `TARGET_CLASS` | empty | Empty = target pose is this frame's best detection; a static class name = its nearest remembered instance |
| `ROS_DOMAIN_ID` | `0` | Must match the robot and the zenoh bridge |

## Static and dynamic objects

Edit `config/object_classes.yaml`. Keys are the class **names** the model
reports (they are logged at startup, with a warning for unknown names):

```yaml
default_type: dynamic          # classes not listed
classes:
  blue_box:
    type: static
    merge_radius_m: 0.5        # detection within this distance of a known instance = same object
    min_observations: 5        # sightings before an instance is confirmed / shown
    max_samples: 30            # position = median of the last N sightings
    unconfirmed_timeout_s: 10  # forget instances never confirmed within this time
  person:
    type: dynamic
```

- **Static**: every detection gets a 3D position and is matched to the nearest
  remembered instance of the same class within `merge_radius_m`. Two boxes in one
  frame are never merged. Instances get stable ids (`blue_box_1`, `blue_box_2`, …),
  stay visible when out of view, are saved every 10 s and on shutdown, and reload
  on restart.
- **Dynamic**: placeholder. Current-frame detections only, no tracking yet
  (`DynamicObjectTracker` in `object_memory.py` is where tracking will go).
- Memory is stored in `TARGET_FRAME`. Use `map` (needs SLAM/localization). `odom`
  works for testing, but odometry drift slowly smears positions. A saved file
  from a different frame is not loaded.

**Reset the memory**

```bash
# while running
docker exec -it yolo_go2 bash -c "source /opt/ros/humble/install/setup.bash && \
  source /opt/cyclonedds_ws/install/setup.bash && \
  ros2 service call /yolo_subscriber_node/clear_static_memory std_srvs/srv/Trigger"

# while stopped
rm ~/go2_yolo_docker/memory/static_objects_*.yaml
```

## Viewing / inspecting

The robot's own ROS 2 is **Foxy**; this container is **Humble**. Humble's
`visualization_msgs/Marker` has extra fields, so a Foxy `ros2 topic echo` /
rviz2 **segfaults** on `/yolo/static_objects` and `/yolo/dynamic_objects`.
Inspect those topics with a Humble environment: your laptop's Humble docker
(via zenoh), or inside the container:

```bash
docker exec -it yolo_go2 bash -c "source /opt/ros/humble/install/setup.bash && \
  source /opt/cyclonedds_ws/install/setup.bash && ros2 topic echo /yolo/static_objects --once"
```

In rviz2 (Humble), set Fixed Frame = `TARGET_FRAME`, then add:
- Image → `/go2/yolo/image_annotated/compressed` (compressed transport)
- MarkerArray → `/yolo/static_objects` and `/yolo/dynamic_objects`
- Pose → `/yolo/target_pose`

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `TF lookup failed ... "map" ... does not exist` | Nothing publishes `map` (start SLAM/AMCL), or `/tf` doesn't reach the container, or no `base_link→camera_link` static TF. Check with `ros2 run tf2_ros tf2_echo map camera_color_optical_frame`. For now use `TARGET_FRAME=odom` or `PUBLISH_CAM_ON_FAIL=true`. On the first failure the node logs the frames it can see. |
| Log still shows old messages after editing code | The node is copied into the image: sync the files to the robot and rebuild. |
| `'str' object has no attribute 'names'` with `.engine` | numpy ≥ 1.24 in the image (see Build), or the engine was built on another machine or TensorRT version. |
| Engine fails to deserialize | Re-export on the robot inside this image. |
| No topics at all | DDS/zenoh discovery: same `ROS_DOMAIN_ID`, `--network host`, Cyclone DDS (`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`). |
| Pose jumps between objects | Use static classes + `TARGET_CLASS=<name>`. |

Utilization on the Jetson: `sudo tegrastats`, `jtop` (`pip3 install jetson-stats`),
`sudo nvpmodel -q`, `docker stats yolo_go2`, `ros2 topic hz /go2/yolo/detections`.
