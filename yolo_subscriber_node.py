#!/usr/bin/env python3
"""
Subscribes to an existing RealSense image topic (published by your already-running
realsense node), runs YOLOv8n inference, and publishes:
  - annotated image (sensor_msgs/CompressedImage, JPEG) -> for rviz2 / rqt_image_view
  - structured detections (vision_msgs/Detection2DArray)
  - 3D target pose in map frame (geometry_msgs/PoseStamped) -> for the SWAP FSM,
    via depth back-projection at the best detection's bbox center

No RealSense SDK / pyrealsense2 dependency needed here anymore -- this node
just consumes whatever your camera driver is already publishing.

Output image is compressed JPEG rather than raw -- this container relays
over the zenoh bridge back to your laptop, so compression matters more
here than it would for a purely local/simulated pipeline.

REAL-HARDWARE DEPTH NOTES (ported from the sim's yolo_detector.py, which
assumed float-meters depth -- neither assumption holds here unmodified):
  - Depth topic here is /camera/aligned_depth_to_color/image_raw, which
    means it's already pixel-aligned to the COLOR image -- so the color
    camera's own intrinsics (camera_info_topic) apply directly, with no
    separate depth-to-color reprojection step needed.
  - RealSense aligned-depth topics are typically encoded 16UC1, i.e.
    integer millimeters, NOT float meters. This code checks msg.encoding
    and converts accordingly -- getting this wrong silently produces
    target poses ~1000x too far away, with no error thrown anywhere.
"""
import colorsys
import math
import time
import zlib
from collections import deque

import numpy as np
import cv2
import yaml

import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException

from ultralytics import YOLO

from object_memory import STATIC, DynamicObjectTracker, StaticObjectMemory, load_class_config

TF_FAIL_LIMIT = 10       # consecutive TF failures before the circuit breaker trips
TF_SKIP_S = 1.0          # ...then skip TF lookups this long (probe once per period)
TF_DIAG_PERIOD_S = 30.0  # at most one TF diagnosis report per this many seconds


def _stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def _set_position(pose, xyz):
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in xyz)


def _quat_to_rot(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _rot_to_quat(R):
    """Rotation matrix -> (x, y, z, w)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)


def _transform_to_rt(transform):
    """geometry_msgs TransformStamped -> (3x3 rotation, translation vector)."""
    q = transform.transform.rotation
    t = transform.transform.translation
    return _quat_to_rot(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])


class YoloSubscriberNode(Node):
    def __init__(self):
        super().__init__('yolo_subscriber_node')

        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('conf_thres', 0.5)
        self.declare_parameter('input_topic', '/camera/color/image_raw')
        self.declare_parameter('use_compressed', False)
        self.declare_parameter('image_topic', '/go2/yolo/image_annotated/compressed')
        self.declare_parameter('detections_topic', '/go2/yolo/detections')
        self.declare_parameter('jpeg_quality', 80)
        # Comma-separated class IDs to keep, e.g. "0" for the first class
        # only. Empty string = keep all classes the loaded model knows.
        # NOTE: this also gates which detections are eligible for
        # /yolo/target_pose below, via ultralytics' own `classes=` predict
        # arg -- result.boxes only ever contains already-filtered classes.
        self.declare_parameter('classes_filter', '')

        # --- 3D back-projection (new) ---
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('target_pose_topic', '/yolo/target_pose')
        self.declare_parameter('target_frame', 'map')
        # Empty = use the incoming image's header.frame_id (what the camera
        # driver actually publishes). Set explicitly only to override it.
        self.declare_parameter('camera_frame', '')
        # Fast-updating frame (odometry) used for tf2 "time travel": robot
        # motion is taken from <fixed_frame> at the exact image stamp, and
        # only the slow <target_frame>-><fixed_frame> correction (AMCL/SLAM)
        # uses its latest value. Empty = plain lookups, no time travel.
        self.declare_parameter('fixed_frame', 'odom')
        # If TF to target_frame fails, still publish the pose expressed in
        # the camera frame (instead of publishing nothing).
        self.declare_parameter('publish_in_camera_frame_on_tf_fail', False)
        # Sanity bounds on accepted depth readings (metres). Real sensors
        # regularly return 0 (no return) or nonsense far values at edges/
        # reflective surfaces -- reject rather than publish garbage poses.
        self.declare_parameter('min_valid_depth_m', 0.15)
        self.declare_parameter('max_valid_depth_m', 8.0)

        # --- static / dynamic object memory ---
        # YAML marking each class static (remembered, stable ids) or dynamic
        # (placeholder). See config/object_classes.yaml. Memory frame =
        # target_frame. Empty/missing file = every class dynamic.
        self.declare_parameter('object_config', '')
        # Where remembered static objects are saved/loaded. Empty = no persistence.
        self.declare_parameter('memory_file', '')
        self.declare_parameter('memory_save_period_s', 10.0)
        # Empty = /yolo/target_pose is the best detection of this frame (old
        # behavior). Set to a static class name = the remembered position of
        # the nearest confirmed instance of that class (stable, no jumping).
        self.declare_parameter('target_class', '')
        self.declare_parameter('static_objects_topic', '/yolo/static_objects')
        self.declare_parameter('dynamic_objects_topic', '/yolo/dynamic_objects')

        # --- pose filtering / measurement quality ---
        # Depth image used must be within this many ms of the color image
        # (picked by timestamp from a short buffer); otherwise skip 3D.
        self.declare_parameter('max_depth_dt_ms', 40.0)
        # Skip static-memory updates while the camera rotates faster than
        # this (rad/s): motion blur + TF latency give large errors.
        self.declare_parameter('max_cam_ang_vel', 0.6)
        # Measurement noise model (camera optical frame):
        #   sigma_depth   = depth_noise_a + depth_noise_b * z^2   [m]
        #   sigma_lateral = z * box_px_sigma / f                  [m]
        self.declare_parameter('depth_noise_a', 0.005)
        self.declare_parameter('depth_noise_b', 0.004)
        self.declare_parameter('box_px_sigma', 3.0)

        # --- TF robustness ---
        # Max age (image stamp - transform stamp) accepted for the camera->fixed_frame
        # hop (odometry, fast) and for the fixed_frame->target_frame hop (SLAM/AMCL,
        # slow). Older data is rejected instead of silently giving wrong poses.
        self.declare_parameter('max_tf_age_s', 0.5)
        self.declare_parameter('max_map_tf_age_s', 5.0)

        # --- instrumentation ---
        # Log per-stage timing / rates every N seconds. 0 = off.
        self.declare_parameter('stats_period_s', 10.0)

        model_path = self.get_parameter('model_path').value
        self.conf_thres = self.get_parameter('conf_thres').value
        input_topic = self.get_parameter('input_topic').value
        self.use_compressed = self.get_parameter('use_compressed').value
        image_topic = self.get_parameter('image_topic').value
        det_topic = self.get_parameter('detections_topic').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value

        classes_filter_str = self.get_parameter('classes_filter').value
        if classes_filter_str.strip():
            self.classes_filter = [int(c) for c in classes_filter_str.split(',')]
        else:
            self.classes_filter = None  # None = all classes

        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        target_pose_topic = self.get_parameter('target_pose_topic').value
        self.target_frame = self.get_parameter('target_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.fixed_frame = self.get_parameter('fixed_frame').value
        self.publish_cam_on_fail = self.get_parameter('publish_in_camera_frame_on_tf_fail').value
        self.tf_ok_logged = False
        self.min_valid_depth_m = self.get_parameter('min_valid_depth_m').value
        self.max_valid_depth_m = self.get_parameter('max_valid_depth_m').value
        object_config = self.get_parameter('object_config').value
        self.memory_file = self.get_parameter('memory_file').value
        memory_save_period_s = float(self.get_parameter('memory_save_period_s').value)
        self.target_class = self.get_parameter('target_class').value
        static_objects_topic = self.get_parameter('static_objects_topic').value
        dynamic_objects_topic = self.get_parameter('dynamic_objects_topic').value
        self.max_depth_dt = float(self.get_parameter('max_depth_dt_ms').value) / 1000.0
        self.max_cam_ang_vel = float(self.get_parameter('max_cam_ang_vel').value)
        self.depth_noise_a = float(self.get_parameter('depth_noise_a').value)
        self.depth_noise_b = float(self.get_parameter('depth_noise_b').value)
        self.box_px_sigma = float(self.get_parameter('box_px_sigma').value)
        self.prev_cam = None  # (rotation 3x3 in target_frame, stamp_s) for motion gating
        self.max_tf_age = float(self.get_parameter('max_tf_age_s').value)
        self.max_map_tf_age = float(self.get_parameter('max_map_tf_age_s').value)
        stats_period_s = float(self.get_parameter('stats_period_s').value)

        # TF circuit breaker: after TF_FAIL_LIMIT failures in a row, skip TF
        # lookups for TF_SKIP_S seconds so a broken TF tree can't slow inference.
        self.tf_fail_streak = 0
        self.tf_skip_until = 0.0
        self.tf_diag_next = 0.0
        self.tf_diag_hist = {}  # link -> (wall_time, age_s) from the previous diagnosis

        # Per-stage timing: name -> [count, total_s, max_s]; counters: name -> int
        self.stats = {}
        self.counters = {}
        self.stats_t0 = time.monotonic()

        self.get_logger().info(f'Loading YOLO model: {model_path}')
        # For real-time speed on Orin Nano, export once to TensorRT and point
        # model_path at the .engine file instead of the .pt:
        #   YOLO('yolov8n.pt').export(format='engine')
        self.model = YOLO(model_path)
        self.get_logger().info(
            f'Model classes: {self.model.names}. '
            f'Class filter: {self.classes_filter if self.classes_filter else "all classes"}'
        )
        self.bridge = CvBridge()

        # --- static / dynamic object memory ---
        self.class_cfg, cfg_found = load_class_config(object_config)
        summary, unknown = self.class_cfg.describe(self.model.names)
        self.get_logger().info(
            f'Object classes ({object_config if cfg_found else "no config file -> all " + self.class_cfg.default_type}):\n{summary}')
        if object_config and not cfg_found:
            self.get_logger().warn(f'object_config "{object_config}" not found')
        if unknown:
            self.get_logger().warn(f'Config classes not in the model (typo?): {unknown}')
        self.static_memory = StaticObjectMemory(self.class_cfg)
        self.dynamic_tracker = DynamicObjectTracker(self.class_cfg)
        if self.memory_file:
            try:
                n, err = self.static_memory.load(self.memory_file, self.target_frame)
                if err:
                    self.get_logger().warn(err)
                else:
                    self.get_logger().info(f'Loaded {n} static objects from {self.memory_file}')
            except Exception as e:  # noqa: BLE001 -- a bad file must not kill the node
                self.get_logger().error(f'Could not load {self.memory_file}: {e!r}')

        # Short buffer so the depth frame can be matched to the color frame by
        # timestamp (latest depth can be one frame off while the robot moves).
        self.depth_buf = deque(maxlen=15)
        self.camera_info = None

        self.tf_buffer = Buffer()
        # spin_thread=True: /tf is received on its own thread. Without it,
        # /tf callbacks share this node's single-threaded executor and can't
        # run while the (slow) inference callback is blocked waiting on a
        # lookup -- so the buffer goes stale and every timeout is wasted.
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # Two separate QoS profiles: BEST_EFFORT for subscribing (matches
        # the camera driver's conventional sensor-data QoS), and RELIABLE
        # for our own outgoing publishers. This matters: a RELIABLE-
        # requesting subscriber (rqt_image_view's default) will NOT match
        # a BEST_EFFORT publisher at all in DDS -- no error, just silence,
        # which looks exactly like "no output" with no clue why.
        sub_qos = QoSProfile(depth=5)
        sub_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sub_qos.history = HistoryPolicy.KEEP_LAST

        pub_qos = QoSProfile(depth=5)
        pub_qos.reliability = ReliabilityPolicy.RELIABLE
        pub_qos.history = HistoryPolicy.KEEP_LAST

        self.image_pub = self.create_publisher(CompressedImage, image_topic, pub_qos)
        self.det_pub = self.create_publisher(Detection2DArray, det_topic, pub_qos)
        self.pose_pub = self.create_publisher(PoseStamped, target_pose_topic, pub_qos)
        self.static_pub = self.create_publisher(MarkerArray, static_objects_topic, pub_qos)
        self.dynamic_pub = self.create_publisher(MarkerArray, dynamic_objects_topic, pub_qos)

        self.create_service(Trigger, '~/clear_static_memory', self.srv_clear_memory)
        self.create_service(Trigger, '~/save_static_memory', self.srv_save_memory)
        # Static markers are republished at 1 Hz even with nothing in view,
        # so rviz keeps showing remembered objects.
        self.create_timer(1.0, self.static_timer_cb)
        if self.memory_file:
            self.create_timer(memory_save_period_s, self.save_memory)
        if stats_period_s > 0:
            self.create_timer(stats_period_s, self.stats_timer_cb)

        # Depth + camera_info: BEST_EFFORT to match the RealSense driver's
        # own publishing convention (same reasoning as sub_qos above).
        self.create_subscription(Image, depth_topic, self.depth_cb, sub_qos)
        self.create_subscription(CameraInfo, camera_info_topic, self.caminfo_cb, sub_qos)

        # Color images: queue depth 1 -- when inference is slower than the
        # camera, always process the NEWEST frame instead of working through
        # a backlog (which only adds latency).
        img_qos = QoSProfile(depth=1)
        img_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        img_qos.history = HistoryPolicy.KEEP_LAST
        if self.use_compressed:
            self.get_logger().info(f'Subscribing (compressed) to: {input_topic}')
            self.sub = self.create_subscription(
                CompressedImage, input_topic, self.compressed_callback, img_qos)
        else:
            self.get_logger().info(f'Subscribing (raw) to: {input_topic}')
            self.sub = self.create_subscription(
                Image, input_topic, self.image_callback, img_qos)

        self.get_logger().info(
            f'3D back-projection: depth_topic={depth_topic}, '
            f'camera_info_topic={camera_info_topic}, target_frame={self.target_frame}, '
            f'camera_frame={self.camera_frame or "<image header.frame_id>"}, '
            f'fixed_frame={self.fixed_frame or "<none>"} -> publishing {target_pose_topic}'
        )

    def depth_cb(self, msg: Image):
        self.depth_buf.append(msg)

    def depth_for(self, stamp):
        """Depth message closest in time to `stamp`, and |dt| in seconds."""
        if not self.depth_buf:
            return None, None
        t = _stamp_s(stamp)
        msg = min(self.depth_buf, key=lambda m: abs(_stamp_s(m.header.stamp) - t))
        return msg, abs(_stamp_s(msg.header.stamp) - t)

    def caminfo_cb(self, msg: CameraInfo):
        self.camera_info = msg

    def compressed_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        self.run_inference(frame, msg.header)

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.run_inference(frame, msg.header)

    # ------------------------------------------------------------ stats
    def _tick(self, name, seconds):
        s = self.stats.get(name)
        if s is None:
            self.stats[name] = [1, seconds, seconds]
        else:
            s[0] += 1
            s[1] += seconds
            if seconds > s[2]:
                s[2] = seconds

    def _count(self, name, n=1):
        self.counters[name] = self.counters.get(name, 0) + n

    def stats_timer_cb(self):
        now = time.monotonic()
        elapsed = max(now - self.stats_t0, 1e-6)
        frames = self.stats.get('frame', [0])[0]
        parts = [f'{frames / elapsed:.1f} FPS']
        for name in ('frame', 'infer', 'draw', '3d', 'tf', 'memory', 'merge', 'save'):
            s = self.stats.get(name)
            if s:
                parts.append(f'{name} {1000 * s[1] / s[0]:.1f}/{1000 * s[2]:.0f}ms')
        c = self.counters
        parts.append(f'tf ok {c.get("tf_ok", 0)} fail {c.get("tf_fail", 0)} '
                     f'skipped {c.get("tf_skipped", 0)}')
        if c.get('draw_skipped'):
            parts.append(f'draw skipped {c["draw_skipped"]} (no subscriber)')
        parts.append(f'objects {len(self.static_memory.instances)} '
                     f'({len(self.static_memory.confirmed())} confirmed)')
        self.get_logger().info(f'stats {elapsed:.0f}s (mean/max): ' + ' | '.join(parts))
        self.stats.clear()
        self.counters.clear()
        self.stats_t0 = now

    # ------------------------------------------------------------ inference
    def run_inference(self, frame, header):
        t_frame = time.perf_counter()
        results = self.model.predict(
            frame, conf=self.conf_thres, classes=self.classes_filter, verbose=False)
        result = results[0]
        t = time.perf_counter()
        self._tick('infer', t - t_frame)

        # Drawing + JPEG costs several ms per frame: only do it when someone
        # (rviz, rqt, the zenoh bridge) actually subscribes to the image.
        if self.image_pub.get_subscription_count() > 0:
            annotated = result.plot()  # BGR numpy array, boxes drawn in
            annotated = np.ascontiguousarray(annotated, dtype=np.uint8)

            # JPEG-encode directly via cv2 rather than cv_bridge.cv2_to_imgmsg --
            # avoids a known cv_bridge/opencv-python type-code conflict when the
            # two coexist (pip-installed opencv-python from ultralytics vs. the
            # apt/source-built OpenCV cv_bridge itself was compiled against),
            # and produces a much smaller message for the zenoh bridge to relay.
            success, encoded = cv2.imencode(
                '.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            if success:
                img_msg = CompressedImage()
                img_msg.header = header  # keep original timestamp/frame_id
                img_msg.format = 'jpeg'
                img_msg.data = encoded.tobytes()
                self.image_pub.publish(img_msg)
            else:
                self.get_logger().warn('JPEG encoding failed, skipping this frame\'s debug image')
            self._tick('draw', time.perf_counter() - t)
        else:
            self._count('draw_skipped')

        det_array = Detection2DArray()
        det_array.header = header

        dets = []  # (class_name, (cx, cy, bw, bh), conf) -- ALL detections, not only the best

        for box in result.boxes:
            cx, cy, bw, bh = box.xywh[0].tolist()

            det = Detection2D()
            det.bbox.center.position.x = cx
            det.bbox.center.position.y = cy
            det.bbox.size_x = bw
            det.bbox.size_y = bh

            hyp = ObjectHypothesisWithPose()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            hyp.hypothesis.class_id = self.model.names[cls_id]
            hyp.hypothesis.score = conf
            det.results.append(hyp)
            det_array.detections.append(det)

            # result.boxes is already restricted to classes_filter by the
            # `classes=` arg passed to model.predict() above.
            dets.append((self.model.names[cls_id], (cx, cy, bw, bh), conf))

        self.det_pub.publish(det_array)

        if dets:
            t = time.perf_counter()
            self.process_3d(dets, header)
            self._tick('3d', time.perf_counter() - t)
        self._tick('frame', time.perf_counter() - t_frame)

    def process_3d(self, dets, header):
        """Back-project every detection, transform to target_frame once,
        route static classes to the memory and dynamic ones to the tracker,
        and publish /yolo/target_pose."""
        if self.camera_info is None:
            return

        depth_msg, depth_dt = self.depth_for(header.stamp)
        if depth_msg is None:
            return
        if depth_dt > self.max_depth_dt:
            self.get_logger().warn(
                f'No depth frame within {self.max_depth_dt * 1000:.0f} ms of the color image '
                f'(closest: {depth_dt * 1000:.0f} ms) -- skipping 3D for this frame.',
                throttle_duration_sec=5.0)
            return
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        scale = self.depth_scale(depth_msg.encoding)

        cam_pts = []  # (class_name, xyz in camera optical frame, conf, 3x3 cov)
        for cls_name, box, conf in dets:
            p = self.pixel_to_camera_point(box, depth_img, scale)
            if p is None:
                continue
            # Optional: move from the seen surface toward the object's center
            # along the viewing ray, so the estimate doesn't depend on the
            # side the object is viewed from.
            offset = float(self.class_cfg.params(cls_name).get('center_offset_m', 0.0))
            norm = float(np.linalg.norm(p))
            if offset and norm > 0:
                p = p * (1.0 + offset / norm)
            cam_pts.append((cls_name, p, conf, self.measurement_cov(p)))
        if not cam_pts:
            return

        cam_frame = self.camera_frame or header.frame_id
        if not cam_frame:
            self.get_logger().warn(
                'Image header has no frame_id and camera_frame param is empty -- '
                'cannot place detections in TF.', throttle_duration_sec=10.0)
            return

        best_i = max(range(len(cam_pts)), key=lambda i: cam_pts[i][2])
        out = PoseStamped()
        out.header.stamp = header.stamp
        out.pose.orientation.w = 1.0

        # One TF lookup per frame, shared by all detections.
        t = time.perf_counter()
        tf = self.lookup_camera_to_target(cam_frame, header.stamp)
        self._tick('tf', time.perf_counter() - t)
        if tf is None:
            if self.publish_cam_on_fail:
                out.header.frame_id = cam_frame
                _set_position(out.pose, cam_pts[best_i][1])
                self.pose_pub.publish(out)
            return

        rot, trans = tf
        moving = self.camera_moving(rot, header.stamp)
        stamp_s = _stamp_s(header.stamp)

        static_dets, dynamic_dets = [], []
        best_static_idx = None
        for i, (cls_name, p, conf, cov) in enumerate(cam_pts):
            d = (cls_name, tuple(rot @ p + trans), conf, rot @ cov @ rot.T)
            if self.class_cfg.type_of(cls_name) == STATIC:
                if i == best_i:
                    best_static_idx = len(static_dets)
                static_dets.append(d)
            else:
                dynamic_dets.append(d)

        matches = []
        if static_dets:
            if moving:
                self.get_logger().debug('Camera rotating fast -- skipping static memory update')
            else:
                t = time.perf_counter()
                matches = self.static_memory.update_frame(static_dets, stamp_s)
                self._tick('memory', time.perf_counter() - t)
        self.dynamic_tracker.update_frame(dynamic_dets, stamp_s)
        if dynamic_dets:
            self.publish_dynamic_markers(dynamic_dets, header.stamp)

        out.header.frame_id = self.target_frame
        best_cls = cam_pts[best_i][0]
        best_xyz = rot @ cam_pts[best_i][1] + trans
        if self.target_class:
            # Stable target: remembered position of the confirmed instance of
            # target_class nearest to the camera right now.
            inst = self.static_memory.nearest_confirmed(self.target_class, tuple(trans))
            if inst is None:
                return
            _set_position(out.pose, inst.position)
        elif best_static_idx is not None:
            # Best detection is a static class: publish the FILTERED estimate
            # of the instance it was associated with, not the raw measurement.
            inst = next((m for m, i, _ in matches if i == best_static_idx), None)
            if inst is None:  # no update this frame (moving) -> nearby known instance
                radius = float(self.class_cfg.params(best_cls)['merge_radius_m'])
                inst = self.static_memory.nearest_confirmed(best_cls, tuple(best_xyz), radius)
            if inst is None:
                return
            _set_position(out.pose, inst.position)
        else:
            _set_position(out.pose, best_xyz)  # dynamic: raw (placeholder, no filter yet)
        self.pose_pub.publish(out)

    def measurement_cov(self, p):
        """3x3 measurement covariance of camera-frame point p (optical frame).

        Depth noise lies along the viewing ray and grows ~z^2 (stereo);
        lateral noise comes from bounding-box jitter of box_px_sigma pixels.
        """
        z = float(p[2])
        f = 0.5 * (self.camera_info.k[0] + self.camera_info.k[4])
        sigma_depth = self.depth_noise_a + self.depth_noise_b * z * z
        sigma_lat = z * self.box_px_sigma / f
        u = p / max(float(np.linalg.norm(p)), 1e-6)
        uu = np.outer(u, u)
        return sigma_lat ** 2 * (np.eye(3) - uu) + sigma_depth ** 2 * uu

    def camera_moving(self, rot, stamp):
        """True if the camera rotated faster than max_cam_ang_vel since the last frame.

        rot: 3x3 rotation target_frame <- camera at this frame.
        """
        t = _stamp_s(stamp)
        prev, self.prev_cam = self.prev_cam, (rot, t)
        if prev is None:
            return False
        dt = t - prev[1]
        if dt <= 1e-3 or dt > 1.0:  # duplicate frame or long gap: can't tell
            return False
        # Angle of the relative rotation prev^T * rot.
        cos_a = (float(np.trace(prev[0].T @ rot)) - 1.0) / 2.0
        angle = math.acos(max(-1.0, min(1.0, cos_a)))
        return angle / dt > self.max_cam_ang_vel

    def depth_scale(self, encoding):
        # 16UC1 = integer millimetres (typical RealSense aligned-depth output).
        # 32FC1 = already float metres. Anything else is unexpected -- warn
        # rather than silently guessing.
        if encoding == '16UC1':
            return 0.001
        if encoding == '32FC1':
            return 1.0
        self.get_logger().warn(
            f'Unrecognized depth encoding "{encoding}", assuming millimetres. '
            'Verify this against your actual depth topic if poses look wrong.',
            throttle_duration_sec=10.0
        )
        return 0.001

    def pixel_to_camera_point(self, box, depth_img, scale):
        """Box (cx, cy, bw, bh) in pixels -> np.array xyz in the camera optical frame, or None."""
        cx, cy, bw, bh = box
        px, py = int(cx), int(cy)
        h, w = depth_img.shape[:2]
        if not (0 <= py < h and 0 <= px < w):
            return None

        # Median over a window in the central part of the box instead of a
        # single pixel: one depth hole (0) or edge pixel exactly at the box
        # center would otherwise drop or corrupt the pose. Window half-size
        # is 25% of the box, clamped to [2, 20] px.
        hx = int(min(max(bw * 0.25, 2), 20))
        hy = int(min(max(bh * 0.25, 2), 20))
        roi = depth_img[max(py - hy, 0):min(py + hy + 1, h),
                        max(px - hx, 0):min(px + hx + 1, w)].astype(np.float32) * scale
        valid = roi[np.isfinite(roi)
                    & (roi >= self.min_valid_depth_m)
                    & (roi <= self.max_valid_depth_m)]
        if valid.size < 5:
            return None
        depth_m = float(np.median(valid))

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        ppx = self.camera_info.k[2]
        ppy = self.camera_info.k[5]

        # Sub-pixel box center (cx, cy), not the int-rounded window center.
        return np.array([(cx - ppx) * depth_m / fx, (cy - ppy) * depth_m / fy, depth_m])

    # ------------------------------------------------------------ markers
    @staticmethod
    def class_color(name, alpha=0.8):
        """Stable distinct color per class name (same across restarts)."""
        hue = (zlib.crc32(name.encode()) % 360) / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
        return r, g, b, alpha

    def make_marker(self, ns, mid, mtype, xyz, scale, color, stamp):
        m = Marker()
        m.header.frame_id = self.target_frame
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = xyz
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = color
        return m

    def static_timer_cb(self):
        now = self.get_clock().now()
        self.static_memory.prune_unconfirmed(now.nanoseconds * 1e-9)
        t = time.perf_counter()
        merged = self.static_memory.merge_duplicates()
        self._tick('merge', time.perf_counter() - t)
        for kept, removed in merged:
            self.get_logger().info(f'Merged duplicate static object {removed} into {kept}')
        stamp = now.to_msg()

        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.target_frame
        clear.action = Marker.DELETEALL  # so cleared / pruned objects disappear in rviz
        arr.markers.append(clear)
        for i, inst in enumerate(sorted(self.static_memory.confirmed(), key=lambda s: s.id)):
            color = self.class_color(inst.class_name)
            arr.markers.append(self.make_marker(
                'static_objects', i, Marker.CUBE, inst.position, 0.3, color, stamp))
            label = self.make_marker(
                'static_labels', i, Marker.TEXT_VIEW_FACING,
                (inst.position[0], inst.position[1], inst.position[2] + 0.35),
                0.15, (1.0, 1.0, 1.0, 1.0), stamp)
            label.text = f'{inst.id} (n={inst.n_obs}, ±{inst.sigma * 100:.1f}cm)'
            arr.markers.append(label)
            arr.markers.append(self.uncertainty_marker(i, inst, color, stamp))
        self.static_pub.publish(arr)

    def uncertainty_marker(self, mid, inst, color, stamp):
        """2-sigma ellipsoid of the instance's Kalman covariance."""
        w, V = inst.eig  # cached at the last filter update
        V = V.copy()      # don't modify the cached eigenvectors below
        if np.linalg.det(V) < 0:
            V[:, 2] *= -1  # keep a right-handed rotation
        m = self.make_marker('static_uncertainty', mid, Marker.SPHERE, inst.position,
                             1.0, (color[0], color[1], color[2], 0.25), stamp)
        m.scale.x, m.scale.y, m.scale.z = (max(4.0 * math.sqrt(max(v, 0.0)), 0.01) for v in w)
        (m.pose.orientation.x, m.pose.orientation.y,
         m.pose.orientation.z, m.pose.orientation.w) = _rot_to_quat(V)
        return m

    def publish_dynamic_markers(self, dynamic_dets, stamp):
        arr = MarkerArray()
        for i, (cls_name, xyz, conf, _cov) in enumerate(dynamic_dets):
            m = self.make_marker('dynamic_objects', i, Marker.SPHERE, xyz, 0.25,
                                 self.class_color(cls_name, 0.6), stamp)
            m.lifetime = Duration(seconds=0.5).to_msg()
            arr.markers.append(m)
            label = self.make_marker('dynamic_labels', i, Marker.TEXT_VIEW_FACING,
                                     (xyz[0], xyz[1], xyz[2] + 0.3), 0.12,
                                     (1.0, 1.0, 1.0, 1.0), stamp)
            label.text = f'{cls_name} {conf:.2f}'
            label.lifetime = m.lifetime
            arr.markers.append(label)
        self.dynamic_pub.publish(arr)

    # ------------------------------------------------------------ memory I/O
    def save_memory(self):
        if not self.memory_file or not self.static_memory.dirty:
            return
        try:
            t = time.perf_counter()
            n = self.static_memory.save(self.memory_file, self.target_frame)
            self._tick('save', time.perf_counter() - t)
            self.get_logger().debug(f'Saved {n} static objects to {self.memory_file}')
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'Could not save {self.memory_file}: {e!r}',
                                    throttle_duration_sec=30.0)

    def srv_clear_memory(self, request, response):
        n = len(self.static_memory.instances)
        self.static_memory.clear()
        self.save_memory()
        response.success = True
        response.message = f'cleared {n} static objects'
        self.get_logger().info(response.message)
        return response

    def srv_save_memory(self, request, response):
        if not self.memory_file:
            response.success = False
            response.message = 'memory_file param is empty -- nothing to save to'
            return response
        self.static_memory.dirty = True
        self.save_memory()
        response.success = True
        response.message = (f'saved {len(self.static_memory.confirmed())} static objects '
                            f'to {self.memory_file}')
        return response

    def lookup_camera_to_target(self, cam_frame, stamp):
        """target_frame <- cam_frame at the image time as (rot 3x3, trans 3), or None.

        Hop by hop instead of one chained lookup, so one slow or stale link
        can't drag the whole chain back to its "latest common time":
          hop 1: fixed_frame (odom) <- camera at the exact image stamp
                 (fallback: that hop's latest, only if at most max_tf_age_s old)
          hop 2: target_frame (map) <- fixed_frame at its own latest
                 (only if at most max_map_tf_age_s old; AMCL/SLAM is slow)
        Without a usable fixed_frame: one hop target <- camera.
        Only the first attempt waits (20 ms); fallbacks never wait. After
        TF_FAIL_LIMIT failures in a row, TF is skipped for TF_SKIP_S seconds
        (detections and images keep running at full rate), then probed again.
        """
        now = time.monotonic()
        if now < self.tf_skip_until:
            self._count('tf_skipped')
            return None

        img_t = rclpy.time.Time.from_msg(stamp)
        img_s = _stamp_s(stamp)
        errors = []
        result, method = None, ''
        use_fixed = bool(self.fixed_frame) and \
            self.fixed_frame not in (self.target_frame, cam_frame)
        if use_fixed:
            hop1 = self._lookup_hop(self.fixed_frame, cam_frame, img_t, img_s,
                                    self.max_tf_age, errors, wait=True)
            hop2 = None if hop1 is None else self._lookup_hop(
                self.target_frame, self.fixed_frame, None, img_s,
                self.max_map_tf_age, errors, wait=False)
            if hop1 is not None and hop2 is not None:
                (r1, t1, m1), (r2, t2, m2) = hop1, hop2
                result = (r2 @ r1, r2 @ t1 + t2)
                method = (f'{self.fixed_frame}<-camera {m1}, '
                          f'{self.target_frame}<-{self.fixed_frame} {m2}')
        else:
            hop = self._lookup_hop(self.target_frame, cam_frame, img_t, img_s,
                                   self.max_tf_age, errors, wait=True)
            if hop is not None:
                result, method = (hop[0], hop[1]), hop[2]

        if result is not None:
            self.tf_fail_streak = 0
            self._count('tf_ok')
            self._tf_ok(cam_frame, method)
            return result

        self.tf_ok_logged = False
        self._count('tf_fail')
        self.tf_fail_streak += 1
        if self.tf_fail_streak >= TF_FAIL_LIMIT:
            # Streak is NOT reset: while TF stays broken, every failed probe
            # re-trips the breaker immediately.
            self.tf_skip_until = now + TF_SKIP_S
        self.get_logger().warn(
            f'TF lookup failed for YOLO target pose: {" | ".join(errors)}',
            throttle_duration_sec=5.0)
        if now >= self.tf_diag_next:
            self.tf_diag_next = now + TF_DIAG_PERIOD_S
            self.get_logger().warn(self.tf_diagnosis(cam_frame, img_s, now))
        return None

    def _lookup_hop(self, target, source, t, img_s, max_age, errors, wait):
        """One TF hop -> (rot, trans, method) or None (reason appended to errors).

        t = image time for an exact lookup (fallback: latest), None = latest only.
        Latest data is accepted only if at most max_age seconds from the image
        time; a stamp of 0 means an all-static chain (always valid).
        """
        timeout = Duration(seconds=0.02) if wait else Duration()
        if t is not None:
            try:
                tf = self.tf_buffer.lookup_transform(target, source, t, timeout=timeout)
                rot, trans = _transform_to_rt(tf)
                return rot, trans, 'exact'
            except TransformException:
                timeout = Duration()  # fallbacks never wait
        try:
            tf = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time(),
                                                 timeout=timeout)
        except TransformException as e:
            errors.append(f'{target}<-{source}: {e}')
            return None
        ts = _stamp_s(tf.header.stamp)
        age = img_s - ts
        if ts > 0 and abs(age) > max_age:
            errors.append(f'{target}<-{source}: newest transform is {age:+.1f} s away from '
                          f'the image time (limit {max_age:.1f} s)')
            return None
        rot, trans = _transform_to_rt(tf)
        return rot, trans, ('static' if ts == 0 else f'latest ({age * 1000:+.0f} ms)')

    def tf_diagnosis(self, cam_frame, img_s, now):
        """Human-readable report of the TF links between camera and target_frame.

        For every link: parent -> child, broadcaster, rate, and how old its
        newest transform is relative to the image. Stale links are flagged;
        comparing with the previous report tells a clock offset (constant age)
        from a stopped publisher (age grows with wall time).
        """
        try:
            frames = yaml.safe_load(self.tf_buffer.all_frames_as_yaml()) or {}
        except Exception as e:  # noqa: BLE001
            return (f'TF diagnosis unavailable ({e!r}). Known frames:\n'
                    f'{self.tf_buffer.all_frames_as_string()}')
        if not frames:
            return ('TF diagnosis: no TF data received at all -- check that /tf and /tf_static '
                    'reach this container (zenoh/DDS, ROS_DOMAIN_ID, RMW_IMPLEMENTATION).')

        def chain(frame):
            links, seen = [], set()
            while frame in frames and frame not in seen:
                seen.add(frame)
                parent = frames[frame].get('parent')
                links.append((frame, parent))
                frame = parent
            return links

        cam_links = chain(cam_frame)
        tgt_links = chain(self.target_frame)
        cam_ancestors = {cam_frame} | {p for _, p in cam_links}
        slow_children = {c for c, _ in chain(self.fixed_frame)} if self.fixed_frame else set()

        lines = [f'TF diagnosis for {self.target_frame} <- {cam_frame}:']
        if cam_frame not in frames:
            lines.append(f'  "{cam_frame}" is not in the TF tree (camera driver TF or the '
                         'static base_link->camera_link transform is missing).')
        tgt_root = tgt_links[-1][1] if tgt_links else self.target_frame
        if self.target_frame not in cam_ancestors and tgt_root not in cam_ancestors:
            cam_root = cam_links[-1][1] if cam_links else cam_frame
            known = set(frames) | {info.get('parent') for info in frames.values()}
            if self.target_frame not in known:
                lines.append(f'  "{self.target_frame}" does not exist -- start SLAM/localization, '
                             'or use TARGET_FRAME=odom.')
            else:
                lines.append(f'  NOT CONNECTED: the camera tree ends at "{cam_root}", '
                             f'"{self.target_frame}" belongs to the tree rooted at "{tgt_root}".')
        for child, parent in cam_links + [l for l in tgt_links if l not in cam_links]:
            info = frames.get(child) or {}
            newest = float(info.get('most_recent_transform', 0) or 0)
            who = info.get('broadcaster', '?')
            if newest == 0:
                lines.append(f'  {parent} -> {child}: static (by {who})')
                continue
            age = img_s - newest
            limit = self.max_map_tf_age if child in slow_children else self.max_tf_age
            flag = ''
            if abs(age) > limit:
                prev = self.tf_diag_hist.get(child)
                if prev is None:
                    hint = ' (next report tells clock offset vs. stopped publisher)'
                else:
                    d_wall, d_age = now - prev[0], age - prev[1]
                    if abs(d_age) < 0.2 * d_wall:
                        hint = (' -> age stays constant: CLOCK OFFSET between machines '
                                '(sync clocks with chrony/NTP)')
                    elif d_age > 0.8 * d_wall:
                        hint = ' -> age keeps growing: PUBLISHER STOPPED (or the bridge dropped /tf)'
                    else:
                        hint = ''
                flag = f'   <-- STALE{hint}'
            self.tf_diag_hist[child] = (now, age)
            rate = float(info.get('rate', 0) or 0)
            lines.append(f'  {parent} -> {child}: newest {age:+.2f} s before image, by {who}, '
                         f'{rate:.1f} Hz{flag}')
        return '\n'.join(lines)

    def _tf_ok(self, cam_frame, method):
        if not self.tf_ok_logged:
            self.tf_ok_logged = True
            self.get_logger().info(
                f'TF OK: {self.target_frame}<-{cam_frame} ({method}); publishing target poses.')


def main(args=None):
    rclpy.init(args=args)
    node = YoloSubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_memory()  # periodic saves cover hard kills; this catches the last few seconds
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()