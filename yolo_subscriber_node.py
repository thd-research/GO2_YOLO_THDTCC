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
import numpy as np
import cv2

import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose, PoseStamped
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_pose

from ultralytics import YOLO


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
        self.tf_frames_logged = False
        self.min_valid_depth_m = self.get_parameter('min_valid_depth_m').value
        self.max_valid_depth_m = self.get_parameter('max_valid_depth_m').value

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

        self.latest_depth = None
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

        # Depth + camera_info: BEST_EFFORT to match the RealSense driver's
        # own publishing convention (same reasoning as sub_qos above).
        self.create_subscription(Image, depth_topic, self.depth_cb, sub_qos)
        self.create_subscription(CameraInfo, camera_info_topic, self.caminfo_cb, sub_qos)

        if self.use_compressed:
            self.get_logger().info(f'Subscribing (compressed) to: {input_topic}')
            self.sub = self.create_subscription(
                CompressedImage, input_topic, self.compressed_callback, sub_qos)
        else:
            self.get_logger().info(f'Subscribing (raw) to: {input_topic}')
            self.sub = self.create_subscription(
                Image, input_topic, self.image_callback, sub_qos)

        self.get_logger().info(
            f'3D back-projection: depth_topic={depth_topic}, '
            f'camera_info_topic={camera_info_topic}, target_frame={self.target_frame}, '
            f'camera_frame={self.camera_frame or "<image header.frame_id>"}, '
            f'fixed_frame={self.fixed_frame or "<none>"} -> publishing {target_pose_topic}'
        )

    def depth_cb(self, msg: Image):
        self.latest_depth = msg

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

    def run_inference(self, frame, header):
        results = self.model.predict(
            frame, conf=self.conf_thres, classes=self.classes_filter, verbose=False)
        result = results[0]

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

        det_array = Detection2DArray()
        det_array.header = header

        best_box = None
        best_conf = 0.0

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
            # `classes=` arg passed to model.predict() above -- no need to
            # re-check class membership here, just take highest confidence.
            if conf > best_conf:
                best_conf = conf
                best_box = (cx, cy, bw, bh)

        self.det_pub.publish(det_array)

        if best_box is not None:
            self.publish_target_pose(best_box, header)

    def publish_target_pose(self, box, header):
        if self.latest_depth is None or self.camera_info is None:
            return

        cx, cy, bw, bh = box
        px, py = int(cx), int(cy)

        depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')
        h, w = depth_img.shape[:2]
        if not (0 <= py < h and 0 <= px < w):
            return

        # 16UC1 = integer millimetres (typical RealSense aligned-depth output).
        # 32FC1 = already float metres. Anything else is unexpected -- warn
        # rather than silently guessing.
        encoding = self.latest_depth.encoding
        if encoding == '16UC1':
            scale = 0.001
        elif encoding == '32FC1':
            scale = 1.0
        else:
            self.get_logger().warn(
                f'Unrecognized depth encoding "{encoding}", assuming millimetres. '
                'Verify this against your actual depth topic if poses look wrong.',
                throttle_duration_sec=10.0
            )
            scale = 0.001

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
            return
        depth_m = float(np.median(valid))

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        ppx = self.camera_info.k[2]
        ppy = self.camera_info.k[5]

        camera_pose = Pose()
        camera_pose.position.x = (px - ppx) * depth_m / fx
        camera_pose.position.y = (py - ppy) * depth_m / fy
        camera_pose.position.z = depth_m
        camera_pose.orientation.w = 1.0

        cam_frame = self.camera_frame or header.frame_id
        if not cam_frame:
            self.get_logger().warn(
                'Image header has no frame_id and camera_frame param is empty -- '
                'cannot place the target pose in TF.', throttle_duration_sec=10.0)
            return

        out = PoseStamped()
        out.header.stamp = header.stamp

        transform = self.lookup_camera_to_target(cam_frame, header.stamp)
        if transform is None:
            if self.publish_cam_on_fail:
                out.header.frame_id = cam_frame
                out.pose = camera_pose
                self.pose_pub.publish(out)
            return

        out.header.frame_id = self.target_frame
        out.pose = do_transform_pose(camera_pose, transform)
        self.pose_pub.publish(out)

    def lookup_camera_to_target(self, cam_frame, stamp):
        """Return target_frame <- cam_frame at the image time, or None.

        Order of attempts (each with a short timeout -- TF is received on its
        own thread, so a failure costs ~50 ms, not a blocked executor):
          1. tf2 "time travel" through fixed_frame (odom): odom<-camera at
             the exact image stamp (fast odometry, always available), then
             map<-odom at the latest available time (AMCL/SLAM only updates
             map->odom at a few Hz, so an exact-stamp map lookup usually
             fails). Or a plain exact-stamp lookup if no fixed_frame.
          2. Latest available target<-camera (small timing error while moving).
        """
        timeout = Duration(seconds=0.05)
        stamp_t = rclpy.time.Time.from_msg(stamp)
        errors = []

        use_time_travel = bool(self.fixed_frame) and \
            self.fixed_frame not in (self.target_frame, cam_frame)
        try:
            if use_time_travel:
                tf = self.tf_buffer.lookup_transform_full(
                    target_frame=self.target_frame, target_time=rclpy.time.Time(),
                    source_frame=cam_frame, source_time=stamp_t,
                    fixed_frame=self.fixed_frame, timeout=timeout)
            else:
                tf = self.tf_buffer.lookup_transform(
                    self.target_frame, cam_frame, stamp_t, timeout=timeout)
            self._tf_ok(cam_frame, 'time travel via ' + self.fixed_frame
                        if use_time_travel else 'exact stamp')
            return tf
        except TransformException as e:
            errors.append(f'{"time-travel" if use_time_travel else "exact"}: {e}')

        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, cam_frame, rclpy.time.Time(), timeout=timeout)
            self.get_logger().warn(
                f'Exact TF lookup failed ({errors[0]}); used latest available '
                f'{self.target_frame}<-{cam_frame} instead -- pose may lag if the robot moves fast.',
                throttle_duration_sec=5.0)
            self._tf_ok(cam_frame, 'latest')
            return tf
        except TransformException as e:
            errors.append(f'latest: {e}')

        self.tf_ok_logged = False
        if not self.tf_frames_logged:
            self.tf_frames_logged = True
            frames = self.tf_buffer.all_frames_as_string() or '  <no TF frames received at all>'
            self.get_logger().warn(
                f'No TF {self.target_frame}<-{cam_frame}. Frames currently known:\n{frames}\n'
                'Check that /tf and /tf_static reach this container, that something publishes '
                f'"{self.target_frame}" (SLAM/AMCL for map), and that the camera is linked to the '
                'robot (static base_link->camera_link). Or set target_frame to an existing frame.')
        self.get_logger().warn(
            f'TF lookup failed for YOLO target pose: {" | ".join(errors)}',
            throttle_duration_sec=5.0)
        return None

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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()