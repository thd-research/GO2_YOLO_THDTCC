"""Static / dynamic object bookkeeping for yolo_subscriber_node.

Pure Python + numpy (no ROS imports) so it can be unit-tested anywhere.

- Class types come from a YAML file (see config/object_classes.yaml).
- STATIC classes: every physical instance gets a stable id (blue_box_1,
  blue_box_2, ...) and a remembered position in a fixed frame (map). Each new
  detection is associated with the nearest known instance of the same class
  within merge_radius_m (greedy one-to-one per frame), then fused with a
  constant-position 3D Kalman filter:
    * measurements carry a full 3x3 covariance (RealSense depth noise lies
      along the viewing ray and grows ~z^2; lateral noise is box jitter), so
      views from different angles shrink each other's depth uncertainty;
    * a Mahalanobis gate rejects outliers (occlusion, box edge on background);
    * after reinit_after consistent rejected measurements in a row the object
      is assumed to have been moved and the filter re-initializes there.
  Instances are only "confirmed" (published) after min_observations accepted
  measurements, which filters out one-off false positives.
- DYNAMIC classes: placeholder. DynamicObjectTracker currently just keeps the
  latest frame's detections. Future work goes there (IoU/Kalman tracking,
  track ids, velocity, prediction).
"""
import math
import os
import re

import numpy as np
import yaml

STATIC = 'static'
DYNAMIC = 'dynamic'

STATIC_DEFAULTS = {
    'merge_radius_m': 0.5,          # same class within this distance = same object
    'min_observations': 5,          # accepted measurements before an instance is confirmed
    'unconfirmed_timeout_s': 10.0,  # drop never-confirmed instances after this
    'process_noise': 1e-4,          # m^2/s: lets the estimate follow slow changes
    'max_predict_dt_s': 30.0,       # cap on uncertainty growth while unseen
    'gate_chi2': 11.34,             # Mahalanobis gate (99% for 3 DOF)
    'reinit_after': 15,             # consecutive rejected measurements -> object moved
    'reinit_spread_m': 0.15,        # ...and those measurements must agree within this
    'center_offset_m': 0.0,         # push the seen surface point this far along the ray (half object depth)
}

DEFAULT_COV = np.eye(3) * 0.05 ** 2


class ClassConfig:
    """Per-class type (static/dynamic) and parameters, keyed by class NAME."""

    def __init__(self, data=None):
        data = data or {}
        self.default_type = str(data.get('default_type', DYNAMIC)).lower()
        self.classes = {}
        for name, cfg in (data.get('classes') or {}).items():
            cfg = dict(cfg or {})
            cfg['type'] = str(cfg.get('type', self.default_type)).lower()
            if cfg['type'] not in (STATIC, DYNAMIC):
                raise ValueError(f'class "{name}": type must be static or dynamic, got {cfg["type"]}')
            self.classes[str(name)] = {**STATIC_DEFAULTS, **cfg}

    def type_of(self, class_name):
        return self.classes.get(class_name, {}).get('type', self.default_type)

    def params(self, class_name):
        cfg = self.classes.get(class_name)
        if cfg is None:
            return dict(STATIC_DEFAULTS, type=self.default_type)
        return cfg

    def describe(self, model_names):
        """Human-readable summary + list of config names not in the model."""
        names = list(model_names.values()) if isinstance(model_names, dict) else list(model_names)
        lines = [f'  {n:<20s} -> {self.type_of(n)}' for n in names]
        unknown = [n for n in self.classes if n not in names]
        return '\n'.join(lines), unknown


def load_class_config(path):
    """Load the YAML config. Missing/empty path -> everything default (dynamic)."""
    if not path or not os.path.isfile(path):
        return ClassConfig(), False
    with open(path, 'r') as f:
        return ClassConfig(yaml.safe_load(f)), True


class StaticInstance:
    """One remembered static object: constant-position Kalman filter in 3D."""

    def __init__(self, inst_id, class_name, xyz, conf, stamp, cov=None):
        self.id = inst_id
        self.class_name = class_name
        self.x = np.asarray(xyz, dtype=float).reshape(3)
        self.P = _as_cov(cov)
        self.n_obs = 1
        self.n_rejected = 0
        self.first_seen = float(stamp)
        self.last_seen = float(stamp)
        self.last_update = float(stamp)
        self.last_conf = float(conf)
        self._rejected = []  # consecutive rejected (z, R) since the last accepted update

    @property
    def position(self):
        return tuple(float(v) for v in self.x)

    @property
    def sigma(self):
        """1-sigma along the most uncertain axis, metres."""
        return float(math.sqrt(max(np.linalg.eigvalsh(self.P).max(), 0.0)))

    def update(self, z, R, conf, stamp, p):
        """Fuse one measurement. Returns True if accepted, False if gated out."""
        z = np.asarray(z, dtype=float).reshape(3)
        R = _as_cov(R)
        stamp = float(stamp)
        self.last_seen = max(self.last_seen, stamp)

        # Predict: constant position, uncertainty grows slowly while time passes.
        dt = min(max(stamp - self.last_update, 0.0), float(p['max_predict_dt_s']))
        P = self.P + np.eye(3) * float(p['process_noise']) * dt

        y = z - self.x
        S = P + R
        d2 = float(y @ np.linalg.solve(S, y))
        if d2 > float(p['gate_chi2']):
            self.n_rejected += 1
            self._rejected.append((z, R))
            if len(self._rejected) >= int(p['reinit_after']):
                self._maybe_reinit(stamp, float(p['reinit_spread_m']))
            return False

        K = P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_K = np.eye(3) - K
        self.P = I_K @ P @ I_K.T + K @ R @ K.T  # Joseph form: stays symmetric PSD
        self.P = 0.5 * (self.P + self.P.T)
        self.n_obs += 1
        self.last_update = stamp
        self.last_conf = float(conf)
        self._rejected.clear()
        return True

    def _maybe_reinit(self, stamp, spread_m):
        zs = np.array([z for z, _ in self._rejected])
        med = np.median(zs, axis=0)
        if np.all(np.linalg.norm(zs - med, axis=1) <= spread_m):
            # Consistent new location -> the object was moved.
            self.x = med
            self.P = np.mean([R for _, R in self._rejected], axis=0) / len(self._rejected)
            self.last_update = stamp
            self._rejected.clear()
        else:
            # Inconsistent rejects (noise/occlusion): keep the estimate, keep a short window.
            self._rejected = self._rejected[-(len(self._rejected) // 2):]

    def to_dict(self):
        P = self.P
        return {
            'id': self.id, 'class': self.class_name,
            'x': round(float(self.x[0]), 4), 'y': round(float(self.x[1]), 4),
            'z': round(float(self.x[2]), 4),
            'sigma_m': round(self.sigma, 4),
            'cov': [round(float(v), 8) for v in
                    (P[0, 0], P[0, 1], P[0, 2], P[1, 1], P[1, 2], P[2, 2])],
            'n_obs': int(self.n_obs), 'n_rejected': int(self.n_rejected),
            'first_seen': float(self.first_seen), 'last_seen': float(self.last_seen),
            'last_conf': round(float(self.last_conf), 3),
        }


class StaticObjectMemory:
    def __init__(self, class_config):
        self.cfg = class_config
        self.instances = {}        # id -> StaticInstance
        self._next_index = {}      # class_name -> next integer suffix
        self.dirty = False

    # ------------------------------------------------------------ update
    def update_frame(self, detections, stamp):
        """Associate one frame of static detections with known instances and fuse them.

        detections: list of (class_name, (x, y, z), conf, cov3x3_or_None) in the memory frame.
        Greedy one-to-one assignment, nearest pairs first, per class: two
        boxes in the same frame can never collapse into one instance.
        Returns a list of (StaticInstance, detection_index, accepted).
        """
        matched = []
        by_class = {}
        for i, det in enumerate(detections):
            by_class.setdefault(det[0], []).append(i)

        for cls, det_idx in by_class.items():
            p = self.cfg.params(cls)
            radius = float(p['merge_radius_m'])
            candidates = [inst for inst in self.instances.values() if inst.class_name == cls]
            pairs = []
            for i in det_idx:
                for inst in candidates:
                    d = _dist(detections[i][1], inst.position)
                    if d <= radius:
                        pairs.append((d, i, inst.id))
            pairs.sort()
            used_det, used_inst = set(), set()
            for _, i, inst_id in pairs:
                if i in used_det or inst_id in used_inst:
                    continue
                used_det.add(i)
                used_inst.add(inst_id)
                inst = self.instances[inst_id]
                _, xyz, conf, cov = _unpack(detections[i])
                accepted = inst.update(xyz, cov, conf, stamp, p)
                matched.append((inst, i, accepted))
            for i in det_idx:
                if i in used_det:
                    continue
                _, xyz, conf, cov = _unpack(detections[i])
                inst = StaticInstance(self._new_id(cls), cls, xyz, conf, stamp, cov)
                self.instances[inst.id] = inst
                matched.append((inst, i, True))
        if detections:
            self.dirty = True
        return matched

    def is_confirmed(self, inst):
        return inst.n_obs >= int(self.cfg.params(inst.class_name)['min_observations'])

    def confirmed(self):
        return [inst for inst in self.instances.values() if self.is_confirmed(inst)]

    def prune_unconfirmed(self, now):
        """Drop instances that were never confirmed and haven't been seen lately."""
        removed = []
        for inst_id, inst in list(self.instances.items()):
            timeout = float(self.cfg.params(inst.class_name)['unconfirmed_timeout_s'])
            if not self.is_confirmed(inst) and now - inst.last_seen > timeout:
                del self.instances[inst_id]
                removed.append(inst_id)
        if removed:
            self.dirty = True
        return removed

    def nearest_confirmed(self, class_name, xyz, max_dist=float('inf')):
        best, best_d = None, max_dist
        for inst in self.confirmed():
            if inst.class_name == class_name:
                d = _dist(inst.position, xyz)
                if d <= best_d:
                    best, best_d = inst, d
        return best

    def clear(self):
        self.instances.clear()
        self._next_index.clear()
        self.dirty = True

    # ------------------------------------------------------- persistence
    def save(self, path, frame):
        """Atomic write (tmp + rename) so a crash never leaves a broken file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            'frame': frame,
            'instances': [inst.to_dict() for inst in
                          sorted(self.instances.values(), key=lambda s: s.id)
                          if self.is_confirmed(inst)],
        }
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)
        os.replace(tmp, path)
        self.dirty = False
        return len(data['instances'])

    def load(self, path, frame):
        """Load saved instances. Returns (count, error_or_None)."""
        if not path or not os.path.isfile(path):
            return 0, None
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        if data.get('frame') != frame:
            return 0, (f'saved memory is in frame "{data.get("frame")}" but the node uses '
                       f'"{frame}" -- not loading it')
        for d in data.get('instances') or []:
            cls = d['class']
            if self.cfg.type_of(cls) != STATIC:
                continue  # class no longer static in config
            cov = None
            if d.get('cov') and len(d['cov']) == 6:
                a, b, c, e, f_, g = d['cov']
                cov = [[a, b, c], [b, e, f_], [c, f_, g]]
            inst = StaticInstance(d['id'], cls, (d['x'], d['y'], d['z']),
                                  d.get('last_conf', 0.0), d.get('last_seen', 0.0), cov)
            inst.n_obs = int(d.get('n_obs', 1))
            inst.n_rejected = int(d.get('n_rejected', 0))
            inst.first_seen = float(d.get('first_seen', inst.last_seen))
            self.instances[inst.id] = inst
            m = re.match(r'^(.*)_(\d+)$', inst.id)
            if m and m.group(1) == cls:
                self._next_index[cls] = max(self._next_index.get(cls, 1), int(m.group(2)) + 1)
        self.dirty = False
        return len(self.instances), None

    def _new_id(self, cls):
        n = self._next_index.get(cls, 1)
        while f'{cls}_{n}' in self.instances:
            n += 1
        self._next_index[cls] = n + 1
        return f'{cls}_{n}'


class DynamicObjectTracker:
    """PLACEHOLDER for dynamic classes (people, other robots, ...).

    Current behavior: no tracking -- keeps only the latest frame's detections.
    Future work (same update_frame() signature so the node doesn't change):
      - associate detections across frames (IoU in image / distance in 3D)
      - assign track ids, estimate velocity with a constant-velocity Kalman filter
      - drop tracks after N missed frames, predict positions while occluded
    """

    def __init__(self, class_config):
        self.cfg = class_config
        self.current = []

    def update_frame(self, detections, stamp):
        """detections: list of (class_name, (x, y, z), conf, cov). Returns list of dicts."""
        self.current = [{'class': d[0], 'position': tuple(float(v) for v in d[1]),
                         'conf': float(d[2]), 'stamp': float(stamp)}
                        for d in detections]
        return self.current


def _unpack(det):
    cls, xyz, conf = det[0], det[1], det[2]
    cov = det[3] if len(det) > 3 else None
    return cls, xyz, conf, cov


def _as_cov(cov):
    if cov is None:
        return DEFAULT_COV.copy()
    c = np.asarray(cov, dtype=float).reshape(3, 3)
    c = 0.5 * (c + c.T)
    return c + np.eye(3) * 1e-8  # keep it invertible


def _dist(a, b):
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))
