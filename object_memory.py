"""Static / dynamic object bookkeeping for yolo_subscriber_node.

Pure Python (no ROS imports) so it can be unit-tested anywhere.

- Class types come from a YAML file (see config/object_classes.yaml).
- STATIC classes: every physical instance gets a stable id (blue_box_1,
  blue_box_2, ...) and a remembered position in a fixed frame (map). Each new
  detection is associated with the nearest known instance of the same class
  within merge_radius_m; the position is the per-axis median of the last
  max_samples observations (robust to depth outliers). Instances are only
  "confirmed" (published) after min_observations frames, which filters out
  one-off false positives.
- DYNAMIC classes: placeholder. DynamicObjectTracker currently just keeps the
  latest frame's detections. Future work goes there (IoU/Kalman tracking,
  track ids, velocity, prediction).
"""
import math
import os
import re
from collections import deque

import yaml

STATIC = 'static'
DYNAMIC = 'dynamic'

STATIC_DEFAULTS = {
    'merge_radius_m': 0.5,        # same class within this distance = same object
    'min_observations': 5,        # frames before an instance is confirmed
    'max_samples': 30,            # position = median of the last N observations
    'unconfirmed_timeout_s': 10.0,  # drop never-confirmed instances after this
}


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
            if cfg['type'] == STATIC:
                cfg = {**STATIC_DEFAULTS, **cfg}
            self.classes[str(name)] = cfg

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
    def __init__(self, inst_id, class_name, xyz, conf, stamp, max_samples):
        self.id = inst_id
        self.class_name = class_name
        self.samples = deque(maxlen=max_samples)
        self.n_obs = 0
        self.first_seen = stamp
        self.last_seen = stamp
        self.last_conf = conf
        self.position = tuple(xyz)
        self.add(xyz, conf, stamp)

    def add(self, xyz, conf, stamp):
        self.samples.append(tuple(xyz))
        self.n_obs += 1
        self.last_seen = stamp
        self.last_conf = conf
        # Per-axis median of recent samples: robust to depth spikes / edges.
        self.position = tuple(_median([s[i] for s in self.samples]) for i in range(3))

    def to_dict(self):
        return {
            'id': self.id, 'class': self.class_name,
            'x': round(self.position[0], 4), 'y': round(self.position[1], 4),
            'z': round(self.position[2], 4),
            'n_obs': self.n_obs, 'first_seen': self.first_seen,
            'last_seen': self.last_seen, 'last_conf': round(self.last_conf, 3),
        }


class StaticObjectMemory:
    def __init__(self, class_config):
        self.cfg = class_config
        self.instances = {}        # id -> StaticInstance
        self._next_index = {}      # class_name -> next integer suffix
        self.dirty = False

    # ------------------------------------------------------------ update
    def update_frame(self, detections, stamp):
        """Associate one frame of static detections with known instances.

        detections: list of (class_name, (x, y, z), conf) in the memory frame.
        Greedy one-to-one assignment, nearest pairs first, per class: two
        boxes in the same frame can never collapse into one instance.
        Returns a list of (StaticInstance, detection_index).
        """
        matched = []
        by_class = {}
        for i, (cls, xyz, conf) in enumerate(detections):
            by_class.setdefault(cls, []).append(i)

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
                inst.add(detections[i][1], detections[i][2], stamp)
                matched.append((inst, i))
            for i in det_idx:
                if i in used_det:
                    continue
                inst = StaticInstance(self._new_id(cls), cls, detections[i][1],
                                      detections[i][2], stamp, int(p['max_samples']))
                self.instances[inst.id] = inst
                matched.append((inst, i))
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

    def nearest_confirmed(self, class_name, xyz):
        best, best_d = None, float('inf')
        for inst in self.confirmed():
            if inst.class_name == class_name:
                d = _dist(inst.position, xyz)
                if d < best_d:
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
            p = self.cfg.params(cls)
            inst = StaticInstance(d['id'], cls, (d['x'], d['y'], d['z']),
                                  d.get('last_conf', 0.0), d.get('last_seen', 0.0),
                                  int(p['max_samples']))
            inst.n_obs = int(d.get('n_obs', 1))
            inst.first_seen = d.get('first_seen', inst.last_seen)
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
        """detections: list of (class_name, (x, y, z), conf). Returns list of dicts."""
        self.current = [{'class': c, 'position': tuple(xyz), 'conf': conf, 'stamp': stamp}
                        for c, xyz, conf in detections]
        return self.current


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])
