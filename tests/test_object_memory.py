#!/usr/bin/env python3
"""Tests for object_memory.py (static object Kalman filter + association).

No ROS needed. Runs with plain python3 (pytest also works):
    python3 tests/test_object_memory.py
On the Go2, inside the image:
    docker run --rm --entrypoint python3 -v "$(pwd)/tests:/workspace/tests" \
      go2-yolo:latest /workspace/tests/test_object_memory.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from object_memory import ClassConfig, StaticObjectMemory  # noqa: E402

CFG = ClassConfig({
    'default_type': 'dynamic',
    'classes': {'blue_box': {'type': 'static', 'merge_radius_m': 0.5, 'min_observations': 5}},
})


def measurement(rng, truth, cam, outlier=0.0):
    """Noisy measurement of `truth` seen from `cam`, with the node's noise model."""
    ray = truth - cam
    z = float(np.linalg.norm(ray))
    u = ray / z
    sd = 0.005 + 0.004 * z * z
    sl = z * 3.0 / 600.0
    R = sl ** 2 * (np.eye(3) - np.outer(u, u)) + sd ** 2 * np.outer(u, u)
    meas = truth + rng.multivariate_normal(np.zeros(3), R) + outlier * u
    return tuple(meas), R


def viewpoints(truth, n, radius=2.0):
    for k in range(n):
        a = 2 * np.pi * k / n * 0.5  # half circle around the object
        yield truth + np.array([radius * np.cos(a), radius * np.sin(a), 0.2])


def test_converges_and_rejects_outliers():
    rng = np.random.default_rng(0)
    mem = StaticObjectMemory(CFG)
    truth = np.array([2.0, 1.0, 0.3])
    n_out = 0
    for k, cam in enumerate(viewpoints(truth, 200)):
        out = 0.0
        if k > 10 and rng.random() < 0.1:  # 10% outliers: box edge on background
            out, n_out = 0.35, n_out + 1
        xyz, R = measurement(rng, truth, cam, out)
        mem.update_frame([('blue_box', xyz, 0.9, R)], stamp=k * 0.1)
    assert len(mem.instances) == 1, mem.instances.keys()
    inst = next(iter(mem.instances.values()))
    err = np.linalg.norm(inst.x - truth)
    assert err < 0.02, f'error {err:.3f} m'
    assert inst.n_rejected >= 0.8 * n_out, (inst.n_rejected, n_out)
    assert inst.sigma < 0.02, inst.sigma
    print(f'  converge: err={err * 100:.2f} cm, sigma={inst.sigma * 100:.2f} cm, '
          f'rejected {inst.n_rejected}/{n_out} outliers')


def test_two_close_objects_stay_separate():
    rng = np.random.default_rng(1)
    mem = StaticObjectMemory(CFG)
    a, b = np.array([2.0, 0.0, 0.3]), np.array([2.0, 0.6, 0.3])
    cam = np.array([0.0, 0.3, 0.5])
    for k in range(50):
        (xa, Ra), (xb, Rb) = measurement(rng, a, cam), measurement(rng, b, cam)
        mem.update_frame([('blue_box', xa, 0.9, Ra), ('blue_box', xb, 0.8, Rb)], stamp=k * 0.1)
    assert len(mem.instances) == 2, list(mem.instances)
    est = sorted((inst.x for inst in mem.instances.values()), key=lambda v: v[1])
    assert np.linalg.norm(est[0] - a) < 0.03 and np.linalg.norm(est[1] - b) < 0.03, est
    print('  two objects: 2 instances, both within 3 cm')


def test_moved_object_reinitializes():
    rng = np.random.default_rng(2)
    mem = StaticObjectMemory(CFG)
    truth = np.array([2.0, 1.0, 0.3])
    cam = np.array([0.0, 0.0, 0.5])
    for k in range(100):
        xyz, R = measurement(rng, truth, cam)
        mem.update_frame([('blue_box', xyz, 0.9, R)], stamp=k * 0.1)
    moved = truth + np.array([0.3, 0.0, 0.0])  # moved, but within merge radius
    for k in range(100, 140):
        xyz, R = measurement(rng, moved, cam)
        mem.update_frame([('blue_box', xyz, 0.9, R)], stamp=k * 0.1)
    assert len(mem.instances) == 1
    inst = next(iter(mem.instances.values()))
    err = np.linalg.norm(inst.x - moved)
    assert err < 0.05, f'did not follow the moved object: error {err:.3f} m'
    print(f'  moved object: re-initialized, err={err * 100:.2f} cm')


def test_save_load_roundtrip():
    rng = np.random.default_rng(3)
    mem = StaticObjectMemory(CFG)
    truth = np.array([1.0, -1.0, 0.2])
    for k in range(20):
        xyz, R = measurement(rng, truth, np.zeros(3))
        mem.update_frame([('blue_box', xyz, 0.9, R)], stamp=k * 0.1)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'mem.yaml')
        assert mem.save(path, 'map') == 1
        mem2 = StaticObjectMemory(CFG)
        n, err = mem2.load(path, 'map')
        assert n == 1 and err is None
        a, b = next(iter(mem.instances.values())), next(iter(mem2.instances.values()))
        assert np.allclose(a.x, b.x, atol=1e-3) and np.allclose(a.P, b.P, atol=1e-6)
        assert a.id == b.id
        n, err = StaticObjectMemory(CFG).load(path, 'odom')
        assert n == 0 and err
        # next new instance continues the numbering
        mem2.update_frame([('blue_box', (5.0, 5.0, 0.0), 0.9, None)], stamp=10.0)
        assert 'blue_box_2' in mem2.instances, list(mem2.instances)
    print('  save/load: round trip OK, frame mismatch refused, ids continue')


def test_merge_duplicates():
    mem = StaticObjectMemory(CFG)
    # Two instances of one object (e.g. second spawned from a bad early estimate).
    mem.update_frame([('blue_box', (2.0, 1.0, 0.3), 0.9, None)], stamp=0.0)
    mem.update_frame([('blue_box', (2.6, 1.0, 0.3), 0.9, None)], stamp=0.1)  # > merge radius -> new
    assert len(mem.instances) == 2
    mem.instances['blue_box_2'].x = np.array([2.2, 1.0, 0.3])  # estimate converged next to #1
    assert mem.merge_duplicates() == [('blue_box_1', 'blue_box_2')]
    assert list(mem.instances) == ['blue_box_1']
    x = mem.instances['blue_box_1'].x
    assert 2.0 < x[0] < 2.2 and mem.instances['blue_box_1'].n_obs == 2, x

    # Two objects seen in the SAME frame are never merged, however close.
    mem = StaticObjectMemory(CFG)
    mem.update_frame([('blue_box', (2.0, 1.0, 0.3), 0.9, None),
                      ('blue_box', (2.0, 1.6, 0.3), 0.9, None)], stamp=0.0)
    for inst in mem.instances.values():
        inst.x = np.array([2.0, 1.0 + 0.1 * (inst.id == 'blue_box_2'), 0.3])
    assert mem.merge_duplicates() == [] and len(mem.instances) == 2
    print('  merge: duplicates fused, co-observed objects kept apart')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t()
            print(f'PASS {t.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL {t.__name__}: {e}')
    print(f'{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
