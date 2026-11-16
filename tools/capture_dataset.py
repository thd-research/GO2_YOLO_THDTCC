#!/usr/bin/env python3
"""Capture a raw image dataset from one or more Intel RealSense cameras.

Images go where notebooks/01_annotate_augment_prepare.ipynb expects them:

    datasets/raw_images/<session>/<serial>_<YYYYmmdd-HHMMSS-fff>_<NNNNN>.jpg

One folder per session (= recording location / run). The notebook splits
train/val/test by that folder, so start a new session for each place or run.
Optional aligned depth (16-bit PNG, millimetres) goes to datasets/raw_depth/<session>/
with the same file stem: it is kept OUT of raw_images on purpose, because the
notebook loads every image under raw_images.

Keys (preview window):
    SPACE / c   capture one frame from every camera
    a           toggle auto-capture every --interval seconds
    q / ESC     quit

Examples:
    .venv/bin/python tools/capture_dataset.py --session lab_corridor
    .venv/bin/python tools/capture_dataset.py --session warehouse --interval 0.5 --min-sharpness 80
    .venv/bin/python tools/capture_dataset.py --list
    .venv/bin/python tools/capture_dataset.py --no-preview --interval 1 --max-images 200
"""

import argparse
import json
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--session', default=None,
                   help='session folder name (default: session_<date>_<time>). Reusing a name appends to it.')
    p.add_argument('--out', type=Path, default=ROOT / 'datasets' / 'raw_images', help='root for color images')
    p.add_argument('--depth-out', type=Path, default=ROOT / 'datasets' / 'raw_depth', help='root for depth images')
    p.add_argument('--serials', nargs='*', default=None, help='camera serial numbers to use (default: all connected)')
    p.add_argument('--list', action='store_true', help='list connected cameras and exit')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--depth', action='store_true', help='also save depth aligned to color (16-bit PNG, mm)')
    p.add_argument('--interval', type=float, default=0.0,
                   help='auto-capture period in seconds; > 0 starts with auto-capture ON (toggle with "a")')
    p.add_argument('--max-images', type=int, default=0, help='stop after this many images per camera (0 = no limit)')
    p.add_argument('--min-sharpness', type=float, default=0.0,
                   help='auto-capture skips frames whose Laplacian variance is below this (0 = off). '
                        'The live value is shown in the preview, use it to pick a threshold.')
    p.add_argument('--jpeg-quality', type=int, default=95)
    p.add_argument('--warmup', type=int, default=30, help='frames to discard at start so auto-exposure settles')
    p.add_argument('--no-preview', action='store_true', help='headless: no window (needs --interval > 0)')
    p.add_argument('--reset', action='store_true', help='hardware-reset the cameras before starting')
    return p.parse_args()


def list_devices(ctx):
    devs = []
    for d in ctx.query_devices():
        info = lambda k: d.get_info(k) if d.supports(k) else '?'
        devs.append(dict(name=info(rs.camera_info.name), serial=info(rs.camera_info.serial_number),
                         firmware=info(rs.camera_info.firmware_version),
                         usb=info(rs.camera_info.usb_type_descriptor)))
    return devs


def sharpness(bgr):
    """Variance of the Laplacian on a downscaled gray image: low = blurry."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if g.shape[1] > 640:
        g = cv2.resize(g, (640, int(640 * g.shape[0] / g.shape[1])), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


class Camera:
    def __init__(self, serial, args):
        self.serial = serial
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        if args.depth:
            cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        self.profile = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color) if args.depth else None
        self.depth_scale = (self.profile.get_device().first_depth_sensor().get_depth_scale()
                            if args.depth else None)
        self.color = self.depth = None
        self.count = 0

    def poll(self):
        """Grab the newest frameset without blocking. Returns True if a new color frame arrived."""
        fs = self.pipe.poll_for_frames()
        if not fs:
            return False
        if self.align is not None:
            fs = self.align.process(fs)
            d = fs.get_depth_frame()
            if d:
                self.depth = np.asanyarray(d.get_data()).copy()
        c = fs.get_color_frame()
        if not c:
            return False
        self.color = np.asanyarray(c.get_data()).copy()
        return True

    def intrinsics(self):
        i = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return dict(width=i.width, height=i.height, fx=i.fx, fy=i.fy, ppx=i.ppx, ppy=i.ppy,
                    model=str(i.model), coeffs=list(i.coeffs))

    def stop(self):
        try:
            self.pipe.stop()
        except RuntimeError:
            pass


def main():
    args = parse_args()
    ctx = rs.context()
    devs = list_devices(ctx)

    if args.list or not devs:
        if not devs:
            print('No RealSense camera found. Check the cable (USB 3 port), `rs-enumerate-devices`, and that no other '
                  'process (ROS realsense node, realsense-viewer, docker container) holds the camera.\n'
                  'If system tools see it but this script does not, unplug/replug the camera.', file=sys.stderr)
            sys.exit(1)
        for d in devs:
            print(f"{d['serial']}  {d['name']}  fw {d['firmware']}  USB {d['usb']}")
        return

    serials = args.serials or [d['serial'] for d in devs]
    missing = set(serials) - {d['serial'] for d in devs}
    if missing:
        sys.exit(f'Camera(s) not connected: {sorted(missing)}. Connected: {[d["serial"] for d in devs]}')
    for d in devs:
        if d['serial'] in serials and d['usb'].startswith('2'):
            print(f"WARNING: {d['serial']} is on USB {d['usb']}; use a USB 3 port/cable for full resolution and fps.")
    if args.no_preview and args.interval <= 0:
        sys.exit('--no-preview needs --interval > 0 (there are no keys to trigger captures)')

    if args.reset:
        for d in ctx.query_devices():
            if d.get_info(rs.camera_info.serial_number) in serials:
                d.hardware_reset()
        print('Resetting cameras...')
        time.sleep(5)

    session = args.session or datetime.now().strftime('session_%Y%m%d_%H%M%S')
    session = re.sub(r'[^A-Za-z0-9_-]+', '_', session)
    img_dir = args.out / session
    dep_dir = args.depth_out / session
    img_dir.mkdir(parents=True, exist_ok=True)
    if args.depth:
        dep_dir.mkdir(parents=True, exist_ok=True)

    cams = []
    try:
        for s in serials:
            cams.append(Camera(s, args))
            print(f'Started {s}: color {args.width}x{args.height}@{args.fps}' + (' + aligned depth' if args.depth else ''))
    except RuntimeError as e:
        for c in cams:
            c.stop()
        sys.exit(f'Failed to start camera: {e}\nTry another --width/--height/--fps (USB 2 limits the modes) or --reset.')

    # Continue numbering when appending to an existing session
    for c in cams:
        c.count = len(list(img_dir.glob(f'{c.serial}_*.jpg')))

    meta_path = img_dir.parent.parent / 'raw_meta' / f'{session}.json'
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(session=session, started=datetime.now().isoformat(timespec='seconds'), args={
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        cameras={c.serial: dict(**next(d for d in devs if d['serial'] == c.serial), color_intrinsics=c.intrinsics(),
                                depth_scale=c.depth_scale) for c in cams})
    meta_path.write_text(json.dumps(meta, indent=2))

    stop = False

    def on_sigint(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, on_sigint)

    # Warm-up: let auto-exposure / white balance settle
    for c in cams:
        for _ in range(args.warmup):
            c.pipe.wait_for_frames(5000)

    auto = args.interval > 0
    period = args.interval if args.interval > 0 else 1.0
    last_auto = 0.0
    flash_until = 0.0
    win = 'RealSense capture  [SPACE] capture  [a] auto  [q] quit'
    if not args.no_preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print(f'Saving to {img_dir}' + (f' (+ depth in {dep_dir})' if args.depth else ''))
    print('SPACE/c = capture, a = toggle auto, q/ESC = quit' if not args.no_preview else 'Headless, Ctrl+C to stop')

    def save_all(check_sharp):
        saved = 0
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')[:-3]
        for c in cams:
            if c.color is None:
                continue
            if args.max_images and c.count >= args.max_images:
                continue
            if check_sharp and args.min_sharpness > 0 and sharpness(c.color) < args.min_sharpness:
                continue
            stem = f'{c.serial}_{stamp}_{c.count:05d}'
            cv2.imwrite(str(img_dir / f'{stem}.jpg'), c.color, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if args.depth and c.depth is not None:
                d = c.depth.astype(np.float32) * (c.depth_scale * 1000.0)  # -> millimetres
                cv2.imwrite(str(dep_dir / f'{stem}.png'), np.clip(d, 0, 65535).astype(np.uint16))
            c.count += 1
            saved += 1
        return saved

    try:
        while not stop:
            got = [c.poll() for c in cams]
            if not any(got):
                time.sleep(0.002)
            now = time.monotonic()

            if auto and now - last_auto >= period:
                if save_all(check_sharp=True):
                    flash_until = now + 0.1
                last_auto = now

            if args.max_images and all(c.count >= args.max_images for c in cams):
                print(f'Reached --max-images {args.max_images}')
                break

            if not args.no_preview and any(got):
                tiles = []
                for c in cams:
                    if c.color is None:
                        continue
                    t = c.color.copy()
                    sh = sharpness(c.color)
                    ok = args.min_sharpness <= 0 or sh >= args.min_sharpness
                    lines = [f'{c.serial}  saved {c.count}', f'sharpness {sh:.0f}',
                             f'AUTO {period:g}s' if auto else 'manual']
                    for k, txt in enumerate(lines):
                        cv2.putText(t, txt, (10, 25 + 25 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
                        cv2.putText(t, txt, (10, 25 + 25 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                    (0, 255, 0) if (k != 1 or ok) else (0, 0, 255), 2)
                    if now < flash_until:
                        cv2.rectangle(t, (0, 0), (t.shape[1] - 1, t.shape[0] - 1), (255, 255, 255), 12)
                    tiles.append(cv2.resize(t, (args.width, args.height)))
                if tiles:
                    cv2.imshow(win, np.hstack(tiles))

            if not args.no_preview:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key in (ord(' '), ord('c')):
                    n = save_all(check_sharp=False)
                    flash_until = now + 0.1
                    print(f'captured {n} image(s); totals: ' + ', '.join(f'{c.serial}={c.count}' for c in cams))
                elif key == ord('a'):
                    auto = not auto
                    last_auto = 0.0
                    print(f'auto-capture {"ON every %gs" % period if auto else "OFF"}')
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        for c in cams:
            c.stop()
        cv2.destroyAllWindows()
        meta['finished'] = datetime.now().isoformat(timespec='seconds')
        meta['images'] = {c.serial: c.count for c in cams}
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f'Done. Images per camera: {meta["images"]} in {img_dir}')


if __name__ == '__main__':
    main()
