#!/usr/bin/env python3

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
import cv2
from ultralytics import YOLO

import os

GROUPS = ('0-empty', '1-low', '2-medium', '3-high')
FIELDS = ('image_id', 'group', 'image_name', 'image_path', 'repeat',
          'persons_detected', 'preprocess_ms', 'inference_ms',
          'postprocess_ms', 'other_ms', 'total_ms',
          'conv_self_cpu_ms', 'conv_self_cpu_pct')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='dataset')
    p.add_argument('--model', default='yolo26n.pt')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=1,
                   help='Measurements per image (default: 1)')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--output', default='data/yolo26_dataset_times.csv')
    p.add_argument('--images-output', default='output',
                   help='Folder for images anotated (default: output)')
    p.add_argument('--profile-conv', action='store_true',
                   help='An adition inferente instrumented for image; slow')
    args = p.parse_args()

    if args.threads < 1 or args.repeats < 1 or args.warmup < 0 or args.imgsz < 1:
        p.error('threads, repeats e imgsz must be >= 1; warmup >= 0')

    root = Path(args.dataset)
    images = []
    for group in GROUPS:
        folder = root / group
        if not folder.is_dir():
            p.error(f'The directory is missing: {folder}')
        paths = sorted(x for x in folder.iterdir()
                       if x.is_file() and x.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        if len(paths) != 25:
            p.error(f'{folder}: 25 images arse expected; there are {len(paths)}')
        images.extend((group, x) for x in paths)
    if len({x.resolve() for _, x in images}) != 100:
        p.error('Duplicated routes have been found')

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    images_output = Path(args.images_output)
    images_output.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    cv2.setNumThreads(args.threads)


    model = YOLO(args.model)



    def predict(path):
        torch.set_num_threads(args.threads)
        return model.predict(source=str(path), device='cpu', imgsz=args.imgsz,
                             batch=1, rect=False, classes=[0], conf=args.conf,
                             verbose=False, save=False)[0]

    print(f'YOLO26: {args.model} | {len(images)} images | '
          f'{args.threads} Thread(s) | {args.repeats} repetitions')
    print('Order: ' + ' -> '.join(GROUPS))
    print('warm-up...')
    for _ in range(args.warmup):
        predict(images[0][1])


    rows = []
    for image_id, (group, path) in enumerate(images):
        for rep in range(args.repeats):
            t0 = time.perf_counter()
            result = predict(path)
            total_ms = (time.perf_counter() - t0) * 1000
            speed = result.speed
            pre = float(speed['preprocess'])
            inf = float(speed['inference'])
            post = float(speed['postprocess'])
            count = len(result.boxes) if result.boxes is not None else 0
            row = dict(image_id=image_id, group=group, image_name=path.name,
                       image_path=str(path), repeat=rep, persons_detected=count,
                       preprocess_ms=pre, inference_ms=inf, postprocess_ms=post,
                       other_ms=total_ms-pre-inf-post, total_ms=total_ms,
                       conv_self_cpu_ms='', conv_self_cpu_pct='')
            rows.append(row)

            if rep == 0:
                group_output = images_output / group
                group_output.mkdir(parents=True, exist_ok=True)
                annotated = result.plot(labels=True, conf=True)
                image_output = group_output / f'{image_id:03d}_{path.stem}.jpg'
                if not cv2.imwrite(str(image_output), annotated):
                    raise OSError(f'It is impossible to store {image_output}')

            print(f'[{image_id:02d}/99] {group:8s} {path.name} '
                  f'rep={rep+1}/{args.repeats} persons={count:2d} '
                  f'inf={inf:9.2f} ms total={total_ms:9.2f} ms;{count:d};{inf.2f};', flush=True)

        if args.profile_conv:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU],
                         record_shapes=False, profile_memory=False) as prof:
                predict(path)
            stats = prof.key_averages()
            conv_us = sum(e.self_cpu_time_total for e in stats
                          if e.key == 'aten::_slow_conv2d_forward')
            total_us = sum(e.self_cpu_time_total for e in stats)
            for row in rows[-args.repeats:]:
                row['conv_self_cpu_ms'] = conv_us / 1000
                row['conv_self_cpu_pct'] = 100 * conv_us / total_us if total_us else 0

        with output.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    print('\n===  Mesurements per image ===')
    for group in GROUPS:
        per_image = []
        for image_id in range(100):
            sub = [r['inference_ms'] for r in rows
                   if r['group'] == group and r['image_id'] == image_id]
            if sub:
                per_image.append(statistics.median(sub))
        print(f'{group:8s}: {statistics.median(per_image):.3f} ms '
              f'(min={min(per_image):.3f}, max={max(per_image):.3f})')
    print(f'\nCSV: {output} ({len(rows)} measurments)')
    print(f'Anotated images: {images_output} ({len(images)} archives)')


if __name__ == '__main__':
    main()
