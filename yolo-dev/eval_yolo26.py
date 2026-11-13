#!/usr/bin/env python3
"""Benchmark ordenado de 100 imágenes YOLO26, detección exclusiva de personas."""
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
                   help='Mediciones por imagen (default: 1)')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--output', default='data/yolo26_dataset_times.csv')
    p.add_argument('--images-output', default='output',
                   help='Carpeta para las imágenes anotadas (default: output)')
    p.add_argument('--profile-conv', action='store_true',
                   help='Una inferencia adicional instrumentada por imagen; lenta')
    args = p.parse_args()

    if args.threads < 1 or args.repeats < 1 or args.warmup < 0 or args.imgsz < 1:
        p.error('threads, repeats e imgsz deben ser >= 1; warmup >= 0')

    root = Path(args.dataset)
    images = []
    for group in GROUPS:
        folder = root / group
        if not folder.is_dir():
            p.error(f'Falta el directorio: {folder}')
        paths = sorted(x for x in folder.iterdir()
                       if x.is_file() and x.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        if len(paths) != 25:
            p.error(f'{folder}: se esperaban 25 imágenes, hay {len(paths)}')
        images.extend((group, x) for x in paths)
    if len({x.resolve() for _, x in images}) != 100:
        p.error('Se han encontrado rutas duplicadas')

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    images_output = Path(args.images_output)
    images_output.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    cv2.setNumThreads(args.threads)

    #import os
    #import ctypes
    #import ctypes.util
    
    #print("\n=== CONFIGURACIÓN DE HILOS ===")
    
    #print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
    #print("OPENBLAS_NUM_THREADS:", os.environ.get("OPENBLAS_NUM_THREADS"))
    #print("PyTorch threads:", torch.get_num_threads())
    #print("Afinidad CPU:", sorted(os.sched_getaffinity(0)))
    
    #blas_path = ctypes.util.find_library("openblas")
    #print("OpenBLAS library:", blas_path)
    
    #if blas_path:
        #blas = ctypes.CDLL(blas_path)
        #blas.openblas_get_num_threads.restype = ctypes.c_int
        #print("OpenBLAS threads:", blas.openblas_get_num_threads())
    #
    #print("==============================\n")

    model = YOLO(args.model)



    def predict(path):
        torch.set_num_threads(args.threads)
        return model.predict(source=str(path), device='cpu', imgsz=args.imgsz,
                             batch=1, rect=False, classes=[0], conf=args.conf,
                             verbose=False, save=False)[0]

    print(f'YOLO26: {args.model} | {len(images)} imágenes | '
          f'{args.threads} hilo(s) | {args.repeats} repetición(es)')
    print('Orden: ' + ' -> '.join(GROUPS))
    print('Calentamiento...')
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

            # Guardar una sola imagen por entrada, fuera del intervalo medido.
            # result.plot() dibuja únicamente las detecciones filtradas (person).
            if rep == 0:
                group_output = images_output / group
                group_output.mkdir(parents=True, exist_ok=True)
                annotated = result.plot(labels=True, conf=True)
                image_output = group_output / f'{image_id:03d}_{path.stem}.jpg'
                if not cv2.imwrite(str(image_output), annotated):
                    raise OSError(f'No se pudo guardar {image_output}')

            print(f'[{image_id:02d}/99] {group:8s} {path.name} '
                  f'rep={rep+1}/{args.repeats} personas={count:2d} '
                  f'inf={inf:9.2f} ms total={total_ms:9.2f} ms;{count:d};{inf.2f};', flush=True)

        if args.profile_conv:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU],
                         record_shapes=False, profile_memory=False) as prof:
                predict(path)
            stats = prof.key_averages()
            # Tiempo propio: no sumar wrappers anidados (aten::conv2d, etc.).
            conv_us = sum(e.self_cpu_time_total for e in stats
                          if e.key == 'aten::_slow_conv2d_forward')
            total_us = sum(e.self_cpu_time_total for e in stats)
            for row in rows[-args.repeats:]:
                row['conv_self_cpu_ms'] = conv_us / 1000
                row['conv_self_cpu_pct'] = 100 * conv_us / total_us if total_us else 0

        # Escritura incremental para no perder mediciones si se interrumpe.
        with output.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    print('\n=== RESUMEN POR GRUPO (medianas por imagen) ===')
    for group in GROUPS:
        per_image = []
        for image_id in range(100):
            sub = [r['inference_ms'] for r in rows
                   if r['group'] == group and r['image_id'] == image_id]
            if sub:
                per_image.append(statistics.median(sub))
        print(f'{group:8s}: {statistics.median(per_image):.3f} ms '
              f'(min={min(per_image):.3f}, max={max(per_image):.3f})')
    print(f'\nCSV: {output} ({len(rows)} mediciones)')
    print(f'Imágenes anotadas: {images_output} ({len(images)} archivos)')


if __name__ == '__main__':
    main()
