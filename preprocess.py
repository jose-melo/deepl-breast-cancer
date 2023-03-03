#!/usr/bin/env python3

import argparse
from typing import Optional, Tuple
from pathlib import Path
import multiprocessing as mp
import warnings
import os
import zipfile

import numpy as np
import pandas as pd

import cv2
import pydicom
from tqdm import tqdm

UINT16_MAX = 2**16 - 1


STATUS_OK = 0
STATUS_ERR = 1
STATUS_SKIP = 2

QUEUE_DONE = "DONE"


class Args(argparse.Namespace):
    csv_file: Path
    input_dir: Path
    output_dir: Path
    force: bool
    output_size: int
    raise_errors: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("-s", "--output_size", type=int, default=512)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-e", "--raise-errors", action="store_true")
    return parser.parse_args(namespace=Args())


def find_border(
    img: np.ndarray, axis: int, color: float, tol: float = 0.02
) -> Tuple[float, float]:
    is_border = (np.abs(img - color) < tol).all(axis=axis)

    # start is the first index which is not a border
    start = is_border.argmin()
    # end is 1 + the last index which is not a border
    end = len(is_border) - is_border[::-1].argmin()

    return start, end


def crop_border(img: np.ndarray, color: float) -> Tuple[bool, np.ndarray]:
    original_img = img

    y0, y1 = find_border(img, axis=1, color=color)
    img = img[y0:y1]
    x0, x1 = find_border(img, axis=0, color=color)
    img = img[:, x0:x1]

    if x0 == y0 == 0 and x1 == original_img.shape[1] and y1 == original_img.shape[0]:
        # No border was found
        return False, img

    if x0 > 0:
        # Ensure the breast is always on the left of the figure
        # (if x0 > 0, then there is a border on the left side)
        img = img[:, ::-1]

    return True, img


def process_image(img: np.ndarray, args: Args) -> np.ndarray:
    # Re-scale image in the range [0, 1]
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()

    # Look for both white and black borders, fixing the colors if required
    ret, img = crop_border(img, color=1)
    if ret:
        # Image color is inverted
        img = 1 - img
    else:
        _, img = crop_border(img, color=0)

    # Resize image to the maximum width and height specified in args
    scale = args.output_size / max(img.shape)
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Convert the image to 16-bit grayscale
    img *= UINT16_MAX
    img = img.astype(np.uint16)
    return img


def process_row(patient_id: int, image_id: int, cancer: int, args: Args) -> int:
    input_file = args.input_dir / str(patient_id) / f"{image_id}.dcm"
    output_file = args.output_dir / str(cancer) / f"{image_id}.png"

    if output_file.exists() and not args.force:
        return STATUS_SKIP

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            file = pydicom.read_file(input_file.open('rb'))
            raw_img = file.pixel_array
    except BaseException as e:
        if args.raise_errors:
            raise
        tqdm.write(
            f"Error processing image {image_id}: {e} "
            f"(run with --raise-errors to see the full error)"
        )
        return STATUS_ERR

    img = process_image(raw_img, args)

    output_file.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(output_file), img)

    return STATUS_OK


def open_as_zipfile(p: Path) -> Optional[zipfile.Path]:
    original_path = p
    while not p.is_file() and p.parents:
        p = p.parent
    
    if p.is_file() and p.suffix.lower() == '.zip':
        zf = zipfile.Path(p, str(original_path.relative_to(p)))
        return zf


def worker(
    task_queue: mp.SimpleQueue, done_queue: mp.SimpleQueue, df: pd.DataFrame, args: Args
):
    # Allow the input images to be inside of a zipfile
    zf = open_as_zipfile(args.input_dir)
    if zf:
        args.input_dir = zf  # type: ignore

    for idx in iter(task_queue.get, QUEUE_DONE):
        row = df.iloc[idx]
        result = process_row(row["patient_id"], row["image_id"], 0, args)
        done_queue.put(result)


def populate_queue(task_queue: mp.SimpleQueue, count: int):
    for idx in range(count):
        task_queue.put(idx)

def main() -> None:
    args = parse_args()

    df = pd.read_csv(str(args.csv_file))
    task_queue = mp.SimpleQueue()
    done_queue = mp.SimpleQueue()

    procs = []
    try:
        for _ in range(os.cpu_count()):
            p = mp.Process(target=worker, args=(task_queue, done_queue, df, args))
            p.start()
            procs.append(p)

        p = mp.Process(target=populate_queue, args=(task_queue, len(df)))
        p.start()
        procs.append(p)

        ok = 0
        err = 0
        skipped = 0
        pbar = tqdm(range(len(df)), desc=f"{args.input_dir}")
        for _ in pbar:
            result = done_queue.get()
            if result == STATUS_OK:
                ok += 1
            elif result == STATUS_ERR:
                err += 1
            else:
                skipped += 1
            pbar.set_postfix(ok=ok, err=err, skipped=skipped)
        pbar.close()
    finally:
        for p in procs:
            p.kill()


if __name__ == "__main__":
    main()
