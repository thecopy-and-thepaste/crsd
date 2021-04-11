#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This script normalize all the pages in a set of PDFs inside a filesystem folder.

    The normalization process consists on:
    1. Obtain all the pages in the PDF document
    2. Skew the text
    3. Shrink or expand the image to have 1000px width
    4. Store single pages
"""
import argparse

import cv2
import threading
import tempfile
import traceback

import numpy as np
import pandas as pd

from pathlib import Path
from pdf2image import convert_from_path
from scipy.ndimage.interpolation import rotate
from concurrent.futures import ThreadPoolExecutor

from pdb import set_trace as bp

# Rememeber to update PYTHON_PATH
# export PYTHONPATH=`pwd`:`pwd`/crsd
from logger import get_logger

log = get_logger(__name__)

IMG_WIDTH = 1200
LOCK = threading.Lock()
MAX_ROTATIONS = 3
MAX_WORKERS = 3

def deskew(image: np.array, delta: int = 1, limit: int = 5):
    """
    The deskew process consists on computing histograms and assign to them a score
    The better the score, the more aligned are the lines

    Parameters
    ----------
    image : np.array
        [Image converted to np.array]
    delta : int, optional
        [Step for ranging the possible angles], by default 1
    limit : int, optional
        [Max angle to make the deskew process], by default 5
    
    Returns
    -------
    [Tuple[int, np.array]]
        [
            best_angle: int 
                [Minimum angle found] 
            rotated: np.array
                [Img matrix rotated]
        ]
    """
    def hist_score(arr, angle):
        # Rotated img matrix
        data = rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)

        return score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    scores = []
    angles = np.arange(-limit, limit + delta, delta)

    for angle in angles:
        score = hist_score(threshold, angle)
        scores.append(score)

    # Best histogram score
    best_angle = angles[int(np.argmax(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated



def pdf_to_images(pdf_file: Path, dest_path: Path):
    """
    Converts a pdf into a folder with images, the process consists on:
    1. Convert pdf to imahges.
    2. Deskew images
    3. Resize to IMG_WIDTH 
    4. Store to dest_path

    Parameters
    ----------
    pdf_file : Path
        [Filepath pf the pdf document]
    dest_path : Path
        [Destination path for images]

    Raises
    ------
    ex
        [Raised if deskew process failed]
    ex
        [General exception]
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            images = convert_from_path(pdf_file, output_folder=temp_dir)
            images_dest = dest_path / pdf_file.stem

            with LOCK:
                images_dest.mkdir(exist_ok=True)


            for i in range(len(images)):
                # images contains an array of PIL images
                image_path = images_dest / f"{i}.jpg"
                img = np.array(images[i])

                # shift of image channel ordering since obtained from PIL
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                try:
                    best_angle = float("inf")
                    rotation_passes = 0

                    while np.abs(best_angle) >= 5 and rotation_passes < MAX_ROTATIONS:
                        best_angle, img = deskew(img)
                        rotation_passes += 1

                    if rotation_passes == 5:
                        log.warning(
                            f"Max number of rotations reached and best_angle: {best_angle}")
                except Exception as ex:
                    log.error(
                        f"An error was found while trying to deskew the image document")
                    raise ex

                # Finally resize to width == IMG_WIDTH and keep aspect ratio
                ratio = IMG_WIDTH / img.shape[1]
                width = int(img.shape[1] * ratio)
                height = int(img.shape[0] * ratio)

                dim = (width, height)
                img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

                with LOCK:
                    cv2.imwrite(str(image_path), img)

    except Exception as ex:
        log.error(ex)
        log.error(traceback.print_exc())
        raise ex


def run(source_path: Path,
        dest_path: Path = None,
        recursive: bool = True):
    """
    It acquires the pdf files to dispatch to the pdf->image converter.

    Parameters
    ----------
    source_path : Path
        [Folder path where pdfs are stored]
    dest_path : Path, optional
        [Destination path to store the images], by default None
    recursive : bool, optional
        [If true, we get in subfolders also], by default True
    """
    search_reg = "*.pdf"
    if recursive:
        search_reg = f'**/{search_reg}'

    pdfs = source_path.glob(search_reg)
    dest_path = dest_path or Path("dataset/img")
    dest_path.mkdir(exist_ok=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for pdf in pdfs:
            try:
                executor.submit(pdf_to_images,
                                pdf_file=pdf,
                                dest_path=dest_path)

            except Exception:
                log.error(traceback.print_exc())
                continue

    # Once finished the process we create a dataset that contains all the resulting
    # images in `dest_path`
    result_images = dest_path.glob("**/*.jpg")
    result_images = set(map(lambda x: "/".join(str(x).split("/")[-2:]),
                            result_images))

    res = pd.DataFrame(result_images, columns=["item"])
    res.to_csv(dest_path.parent / "dataset.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("-s", "--source_path",
                        help="Source path to search for PDF documents")
    parser.add_argument("-d", "--dest_path",
                        default=None,
                        help="Destination path to store the resulting jpgs from PDF documents")
    parser.add_argument("-r", "--recursive",
                        action="store_false",
                        help="True to search for PDF files also in subfolders")

    ARGS = parser.parse_args()

    assert ARGS.source_path, "Source path of PDF documents is not defined"
    source_path = Path(ARGS.source_path)

    assert source_path.is_dir(), \
        "Source path does not exist"

    run(source_path=source_path,
        dest_path=ARGS.dest_path,
        recursive=not ARGS.recursive)
