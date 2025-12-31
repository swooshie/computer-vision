# Team - Binary Stars

# Aditya Jhaveri - aaj6301
# Arsh Panesar - ap9332

import os
from pathlib import Path
import numpy as np
import cv2
import tifffile
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from itertools import combinations
from scipy.spatial.distance import pdist
from scipy.spatial import procrustes
import argparse


# Set Up Paths for Loading in Images
ROOT_PATH = '/Users/swooshie/Desktop/User/Aditya/NYU/Semester/Fall-2025/Courses/Computer Vision/Projects/Repository/computer-vision/project01/Data_Project1/'

PATTERNS_SUBPATH = 'patterns'
TRAIN_SUBPATH = 'train'
VALIDATION_SUBPATH = 'validation'

# K:Name, V:Image
pattern_img_table = {}

# K: Name, V: (<SkyImage>.tif, [... <Patches>.png ...])
constellation_table = {}

# Maximum Number of Patches available per Constelation
max_patches = 0

# Need this to sort Constellation Directories according to Numbers, not lexicographically
def natural_sort_key(s):
    # Split text into digits and non-digits
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def load_data(root_dir: str, constellation_subdir: str):
    # First Load In the Patterns
    pattern_dir = root_dir + PATTERNS_SUBPATH
    pattern_dir_files = sorted(Path(pattern_dir).glob("*.png"))
    
    for pattern_file in pattern_dir_files:
        img = cv2.imread(str(pattern_file), cv2.IMREAD_UNCHANGED)
        img_name = pattern_file.name
        pattern_img_table[img_name] = img

    print("Pattern Images Loaded: ", pattern_img_table.keys())
    print("Number of Patterns: ", len(pattern_img_table))

    # Load In the Constellation Data
    constellation_dir = root_dir + constellation_subdir
    constellation_dir_list = sorted([str(p) for p in Path(constellation_dir).iterdir() if p.is_dir()], key=natural_sort_key)
    constellation_dir_list = [Path(p) for p in constellation_dir_list]
    # constellation_dir_sky_img = Path(constellation_dir).glob("*.tif")
    # constellation_dir_patches = sorted(Path(constellation_dir + '/patches').glob("*.png"))

    print("Loading Constellation Data...")

    max_patches = 0
    for dir in constellation_dir_list:
        sky_img_path = sorted(dir.glob("*.tif"))
        sky_img = tifffile.imread(sky_img_path[0])
        print("Sky Image: ", sky_img_path)

        patches_list = []
        patches_dir_files = sorted(Path(str(dir) + '/patches').glob("*.png"))
        print("Patches Found: ", patches_dir_files)
        
        for patch_file in patches_dir_files:
            patch_img = cv2.imread(str(patch_file), cv2.IMREAD_UNCHANGED)
            patches_list.append(patch_img)
        
        constellation_table[dir.name] = ( sky_img, patches_list )
        max_patches = max(max_patches, len(patches_list))

        print("Constellation " + dir.name + " data loaded.")
    
    print("Total Constellations: ", len(constellation_table))
    print("Maximum Patches: ", max_patches)

    return pattern_img_table, constellation_table, max_patches

# Preprocessing Images first

def preprocess_sky_image(img):

    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to 8-bit
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Enhance Contrast using Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)


    # Top-hat filtering to highlight stars
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    # Optional: light sharpening
    gaussian = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

    return img

def preprocess_patch_image(img):

    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize patch
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Enhance Contrast using Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)


    # Top-hat filtering to highlight stars
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    # Optional: light sharpening
    gaussian = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

    return img

def perform_image_preprocessing(constellation_table):
    for const_name, (sky_img, patch_img_list) in constellation_table.items():
        preprocessed_sky_img = preprocess_sky_image(sky_img)

        preprocessed_patch_img_list = [preprocess_patch_image(p) for p in patch_img_list]
        
        constellation_table[const_name] = (preprocessed_sky_img, preprocessed_patch_img_list)

# # Matching a Patch to a Sky Image

# Matching a Patch to a Sky Image
def match_patch_to_sky(sky_img, patch, scales=[0.8, 1.0, 1.2]):
    best_score = -1
    best_center = None

    for scale in scales:
        # Resize patch
        scaled_patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        if scaled_patch.shape[0] > sky_img.shape[0] or scaled_patch.shape[1] > sky_img.shape[1]:
            continue  # skip if patch is larger than sky

        # Template matching
        res = cv2.matchTemplate(sky_img, scaled_patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            top_left = max_loc
            center = (top_left[0] + scaled_patch.shape[1] // 2,
                      top_left[1] + scaled_patch.shape[0] // 2)
            best_center = center

    return best_center, best_score


# Run Matching for All Constellations with adaptive threshold
def run_patch_matching(constellation_table, scales=[0.5, 1.0, 1.5, 2.0], base_threshold=0.6):
    results = {}

    for const_name, (sky_img, patch_list) in constellation_table.items():
        print(f"\nPatch Matching for constellation: {const_name}")
        patch_scores = []
        patch_coords = []

        # Step 1: collect scores
        for patch in patch_list:
            coords, score = match_patch_to_sky(sky_img, patch, scales)
            patch_coords.append((coords, score))
            patch_scores.append(score)

        # Step 2: adaptive threshold (relative to max)
        # max_score = max(patch_scores) if patch_scores else 0
        # dynamic_thresh = max(base_threshold, 0.75 * max_score)
        mean_score = np.mean(patch_scores)
        std_score  = np.std(patch_scores)

        # Lower threshold to reduce false negatives
        dynamic_thresh = max(base_threshold, mean_score - 0.3 * std_score)

        # Step 3: accept/reject using adaptive threshold
        const_results = []
        for idx, (coords, score) in enumerate(patch_coords):
            if score >= dynamic_thresh and coords != -1:
                const_results.append(coords)
                print(f"  Patch {idx+1}: ACCEPTED at {coords} (score={score:.2f})")
            else:
                const_results.append(-1)
                print(f"  Patch {idx+1}: REJECTED (score={score:.2f})")

        results[const_name] = const_results

    return results

# Using the Accepted Patches, we will create a Sky Image mask of our own
def build_constellation_mask(sky_shape, patch_coords, patch_sizes):
    mask = np.zeros(sky_shape[:2], dtype=np.uint8)

    for (coord, size) in zip(patch_coords, patch_sizes):
        if coord == -1:
            continue
        x, y = coord
        radius = max(size) // 4  # heuristic: quarter of patch size
        cv2.circle(mask, (x, y), radius, 255, -1)

    return mask

# Constellation Classifier

# First, we have to preprocess the pattern images
def preprocess_pattern_image(img):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold: keep bright (stars), remove gray background/green lines
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Clean Background noise
    mask = cv2.medianBlur(mask, 3)

    return mask


def preprocess_sky_image_for_classification(img):

    # Ensure Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Adaptive threshold to isolate stars
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 15, -5
    )

    # Clean Background Noise
    mask = cv2.medianBlur(mask, 3)

    return mask

def distance_ratios(coords):
    D = pdist(coords)
    if D.max() > 0:
        D = D / D.max()
    return np.sort(D)

# Match Geometry
def extract_star_centroids(mask_img, thresh=200):
    # Ensure grayscale
    if len(mask_img.shape) == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, binary = cv2.threshold(mask_img, thresh, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # Ignore background (label 0)
    star_coords = []
    for i in range(1, num_labels):
        x, y = centroids[i]
        star_coords.append((x, y))

    return np.array(star_coords, dtype=np.float32)

def normalize_points_translation_and_scale(points):
    # Center
    centered = points - np.mean(points, axis=0)

    # Scale
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    if max_dist > 0:
        centered /= max_dist

    return centered


def match_with_ratios(sky_coords, pattern_coords, tol=0.05):
    n_stars = len(pattern_coords)
    if n_stars < 2 or len(sky_coords) < n_stars:
        return None, float("inf")

    pattern_ratios = distance_ratios(pattern_coords)

    best_score = float("inf")
    best_group = None

    # Try all combinations of sky stars of same size
    for group in combinations(range(len(sky_coords)), n_stars):
        candidate = np.array([sky_coords[i] for i in group])
        cand_ratios = distance_ratios(candidate)

        if len(cand_ratios) != len(pattern_ratios):
            continue

        # quick ratio check
        ratio_err = np.mean(np.abs(pattern_ratios - cand_ratios))
        if ratio_err > tol:
            continue

        # refine with Procrustes alignment
        sky_norm = normalize_points_translation_and_scale(candidate)
        pattern_norm = normalize_points_translation_and_scale(pattern_coords)

        try:
            _, _, disparity = procrustes(sky_norm, pattern_norm)
        except ValueError:
            continue

        if disparity < best_score:
            best_score = disparity
            best_group = candidate

    return best_group, best_score

def classify_constellation_using_geometry(sky_img, patch_coords, patches, pattern_img_table):

    # Preprocess sky
    # sky_mask = build_constellation_mask(sky_img.shape, patch_coords, )
    sky_mask = build_constellation_mask(sky_img.shape, patch_coords, [(p.shape[0], p.shape[1]) for p in patches])
    
    sky_coords = extract_star_centroids(sky_mask)

    best_score = float("inf")  # lower is better for disparity
    best_label = None

    for pattern_name, pattern_img in pattern_img_table.items():
        # Preprocess pattern
        pattern_mask = preprocess_pattern_image(pattern_img)

        pattern_coords = extract_star_centroids(pattern_mask)
        
        if len(pattern_coords) < 2:
            continue

        matched_group, score = match_with_ratios(sky_coords, pattern_coords, tol=0.05)

        if score < best_score:
            best_score = score
            best_label = pattern_name.replace("_pattern.png", "")

    return best_label, best_score

def write_submission_csv(all_patch_results, max_patches, constellation_table, pattern_img_table,
                         output_name="train_results.csv"):
    rows = []
    serial = 1

    for folder, patch_results in all_patch_results.items():
        row = {
            "S.no": serial,
            "Folder No.": folder,
        }

        # Fill patch columns
        for i in range(max_patches):
            if i < len(patch_results):
                val = patch_results[i]
                if val == -1:
                    row[f"patch {i+1}"] = -1
                else:
                    row[f"patch {i+1}"] = f"({val[0]},{val[1]})"  # (x,y) coords
            else:
                row[f"patch {i+1}"] = -1  # padding if fewer patches

        # Run classifier for this folder
        sky_img, patches = constellation_table[folder]
        label, score = classify_constellation_using_geometry(
            sky_img,
            patch_results,
            patches,
            pattern_img_table
        )

        row["Constellation prediction"] = label if label else "unknown"

        rows.append(row)
        serial += 1

    df = pd.DataFrame(rows)
    df.to_csv(output_name, index=False)
    print(f"CSV written to {output_name}")
    return df

# Need an Argument Parser so this script runs for the required input format
def parse_args():
    parser = argparse.ArgumentParser(description="Binary Stars: Constellation Classifier")

    # Root Folder for the Script
    parser.add_argument(
        "root_folder",
        type=str,
        help="Path to dataset root (contains 'patterns', 'train', 'validation', etc.)"
    )

    # Target Folder for the Script (train/validation/test)
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        required=True,
        help="Target folder name to process (e.g., train, validation, test)"
    )

    # Verbose flag
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    if args.verbose:
        print("Entered Arguments:")
        print(f"Root folder: {args.root_folder}")
        print(f"Target folder: {args.folder}")
    
    results_file_name = str(args.folder) + ("_results.csv")
    print("Will Write to: ", results_file_name)

    ROOT_PATH = args.root_folder + "/"
    FOLDER_PATH = args.folder

    pattern_img_table, constellation_table, max_patches = load_data(ROOT_PATH, FOLDER_PATH)

    perform_image_preprocessing(constellation_table)
    print("Preprocessed Images using Histogram Equalization and Gaussian Blur.")
    
    # Run matching over your preprocessed constellation_table
    all_patch_results = run_patch_matching(constellation_table, base_threshold=0.6)
    
    # Write the Output
    df = write_submission_csv(
        all_patch_results,
        max_patches,
        constellation_table,
        pattern_img_table,
        output_name=results_file_name
    )
    
    print(df.head())
