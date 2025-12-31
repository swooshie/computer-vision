# Computer Vision Portfolio

This repository aggregates my coursework for CS-GY 6643 (Computer Vision). Each project explores a different part of the curriculum—classical image processing, detection and tracking, medical imaging, multi-modal Kaggle challenges, and geo-localization. The repo also carries the virtual environment that I used locally, model checkpoints, and write-ups delivered with each submission.

> **Note:** The `.gitignore` file keeps large artifacts (models, datasets, intermediate outputs, envs) out of version control. If you add new experiments, please store any heavy data in the existing `/data`, `/output`, or `/checkpoints` folders instead of committing it.

## Getting Started

```bash
# optional: recreate the provided virtual environment
python3 -m venv cell_project_env
source cell_project_env/bin/activate
pip install --upgrade pip

# install project-specific requirements as needed
pip install -r project-2/requirements.txt
```

Python 3.12 was used for the original work (`cell_project_env/pyvenv.cfg` shows the exact interpreter path). Each sub-project pins its own dependencies in notebooks or requirement files—activate the env and install only what you need right before running a notebook or script.

## Repository Layout

| Path | Description |
| --- | --- |
| `project01/` | Project 1 notebooks/scripts for astronomy-style image restoration, star template matching, and detection (Q1–Q5 deliverables plus final figures). |
| `project-2/` | Project 2: Multi-Organ Nuclei Segmentation & Classification (Kaggle). Contains HPC-ready training script (`main.py`), data folders, submissions, and helper shell scripts. |
| `project-3/` | Project 3: Kaggle baseball pitch-tracking challenge (classification/regression) plus assignment Q1–Q6 write-ups and notebooks. |
| `project-4/` | Project 4: GeoGuessr Street View geolocation challenge. Includes data, training notebook, checkpoints, and submissions. |
| `cell_project_env/` | Local Python 3.12 virtual environment used while developing the notebooks. Recreate or remove as needed. |
| `problem-description.txt`, `Project*_*.pdf`, `PRD_greenfield.md` | Assignment briefs, reports, and a PRD captured during ideation. |

## Core Problems & Skill Highlights

| Project | Problem Solved | Skills Demonstrated |
| --- | --- | --- |
| Project 01 | Restored degraded astronomical plates, matched noisy telescope patches back to sky panoramas, and validated detections with geometric alignment. | Image restoration, histogram/CLAHE contrast tricks, morphological filtering, template matching, Procrustes analysis, scientific visualization. |
| Project 2 | Built an end-to-end nuclei segmentation system that parses XML polygons, trains CellPose on multi-organ tissue, and exports weighted Panoptic Quality submissions. | Label parsing, Albumentations, PyTorch fine-tuning, HPC orchestration, RLE encoding, class-imbalance strategies. |
| Project 3 | Combined course assignments with a Kaggle baseball pitch tracker: visualized CNN abstraction, ran multi-object tracking, and fused tabular + vision models for pitch outcome prediction. | Transfer learning diagnostics, Faster R-CNN tracking, Kalman filtering, YOLO/Ultralytics, multimodal feature engineering, Kaggle workflow management. |
| Project 4 | Tackled a GeoGuessr-style street-view localization challenge by training torchvision backbones to classify U.S. states from panoramas and resume training from checkpoints. | Geospatial data handling, large-image preprocessing, PyTorch training loops, transfer learning, inference pipelines, submission automation. |

## Data & Storage Conventions

- Datasets live inside each project directory (`project01/Data_Project1`, `project-2/kaggle-data`, `project-3/data/Question4/...`, `project-4/data/kaggle_dataset`, etc.). These paths mirror the structure distributed by the instructors or Kaggle.
- Intermediate results (`output`, `output-q3`, `output_dir_keras_*`, submission CSVs, `checkpoints`, etc.) stay beside the code that generated them. `.gitignore` already excludes these directories, so add any new cache/output folders underneath the matching project root.
- Large checkpoints (e.g., `best_model.pth`, `.pt` YOLO weights) are referenced but not meant to be tracked. Keep only symlinks or download scripts if you need reproducibility on another machine.

## Project Guides

### Project 01 – Astronomical Image Processing & Template Matching

- **Problem statement:** Given low-contrast telescope scans plus small template patches, reconstruct the clean sky panorama, locate each pattern instance, and quantify alignment quality for a midterm report.
- **Solution outline:** Designed a classical vision pipeline that denoises and enhances plates (CLAHE + top-hat filters), normalizes all constellation patches, runs multi-scale template matching with adaptive cutoffs, then applies Procrustes analysis to validate the recovered constellations. Final detections, histograms, and matching overlays are saved as `.jpg`.
- **Highlights:** `aaj6301_Q5.py` contains the reusable CLI pipeline (data loading, preprocessing, adaptive matching, geometric alignment). Deliverables in `Final/` show restored imagery (`astronomical_*`, `processed_*`, `restored_*`).
- **Data layout:** Place the provided train/pattern datasets under `project01/Data_Project1/` exactly as distributed. The script automatically scans `patterns/` and the numbered constellation folders.
- **Running the pipeline:**  
  ```bash
  cd project01/aaj6301-Project01
  python3 aaj6301_Q5.py --help        # view CLI flags such as --root_dir, --threshold, --scales
  python3 aaj6301_Q5.py --root_dir ../Data_Project1/train/
  ```
- **Outputs:** Processed sky images and matching visualizations land in `project01/output*` and `project01/Final/outputs-4/`. Figures referenced in the report are already rendered into `.jpg` files.

### Project 2 – Multi-Organ Nuclei Segmentation & Classification

- **Problem statement:** For H&E-stained tissue from four organs, detect every nucleus, segment its boundary, and classify it into four immune/tumor classes so the submission maximizes weighted PQ on Kaggle.
- **Solution outline:** Parsed the XML polygons provided with the course dataset, generated instance/type maps on the fly, and fine-tuned a CellPose backbone using Albumentations-based resizing and normalization. The training script supports HPC scheduling, class-specific losses, and exports RLE submissions directly.
- **Goal:** Instance-segment and classify epithelial, lymphocyte, macrophage, and neutrophil nuclei for the Kaggle/CS-GY challenge (see `project-2/problem-description.txt`).
- **Core script:** `project-2/main.py` fine-tunes `cellseg-models-pytorch` (CellPose) on the provided XML annotations. It handles XML parsing, Albumentations transforms, PyTorch dataset/dataloader creation, fine-tuning, and RLE submission generation.
- **Environment:** Install packages from `project-2/requirements.txt`. The script was designed for an HPC cluster (see `send.sh`, `hcp.sh`, and `submit.sbatch`), but it also runs locally as long as CUDA is available.
- **Training example:**
  ```bash
  cd project-2
  pip install -r requirements.txt
  python3 main.py \
    --base_dir /path/to/kaggle-data \
    --model_save_path checkpoints/finetuned_cellpose.pth \
    --img_size 512 --batch_size 4 --epochs 20 --lr 1e-5
  ```
- **Artifacts:** Multiple experiment directories (`combinations-maskrcnn*`, `output_dir_keras*`, `output-resnet50`, `output`) capture ablations with Mask R-CNN, Keras, and metric learning approaches. Submission CSVs (`submission_*.csv`) correspond to Kaggle uploads and are ignored going forward.

### Project 3 – Baseball Pitch Tracking (Kaggle) + Assignment Q1–Q6

- **Problem statement:** Interpret convolutional features, build a multi-car tracking system, and compete in a baseball pitch-tracking Kaggle challenge predicting pitch types/locations from video and tabular data.
- **Assignment portion:** `notebook.ipynb` walks through Q1 (feature visualization via ResNet/DenseNet), Q2 (vehicle detection, Kalman-filter tracking, and speed estimation), and Q6 (PDF write-ups in `Project03-Q6.pdf`). Supporting markdown answers and plots are embedded inside the notebook.
- **Kaggle challenge:** Subdirectory notebooks (`kaggle_gpt.ipynb`, `kaggle_gemini.ipynb`, `kaggle_claude.ipynb`, `naman.ipynb`) explore different workflows—classical pipelines, YOLOv8, baseballcv toolkit, and hybrid tabular/NN models. Data is expected under `project-3/data/Question4/baseball-pitch-tracking-cs-gy-6643/...`.
- **Models:** pretrained YOLO weights (`yolov5s.pt`, `yolov8*.pt`) plus `best_pitch_model.pth`. A small helper script writes submissions (e.g., `submission_hybrid_final_lower_lr5e6.csv`).
- **Running notebooks:** Use Jupyter or VS Code, ensuring that `BASE_DIR` and `SAMPLE_SUB_PATH` point to the extracted Kaggle dataset. Generated features (`train_video_features_nn.csv`, `test_video_features_nn.csv`) are ignored for future commits.

### Project 4 – GeoGuessr Street View Geo-Localization

- **Problem statement:** Predict the U.S. state from a single Google Street View style panorama—essentially a GeoGuessr Kaggle clone emphasizing large-image classification and class imbalance.
- **Solution outline:** Implemented a transfer-learning pipeline driven by configurable modes (train, resume, infer). Leveraged torchvision encoders, stratified splits, augmentation hooks, checkpointing, and reproducible submission writers. This project showcases ability to move from dataset ingestion through resume-able training to final inference.
- **Dataset:** `project-4/data/geo-guessr-street-view-cs-gy-6643.zip` (raw) expands to `data/kaggle_dataset/` with `train_images/`, `test_images/`, `train_ground_truth.csv`, and `state_mapping.csv`.
- **Notebook:** `project-4/notebook.ipynb` contains a configurable PyTorch pipeline for training/resuming/inference (`MODE = "train"|"resume"|"infer"`). It uses transfer learning on torchvision backbones, wraps datasets/dataloaders, and exports Kaggle-ready CSVs.
- **Checkpoints & submissions:** Saved under `project-4/checkpoints/` (`best_model*.pth`) and `submission*.csv`. Use `BEST_CKPT_PATH` in the notebook to resume training or generate predictions.

## Virtual Environment Notes

- `cell_project_env/` is a Python 3.12 virtual environment captured for reproducibility. You can delete it (git ignores it) or recreate it via `python3 -m venv cell_project_env`.
- When switching between projects, install only the dependencies you need. For notebooks that install extra libraries inline (e.g., `baseballcv`, `ultralytics`), re-run the pip cells inside the activated environment.

## Contribution & Maintenance Tips

1. **Keep large files out of Git:** The `.gitignore` already covers dataset folders, outputs, submissions, and checkpoints. If you spin up a new experiment, add its cache directory to `.gitignore` before training.
2. **Document runs:** Each notebook already logs parameters in markdown. Continue the pattern so teammates can reproduce your settings.
3. **Re-use helpers:** Scripts such as `project-2/main.py` or `project01/aaj6301_Q5.py` accept CLI flags—prefer updating those flags rather than editing constants in the code.
4. **Testing:** Run lightweight sanity checks (visualizations, small validation splits) before launching full training, especially on HPC resources referenced in `project-2/submit.sbatch`.

Feel free to open an issue or leave comments in the notebooks if additional clarification is needed about any experiment. This README should provide enough context to onboard quickly and stay organized across all four course projects.
