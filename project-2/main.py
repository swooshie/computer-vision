# =================================================================================================
# NYU CS-GY 6643: Multi-Organ Nuclei Segmentation & Classification
#
# Description:
# This script fine-tunes a pre-trained CellPose model for instance segmentation and
# classification of cell nuclei from H&E-stained tissue images. It is designed to be
# run on an HPC environment with an NVIDIA GPU.
#
# Author: Aditya Jhaveri
# Date: October 30, 2025
# =================================================================================================

import os
import cv2
import torch
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from albumentations import Resize, Compose, Normalize
from albumentations.pytorch import ToTensorV2
import argparse
import warnings

# --- Import the Cell Segmentation Model ---
from cellseg_models_pytorch.models.cellpose import CellPose
from cellseg_models_pytorch.utils import FileHandler

# --- Suppress unnecessary warnings ---
warnings.filterwarnings("ignore", category=UserWarning)

# =================================================================================================
# 1. Configuration & Argument Parsing
# =================================================================================================

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a CellPose model for nuclei segmentation.")
    parser.add_argument('--base_dir', type=str, default="/vast/aaj6301/kaggle-data",
                        help='Base directory for the dataset.')
    parser.add_argument('--model_save_path', type=str, default="finetuned_cellpose_hpc.pth",
                        help='Path to save the fine-tuned model.')
    parser.add_argument('--img_size', type=int, default=512, help='Image size for training and inference.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for the optimizer.')
    return parser.parse_args()

class CONFIG:
    """Static configuration class."""
    CLASSES = ["Epithelial", "Lymphocyte", "Macrophage", "Neutrophil"]
    CLASS_MAP = {name: i + 1 for i, name in enumerate(CLASSES)}  # 1-based indexing for classes
    INV_CLASS_MAP = {i + 1: name for i, name in enumerate(CLASSES)}

# =================================================================================================
# 2. Data Handling and Utilities
# =================================================================================================

def rle_encode_instance_mask(mask: np.ndarray) -> str:
    """Encodes a 2D instance mask to a RLE triple string."""
    pixels = mask.flatten(order="F").astype(np.int32)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    rle = []
    for i in range(0, len(runs) - 1):
        start, val = runs[i], pixels[runs[i]]
        length = runs[i + 1] - start
        if val > 0:
            rle.extend([val, start, length])

    return " ".join(map(str, rle)) if rle else "0"

def create_training_maps(xml_path, shape):
    """Parses XML to create instance and type maps for training."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    inst_map = np.zeros(shape, dtype=np.int16)
    type_map = np.zeros(shape, dtype=np.int16)
    inst_id_counter = 1

    for ann_elem in root.findall('Annotation'):
        class_name = ann_elem.get('Name')
        if class_name not in CONFIG.CLASS_MAP:
            continue

        class_id = CONFIG.CLASS_MAP[class_name]

        for region_elem in ann_elem.findall('.//Region'):
            vertices = [(float(v.get('X')), float(v.get('Y'))) for v in region_elem.findall('.//Vertex')]
            polygon = np.array(vertices, dtype=np.int32)

            cv2.fillPoly(inst_map, [polygon], inst_id_counter)
            cv2.fillPoly(type_map, [polygon], class_id)
            inst_id_counter += 1

    return inst_map, type_map

class NucleiDataset(Dataset):
    """Custom PyTorch Dataset for loading nuclei images and masks."""
    def __init__(self, image_dir, xml_dir, file_ids, transform=None):
        self.image_dir = image_dir
        self.xml_dir = xml_dir
        self.file_ids = file_ids
        self.transform = transform

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        image_id = self.file_ids[idx]
        img_path = os.path.join(self.image_dir, f"{image_id}.tif")
        xml_path = os.path.join(self.xml_dir, f"{image_id}.xml")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape
        inst_map, type_map = create_training_maps(xml_path, (h, w))

        if self.transform:
            transformed = self.transform(image=image, masks=[inst_map, type_map])
            image = transformed["image"]
            inst_map, type_map = transformed["masks"]

        # The model expects a stacked label map
        labels = np.stack([inst_map, type_map], axis=0)
        return image, torch.from_numpy(labels).long()

# =================================================================================================
# 3. Main Training and Inference Functions
# =================================================================================================

def fine_tune_model(args, device):
    """Main function to handle model fine-tuning."""
    print("--- Starting Model Fine-Tuning ---")
    
    # --- Setup Model ---
    print("Loading pre-trained CellPose model...")
    model = CellPose.from_pretrained("hgsc_v1_efficientnet_b5")

    # Manually replace the final classification layer to match the new dataset
    in_channels = model.decoder.type_branch[-1].in_channels
    out_channels = len(CONFIG.CLASSES) + 1  # 4 classes + 1 background
    model.decoder.type_branch[-1] = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
    model.to(device)
    
    # --- Setup Data ---
    train_img_dir = os.path.join(args.base_dir, "train")
    train_xml_dir = os.path.join(args.base_dir, "train")

    train_transform = Compose([
        Resize(args.img_size, args.img_size),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

    all_files = [f.split('.')[0] for f in os.listdir(train_img_dir) if f.endswith(".tif")]
    train_dataset = NucleiDataset(train_img_dir, train_xml_dir, all_files, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # --- Training Loop ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        # Use tqdm for a progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            loss_dict = model(images, labels)
            loss = sum(l for l in loss_dict.values())
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {avg_loss:.4f}")
        
        # Save model checkpoint
        torch.save(model.state_dict(), args.model_save_path)
        print(f"Model checkpoint saved to {args.model_save_path}")

def run_inference(args, device):
    """Main function to run inference and generate submission."""
    print("\n--- Running Inference on Test Set ---")
    
    # --- Load Fine-Tuned Model ---
    model = CellPose(n_nuc_classes=len(CONFIG.CLASSES) + 1, enc_name="efficientnet-b5")
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    model.to(device)
    model.set_inference_mode()
    
    # --- Setup Data and Transforms ---
    test_img_dir = os.path.join(args.base_dir, "test_final")
    test_files = [f for f in os.listdir(test_img_dir) if f.endswith(".tif")]
    
    inf_transform = Compose([
        Resize(args.img_size, args.img_size),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    submission_data = []
    for filename in tqdm(test_files, desc="Generating Predictions"):
        image_id = filename.split('.')[0]
        img_path = os.path.join(test_img_dir, filename)
        
        image_orig = FileHandler.read_img(img_path)
        h_orig, w_orig, _ = image_orig.shape
        
        transformed = inf_transform(image=image_orig)
        image_tensor = torch.from_numpy(transformed["image"]).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            prob_map = model.predict(image_tensor)
            output = model.post_process(prob_map)["nuc"][0]
            pred_inst_map, pred_type_map = output[0], output[1]
            
            pred_inst_map = cv2.resize(pred_inst_map, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            pred_type_map = cv2.resize(pred_type_map, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        final_masks = {cls: np.zeros((h_orig, w_orig), dtype=np.int32) for cls in CONFIG.CLASSES}
        
        for inst_id in range(1, pred_inst_map.max() + 1):
            inst_mask = (pred_inst_map == inst_id)
            if inst_mask.sum() == 0: continue
            
            inst_type = np.median(pred_type_map[inst_mask]).astype(int)
            
            if inst_type in CONFIG.INV_CLASS_MAP:
                class_name = CONFIG.INV_CLASS_MAP[inst_type]
                final_masks[class_name][inst_mask] = inst_id

        row = {"image_id": image_id}
        for class_name in ["Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]:
            row[class_name] = rle_encode_instance_mask(final_masks[class_name])
        submission_data.append(row)

    # --- Create Submission File ---
    submission_df = pd.DataFrame(submission_data)
    submission_df = submission_df[["image_id", "Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]]
    submission_df.to_csv("submission.csv", index=False)

    print("\nSubmission file 'submission.csv' created successfully!")
    print(submission_df.head())

# =================================================================================================
# 4. Script Execution
# =================================================================================================

if __name__ == '__main__':
    args = get_args()

    # --- Set Device (CUDA for HPC) ---
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
        print("CUDA not available, using CPU. This will be very slow.")

    # --- Run Training ---
    fine_tune_model(args, DEVICE)
    
    # --- Run Inference ---
    run_inference(args, DEVICE)