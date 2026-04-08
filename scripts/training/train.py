import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
import cv2
import rasterio
import geopandas as gpd
from shapely.geometry import box
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
TILE_SIZE            = 256
BATCH_SIZE           = 2
LEARNING_RATE        = 1e-4
MIN_LEARNING_RATE    = 1e-5
NUM_EPOCHS           = 50
WARMUP_EPOCHS        = 30      
LR_REDUCTION_FACTOR  = 0.85      
PATIENCE             = 6        
SEED                 = 100
NUM_WORKERS          = 0
VALIDATION_SPLIT     = 0.2  # 20% for validation

# # Paths - adjust these for your data structure
# IMG_DIR              = Path(__file__).parent / "data" / "planet" / "training"  # Contains TIF files
# MSK_DIR              = Path(__file__).parent / "data" / "planet" / "training"   # Contains PNG mask files  
# COASTLINE_SHP        = Path(__file__).parent / "data" / "shapefiles" / "coastline.shp"

# Paths - adjust these for your data structure
IMG_DIR              = Path(__file__).parent / "data"   # Contains TIF files
MSK_DIR              = Path(__file__).parent / "data"   # Contains PNG mask files  
COASTLINE_SHP        = Path(__file__).parent.parent.parent / "data" / "shapefiles" / "coastline.shp"

# ─── GLOBAL VARIABLES FOR TRACKING ─────────────────────────────────────────
TIMESTAMP = None
TRAINING_CONFIG = {}

# ─── DEVICE SELECTION ───────────────────────────────────────────────────────
def select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA (GPU)")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("🍏 Using Apple MPS (Metal)")
    else:
        device = torch.device("cpu")
        print("🧠 Using CPU")
    return device

# ─── SET SEED ───────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)

# ─── INITIALIZE TRAINING CONFIG ────────────────────────────────────────────
def initialize_training_config():
    """Initialize training configuration with timestamp"""
    global TIMESTAMP, TRAINING_CONFIG
    
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    TRAINING_CONFIG = {
        "timestamp": TIMESTAMP,
        "hyperparameters": {
            "tile_size": TILE_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "min_learning_rate": MIN_LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "lr_reduction_factor": LR_REDUCTION_FACTOR,
            "patience": PATIENCE,
            "seed": SEED,
            "num_workers": NUM_WORKERS,
            "validation_split": VALIDATION_SPLIT
        },
        "paths": {
            "img_dir": str(IMG_DIR.absolute()),
            "msk_dir": str(MSK_DIR.absolute()),
            "coastline_shp": str(COASTLINE_SHP.absolute()),
            "model_path": str(Path(__file__).parent / f"best_sediment_model_{TIMESTAMP}.pth"),
            "config_path": str(Path(__file__).parent / f"training_config_{TIMESTAMP}.json")
        },
        "data": {
            "images_processed": [],
            "train_tiles": [],
            "val_tiles": [],
            "total_tiles": 0,
            "tiles_with_signal": 0,
            "tiles_without_signal": 0
        },
        "results": {
            "best_val_iou": 0.0,
            "best_epoch": 0,
            "final_train_metrics": {},
            "final_val_metrics": {},
            "training_history": {
                "train_loss": [],
                "val_loss": [],
                "train_iou": [],
                "val_iou": [],
                "train_dice": [],
                "val_dice": [],
                "train_acc": [],
                "val_acc": []
            }
        }
    }
    
    print(f"🕒 Training session timestamp: {TIMESTAMP}")

# ─── MASK VERIFICATION HELPER ───────────────────────────────────────────────
def verify_binary_mask(mask_path: Path) -> np.ndarray:
    """Load and ensure mask is binary (0s and 255s)"""
    mask_img = Image.open(mask_path)
    
    # Convert to grayscale if needed
    if mask_img.mode != 'L':
        mask_img = mask_img.convert('L')
    
    mask_array = np.array(mask_img)
    
    # Check if already binary
    unique_vals = np.unique(mask_array)
    if len(unique_vals) <= 2 and all(v in [0, 255] for v in unique_vals):
        return mask_array
    
    # Apply threshold to binarize
    threshold = 128
    binary_mask = (mask_array >= threshold).astype(np.uint8) * 255
    return binary_mask

# ─── COASTLINE-BASED TILING FUNCTION ───────────────────────────────────────
def create_coastline_tiles(img_dir, msk_dir, coastline_shp, tile_size):
    """
    Create tiles based on coastline intersection logic.
    Returns metadata about tiles created for each image.
    """
    # Find all TIF files and their corresponding mask files
    tif_files = list(img_dir.glob("*.tif"))
    print(f"Found {len(tif_files)} TIF files")
    
    all_tiles_metadata = []
    
    for tif_path in tif_files:
        stem = tif_path.stem
        
        # Try different mask naming conventions
        possible_mask_names = [
            f"{stem}.png",
            f"{stem}_mask.png", 
            f"mask_{stem}.png",
            f"{stem}_plume_mask.png"
        ]
        
        mask_path = None
        for mask_name in possible_mask_names:
            potential_path = msk_dir / mask_name
            if potential_path.exists():
                mask_path = potential_path
                break
        
        if mask_path is None:
            print(f"⚠️ No mask found for {stem}, skipping")
            continue
            
        print(f"Processing {stem}...")
        
        # Store image info in config
        TRAINING_CONFIG["data"]["images_processed"].append({
            "image_stem": stem,
            "tif_path": str(tif_path.absolute()),
            "mask_path": str(mask_path.absolute())
        })
        
        # Process this image using coastline logic
        tiles_metadata = process_single_image_with_coastline(
            tif_path, mask_path, coastline_shp, tile_size, stem
        )
        
        all_tiles_metadata.extend(tiles_metadata)
    
    print(f"✅ Coastline-based tiling complete! Created {len(all_tiles_metadata)} tiles")
    
    # Update config with tile statistics
    TRAINING_CONFIG["data"]["total_tiles"] = len(all_tiles_metadata)
    TRAINING_CONFIG["data"]["tiles_with_signal"] = sum(1 for t in all_tiles_metadata if t["has_signal"])
    TRAINING_CONFIG["data"]["tiles_without_signal"] = TRAINING_CONFIG["data"]["total_tiles"] - TRAINING_CONFIG["data"]["tiles_with_signal"]
    
    return all_tiles_metadata

def process_single_image_with_coastline(tif_path, mask_path, coastline_shp, tile_size, stem):
    """Process a single TIF/mask pair using coastline intersection logic"""
    
    tiles_metadata = []
    
    # Step 1: Load TIF and get bounds/CRS
    with rasterio.open(tif_path) as src:
        n_cols, n_rows = src.width, src.height
        profile = src.profile
        tif_bounds = box(*src.bounds)
        tif_crs = src.crs
        
        # Step 2: Load and process coastline
        coastline = gpd.read_file(coastline_shp)
        if coastline.crs != tif_crs:
            coastline = coastline.to_crs(tif_crs)
        
        coastline = coastline[coastline.is_valid & coastline.geometry.notnull()]
        coastline_outline = coastline.copy()
        coastline_outline["geometry"] = coastline_outline.boundary
        coastline_union = coastline_outline.geometry.union_all()
        landmass_union = coastline.geometry.union_all()
        
        # Step 3: Create all possible tiles and check coastline intersection
        tile_records = []
        
        for i in range(0, n_rows, tile_size):
            for j in range(0, n_cols, tile_size):
                row, col = i // tile_size, j // tile_size
                width = min(tile_size, n_cols - j)
                height = min(tile_size, n_rows - i)
                
                # Only keep square tiles
                if width != tile_size or height != tile_size:
                    continue
                    
                window = Window(j, i, width, height)
                transform = src.window_transform(window)
                bounds = rasterio.windows.bounds(window, src.transform)
                tile_geom = box(*bounds)
                
                tile_records.append({
                    "row": row,
                    "col": col, 
                    "window": window,
                    "transform": transform,
                    "geometry": tile_geom,
                    "bounds": bounds
                })
        
        # Step 4: Filter tiles that intersect coastline and have content
        intersecting_tiles = []
        
        for tile_rec in tile_records:
            # Check coastline intersection
            if not tile_rec["geometry"].intersects(coastline_union):
                continue
                
            # Check if tile has actual content
            tile_data = src.read(window=tile_rec["window"])
            if np.all(tile_data == 0) or np.all(np.isnan(tile_data)):
                continue
                
            # Check mask coverage
            mask_data = src.read_masks(1, window=tile_rec["window"])
            if mask_data.mean() < 10:  # Skip tiles with very low mask coverage
                continue
                
            intersecting_tiles.append(tile_rec)
        
        # Step 5: Find adjacent tiles (touching intersecting tiles but not over land)
        adjacent_tiles = []
        for tile_rec in tile_records:
            if tile_rec in intersecting_tiles:
                continue
                
            # Check if touches any intersecting tile
            touches_coast_tile = any(
                tile_rec["geometry"].touches(coast_tile["geometry"]) 
                for coast_tile in intersecting_tiles
            )
            
            if touches_coast_tile and not tile_rec["geometry"].within(landmass_union):
                adjacent_tiles.append(tile_rec)
        
        # Step 6: Process tiles and extract data to memory
        all_selected_tiles = intersecting_tiles + adjacent_tiles
        
        # Load mask data once
        mask_array = verify_binary_mask(mask_path)
        
        for tile_rec in all_selected_tiles:
            row, col = tile_rec["row"], tile_rec["col"]
            tile_name = f"{stem}__{row:05d}_{col:05d}"
            
            # Extract image tile data
            tile_data = src.read(window=tile_rec["window"])
            
            # Convert to 8-bit RGB if needed
            if tile_data.dtype != np.uint8:
                tile_data = (tile_data >> 8).astype(np.uint8)
            
            # Ensure RGB (3 channels)
            if tile_data.shape[0] == 1:
                tile_data = np.repeat(tile_data, 3, axis=0)
            elif tile_data.shape[0] > 3:
                tile_data = tile_data[:3]
                
            # Convert to HWC format for PIL
            tile_img = np.transpose(tile_data, (1, 2, 0))
            
            # Extract mask tile
            i, j = row * tile_size, col * tile_size
            mask_tile = mask_array[i:i+tile_size, j:j+tile_size]
            
            # Check for signal (any white pixels)
            has_signal = np.any(mask_tile > 0)
            
            tiles_metadata.append({
                "image_stem": stem,
                "tile_name": tile_name,
                "category": "intersecting" if tile_rec in intersecting_tiles else "adjacent",
                "row": row,
                "col": col,
                "has_signal": has_signal,
                "image_data": tile_img,  # Store in memory
                "mask_data": mask_tile   # Store in memory
            })
    
    print(f"  → Created {len(tiles_metadata)} tiles for {stem}")
    return tiles_metadata

# ─── STRATIFIED TILE SPLITTING ─────────────────────────────────────────────
def create_stratified_splits(tiles_metadata, validation_split=0.2):
    """Create stratified train/validation splits maintaining signal/no-signal ratios."""
    # Separate tiles by signal presence
    signal_tiles = [t for t in tiles_metadata if t["has_signal"]]
    no_signal_tiles = [t for t in tiles_metadata if not t["has_signal"]]
    
    print(f"Total tiles: {len(tiles_metadata)}")
    print(f"  - With signal: {len(signal_tiles)}")
    print(f"  - Without signal: {len(no_signal_tiles)}")
    
    # Stratified split for each group
    if len(signal_tiles) > 0:
        signal_train, signal_val = train_test_split(
            signal_tiles, test_size=validation_split, random_state=SEED
        )
    else:
        signal_train, signal_val = [], []
    
    if len(no_signal_tiles) > 0:
        no_signal_train, no_signal_val = train_test_split(
            no_signal_tiles, test_size=validation_split, random_state=SEED
        )
    else:
        no_signal_train, no_signal_val = [], []
    
    # Combine splits
    train_tiles = signal_train + no_signal_train
    val_tiles = signal_val + no_signal_val
    
    # Shuffle
    random.shuffle(train_tiles)
    random.shuffle(val_tiles)
    
    print(f"Train split: {len(train_tiles)} tiles")
    print(f"  - With signal: {len(signal_train)}")
    print(f"  - Without signal: {len(no_signal_train)}")
    print(f"Validation split: {len(val_tiles)} tiles")
    print(f"  - With signal: {len(signal_val)}")
    print(f"  - Without signal: {len(no_signal_val)}")
    
    # Store split information in config (without image/mask data to keep JSON manageable)
    TRAINING_CONFIG["data"]["train_tiles"] = [
        {k: v for k, v in tile.items() if k not in ["image_data", "mask_data"]} 
        for tile in train_tiles
    ]
    TRAINING_CONFIG["data"]["val_tiles"] = [
        {k: v for k, v in tile.items() if k not in ["image_data", "mask_data"]} 
        for tile in val_tiles
    ]
    
    return train_tiles, val_tiles

# ─── DATASET CLASS ─────────────────────────────────────────────────────────
class SedimentSegmentationDataset(Dataset):
    def __init__(self, tiles_metadata, transform=None):
        self.tiles_metadata = tiles_metadata
        self.transform = transform

    def __len__(self):
        return len(self.tiles_metadata)

    def __getitem__(self, idx):
        tile_meta = self.tiles_metadata[idx]
        
        # Get data from memory
        img = tile_meta["image_data"]
        msk = tile_meta["mask_data"]

        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img, msk = aug['image'], aug['mask']

        msk = msk.unsqueeze(0).float() / 255.0
        return img, msk

# ─── LOSS FUNCTIONS ────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits, targets):
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt    = targets*probs + (1-targets)*(1-probs)
        w     = self.alpha * (1-pt).pow(self.gamma)
        loss  = w * bce
        return loss.mean() if self.reduction=="mean" else loss.sum()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce     = nn.BCEWithLogitsLoss()
        self.dice    = smp.losses.DiceLoss(mode="binary")
        self.focal   = FocalLoss(alpha=0.25, gamma=2.0)
        self.tversky = smp.losses.TverskyLoss(mode="binary", alpha=0.7)

    def forward(self, outputs, targets):
        return (
            self.bce(outputs, targets)
          + self.dice(outputs, targets)
          + 0.5 * self.focal(outputs, targets)
          + 0.5 * self.tversky(outputs, targets)
        )

# ─── TRAIN & VALIDATE ──────────────────────────────────────────────────────
def train_one_epoch(model, loader, optim, loss_fn, device):
    model.train()
    running, total_iou, total_dice, total_acc, batches = 0, 0, 0, 0, 0
    iou_batches = 0
    
    for imgs, msks in tqdm(loader, desc="Training"):
        imgs, msks = imgs.to(device), msks.to(device)
        optim.zero_grad()
        outs = model(imgs)
        loss = loss_fn(outs, msks)
        loss.backward()
        optim.step()
        running += loss.item()

        with torch.no_grad():
            preds = (torch.sigmoid(outs)>0.5).float()

            # Compute per-sample intersection/union over the full batch
            inter_all = (preds * msks).sum(dim=(1,2,3))
            union_all = ((preds + msks) > 0).sum(dim=(1,2,3))

            # Only accumulate IoU on samples that actually have GT signal
            mask_area = msks.sum(dim=(1,2,3))
            valid     = mask_area > 0
            if valid.any():
                total_iou  += (inter_all[valid] / (union_all[valid] + 1e-7)).mean().item()
                iou_batches += 1

            # Dice & accuracy on full batch
            total_dice += (2 * inter_all /
                           (preds.sum((1,2,3)) + msks.sum((1,2,3)) + 1e-7)
                          ).mean().item()
            total_acc  += (preds == msks).float().mean().item()
            batches    += 1

    avg_iou = total_iou / iou_batches if iou_batches > 0 else 0.0
    return {
        "loss":     running/batches,
        "iou":      avg_iou,
        "dice":     total_dice/batches,
        "accuracy": total_acc/batches
    }

def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss, total_iou, total_dice, total_acc, batches = 0, 0, 0, 0, 0
    iou_batches = 0
    
    with torch.no_grad():
        for imgs, msks in tqdm(loader, desc="Validation"):
            imgs, msks = imgs.to(device), msks.to(device)
            outs = model(imgs)
            val_loss += loss_fn(outs, msks).item()

            preds = (torch.sigmoid(outs)>0.5).float()

            # Full‐batch intersection/union
            inter_all = (preds * msks).sum(dim=(1,2,3))
            union_all = ((preds + msks) > 0).sum(dim=(1,2,3))

            # Only accumulate IoU where GT has signal
            mask_area = msks.sum(dim=(1,2,3))
            valid     = mask_area > 0
            if valid.any():
                total_iou  += (inter_all[valid] / (union_all[valid] + 1e-7)).mean().item()
                iou_batches += 1

            # Dice & accuracy on full batch
            total_dice += (2 * inter_all /
                            (preds.sum((1,2,3)) + msks.sum((1,2,3)) + 1e-7)
                            ).mean().item()
            total_acc  += (preds == msks).float().mean().item()
            batches    += 1

    avg_iou = total_iou / iou_batches if iou_batches > 0 else 0.0
    return {
        "loss":     val_loss/batches,
        "iou":      avg_iou,
        "dice":     total_dice/batches,
        "accuracy": total_acc/batches
    }

# ─── WARMUP + PLATEAU SCHEDULERS ───────────────────────────────────────────
def get_schedulers(optimizer):
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch+1) / float(WARMUP_EPOCHS)
        return 1.0

    warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)
    plateau = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=LR_REDUCTION_FACTOR,
        patience=PATIENCE,
        verbose=True,
        min_lr=MIN_LEARNING_RATE
    )
    return warmup, plateau

# ─── SAVE TRAINING CONFIG ──────────────────────────────────────────────────
def save_training_config():
    """Save complete training configuration and results to JSON"""
    config_path = TRAINING_CONFIG["paths"]["config_path"]
    
    with open(config_path, 'w') as f:
        json.dump(TRAINING_CONFIG, f, indent=2)
    
    print(f"✅ Training configuration saved to {config_path}")

# ─── MAIN EXECUTION ────────────────────────────────────────────────────────
def main():
    # Initialize timestamp and config at the very beginning
    initialize_training_config()
    
    device = select_device()
    set_seed(SEED)

    # Step 1: Create coastline-based tiles (in memory)
    print("🌊 Creating coastline-based tiles...")
    tiles_metadata = create_coastline_tiles(
        IMG_DIR, MSK_DIR, COASTLINE_SHP, TILE_SIZE
    )
    
    if len(tiles_metadata) == 0:
        print("❌ No tiles created! Check your data paths and coastline file.")
        return

    # Step 2: Create stratified train/validation splits
    print("\n📊 Creating stratified train/validation splits...")
    train_tiles, val_tiles = create_stratified_splits(tiles_metadata, VALIDATION_SPLIT)
    
    if len(val_tiles) == 0:
        print("⚠️ No validation tiles! Consider reducing validation split or checking data.")
        return

    # Step 3: Create datasets and data loaders
    print("\n🔄 Setting up data loaders...")
    train_tf = A.Compose([
        A.HorizontalFlip(p=0.4), 
        A.RandomRotate90(p=0.4),
        A.RandomBrightnessContrast(p=0.4),
        A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ToTensorV2()],
        additional_targets={'mask':'mask'}
    )
    val_tf = A.Compose([
        A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ToTensorV2()],
        additional_targets={'mask':'mask'}
    )

    train_ds = SedimentSegmentationDataset(train_tiles, transform=train_tf)
    val_ds   = SedimentSegmentationDataset(val_tiles,   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Step 4: Initialize model, loss, optimizer, and schedulers
    print("\n🤖 Initializing model...")
    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    loss_fn   = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    warmup_sched, plateau_sched = get_schedulers(optimizer)

    # Step 5: Training loop
    print(f"\n🚀 Starting training for {NUM_EPOCHS} epochs...")
    best_iou = 0.0
    history = TRAINING_CONFIG["results"]["training_history"]

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
        tm = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        vm = validate(model, val_loader,   loss_fn, device)
        
        # Store metrics
        history["train_loss"].append(tm["loss"])
        history["val_loss"].append(vm["loss"])
        history["train_iou"].append(tm["iou"])
        history["val_iou"].append(vm["iou"])
        history["train_dice"].append(tm["dice"])
        history["val_dice"].append(vm["dice"])
        history["train_acc"].append(tm["accuracy"])
        history["val_acc"].append(vm["accuracy"])

        # Step schedulers
        if epoch < WARMUP_EPOCHS:
            warmup_sched.step()
        else:
            plateau_sched.step(vm["iou"])

        lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss: {tm['loss']:.4f} | Train IoU: {tm['iou']:.4f} | Train Dice: {tm['dice']:.4f}")
        print(f"Val Loss: {vm['loss']:.4f} | Val IoU: {vm['iou']:.4f} | Val Dice: {vm['dice']:.4f} | LR: {lr:.6f}")

        # Save best model
        if vm["iou"] > best_iou:
            best_iou = vm["iou"]
            model_path = TRAINING_CONFIG["paths"]["model_path"]
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved best model (IoU={best_iou:.4f}) to {model_path}")
            
            # Update config with best results
            TRAINING_CONFIG["results"]["best_val_iou"] = best_iou
            TRAINING_CONFIG["results"]["best_epoch"] = epoch + 1
            TRAINING_CONFIG["results"]["final_train_metrics"] = tm
            TRAINING_CONFIG["results"]["final_val_metrics"] = vm

    # Step 6: Plot training history
    print("\n📈 Plotting training history...")
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2,3,1)
    plt.plot(range(1,NUM_EPOCHS+1), history["train_loss"], label="Train Loss")
    plt.plot(range(1,NUM_EPOCHS+1), history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss")

    plt.subplot(2,3,2)
    plt.plot(range(1,NUM_EPOCHS+1), history["train_iou"], label="Train IoU")
    plt.plot(range(1,NUM_EPOCHS+1), history["val_iou"],   label="Val IoU")
    plt.xlabel("Epoch"); plt.ylabel("IoU"); plt.legend(); plt.title("IoU")

    plt.subplot(2,3,3)
    plt.plot(range(1,NUM_EPOCHS+1), history["train_dice"], label="Train Dice")
    plt.plot(range(1,NUM_EPOCHS+1), history["val_dice"],   label="Val Dice")
    plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.legend(); plt.title("Dice Score")

    plt.subplot(2,3,4)
    plt.plot(range(1,NUM_EPOCHS+1), history["train_acc"], label="Train Acc")
    plt.plot(range(1,NUM_EPOCHS+1), history["val_acc"],   label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy")

    plt.tight_layout()
    history_plot_path = Path(__file__).parent / f"training_history_{TIMESTAMP}.png"
    plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Add plot path to config
    TRAINING_CONFIG["paths"]["training_history_plot"] = str(history_plot_path)

    # Step 7: Save complete training configuration
    print(f"\n💾 Saving training configuration...")
    save_training_config()
    
    print(f"\n🎉 Training complete! Best IoU: {best_iou:.4f}")
    print(f"📁 Files saved:")
    print(f"  - Model: {TRAINING_CONFIG['paths']['model_path']}")
    print(f"  - Config: {TRAINING_CONFIG['paths']['config_path']}")
    print(f"  - History: {TRAINING_CONFIG['paths']['training_history_plot']}")

if __name__ == "__main__":
    main()