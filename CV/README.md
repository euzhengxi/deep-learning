# 🪨 Computer Vision: Rock Identification

This README documents key implementation details, design decisions, and changes for the **Rock Identification** project using Computer Vision techniques.

---

## 📄 Contents

1. [General Overview](#general-overview)  
2. [Pre-run Instructions](#pre-run-instructions)  
3. [Optimizations Implemented](#optimizations-implemented)  
4. [Design Rationale](#design-rationale)  
5. [Changelog](#changelog)

---

## 🧠 General Overview

This project applies computer vision techniques to identify different types of rocks based on images. It involves data preprocessing, CNN-based training, and deployment optimizations.

**Supported Rock Types**:
- Anthracite  
- Conglomerate  
- Flint  
- Granite  
- Limestone  
- Marble  
- Obsidian  
- Sandstone  
- Slate  

---

## ⚠️ Pre-run Instructions

Please take note of the following **before running any scripts**:

1. **Dataset Structure**:  
   Organize your training and testing folders in the following structure:  
   ```
   train/Anthracite, train/Conglomerate, ..., test/Slate
   ```

2. **Image Modification Warning**:  
The `preprocessing` function **modifies images directly** (resizing and converting to `.jpg`).  
> 💾 Please back up the original images if needed.

3. **Adding New Data**:  
If new images are added, re-run the preprocessing or amend the appropriate function accordingly.

---

## ⚙️ Optimizations Implemented

### 1. Parallel Data Loading
- **Multiprocessing Compatibility**:  
On macOS, the DataLoader spawns new processes.  
→ Functions and arguments passed to the DataLoader must be **picklable**. This is handled in `preprocessing.py`.

- **Pinned Memory**:  
Although **Metal (MPS)** on macOS doesn't support pinned memory, support is available for **CUDA** devices.  
→ Pinned memory is **optional and can be enabled MANUALLY only if RAM is sufficient**. After pinned memory is enabled, set non_blocking=true in devices to enable asynchronous data transfer. 

- **Worker Initialization**:  
Custom `worker_init_fn` is used to prevent issues such as data duplication when using multiple DataLoaders.

### 2. Apple Silicon Acceleration
- Automatically uses **MPS backend** when available for faster training and inference.

---

## 💡 Design Rationale

### Why modify and save images directly?
- Performing image format conversion and resizing **only once** helps reduce preprocessing time during repeated training/inference sessions.
- Converting all images to `.jpg` also ensures **consistent dimensions and file type**, simplifying data handling.

### Handling Small and Unbalanced Datasets
- Techniques like **data augmentation**, **oversampling**, and **weighted loss functions** can be considered to mitigate class imbalance and overfitting.

---

## 📝 Changelog

| Date       | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| 06-07-2025 | Improved preprocessing for multi-processing: pickling, pin memory, and `worker_init_fn` |
| 19-06-2025 | Updated training pipeline                                                   |
| 16-06-2025 | Uploaded working training pipeline                                           |

---
