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

This project applies computer vision techniques to identify different types of rocks based on images. It involves data preprocessing, CNN-based training, and deployment optimizations. A total of ~500 images were used, with ~50 images per class.

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

### 1. Apple Silicon Acceleration
- Automatically uses **MPS backend** when available for faster training and inference.

### 2. Parallel Data Loading
- **Multiprocessing Compatibility**:  
On macOS, the DataLoader spawns new processes.  
→ Functions and arguments passed to the DataLoader must be **picklable**. This is handled in `preprocessing.py`.

- **Pinned Memory**:  
Although **Metal (MPS)** on macOS doesn't support pinned memory, support is available for **CUDA** devices.  
→ Pinned memory is **optional and can be enabled MANUALLY only if RAM is sufficient**. After pinned memory is enabled, set non_blocking=true in devices to enable asynchronous data transfer. 

- **Worker Initialization**:  
Custom `worker_init_fn` is used to prevent issues such as data duplication when using multiple DataLoaders.

### 3. Autograd optimisations
By encasing the evaluation and inference block within the torch.inference_mode() decorator, the creation of the computation graph and tracking of computations (by autograd for backward pass) are omitted. This results in better performance. Setting a model to evaluation only modifies certain parameters such as dropout and does not disable gradient tracking. Hence, it is not sufficient to set the model to evaluation. 

### 4.Experimenting with different model architectures, transfer learning and ensemble inference. 
A major performance bottleneck was encountered while using the custom CNN model. The performance of the model did not improve beyond 5/11 despite numerous attempts of hyperparameter finetuning. Hence, other model architectures were tested, including resnet, mobilenet, mobileViT and deity_tiny_patch. The resnet and mobilenet architectures were tested as they promised better receptive fields - which was crucial for textual details in rock classification. mobileViT and deity_tiny_patch were tested due to their performance in both field reception and global - local fusion, ie capturing dependencies between finer grain details and features that were higher up in the hierarchy.

Transfer learning was also attempted. The benefits of transfer learning are only observed for the transformer models: mobileViT and deity_tiny_patch. 


---

## 💡 Design Rationale

### Why modify and save images directly?
- Performing image format conversion **only once** helps reduce preprocessing time during repeated training/inference sessions.
- Converting all images to `.jpg` also ensures **consistent dimensions and file type**, simplifying data handling.

### Why not train with more images? 
The main rationale behind sticking to a small dataset is to challenge myself to innovate workarounds for a small dataset. A total of ~500 images are used in this project(~50 per class) and the following approaches were tested:

- **Aggresive but nuanced image augmentation**:  
Cropping, affine transformations, color jitter etc were used in the transformation pipeline. 

- **Using sum of training loss as opposed to mean (default configuration in most loss functions)**:  
It was thought that using the sum of training errors across the batch will be more effective at reducing biase (as the dataset is small) as compared to just using mean. Therefore, the initial plan is to start training the model using sum and finetune its performance subsequenrtly using mean. However, there were no compelling evidence to suggest this approach worked. 

- **Using KLDiv instead of CrossEntropyLoss**:  
Similar to the previous points, the decision to use KLDiv is motivated by the desire to overcome the constraints of having a small dataset by introducing more data points as KLDiv aims to minimise the difference in the 2 distribution. While this did not introduce a significant improvement in terms of the overall accuracy, it inspired subsequent attempts combine the strengths of both models using ensemble inference. 

- **Label smoothing in both Cross Entropy and KLdiv to reduce overfitting**:  
There wasn't a significant improvement.  

---

## 📝 Changelog

| Date       | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| 16-07-2025 | Autograd optimisation, hyperparameter finetuning & model architecture refining |
| 06-07-2025 | Improved preprocessing for multi-processing: pickling, pin memory, and `worker_init_fn` |
| 19-06-2025 | Updated training pipeline                                                   |
| 16-06-2025 | Uploaded working training pipeline                                           |

---
