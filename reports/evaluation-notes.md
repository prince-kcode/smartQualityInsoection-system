# Initial PatchCore evaluation

Run date: 2026-10-31  
Dataset: MVTec AD, bottle and metal_nut  
Model: Anomalib PatchCore, pretrained Wide ResNet-50, feature layers `layer2` and `layer3`  
Input size: 256 x 256  
Seed: 42  
Device: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM  
Validation: synthetic anomalies from the training split; the official test split was reserved for evaluation.

| Category | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 |
| --- | ---: | ---: | ---: | ---: |
| Bottle | 1.000000 | 0.992000 | 0.985560 | 0.692171 |
| Metal nut | 0.998045 | 0.994595 | 0.987006 | 0.810754 |

These are single-seed baseline results, not a final comparison or a guarantee of factory performance. F1 depends on the threshold calibrated during validation. Bottle localization has a notably lower pixel F1 than its AUROC, so defect-region thresholding and visual inspection need further work.

The standalone inference path correctly classified one normal bottle image as normal, one defective bottle image as defective, and one defective metal nut image as defective. Inference took about 1.16–1.20 seconds per image after the model was loaded. These timings exclude model startup and checkpoint loading.
