# IndustrialGuard AI Viva Preparation

## Project overview

**Q: What does IndustrialGuard AI do?**  
A: It learns the normal visual appearance of a product category, scores a new image for anomaly, highlights suspicious regions, records the inspection locally, and produces a PDF report.

**Q: Why is it a mini-project rather than a production inspection system?**  
A: It is evaluated on two benchmark categories and has not been validated with a factory camera, production lighting, real operating speeds, or safety requirements.

**Q: What makes the project integrate three subjects?**  
A: OpenCV demonstrates image processing; the pretrained CNN features and PatchCore pipeline demonstrate deep-learning-based anomaly detection; the CNN autoencoder demonstrates neurons, activations, convolution, forward propagation, backpropagation, loss, and optimization.

## Dataset and experimental design

**Q: Why did you choose MVTec AD?**  
A: Its normal-only training split fits anomaly detection, it contains several defect types, and it supplies pixel-precise masks for localization evaluation.

**Q: How many images did you use?**  
A: The bottle and metal nut subsets contain 627 images total: 429 normal training images, 42 normal test images, and 156 defective test images with masks.

**Q: Were defective test images used in training?**  
A: No. PatchCore training uses only defect-free training images. The autoencoder trains on a seeded subset of defect-free training images and uses a separate normal-only validation subset. The official test split is held out for evaluation.

**Q: What is data leakage?**  
A: It occurs when information from evaluation examples influences training or threshold choices. This would make reported performance too optimistic. The project keeps the official test split out of model fitting and threshold selection.

**Q: Why use two separate models for bottle and metal nut?**  
A: Their normal appearances differ. One model per category builds a separate reference of normal patches for that product type.

## Image processing

**Q: What is image enhancement?**  
A: An operation that changes image appearance to make some details easier to inspect. It does not guarantee that a defect becomes easier for a model to detect.

**Q: What does a histogram show?**  
A: The number of pixels at each intensity level. Separate RGB histograms summarize channel intensity distributions.

**Q: What is CLAHE?**  
A: Contrast Limited Adaptive Histogram Equalization. It improves local contrast in tiles and clips histogram amplification to limit noise growth.

**Q: What does Gaussian blur do?**  
A: It replaces each pixel with a Gaussian-weighted neighborhood average. It reduces high-frequency noise but can soften small defects.

**Q: How is median filtering different?**  
A: It uses the median value in a local neighborhood. It is useful for isolated impulse noise and can preserve edges better than averaging.

**Q: How does Canny edge detection work?**  
A: It smooths the image, measures gradients, thins edges with non-maximum suppression, and uses two thresholds to link strong and weak edges.

**Q: Why not preprocess every model input with all filters?**  
A: Filters can erase small defects or alter texture. The app keeps the model input original and presents transformations as comparison views until controlled evaluation supports a change.

## Deep learning and neural networks

**Q: What is a CNN?**  
A: A neural network that uses convolution kernels to learn local spatial patterns such as edges, textures, and object parts.

**Q: What is transfer learning?**  
A: Reusing features from a model pretrained on a large dataset. PatchCore uses a pretrained Wide ResNet-50 feature extractor to represent image patches.

**Q: What is PatchCore?**  
A: A normal-only anomaly detector. It stores representative patch features from normal images and scores test patches by their distance from the stored normal features.

**Q: Why does PatchCore use a memory bank?**  
A: It preserves reference examples of normal local appearances. A coreset reduces redundant entries so inference uses less memory and time than a full bank.

**Q: What is an autoencoder?**  
A: An encoder compresses an input image into lower-resolution features; a decoder reconstructs the image. If an abnormal region is poorly reconstructed, its reconstruction error may indicate an anomaly.

**Q: What is the autoencoder loss?**  
A: Mean squared error between the input and its reconstruction, averaged across pixels and channels.

**Q: What are weights and biases?**  
A: Trainable parameters in a neural layer. Weights control input contributions and bias shifts a neuron's response.

**Q: What is an activation function?**  
A: A nonlinear function applied after a layer. ReLU outputs `max(0, x)` and lets stacked layers learn nonlinear patterns.

**Q: What is backpropagation?**  
A: The chain-rule procedure that computes how each parameter contributed to the loss and obtains its gradient.

**Q: What is gradient descent?**  
A: An optimization method that adjusts parameters in the direction that reduces loss. Adam adapts update sizes using running gradient statistics.

**Q: What is overfitting?**  
A: When a model performs well on training images but generalizes poorly. A held-out validation split and validation loss help detect it.

## Evaluation and interpretation

**Q: What is AUROC?**  
A: The area under the receiver operating characteristic curve. It summarizes ranking quality across thresholds; 1.0 indicates perfect ranking on that evaluated split.

**Q: What is F1-score?**  
A: The harmonic mean of precision and recall at a particular threshold. It is useful when both missed defects and false alarms matter.

**Q: What is pixel AUROC?**  
A: It evaluates how well anomaly-map values rank defective pixels above normal pixels across thresholds.

**Q: Why can high pixel AUROC coexist with lower pixel F1?**  
A: AUROC evaluates ranking over all thresholds, while F1 depends on one chosen threshold. A map may rank pixels well but produce an imperfect binary mask at its fixed threshold.

**Q: Is the anomaly score a probability?**  
A: No. PatchCore's score is distance-based. It must not be described as a probability or confidence without a separate calibration procedure.

**Q: What does a heatmap prove?**  
A: It shows where the model assigns higher anomaly scores. It does not prove the model's causal reasoning or guarantee an exact defect boundary.

**Q: What were the PatchCore results?**  
A: Bottle: image AUROC 1.000000, image F1 0.992000, pixel AUROC 0.985560, pixel F1 0.692171. Metal nut: 0.998045, 0.994595, 0.987006, and 0.810754 respectively. These are single-seed benchmark results.

**Q: What is the most visible weakness in the baseline?**  
A: Bottle pixel F1 is lower than its pixel AUROC, so the thresholded localization mask needs improvement.

## Application and engineering

**Q: Why use SQLite?**  
A: It provides local relational storage without a server. It is sufficient for a single-user academic prototype; a multi-user cloud service would need a server database and access controls.

**Q: Why keep images out of the database?**  
A: Images and reports are large binary artifacts. The database stores small metadata records and file paths, which is simpler to back up and query.

**Q: How does the app validate uploads?**  
A: It limits file size and pixel count, restricts extensions, decodes the image with Pillow, and converts valid images to RGB before inference.

**Q: What is the purpose of the evaluation report?**  
A: It records category, model settings, split rules, thresholds, metrics, timing caveats, checkpoint size, and known limitations so results can be interpreted and reproduced.

**Q: What would you do before real factory deployment?**  
A: Collect authorized production images, validate camera and lighting conditions, define acceptable false-negative and false-positive rates, run a prospective pilot with human review, measure latency and GPU memory, add access controls and monitoring, and obtain relevant safety/quality approvals.

## Quick answers for the demonstration

- Normal bottle sample: score `0.215387`, result `NORMAL`.
- Broken bottle sample: score `0.943801`, result `DEFECTIVE`.
- Metal nut bent sample: score `0.842154`, result `DEFECTIVE`.
- Scores are not probabilities.
- MVTec benchmark data is for non-commercial research use under its published license.
- Local start command: `$env:PYTHONPATH = "src"; .\.venv\Scripts\streamlit.exe run app.py`.
