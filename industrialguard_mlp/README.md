# IndustrialGuard AI — clean MLP rebuild

This clean rebuild follows the core two-model project design:

1. **Product recognition:** an MLP category classifier using the industrial datasets, PharmaVision tablet photos, ETID e-waste product photos, and a dedicated plastic-shampoo-bottle subtype. The currently configured label set has 41 categories, including generic `bottle` and separate `plastic_shampoo_bottle` labels, plus cable, keyboard, laptop, mobile phone, mouse, and printed circuit board.
2. **Wear/damage inspection:** a separate ResNet-18 U-Net trained from industrial defect masks, producing a highlighted candidate region and an image-level screening score. For plastic shampoo bottles only, an optional YOLO11n detector uses the source's real defect boxes and five type × two grade labels. Other categories continue through the mask screen and MVTec-trained broad type head. These outputs are domain-limited and do not assign universal severity or sale readiness.

The app can analyze multiple uploaded or camera-captured images in one run. The **Industry focus** selector groups manual product choices into electronics/e-waste, packaging, food, pharma, and manufacturing/materials; **All industries** shows every trained category. It only filters the manual list, while automatic detection continues to check all 41 labels. Product category and damage score are shown separately; a manual category selection overrides automatic routing. OpenCLIP can suggest familiar product labels outside the trained MLP categories, and **Other product (manual)** lets the user enter a label and run the damage model. For either path, damage regions are exploratory because the threshold was calibrated on industrial benchmark images. This is not a universal object classifier.

The damage model localizes candidate regions from industrial benchmarks. The MVTec-trained type head can suggest scratch, crack, hole/puncture, cut/tear, broken/damaged, missing/misplaced part, deformation, or surface/contamination labels when its validation threshold is met. Those labels come from dataset folder names and are not universal; the bottle head has separate bottle-only types and marginal/unacceptable grades. General minor/major severity and sale suitability still require broader product-specific labels and human review.

## Data folders

Place the downloaded/extracted sources under `datasets/`:

- `mvtec_ad/` — linked to the existing local MVTec AD extraction.
- `visa/` — VisA images plus official `split_csv/1cls.csv`.
- `mvtec_ad_2/` — MVTec AD 2 public data; private test data is excluded.
- `additional/etid/` — verified ETID product photos and a global source-group-held-out manifest; used only for category training and evaluation.
- `neu_cls/NEU-CLS/` — six-class NEU-CLS BMP images.
- `additional/PharmaVision/` — supplemental pill images; official train, validation, and test splits are used, while augmented copies are retained but excluded.
- `additional/Plastic Shampoo Bottle Dataset/` — official YOLO train/valid/test annotations for five bottle defect types with marginal/unacceptable grades.

The training script reads the benchmark folders and ETID manifest, keeps category-only ETID and shampoo-bottle subtype images out of normal-reference banks, and writes a dataset inventory and evaluation metrics. The generic `bottle` class and shampoo subtype have separate inference routes. PharmaVision adds pill-category examples, but its archive contains no per-image severity labels. Original archive names, sizes, source URLs, and license notes are recorded in `reports/data_sources.md`.

## Train

From this directory, using the existing project environment:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m src.industrialguard_mlp.train
```

The first run downloads torchvision's public ResNet-18 ImageNet weights if they are not cached. Product and optional NEU model files are written to `models/` and their results are written to `reports/training_metrics.json`. Use `--product-only` to update the product classifier without retraining the separate NEU steel model.

Train the separate damage-localization model after the datasets have been extracted:

```powershell
..\.venv\Scripts\python.exe -m src.industrialguard_mlp.train_defect_segmenter --epochs 12 --batch-size 8
```

Its checkpoint is `models/defect_segmenter.pt`; held-out pixel and image screening measurements are saved in `reports/defect_segmentation_metrics.json`.

Train the localized defect-type head from MVTec's annotated mask crops:

```powershell
..\.venv\Scripts\python.exe -m src.industrialguard_mlp.train_damage_type --epochs 50
```

This writes `models/damage_type_mlp.pt` and `reports/damage_type_metrics.json`. Its broad labels are mapped from the dataset's fault-folder names and are not severity grades.

Train the bottle-specific damage type/grade head from its official split labels:

```powershell
..\.venv\Scripts\python.exe -m src.industrialguard_mlp.train_bottle_grade --epochs 50
```

This writes `models/bottle_damage_mlp.pt` and `reports/bottle_damage_metrics.json`. It is a bottle-subtype fallback for `plastic_shampoo_bottle`, not generic glass or plastic bottles. The source has no clean-bottle examples, so this head cannot certify that a bottle is undamaged.

Train the bottle-only YOLO defect localizer from the same official bounding-box labels:

```powershell
..\.venv\Scripts\python.exe -m src.industrialguard_mlp.train_bottle_yolo --epochs 30 --imgsz 640 --batch 16 --workers 4
```

This writes `models/bottle_defect_yolo.pt` and `reports/bottle_yolo_metrics.json`, then evaluates on the official test split. When that checkpoint is present, the app draws its candidate boxes and reports dataset-specific defect and grade clues only for `plastic_shampoo_bottle`. Generic MVTec bottles and every other category stay on the mask-based damage screen. The bottle dataset contains no clean-bottle images, so no detection cannot mean “defect-free.”

## Run the browser demo

```powershell
..\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8765
```

Choose one or more uploaded photos (or capture one with the camera), press **Analyze images**, then review each result. A category can be selected manually per image, including **Other product (manual)**. Model 1 reports the product category; Model 2 runs the mask-based damage screen and uses the separate YOLO defect detector only for the recognized `plastic_shampoo_bottle` subtype. Generic bottles do not route to that specialized detector. Open-vocabulary and manual-other damage results are exploratory. The optional NEU mode runs the six-class steel-surface classifier. The downloadable report includes each analyzed image and preserves manual selections.

## Evaluation notes

Product-category validation uses held-out normal images; NEU metrics use a deterministic stratified test split. Damage localization is evaluated on held-out image/mask examples, and its metrics and calibrated thresholds are reported separately. The model can miss subtle damage or flag harmless variation. The open-vocabulary label matcher is not calibrated for every product or background; an uncertain result should be manually corrected or treated as unknown. Severity and sale suitability are not inferred as facts by this prototype.
