# IndustrialGuard AI — Core Project Requirement

## Goal
Build an inspection app for checking multiple different products in one session. It should identify each product, localize visible wear or damage, and create a per-item report.

## Required two-model design
1. **Product classifier:** classify the supported product category before inspection. The trained MLP now covers 41 labels from MVTec AD, VisA, MVTec AD 2, ETID, PharmaVision, and the labeled plastic shampoo-bottle source. The generic `bottle` label is separate from `plastic_shampoo_bottle` so the specialized defect model is not routed to every kind of bottle. Keep category confidence separate from defect scores and allow a manual category override when uncertain. OpenCLIP may suggest a label outside the trained camera/image domain, but that suggestion is not a calibrated class probability.
2. **Wear-and-tear inspection:** a separate defect stage. The general ResNet-18 U-Net learns candidate regions from industrial masks and reports an image-level screening score. The `plastic_shampoo_bottle` subtype can additionally use a bottle-only YOLO11 detector trained on its official box labels. Its localized candidates include five defect types × two dataset grades. Other supported products continue through the mask screen and limited MVTec crop-type head; open-vocabulary/manual products remain exploratory. None of these outputs is a universal defect classifier or sale decision.
   - A defect-type head classifies localized crops into broad types mapped from MVTec fault-folder names (scratch, crack, hole/puncture, cut/tear, broken/damaged, missing/misplaced part, deformation, and surface/contamination). It is trained only on MVTec-style defect crops.

## User flow
Industry focus (filters manual category choices only) → multiple uploads or a camera capture → Analyze button → product category model or open-vocabulary suggestion → wear/damage localization model → per-item results and downloadable report. Automatic detection still evaluates all trained labels. A manual “Other product” selection allows the user to name an item and run exploratory screening when no learned category is available. The report should keep industry focus, category confidence, defect score, highlighted region, possible defect clue, and sale/review guidance separate.

## Evidence and limitations
- Available pixel-mask sources: MVTec AD, VisA, and public MVTec AD 2. These support learning where anomalous regions appear, but do not provide reliable scratch/dent class labels across every product, minor/major severity labels, or sale-readiness labels.
- The downloaded PharmaVision tablet archive supplies only per-image normal/defective folders, not per-image severity/type metadata; its official original splits are used to supplement the pill category.
- The downloaded Plastic Shampoo Bottle dataset supplies per-instance defect boxes, five defect types, and marginal/unacceptable grades. Its labels support a bottle-only YOLO head and subtype examples; it has no clean shampoo-bottle examples and does not represent all products. A separate MVTec `bottle` class prevents automatic routing of glass/other bottles to the shampoo detector.
- OpenCLIP is used only to suggest familiar product labels when the 41-class MLP is uncertain; those suggestions are not calibrated probabilities. The separate trained MVTec type head is limited by folder-name-derived labels, its 178-region test set, and benchmark-style imagery.
- Visible area is not functional severity. A model must not certify an item as fit for sale without product-specific severity and sale-grade labels; human review is required.
- “Any object” is not guaranteed by the 41 trained category set. Open-vocabulary suggestions are exploratory; unsupported or uncertain objects can be sent to manual category selection and are not treated as confirmed automatically.

## Saved on
2026-11-01
