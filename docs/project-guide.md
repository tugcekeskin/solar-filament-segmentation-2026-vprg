# ☀️ Solar Filament Segmentation Challenge 2026 — Project Guide

## 📋 Competition Summary

| | |
|---|---|
| **Platform** | Kaggle (IEEE BigData Cup 2026) |
| **Link** | https://www.kaggle.com/competitions/filament-segmentation-2026 |
| **Sponsor** | NSF National Solar Observatory (NSO) |
| **Prize** | $3,000 (top 3 teams, proportional to score) |
| **Team Limit** | Maximum 5 people |
| **Daily Submissions** | 5 |
| **Final Submission** | 2 submissions will be selected |
| **Repo** | https://github.com/tugcekeskin/solar-filament-segmentation-2026-vprg |

---

## 📅 Important Dates

| Date | Event |
|---|---|
| July 10, 2026 | Competition started |
| **November 15, 2026, 23:59 CDT** | **Final report + solution submission deadline** |
| November 30, 2026 | Winners announced |
| December 14-17, 2026 | IEEE BigData Conference, Phoenix AZ (optional) |

---

## 🎯 Task

Detect and mask **solar filaments** at the pixel level in **2048×2048 pixel** solar images taken by GONG H-Alpha telescopes.

```
Input:  2048×2048 grayscale H-Alpha JPEG
Output: Per-filament RLE mask → submission.csv
```

**What is a solar filament?**
Dark, elongated structures on the solar surface. A precursor of solar eruptions (CME). Critical for space weather forecasting.

---

## 📏 Evaluation Criteria

### Quantitative (70%)
**Main metric: Panoptic Quality (PQ)**

```
PQ = SQ × DQ

SQ (Segmentation Quality) = How accurate is the mask? (IoU)
DQ (Detection Quality)    = How many filaments did you find? (F1 score)

PQ formula:
PQ(Y, Ŷ) = Σ IoU(y, ŷ) / (|TP| + 0.5|FP| + 0.5|FN|)
```

Also to be looked at:
- Dice score distribution
- IoU score distribution
- One-to-many / many-to-one relationship distribution

### Qualitative (30%)
- Pipeline description (preprocessing → model → postprocessing)
- Visual quality of the predicted masks
- Modularity and documentation of the code

> ⚠️ PQ alone is not enough! Clean code + a good report matter too.

---

## 🎯 Realistic Score Targets

| Level | PQ | Note |
|---|---|---|
| 🏁 First baseline | ~0.18 | U-Net 5 epochs |
| 👍 Good start | ~0.36 | YOLO + U-Net refiner |
| 🥇 **Our target** | **0.40–0.45** | — |

> Organizers: **"PQ > 0.35 is valuable for us"**

---

## 📦 Dataset — MAGFiLO 1.0

**MAGFiLO** = Manually Annotated GONG Filaments from H-Alpha Observations

```
MAGFiLO_1.0_Kaggle_2026/          (750.97 MB, 888 files)
├── train/
│   ├── train_images/              ← 707 JPEG images
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
└── test/
    └── test_images/               ← 180 JPEG images (no annotations)
```

### Image Properties
- 2048×2048 pixels, grayscale JPEG (saved as RGB — 3 channels, all identical)
- File name format: `YYYYMMDDHHMMSSII.jpeg`
- Example: `20140609195854Bh.jpeg` → June 9, 2014, 19:58:54, Big Bear Observatory

### GONG Observatories (6 Stations)

| Code | Observatory | Location | Image Count |
|---|---|---|---|
| **Bh** | Big Bear Solar Observatory | California, USA | 198 |
| **Ch** | Cerro Tololo Interamerican Observatory | Chile | 195 |
| **Lh** | Learmonth Solar Observatory | Australia | 210 |
| **Mh** | Mauna Loa Solar Observatory | Hawaii, USA | 205 |
| **Th** | Teide Observatory | Canary Islands, Spain | 180 |
| **Uh** | Udaipur Solar Observatory | India | 166 |

> The distribution is very balanced — the model will not memorize a particular observatory.

### EDA Findings

- **707 train, 180 test** images, all 2048×2048 grayscale JPEG
- **37 annotators**, 8199 annotations in total
- **Filaments per image:** min=1, max=58, mean=11.6
- **Multi-annotator distribution:**
  - 1 annotator: 411 images (58.1%)
  - 2 annotators: 145 images (20.5%)
  - 3 annotators: 151 images (21.4%)
- **Filament area:** min=9, max=37739, mean=2120, median=1228 pixels → most are very small!
- **Pixel values:** mean ~80, std ~58 (dark solar disk, bright background)
- **Spine:** present in 8199/8199 annotations — every filament has one
- **Spine length:** min=12, max=1105, mean=147 pixels → filaments are thin and long

### Category Distribution

| Category | Count | Ratio | Description |
|---|---|---|---|
| Unidentifiable | 3074 | 37.5% | Filament exists but chirality cannot be determined |
| Right | 2590 | 31.6% | Right chirality (certain) |
| Left | 2535 | 30.9% | Left chirality (certain) |
| Ambiguous | 0 | 0% | Unclear whether it is a filament or not |

> **What is chirality?** The helical direction of the filament's magnetic field. We do not predict it, but it can be used as information during training.

### Annotation Format (COCO-style JSON)

```json
{
  "id": "annotatorID-imageFilename",
  "segmentation": [[x0,y0,x1,y1,...]],
  "bbox": [x, y, width, height],
  "spine": [[x0,y0,x1,y1,...]],
  "area": float,
  "category_id": 1|2|3|4,
  "iscrowd": 0
}
```

- **segmentation:** polygon coordinates (closed path)
- **spine:** the filament's backbone / center line — can be used as an additional cue
- **iscrowd:** always 0 (single filament)

### Local Path

```
data/raw/MAGFiLO_1.0_Kaggle_2026
```

### Kaggle Path

```
/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026
```

---

## 📤 Submission Format

```csv
filament_id,segmentation_rle
20150125172714Mh_1,f8uSDds...VQNC
20150125172714Mh_2,KHT%$HD...9>km
20150125172714Mh_3,YQNEgn1...BH6^
```

- Each row = 1 filament prediction
- `filament_id` = `{image_name}_{sequence_no}` (order does not matter, matching is done by overlap)
- `segmentation_rle` = pycocotools COCO RLE format (Fortran order!)
- Image size is fixed: 2048×2048

**RLE conversion:**
```python
from pycocotools import mask as coco_mask
import numpy as np

rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
rle_string = rle["counts"].decode("utf-8")
```

> **Important:** The organizers do not look at the sequence number in filament_id. They match the predicted masks to the ground truth **by IoU overlap**.

---

## 📝 Final Submission Requirements

### 1) Kaggle Submission (for the score)
- Upload the CSV file to Kaggle
- 5 submissions per day
- Deadline: **November 15, 2026**

### 2) Google Form (for the final evaluation)
**Form link:** https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform

What the form asks for:
- Email address
- Full names of all team members (separated by `;`)
- Email addresses of all members
- Institution/university (optional)
- Countries
- **Public GitHub repo URL** (required)
- **Kaggle submission CSV** (can be updated 10 times, max 10 MB)
- **Technical report PDF** (can be updated 10 times, max 10 MB)

### 3) Technical Report (PDF)
- **4 pages** (excluding Acknowledgment and References)
- Overleaf template: https://www.overleaf.com/project/6a735222766a5d9bbc55c686
- Sections: Abstract, Introduction, Methodology, Evaluation, Acknowledgment, References

### 4) GitHub Repo (Required)
**Repo:** https://github.com/tugcekeskin/solar-filament-segmentation-2026-vprg

The final repository layout is described in the README ("Repository structure").

---

## 🛠️ Technical Plan

### Pipeline

```
1. Preprocessing
   ├── Grayscale → 3 channels (for ImageNet normalization)
   ├── CLAHE (contrast enhancement)
   ├── Resize (2048 → 1024)
   └── Spine mask (as an extra channel or as a weight in the loss function — inspired by OTH)

2. Model
   ├── Encoder: ResNet34 (pretrained ImageNet)
   ├── Decoder: U-Net
   └── Loss: Focal Loss + Dice Loss                (OTH)

3. Postprocessing
   ├── Threshold
   ├── Connected Components (instance separation)
   ├── Min area filter (discard small noise)
   ├── Filter out Unidentifiable filaments         (OTH)
   └── RLE encoding → submission.csv
```

### Organizer Technical Hints (OTH)

1. **Focus on fine-scale structures** — capturing barbs (thin protrusions on the filament edge) is critical
2. The **Focal Loss + Dice Loss** combination works well
3. You can use the **spine annotations** as an additional cue during training
4. **Filter out Unidentifiable filaments** — predicting them increases FP and lowers PQ
5. **Focus on consensus** — predicting filaments marked by only a few annotators increases FP
6. **Quality > PQ score** — the qualitative evaluation matters, the pipeline must be well explained
7. **Computational efficiency** is also evaluated (~1 s/image target)

### Libraries to Use

```
torch==2.15.0               ← deep learning
segmentation_models_pytorch  ← U-Net
albumentations               ← augmentation
pycocotools                  ← RLE conversion
opencv-python                ← image processing
```

---

## 🖥️ Setup

```bash
conda create -n filament python=3.12 -y
conda activate filament

# PyTorch (depending on CUDA version)
# CUDA 13.0:
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu130
# CUDA 12.1:
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu121

# The rest
pip install -r requirements.txt
```

### Test

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### Working Environment

| | Local (4090) | Kaggle |
|---|---|---|
| Usage | Model training | Final inference + submit |
| GPU | RTX 4090 (24GB) | T4 x2 (16GB) |
| Speed | 3-4x faster | Slow |
| Limit | Unlimited | ~30 hours/week |

---

## 🧪 Experiment Plan

### Experiment Order

**Run 1 — YOLO11s-seg Baseline**
- imgsz: 1024, epochs: 80, batch: 4
- conf: 0.30, NMS: 0.3, min_area: 20
- Mask thinning: 5×5 cross erosion
- Target: ~0.35 PQ

**Run 2 — YOLO11s-seg + Thinning Grid**
- Try different erosion parameters on top of Run 1 (3×3, 5×5, 7×7)
- conf and min_area grid
- Target: ~0.38 PQ

**Run 3 — YOLO11m-seg**
- imgsz: 1280, epochs: 80
- Best thinning parameters
- Target: ~0.42 PQ

**Run 4 — YOLO11m-seg + Flip TTA**
- id+h+v flip TTA
- Target: ~0.45 PQ

**Run 5 — U-Net + Spine-guided Loss (Original Contribution)**
- Encoder: ResNet34, pretrained ImageNet
- Spine mask as a per-pixel weight in the loss
- Loss: Focal + Dice + Spine-weighted
- Target: an original loss that uses the annotated spines

**Run 6+ — Free Experimentation**
- Further experiments chosen from the results of runs 1-5

---

## 📜 Important Rules

| Rule | Detail |
|---|---|
| Team size | Max. 5 people |
| Daily submissions | 5 |
| Final submission selection | 2 submissions |
| External data | ✅ Allowed (must be publicly available) |
| Pretrained models | ✅ Allowed (ImageNet etc.) |
| GONG archive | ✅ Approved (for pretraining) |
| Writing code with AI | ✅ Allowed (but you must understand and be able to defend the solution) |
| Hand-labeling test data | ❌ Prohibited |
| Multiple Kaggle accounts | ❌ Prohibited |

---

## 🔗 Important Links

| | Link |
|---|---|
| Competition page | https://www.kaggle.com/competitions/filament-segmentation-2026 |
| Leaderboard | https://www.kaggle.com/competitions/filament-segmentation-2026/leaderboard |
| Discussion | https://www.kaggle.com/competitions/filament-segmentation-2026/discussion |
| **Final Submission Formu** | https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform |
| **Final Submission Form** | https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform |
| **Overleaf Report Template** | https://www.overleaf.com/project/6a735222766a5d9bbc55c686 |
| Self-evaluation Notebook | https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook |
| SABID Workshop | https://sites.google.com/view/sabid2026/home |
| IEEE BigData 2026 | https://bigdataieee.org/BigData2026/cup/ |
| MAGFiLO Dataset Paper | https://doi.org/10.1038/s41597-024-03876-y |
