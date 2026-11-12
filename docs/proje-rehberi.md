# ☀️ Solar Filament Segmentation Challenge 2026 — Proje Rehberi

## 📋 Yarışma Özeti

| | |
|---|---|
| **Platform** | Kaggle (IEEE BigData Cup 2026) |
| **Link** | https://www.kaggle.com/competitions/filament-segmentation-2026 |
| **Sponsor** | NSF National Solar Observatory (NSO) |
| **Ödül** | $3.000 (ilk 3 takım, skor oranına göre) |
| **Takım Limiti** | Maksimum 5 kişi |
| **Günlük Submit** | 5 hak |
| **Final Submit** | 2 submission seçilecek |
| **Repo** | https://github.com/tugcekeskin/solar-filament-segmentation-2026-vprg |

---

## 📅 Önemli Tarihler

| Tarih | Olay |
|---|---|
| 10 Temmuz 2026 | Yarışma başladı |
| **15 Kasım 2026, 23:59 CDT** | **Final rapor + çözüm teslim deadline'ı** |
| 30 Kasım 2026 | Kazananlar açıklanacak |
| 14-17 Aralık 2026 | IEEE BigData Konferansı, Phoenix AZ (opsiyonel) |

---

## 🎯 Görev

GONG H-Alpha teleskoplarından çekilmiş **2048×2048 piksel** güneş fotoğraflarında **solar filamentleri** piksel düzeyinde tespit et ve maskele.

```
Input:  2048×2048 grayscale H-Alpha JPEG
Output: Per-filament RLE mask → submission.csv
```

**Solar filament nedir?**
Güneş yüzeyindeki karanlık, uzun yapılar. Güneş patlamalarının (CME) habercisi. Uzay hava tahmini için kritik.

---

## 📏 Değerlendirme Kriterleri

### Nicel (%70)
**Ana metrik: Panoptic Quality (PQ)**

```
PQ = SQ × DQ

SQ (Segmentation Quality) = Maske ne kadar doğru? (IoU)
DQ (Detection Quality)    = Kaç filament buldun? (F1 skoru)

PQ formülü:
PQ(Y, Ŷ) = Σ IoU(y, ŷ) / (|TP| + 0.5|FP| + 0.5|FN|)
```

Ayrıca bakılacaklar:
- Dice score dağılımı
- IoU score dağılımı
- One-to-many / many-to-one ilişki dağılımı

### Nitel (%30)
- Pipeline açıklaması (preprocessing → model → postprocessing)
- Tahmin edilen maskelerin görsel kalitesi
- Kodun modülerliği ve dokümantasyonu

> ⚠️ Sadece PQ yetmez! Temiz kod + iyi rapor da önemli.

---

## 🎯 Gerçekçi Skor Hedefleri

| Seviye | PQ | Not |
|---|---|---|
| 🏁 İlk baseline | ~0.18 | U-Net 5 epoch |
| 👍 İyi başlangıç | ~0.36 | YOLO + U-Net refiner |
| 🥇 **Hedefimiz** | **0.40–0.45** | — |
| 🏆 En iyi dürüst skor | ~0.49 | noisyislands |

> Organizatörler: **"PQ > 0.35 bizim için değerli"**
> Leaderboard'daki 0.70+ skorlar sistem açığı kullanan takımlar — leaderboard temizlenecek.

---

## 📦 Veri Seti — MAGFiLO 1.0

**MAGFiLO** = Manually Annotated GONG Filaments from H-Alpha Observations

```
MAGFiLO_1.0_Kaggle_2026/          (750.97 MB, 888 dosya)
├── train/
│   ├── train_images/              ← 707 JPEG görüntü
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
└── test/
    └── test_images/               ← 180 JPEG görüntü (annotation yok)
```

### Görüntü Özellikleri
- 2048×2048 piksel, grayscale JPEG (RGB olarak kaydedilmiş — 3 kanal, hepsi aynı)
- Dosya adı formatı: `YYYYMMDDHHMMSSII.jpeg`
- Örnek: `20140609195854Bh.jpeg` → 9 Haziran 2014, 19:58:54, Big Bear Gözlemevi

### GONG Gözmeveyleri (6 İstasyon)

| Kod | Gözlemevi | Konum | Görüntü Sayısı |
|---|---|---|---|
| **Bh** | Big Bear Solar Observatory | California, ABD | 198 |
| **Ch** | Cerro Tololo Interamerican Observatory | Şili | 195 |
| **Lh** | Learmonth Solar Observatory | Avustralya | 210 |
| **Mh** | Mauna Loa Solar Observatory | Hawaii, ABD | 205 |
| **Th** | Teide Observatory | Kanarya Adaları, İspanya | 180 |
| **Uh** | Udaipur Solar Observatory | Hindistan | 166 |

> Dağılım çok dengeli — model belirli bir gözmeveyi ezberlemez.

### EDA Bulguları

- **707 train, 180 test** görüntü, tümü 2048×2048 grayscale JPEG
- **37 annotator**, 8199 toplam annotation
- **Görüntü başına filament:** min=1, max=58, ortalama=11.6
- **Multi-annotator dağılımı:**
  - 1 annotator: 411 görüntü (%58.1)
  - 2 annotator: 145 görüntü (%20.5)
  - 3 annotator: 151 görüntü (%21.4)
- **Filament alanı:** min=9, max=37739, ortalama=2120, median=1228 piksel → çoğu çok küçük!
- **Piksel değerleri:** mean ~80, std ~58 (karanlık güneş diski, parlak arka plan)
- **Spine:** 8199/8199 annotation'da mevcut — her filamentte var
- **Spine uzunluğu:** min=12, max=1105, ortalama=147 piksel → filamentler ince ve uzun

### Kategori Dağılımı

| Kategori | Sayı | Oran | Açıklama |
|---|---|---|---|
| Unidentifiable | 3074 | %37.5 | Filament var ama chirality belirlenemiyor |
| Right | 2590 | %31.6 | Sağ chiralite (kesin) |
| Left | 2535 | %30.9 | Sol chiralite (kesin) |
| Ambiguous | 0 | %0 | Filament mi değil mi belirsiz |

> **Chirality nedir?** Filamentin manyetik alan sarmal yönü. Biz tahmin etmiyoruz ama eğitimde bilgi olarak kullanılabilir.

### Annotation Formatı (COCO-style JSON)

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

- **segmentation:** polygon koordinatları (kapalı yol)
- **spine:** filament omurgası / merkez çizgisi — ek ipucu olarak kullanılabilir
- **iscrowd:** her zaman 0 (tek filament)

### Lokal Path

```
/home/baykar/Desktop/FLAMENT/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026
```

### Kaggle Path

```
/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026
```

---

## 📤 Submission Formatı

```csv
filament_id,segmentation_rle
20150125172714Mh_1,f8uSDds...VQNC
20150125172714Mh_2,KHT%$HD...9>km
20150125172714Mh_3,YQNEgn1...BH6^
```

- Her satır = 1 filament tahmini
- `filament_id` = `{görüntü_adı}_{sıra_no}` (sıra önemli değil, overlap ile eşleştiriliyor)
- `segmentation_rle` = pycocotools COCO RLE formatı (Fortran order!)
- Görüntü boyutu sabit: 2048×2048

**RLE dönüşümü:**
```python
from pycocotools import mask as coco_mask
import numpy as np

rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
rle_string = rle["counts"].decode("utf-8")
```

> **Önemli:** Organizatörler filament_id'deki sıra numarasına bakmaz. Tahmin edilen maskeleri ground truth ile **IoU overlap'e göre** eşleştirir.

---

## 📝 Final Teslim Gereksinimleri

### 1) Kaggle Submit (Skor için)
- CSV dosyasını Kaggle'a yükle
- Günde 5 hak
- Deadline: **15 Kasım 2026**

### 2) Google Form (Final değerlendirme için)
**Form linki:** https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform

Form'da istenenler:
- E-posta adresi
- Tüm takım üyelerinin tam adı (`;` ile ayır)
- Tüm üyelerin email adresleri
- Kurum/üniversite (opsiyonel)
- Ülkeler
- **Public GitHub repo URL** (zorunlu)
- **Kaggle submission CSV** (10 kez güncellenebilir, max 10 MB)
- **Teknik rapor PDF** (10 kez güncellenebilir, max 10 MB)

### 3) Teknik Rapor (PDF)
- **4 sayfa** (Acknowledgment ve References hariç)
- Overleaf şablonu: https://www.overleaf.com/project/6a735222766a5d9bbc55c686
- Bölümler: Abstract, Introduction, Methodology, Evaluation, Acknowledgment, References

### 4) GitHub Repo (Zorunlu)
**Repo:** https://github.com/tugcekeskin/solar-filament-segmentation-2026-vprg

```
solar-filament-segmentation-2026-vprg/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   └── 03_inference.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── requirements.txt
├── README.md
└── LICENSE (MIT)
```

---

## 🛠️ Teknik Plan

### Pipeline

```
1. Preprocessing
   ├── Grayscale → 3 kanal (ImageNet normalization için)
   ├── CLAHE (kontrast artırma)
   ├── Resize (2048 → 1024)
   └── Spine mask (ek kanal olarak veya loss fonksiyonunda ağırlık olarak — OTİ'den ilham alındı)

2. Model
   ├── Encoder: ResNet34 (pretrained ImageNet)
   ├── Decoder: U-Net
   └── Loss: Focal Loss + Dice Loss                (OTİ)

3. Postprocessing
   ├── Threshold
   ├── Connected Components (instance ayırma)
   ├── Min area filtresi (küçük gürültüyü at)
   ├── Unidentifiable filamentleri filtrele         (OTİ)
   └── RLE encoding → submission.csv
```

### Organizatörden Teknik İpuçları (OTİ)

1. **Fine-scale yapılara odaklan** — barbs (filament kenarındaki ince çıkıntılar) yakalamak kritik
2. **Focal Loss + Dice Loss** kombinasyonu iyi çalışıyor
3. **Spine annotation'larını** eğitimde ek ipucu olarak kullanabilirsin
4. **Unidentifiable filamentleri filtrele** — tahmin etmek FP artırır, PQ düşürür
5. **Consensus'a odaklan** — az annotator'ın işaretlediği filamentleri tahmin etmek FP artırır
6. **Kalite > PQ skoru** — nitel değerlendirme önemli, pipeline iyi açıklanmalı
7. **Hesaplama verimliliği** de değerlendiriliyor (~1 sn/görüntü hedef)

### Kullanılacak Kütüphaneler

```
torch==2.14.1               ← deep learning
segmentation_models_pytorch  ← U-Net
albumentations               ← augmentation
pycocotools                  ← RLE dönüşümü
opencv-python                ← görüntü işleme
```

---

## 🖥️ Kurulum

```bash
conda create -n filament python=3.12 -y
conda activate filament

# PyTorch (CUDA versiyonuna göre)
# CUDA 13.0:
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
# CUDA 12.1:
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu121

# Geri kalanlar
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

### Çalışma Ortamı

| | Lokal (4090) | Kaggle |
|---|---|---|
| Kullanım | Model eğitimi | Final inference + submit |
| GPU | RTX 4090 (24GB) | T4 x2 (16GB) |
| Hız | 3-4x daha hızlı | Yavaş |
| Limit | Sınırsız | ~30 saat/hafta |

---

## 🏆 Mevcut En İyi Skorlar (Dürüst Yöntemler)

| Skor | Yöntem |
|---|---|
| 0.49 | Bilinmiyor (noisyislands — Kaggle Discussion) |
| 0.48 | Fine-scale odaklı U-Net (Spaghetto99 — Kaggle Discussion) |
| 0.45 | ResNet34 + EfficientNetV2-S ensemble (OOF Notebook — Kaggle Code sekmesi) |
| 0.36 | YOLO11s + SeedRefiner U-Net (Sunjnn — GitHub) |
| 0.27 | U-Net ResNet34, 35 epoch (Sunjnn — GitHub) |
| 0.18 | U-Net ResNet34, 5 epoch (Baseline — Kaggle Code sekmesi) |

**En iyi doğrulanmış sonuç (Spaghetto99, Discussion):**
```
PQ:  0.4843
SQ:  0.6870  (maske kalitesi)
RQ:  0.7049  (detection F1)
TP: 855 | FP: 313 | FN: 403
Inference: ~1.13 sn/görüntü
```

---

## 🤖 Claude Code İçin Talimatlar

### Mevcut Durum
- EDA tamamlandı (`notebooks/01_eda.ipynb`)
- Ortam kurulu (conda: `filament`, PyTorch 2.14.1, CUDA 13.0, RTX 4090)
- Veri hazır: `/home/baykar/Desktop/FLAMENT/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026`
- Repo: `/home/baykar/Desktop/FLAMENT/solar-filament-segmentation-2026-vprg`

### Görev
Aşağıdaki deney planını sırayla uygula. Her deney bitince:
1. Sonuçları `results.md`'ye yaz (format aşağıda)
2. GitHub'a push et (`git add . && git commit -m "results: run X" && git push`)
3. Sonuçları analiz et
4. Bir sonraki deneye geç veya optimize et

### Karar Yetkisi
- Deney planı biterse kendi inisiyatifinle yeni deneyler tasarla
- PQ artışı < 0.005 ise o yönü bırak, farklı yaklaşım dene
- Arkadaşımızın bulgularını dikkate al ama kendi özgün katkımızı da ekle
- **Deadline: 15 Kasım 09:00 TR saati** — son 3 saati rapor + form için bırak
- Tüm kararlarını `results.md`'ye "Neden bu değişikliği yaptım" notu ile kaydet

### Arkadaşımızın Bulgularından Öğrendiklerimiz
- CLAHE → no gain (deneme)
- Mask thinning (5×5 cross erosion) → +0.073 PQ (kritik!)
- YOLO11m > YOLO11s (+0.005 PQ)
- 1536px > 1280px > 1024px
- Flip TTA → +0.006
- U-Net crop refiner → +0.004
- 150 epoch → no gain (80 yeterli)
- Consensus filtering → noise seviyesinde

### Özgün Katkımız
Spine-guided loss — spine maskesini loss fonksiyonunda piksel ağırlığı olarak kullanmak. Kimse denememiş, raporda güçlü bir katkı olur.

---

## 🧪 Deney Planı

### Deney Sırası

**Run 1 — YOLO11s-seg Baseline**
- imgsz: 1024, epochs: 80, batch: 4
- conf: 0.30, NMS: 0.3, min_area: 20
- Mask thinning: 5×5 cross erosion
- Hedef: ~0.35 PQ

**Run 2 — YOLO11s-seg + Thinning Grid**
- Run 1 üzerine farklı erosion parametreleri dene (3×3, 5×5, 7×7)
- conf ve min_area grid'i
- Hedef: ~0.38 PQ

**Run 3 — YOLO11m-seg**
- imgsz: 1280, epochs: 80
- En iyi thinning parametreleri
- Hedef: ~0.42 PQ

**Run 4 — YOLO11m-seg + Flip TTA**
- id+h+v flip TTA
- Hedef: ~0.45 PQ

**Run 5 — U-Net + Spine-guided Loss (Özgün Katkı)**
- Encoder: ResNet34, pretrained ImageNet
- Spine maskesi loss'ta piksel ağırlığı olarak
- Loss: Focal + Dice + Spine-weighted
- Hedef: Arkadaşından farklı özgün yaklaşım

**Run 6+ — Serbest Deney**
- Önceki sonuçlara göre kendi inisiyatifinle devam et

### Results.md Formatı

```markdown
# Experiment Results

## Run Log

| # | Date | Model | Config | Val PQ | Val SQ | Val RQ | TP | FP | FN | Notes |
|---|------|-------|--------|--------|--------|--------|----|----|----|-------|
| 1 | 2026-11-12 | YOLO11s-seg | 1024px, 80ep | 0.xxx | 0.xxx | 0.xxx | - | - | - | baseline |

---

## Run Details

### Run 1 — ...
**Date:** ...

#### Preprocessing
- ...

#### Model
- Architecture: ...
- Pretrained: ...
- imgsz: ...
- Epochs: ...
- Batch: ...
- LR: ...

#### Augmentation
- ...

#### Postprocessing
- Confidence threshold: ...
- Mask NMS IoU: ...
- Min area: ...
- Mask thinning: ...
- TTA: ...

#### Results
- Val PQ: ...
- Val SQ: ...
- Val RQ: ...
- TP: ... | FP: ... | FN: ...
- Train time: ... min
- Inference: ... sn/görüntü
- GPU memory: ... GB

#### Neden Bu Değişikliği Yaptım
...
```

---

## 📜 Önemli Kurallar

| Kural | Detay |
|---|---|
| Takım büyüklüğü | Maks. 5 kişi |
| Günlük submit | 5 hak |
| Final submit seçimi | 2 submission |
| Dış veri | ✅ Serbest (herkese açık olmalı) |
| Pretrained model | ✅ Serbest (ImageNet vb.) |
| GONG arşivi | ✅ Onaylandı (pretraining için) |
| AI ile kod yazma | ✅ Serbest (ama çözümü anlayıp savunabilmelisin) |
| Test verisini elle etiketlemek | ❌ Yasak |
| Birden fazla Kaggle hesabı | ❌ Yasak |

---

## 🔗 Önemli Linkler

| | Link |
|---|---|
| Yarışma sayfası | https://www.kaggle.com/competitions/filament-segmentation-2026 |
| Leaderboard | https://www.kaggle.com/competitions/filament-segmentation-2026/leaderboard |
| Discussion | https://www.kaggle.com/competitions/filament-segmentation-2026/discussion |
| **Final Submission Formu** | https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform |
| **Overleaf Rapor Şablonu** | https://www.overleaf.com/project/6a735222766a5d9bbc55c686 |
| **GitHub Repo** | https://github.com/tugcekeskin/solar-filament-segmentation-2026-vprg |
| Self-evaluation Notebook | https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook |
| SABID Workshop | https://sites.google.com/view/sabid2026/home |
| IEEE BigData 2026 | https://bigdataieee.org/BigData2026/cup/ |
| MAGFiLO Dataset Paper | https://doi.org/10.1038/s41597-024-03876-y |
