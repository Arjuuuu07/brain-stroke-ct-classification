# Brain Stroke CT Classification

Detects **intracranial hemorrhage (bleeding vs. non-bleeding)** on non-contrast head CT slices (DICOM). Each scan is preprocessed in Hounsfield units (skull stripping, adaptive HU binarization), and a ResNet18 is trained with 5-fold cross-validation on a 3-channel input built from the preprocessed planes.

> Research code, not a medical device. Not clinically validated.

## Method

1. **Preprocessing** (`src/stroke_ct/preprocessing.py`): DICOM to HU, skull strip (>= 100 HU), smoothed 1 mm border erosion, speck cleanup, tissue bounding box, adaptive HU binarization (white area 16-18% of the box), guide mask, 4 px HU-constrained extension. Five 256x256 planes per scan are cached on disk: `raw, clean, guide, ext, guide_hu`.
2. **Input channels** (`channels:` in the config):

   | Preset | Channel 1 | Channel 2 | Channel 3 |
   |--------|-----------|-----------|-----------|
   | `t1` (control) | clean | clean | clean |
   | `t2` (default) | clean | guide_hu | ext |
   | `t2clean` | clean | guide (binary) | ext |

3. **Training** (`scripts/train.py`): ResNet18 (ImageNet weights), no augmentation, AdamW (lr 1e-4), up to 50 epochs with early stopping on validation AUC, stratified 5-fold CV on a balanced pool of 500 bleeding + 500 non-bleeding scans.
4. **Evaluation:** the best fold (highest validation AUC) is tested once on every scan outside the pool, at threshold 0.5 and at a threshold chosen on that fold's validation set (Youden).

## Dataset

Not included (patient data). Download: `<ADD DATASET LINK>`

```
Brain_Stroke_CT_Dataset/
├── Bleeding/DICOM/
├── Ischemia/DICOM/
├── Normal/DICOM/
└── External_Test/ (DICOM/ and labels Copy.csv)
```

Bleeding = 1; Ischemia and Normal = 0.

## Installation

```bash
git clone https://github.com/Arjuuuu07/brain-stroke-ct-classification.git
cd brain-stroke-ct-classification
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -e .
```

## Configuration

Settings are in `configs/default.yaml`. Put your machine-specific paths in `configs/local.yaml` (gitignored):

```bash
copy configs\local.example.yaml configs\local.yaml     # Linux/macOS: cp
```

```yaml
paths:
  data_root: 'D:\path\to\Brain_Stroke_CT_Dataset'
  output_dir: 'D:\path\to\outputs'
```

Precedence: `default.yaml` < `local.yaml` < command-line flags.

## Usage

```bash
python scripts/train.py --channels t2 --tag t2           # main experiment
python scripts/train.py --channels t1 --tag t1           # control
python scripts/train.py --channels t2clean --tag t2clean # variant

python scripts/threshold_tune.py --channels t2
