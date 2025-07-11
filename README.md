# MedSAM2 Inference

This is a simple inference script for MedSAM2.

## Installation

Clone the repository:
```bash
git clone https://github.com/JoshuaSiraj/medsam2_inference.git
cd medsam2_inference
```

Clone the MedSAM2 repository, while in the medsam2_inference directory:
```bash
git clone https://github.com/bowang-lab/MedSAM2.git
```

This project uses [Pixi](https://pixi.sh/dev/) to manage dependencies. Install it by following the instructions [here](https://pixi.sh/dev/installation/).

Install the dependencies by running:
```bash
pixi install
```

## Download Weights

The weights for MedSAM2 are not included in this repository. You can download them by running the following command:

```bash
pixi run download-weights
```

## Example Usage

```python
from inference import MedSAM3DInference, MedSAM3DInferenceConfig

config = MedSAM3DInferenceConfig(
    dataset_csv="data/dataset.csv",
    model_config_path="configs/sam2.1_hiera_t512.yaml",
    checkpoint_path="checkpoints/MedSAM2_latest.pt", 
    output_dir="output",
    window_level=500.0,
    window_width=2500.0,
)

inference_module = MedSAM3DInference(config)
inference_module.run()
```

## Important Notes

- GPU is required.
- Performance is reliant on the right window level and width. Therefore only one Region of Interest (ROI) should be present in the mask.

## Example Dataset CSV

```csv
ID,image_path,mask_path
1,/path/to/image.nii.gz,/path/to/mask.nii.gz
2,/path/to/image.nii.gz,/path/to/mask.nii.gz
```
