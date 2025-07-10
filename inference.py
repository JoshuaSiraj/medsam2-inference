import pandas as pd
import numpy as np  
from pydantic import BaseModel, Field
from pathlib import Path
import json

import SimpleITK as sitk
import torch
from sam2.build_sam import build_sam2_video_predictor_npz

from utils import dice_multi_class, resize_grayscale_to_rgb_and_resize, mask3D_to_bbox, preprocess, AddedPathLength

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

class MedSAM3DInferenceConfig(BaseModel):
    """Configuration for MedSAM3D runner."""

    dataset_csv: str = Field(
        description="Path to the dataset CSV file. Must contain columns: "
        "'ID' (unique identifier), "
        "'image_path' (path to input image), and "
        "'mask_path' (path to ground truth mask). "
    )
    model_config_path: str = Field(description="Path to the model configuration file.")
    checkpoint_path: str = Field(description="Path to the checkpoint file.")
    output_dir: str = Field(description="Path to the output directory.")
    window_level: float = Field(
        description="Window Level. This must be configured to match Target Window Width."
    )
    window_width: float = Field(
        description="Window Width. This must be configured to match Target Window Level."
    )

    image_size: int = Field(default=512, description="Size of the image to be processed.")  
    mean: tuple[float, float, float] = Field(default=(0.485, 0.456, 0.406), description="Mean values for each channel.")
    std: tuple[float, float, float] = Field(default=(0.229, 0.224, 0.225), description="Standard deviation values for each channel.")

    propagate_with_bbox: bool = Field(default=False, description="Whether to propagate the mask with the bounding box.")

    def post_init(self):
        """Post-initialization hook."""
        # Check if dataset file exists
        if not Path(self.dataset_csv).exists():
            raise FileNotFoundError(f"Dataset CSV file not found: {self.dataset_csv}")
        
        # Check if model config file exists
        if not Path(self.model_config_path).exists():
            raise FileNotFoundError(f"Model config file not found: {self.model_config_path}")
        
        # Check if checkpoint file exists
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

class MedSAM3DInference:
    """Class for running MedSAM3D segmentation model."""

    def __init__(self, config: MedSAM3DInferenceConfig):
        self.config = config

        self.predictor = build_sam2_video_predictor_npz(
            self.config.model_config_path,
            self.config.checkpoint_path,
        )

    def run(self):

        dataset = pd.read_csv(self.config.dataset_csv)

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        masks_dir = Path(self.config.output_dir) / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        with open(Path(self.config.output_dir) / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=4)

        results_df = []

        for _, row in dataset.iterrows():
            patient_id = row["ID"]
            image_path = Path(str(row["image_path"]))
            mask_path = Path(str(row["mask_path"]))

            print(f"Inferring {patient_id}...")
            
            # Check if image and mask files exist
            if not image_path.exists():
                print(f"Warning: Image file not found: {image_path}, skipping...")
                continue
            if not mask_path.exists():
                print(f"Warning: Mask file not found: {mask_path}, skipping...")
                continue
            
            # Read image and mask
            try:
                image = sitk.ReadImage(str(image_path))
                image_array = sitk.GetArrayFromImage(image)
                mask = sitk.ReadImage(str(mask_path))
                mask_array = sitk.GetArrayFromImage(mask)
            except Exception as e:
                print(f"Error: {e}")
                continue

            # Preprocess image
            spacing = image.GetSpacing()
            image_array = preprocess(
                image_array, 
                window_level=self.config.window_level, 
                window_width=self.config.window_width
            )

            # Initialize segmentation mask
            segs_3D = np.zeros(image_array.shape, dtype=np.uint8)

            # Get bounding box
            x_min, y_min, z_min, x_max, y_max, z_max = mask3D_to_bbox(mask_array, mask_path)
            bbox2d = [x_min, y_min, x_max, y_max] 

            # Get z-axis coordinates
            zs, _, _ = np.where(mask_array>0)
            zs = np.unique(zs)
            assert z_min == min(zs)
            assert z_max == max(zs)
            z_mid_orig = (z_min + z_max)//2 
            z_mid = z_mid_orig - z_min

            # Crop image and mask
            cropped_image = image_array[z_min:z_max+1]
            cropped_mask = mask_array[z_min:z_max+1]
            cropped_mask = cropped_mask.astype(np.uint8)

            # Get video dimensions
            video_height = cropped_image.shape[1]
            video_width = cropped_image.shape[2]

            # Resize image if necessary and convert to RGB tensor
            if video_height != 512 or video_width != 512:
                cropped_image = resize_grayscale_to_rgb_and_resize(cropped_image, 512)
            else:
                cropped_image = cropped_image[:,None].repeat(3, axis=1)

            # Normalize image
            cropped_image = cropped_image / 255.0
            cropped_image = torch.from_numpy(cropped_image).cuda()
            img_mean = torch.tensor(self.config.mean, dtype=torch.float32)[:, None, None].cuda()
            img_std = torch.tensor(self.config.std, dtype=torch.float32)[:, None, None].cuda()
            cropped_image -= img_mean
            cropped_image /= img_std

            # Run inference
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                # Create initial inference state
                inference_state = self.predictor.init_state(cropped_image, video_height, video_width)

                # Create mask prompt
                if self.config.propagate_with_bbox:
                    _, _, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=z_mid,
                        obj_id=1,
                        box=bbox2d,
                    )
                    mask_prompt = (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
                else: # gt
                    mask_prompt = (cropped_mask[z_mid] == 1).astype(np.uint8)

                _, _, masks = self.predictor.add_new_mask(
                    inference_state, 
                    frame_idx=z_mid, 
                    obj_id=1,
                    mask=mask_prompt
                )
                segs_3D[z_mid_orig, ((masks[0] > 0.0).cpu().numpy())[0]] = 1

                # Forward Pass
                for out_frame_idx, _, out_mask_logits in self.predictor.propagate_in_video(
                    inference_state, 
                    start_frame_idx=z_mid, 
                    reverse=False
                ):
                    segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1

                # Reverse Pass
                self.predictor.reset_state(inference_state)
                inference_state = self.predictor.init_state(cropped_image, video_height, video_width)
                _, _, masks = self.predictor.add_new_mask(
                    inference_state, 
                    frame_idx=z_mid, 
                    obj_id=1, 
                    mask=mask_prompt
                )

                for out_frame_idx, _, out_mask_logits in self.predictor.propagate_in_video(
                    inference_state, 
                    start_frame_idx=z_mid, 
                    reverse=True
                ):
                    segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1

                self.predictor.reset_state(inference_state)

                # Calculate metrics
                dice = dice_multi_class((segs_3D == 1).astype(np.uint8), (mask_array == 1).astype(np.uint8))
                apl = AddedPathLength(segs_3D == 1, mask_array == 1)
                results_df.append({
                    "ID": patient_id,
                    "Volume_Dice": dice,
                    "Added_Path_Length": apl
                })
                print(f'Metrics for {patient_id}: Dice: {dice}, APL: {apl}')

            # Save mask
            save_mask = sitk.GetImageFromArray(segs_3D)
            save_mask.SetSpacing(spacing)
            sitk.WriteImage(save_mask, masks_dir / f'{patient_id}.nii.gz')

        # Save results
        results_df = pd.DataFrame(results_df)
        results_df.to_csv(Path(self.config.output_dir) / 'results.csv', index=False)


if __name__ == '__main__':
    config = MedSAM3DInferenceConfig(
        dataset_csv="data/temp_dataset_mandible.csv",
        model_config_path="configs/sam2.1_hiera_t512.yaml",
        checkpoint_path="checkpoints/MedSAM2_latest.pt",
        output_dir="output_mandible",
        window_level=500.0,
        window_width=2500.0,
    )

    inference_module = MedSAM3DInference(config)
    inference_module.run()