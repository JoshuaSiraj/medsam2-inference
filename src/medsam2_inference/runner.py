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

        df = []

        for _, row in dataset.iterrows():
            patient_id = row["ID"]
            image_path = Path(str(row["image_path"]))
            mask_path = Path(str(row["mask_path"]))
            
            # Check if image and mask files exist
            if not image_path.exists():
                print(f"Warning: Image file not found: {image_path}, skipping...")
                continue
            if not mask_path.exists():
                print(f"Warning: Mask file not found: {mask_path}, skipping...")
                continue
            
            try:
                image = sitk.ReadImage(str(image_path))
                image_array = sitk.GetArrayFromImage(image)
                mask = sitk.ReadImage(str(mask_path))
                mask_array = sitk.GetArrayFromImage(mask)
            except Exception as e:
                print(f"Error: {e}")
                continue

            spacing = image.GetSpacing()
            image_array = preprocess(
                image_array, 
                window_level=self.config.window_level, 
                window_width=self.config.window_width
            )

            segs_3D = np.zeros(image_array.shape, dtype=np.uint8)

            unique_labels = np.unique(mask_array)
            unique_labels = unique_labels[unique_labels != 0]
            print(f'Unique labels: {unique_labels}')

            # dsc_organ_dict = {}

            for label in unique_labels:
                mask_array_per_label = (mask_array == label)*label
                bbox3d = mask3D_to_bbox(mask_array_per_label, mask_path)
                bbox2d = bbox3d[[0,1,3,4]] # [x_min, y_min, x_max, y_max]

                zs, _, _ = np.where(mask_array_per_label>0)
                zs = np.unique(zs)
                z_min = min(zs)
                z_max = max(zs)
                z_mid_orig = (z_min + z_max)//2 
                z_mid = z_mid_orig - z_min

                cropped_image = image_array[z_min:z_max+1]
                cropped_mask = mask_array_per_label[z_min:z_max+1]
                cropped_mask = cropped_mask.astype(np.uint8)

                video_height = cropped_image.shape[1]
                video_width = cropped_image.shape[2]

                if video_height != 512 or video_width != 512:
                    cropped_image = resize_grayscale_to_rgb_and_resize(cropped_image, 512)
                else:
                    cropped_image = cropped_image[:,None].repeat(3, axis=1)
                cropped_image = cropped_image / 255.0
                cropped_image = torch.from_numpy(cropped_image).cuda()
                img_mean = torch.tensor(self.config.mean, dtype=torch.float32)[:, None, None].cuda()
                img_std = torch.tensor(self.config.std, dtype=torch.float32)[:, None, None].cuda()
                cropped_image -= img_mean
                cropped_image /= img_std

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    inference_state = self.predictor.init_state(cropped_image, video_height, video_width)
                    if self.config.propagate_with_bbox:
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                                                            inference_state=inference_state,
                                                            frame_idx=z_mid,
                                                            obj_id=1,
                                                            box=bbox2d,
                                                        )
                        mask_prompt = (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
                    else: # gt
                        mask_prompt = (cropped_mask[z_mid] == label).astype(np.uint8)

                    _, _, masks = self.predictor.add_new_mask(
                        inference_state, 
                        frame_idx=z_mid, 
                        obj_id=1,
                        mask=mask_prompt
                    )
                    segs_3D[z_mid_orig, ((masks[0] > 0.0).cpu().numpy())[0]] = label

                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state, start_frame_idx=z_mid, reverse=False):
                        segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = label

                    # reverse process, delete old memory and initialize new predictor
                    self.predictor.reset_state(inference_state)
                    inference_state = self.predictor.init_state(cropped_image, video_height, video_width)
                    _, _, masks = self.predictor.add_new_mask(
                        inference_state, 
                        frame_idx=z_mid, 
                        obj_id=1, 
                        mask=mask_prompt
                    )

                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state, start_frame_idx=z_mid, reverse=True):
                        segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = label

                    self.predictor.reset_state(inference_state)

                    dice = dice_multi_class((segs_3D == label).astype(np.uint8), (mask_array == label).astype(np.uint8))
                    dice = np.round(dice, 4)
                    # dsc_organ_dict.update({label.item(): dice.item()})
                    apl = AddedPathLength(segs_3D == label, mask_array == label)
                    df.append({
                        "ID": patient_id,
                        "label": label,
                        "dice": dice,
                        "apl": apl
                    })
                    print(f'Dice for {patient_id} and label {label}: {dice}, APL: {apl}')

            save_mask = sitk.GetImageFromArray(segs_3D)
            save_mask.SetSpacing(spacing)
            sitk.WriteImage(save_mask, masks_dir / f'{patient_id}.nii.gz')

        df = pd.DataFrame(df)
        df.to_csv(Path(self.config.output_dir) / 'metrics.csv', index=False)


if __name__ == '__main__':
    config = MedSAM3DInferenceConfig(
        dataset_csv="/home/gpudual/bhklab/josh/auto-seg-bias/data/temp_dataset_mandible.csv",
        model_config_path="configs/sam2.1_hiera_t512.yaml",
        checkpoint_path="/home/gpudual/bhklab/josh/auto-seg-bias/workflow/scripts/medsam2_runner/MedSAM2/checkpoints/MedSAM2_latest.pt",
        output_dir="tmp_output_mandible",
        window_level=500.0,
        window_width=2500.0,
    )

    inference_module = MedSAM3DInference(config)
    inference_module.run()