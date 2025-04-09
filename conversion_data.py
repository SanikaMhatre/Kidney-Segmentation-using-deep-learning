import nibabel as nib
import numpy as np
from pathlib import Path
from dataset.kits19 import KiTS19  # Ensure correct import

def conversion_all():
    # Define dataset and output paths
    data = Path(r"F:\kidney_tumor_segmentation\kits19\data")
    output = Path(r"F:\kidney_tumor_segmentation\output")
    
    # Get all case directories
    cases = sorted([d for d in data.iterdir() if d.is_dir() and d.name.startswith("case_")])
    
    for case in cases:
        output_file = output / case.name / "data.npz"
        
        # ✅ Skip cases that are already processed
        if output_file.exists():
            print(f"Skipping {case.name} (already converted)")
            continue  
        
        print(f"Processing {case.name}...")
        try:
            conversion(case, output)
            print(f"✅ Completed: {case.name}")
        except Exception as e:
            print(f"❌ Error processing {case.name}: {e}")

def conversion(case, output):
    # Load imaging data
    vol_nii = nib.load(str(case / "imaging.nii.gz"))
    vol = vol_nii.get_fdata()
    vol = KiTS19.normalize(vol)
    
    # Load segmentation if available
    segmentation_file = case / "segmentation.nii.gz"
    if segmentation_file.exists():
        seg = nib.load(str(segmentation_file)).get_fdata()
    else:
        seg = None
    
    # Save all data as .npz (compressed format)
    output_dir = output / case.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if seg is not None:
        np.savez_compressed(str(output_dir / "data.npz"), images=vol, segmentations=seg, affine=vol_nii.affine)
    else:
        np.savez_compressed(str(output_dir / "data.npz"), images=vol, affine=vol_nii.affine)
    
if __name__ == "__main__":
    conversion_all()
