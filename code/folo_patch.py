import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm
import argparse
import os

def get_patches_from_eeg(eeg, patch_size=32):
    """Convert EEG data from electrode space to 2D grid space
    Args:
        eeg: numpy array of shape (time_points, channels)
        patch_size: size of the output grid (patch_size x patch_size)
    Returns:
        patches: numpy array of shape (time_points, patch_size, patch_size)
    """
    # Load 2D locations
    locs_2d = np.loadtxt("folo_2d.csv", delimiter=",")
    x_min, x_max = locs_2d[:, 0].min(), locs_2d[:, 0].max()
    y_min, y_max = locs_2d[:, 1].min(), locs_2d[:, 1].max()

    grid_x, grid_y = np.mgrid[
        x_min : x_max : patch_size * 1j, y_min : y_max : patch_size * 1j
    ]

    patches = []
    for i in range(eeg.shape[0]):  # For each time point
        v_min = np.min(eeg[i])
        patch = griddata(
            locs_2d, eeg[i], (grid_x, grid_y), method="cubic", fill_value=v_min
        )
        patches.append(patch)

    # Stack all patches: (time_points, patch_size, patch_size)
    patches = np.stack(patches, axis=0)
    return patches

def process_and_save_data(eeg_path, save_path, patch_size=32):
    """Process all EEG data and save to new file
    Args:
        eeg_path: path to original EEG data
        save_path: path to save processed data
        patch_size: size of the output grid
    """
    # Load original data
    print("Loading original data...")
    data = torch.load(eeg_path)
    
    # Initialize lists to store processed data
    processed_dataset = []
    
    # Process each sample
    print("Processing EEG data to patches...")
    for idx in tqdm(range(len(data['dataset']))):
        sample = data['dataset'][idx]
        
        # Get EEG data and transpose to (time, channels)
        eeg = sample['eeg'].float().t().numpy()[20:460]  # Now shape is (440, 128)
        
        # Generate patches
        patches = get_patches_from_eeg(eeg, patch_size)  # Shape becomes (440, 32, 32)
        
        # Create new sample dictionary
        new_sample = {
            'eeg': torch.from_numpy(patches).float(),  # Convert back to tensor
            'label': sample['label'],
            'subject': sample['subject'] if 'subject' in sample else None
        }
        processed_dataset.append(new_sample)
    
    # Create save directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save processed data
    print(f"Saving processed data to {save_path}")
    save_dict = {
        'dataset': processed_dataset,
        'labels': data['labels'],
        'images': data['images'] if 'images' in data else None
    }
    torch.save(save_dict, save_path)
    print("Done!")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="EEG Patch Generation")
    parser.add_argument('-ed', '--eeg-dataset', default=r"block/eeg_55_95_std.pth", help="Input EEG dataset path")
    parser.add_argument('-sp', '--save-path', default=r"block/eeg_55_95_std_patch.pth", help="Path to save processed data")
    parser.add_argument('-ps', '--patch-size', default=32, type=int, help="patch size")
    args = parser.parse_args()

    # Process and save data
    process_and_save_data(
        eeg_path=args.eeg_dataset,
        save_path=args.save_path,
        patch_size=args.patch_size
    )
