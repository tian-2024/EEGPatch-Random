import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn
import numpy as np
import argparse

from resnet import resnet

cudnn.benchmark = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Define options
parser = argparse.ArgumentParser(description="EEG Patch Classification")
# Data
parser.add_argument('-ed', '--eeg-dataset', default=r"block/eeg_55_95_std_patch.pth", help="EEG dataset path",choices=["block/eeg_55_95_std_patch.pth","block/eeg_55_95_std_large_patch.pth"])
parser.add_argument('-sp', '--splits-path', default=r"block/block_splits_by_image_all.pth", help="splits path",choices=["block/eeg_55_95_std_small_splits.pth","block/block_splits_by_image_all.pth"])
parser.add_argument('-b', '--batch_size', default=32, type=int, help="batch size")
parser.add_argument('-o', '--optim', default="Adam", help="optimizer")
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help="learning rate")
parser.add_argument('-e', '--epochs', default=100, type=int, help="training epochs")
parser.add_argument('-dw', '--data-workers', default=4, type=int, help="data loading workers")

def create_image_from_patches(patches, dct_ratio=0.2, show_coeffs=True):
    """Convert EEG patches into a single image with partial DCT coefficients
    Args:
        patches: tensor of shape (440, 32, 32)
        dct_ratio: ratio of patches to replace with DCT coefficients
        show_coeffs: if True, show DCT coefficients; if False, show reconstructed image
    """
    # Group patches into 64 groups using numpy array_split
    num_groups = 64
    patches_np = patches.numpy()
    grouped_patches = np.array_split(patches_np, num_groups, axis=0)
    averaged_patches = torch.from_numpy(np.stack([group.mean(axis=0) for group in grouped_patches]))  # Shape: (64, 32, 32)
    
    # 随机选择要替换的patch索引
    num_to_replace = int(64 * dct_ratio)
    replace_indices = np.random.choice(64, num_to_replace, replace=False)
    
    # 对选中的patches进行DCT变换
    for idx in replace_indices:
        patch = averaged_patches[idx].numpy()
        # 进行2D DCT变换
        dct_coeffs = np.fft.fft2(patch).real
        
        if show_coeffs:
            # 直接使用完整的DCT系数（不截断）
            coeffs_vis = np.log(np.abs(dct_coeffs) + 1)
            # 归一化到原patch的数值范围
            patch_min, patch_max = patch.min(), patch.max()
            coeffs_vis = (coeffs_vis - coeffs_vis.min()) / (coeffs_vis.max() - coeffs_vis.min())
            coeffs_vis = coeffs_vis * (patch_max - patch_min) + patch_min
            averaged_patches[idx] = torch.from_numpy(coeffs_vis)
        else:
            # 重建低频图像时才截断
            dct_coeffs_truncated = dct_coeffs.copy()
            cutoff = 8
            dct_coeffs_truncated[cutoff:, :] = 0
            dct_coeffs_truncated[:, cutoff:] = 0
            patch_dct = np.fft.ifft2(dct_coeffs_truncated).real
            averaged_patches[idx] = torch.from_numpy(patch_dct)
    
    # Arrange 64 patches into 8x8 grid
    grid = averaged_patches.view(8, 8, 32, 32)
    image = grid.permute(0, 2, 1, 3).reshape(256, 256)
    image = image.unsqueeze(0)
    return image

class EEGPatchDataset:
    def __init__(self, eeg_signals_path, split_path, split_num=0, split_name="train", dct_ratio=0.5):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        self.dataset = loaded['dataset']
        self.dct_ratio = dct_ratio
        
        # Load split
        loaded_splits = torch.load(split_path)
        self.split_idx = loaded_splits['splits'][split_num][split_name]
    
    def __len__(self):
        return len(self.split_idx)
    
    def __getitem__(self, i):
        # Get sample from dataset
        sample = self.dataset[self.split_idx[i]]
        patches = sample['eeg']  # Shape: (440, 32, 32)
        label = sample['label']
        
        # Convert patches to image with DCT
        image = create_image_from_patches(patches, self.dct_ratio, show_coeffs=False)
        return image, label

def main():
    args = parser.parse_args()
    
    # 添加DCT ratio参数
    dct_ratio = 0.5  # 可以根据需要修改这个值：0.25, 0.125等
    
    # Create data loaders
    loaders = {
        split: torch.utils.data.DataLoader(
            EEGPatchDataset(
                args.eeg_dataset,
                args.splits_path,
                split_num=0,
                split_name=split,
                dct_ratio=dct_ratio  # 添加这个参数
            ),
            batch_size=args.batch_size,
            shuffle=(split == 'train'),
            num_workers=args.data_workers
        )
        for split in ['train', 'val', 'test']
    }
    
    # Initialize model
    model = resnet(attn_name="se", act_name="elu", blocks=[2, 2, 2, 2])
    
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = getattr(torch.optim, args.optim)(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        # Initialize metrics
        metrics = {'train': {'loss': 0, 'correct': 0, 'total': 0},
                  'val': {'loss': 0, 'correct': 0, 'total': 0},
                  'test': {'loss': 0, 'correct': 0, 'total': 0}}
        
        for split in ['train', 'val', 'test']:
            model.train() if split == 'train' else model.eval()
            torch.set_grad_enabled(split == 'train')
            
            for data, target in loaders[split]:
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                # Update metrics
                metrics[split]['loss'] += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                metrics[split]['correct'] += pred.eq(target).sum().item()
                metrics[split]['total'] += data.size(0)
                
                # Backward pass (only for training)
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        # Calculate epoch metrics
        for split in metrics:
            metrics[split]['loss'] /= metrics[split]['total']
            metrics[split]['acc'] = 100. * metrics[split]['correct'] / metrics[split]['total']
        
        # Update best accuracy
        if metrics['val']['acc'] > best_val_acc:
            best_val_acc = metrics['val']['acc']
            best_test_acc = metrics['test']['acc']
            best_epoch = epoch
        
        # Print progress
        print(f'Epoch {epoch}:')
        for split in metrics:
            print(f'{split.capitalize()}: Loss={metrics[split]["loss"]:.4f}, Acc={metrics[split]["acc"]:.2f}%')
        print(f'Best val acc: {best_val_acc:.2f}% (epoch {best_epoch}, test acc: {best_test_acc:.2f}%)\n')

if __name__ == "__main__":
    main() 