import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
from pathlib import Path
import random
import matplotlib.pyplot as plt

# Define the Dataset class with the corrected loading mechanism
class KidneyDataset(Dataset):
    def __init__(self, data_path, transform=None, split='train', split_ratio=(0.7, 0.15, 0.15), seed=42):
        """
        Args:
            data_path: Root directory of the dataset
            transform: Optional transform to be applied to samples
            split: One of 'train', 'val', 'test'
            split_ratio: Tuple of (train_ratio, val_ratio, test_ratio)
            seed: Random seed for reproducibility
        """
        self.data_files = []
        for case_folder in os.listdir(data_path):
            case_path = os.path.join(data_path, case_folder)
            npz_file = os.path.join(case_path, "data.npz")
            if os.path.isdir(case_path) and os.path.exists(npz_file):
                self.data_files.append(npz_file)
        
        print(f"Found {len(self.data_files)} cases in {data_path}")
        self.transform = transform
        
        # Split the dataset
        random.seed(seed)
        self.data_files.sort()  # Ensure deterministic order before shuffling
        random.shuffle(self.data_files)
        
        train_size = int(len(self.data_files) * split_ratio[0])
        val_size = int(len(self.data_files) * split_ratio[1])
        
        if split == 'train':
            self.data_files = self.data_files[:train_size]
        elif split == 'val':
            self.data_files = self.data_files[train_size:train_size+val_size]
        elif split == 'test':
            self.data_files = self.data_files[train_size+val_size:]
        
        print(f"Using {len(self.data_files)} samples for {split} set")
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        npz_path = self.data_files[idx]
        
        # Load the npz file
        data = np.load(npz_path)
        image = data['images']
        segmentation = data['segmentations']
        
        # Convert to correct shape and type
        # Assuming the images and segmentations are 3D volumes
        # Select a random slice for training
        if image.shape[0] > 1:  # If we have multiple slices
            slice_idx = random.randint(0, image.shape[0]-1)
        else:
            slice_idx = 0
        
        # Get 2D slice
        image_slice = image[slice_idx].astype(np.float32)
        mask_slice = segmentation[slice_idx].astype(np.float32)
        
        # Normalize image to [0, 1] range if not already
        if image_slice.max() > 0:  # Avoid division by zero
            image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
        
        # Ensure mask is binary
        mask_slice = (mask_slice > 0).astype(np.float32)
        
        # Create a dict for albumentations
        sample = {
            'image': image_slice,
            'mask': mask_slice
        }
        
        # Apply transformations
        if self.transform:
            sample = self.transform(**sample)
        
        # Ensure image has the right dimensions [C, H, W]
        if isinstance(sample['image'], np.ndarray):
            sample['image'] = torch.from_numpy(sample['image']).unsqueeze(0)
            sample['mask'] = torch.from_numpy(sample['mask']).long()
        elif sample['image'].ndim == 2:
            sample['image'] = sample['image'].unsqueeze(0)
        
        # Output format expected by the model
        return {
            'image': sample['image'],
            'label': sample['mask'].long()
        }

# Define transformations for data augmentation with fixed parameters
class MedicalTransform:
    def __init__(self, output_size=(128, 128), is_train=True):
        self.output_size = output_size
        
        if is_train:
            self.transform = A.Compose([
                A.Resize(height=output_size[0], width=output_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
                A.OneOf([
                    A.GridDistortion(p=0.5),
                    A.ElasticTransform(alpha=1, sigma=50, p=0.5),
                ], p=0.25),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.GaussNoise(var_limit=0.05, mean=0, p=0.2),
                A.Normalize(mean=0.0, std=1.0),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=output_size[0], width=output_size[1]),
                A.Normalize(mean=0.0, std=1.0),
                ToTensorV2(),
            ])
    
    def __call__(self, **kwargs):
        return self.transform(**kwargs)

# Define Double U-Net Model
class DoubleUNet(nn.Module):
    def __init__(self, enc_name='efficientnet-b1', classes=2, pretrain='imagenet'):
        super(DoubleUNet, self).__init__()
       
        self.unet_plus = smp.UnetPlusPlus(
            encoder_name=enc_name,
            encoder_weights=pretrain,
            in_channels=1,
            classes=classes,
        )
       
        self.unet = smp.Unet(
            encoder_name=enc_name,
            encoder_weights=pretrain,
            in_channels=1,
            classes=classes,
        )
       
        self.final_conv = nn.Conv2d(classes * 2, classes, kernel_size=1)
   
    def forward(self, x):
        out1 = self.unet_plus(x)
        out2 = self.unet(x)
        out = torch.cat((out1, out2), dim=1)
        return self.final_conv(out)

# Evaluator class for metrics
class Evaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))
    
    def add(self, pred, target):
        # Flatten the arrays
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        
        # Accumulate confusion matrix
        mask = (target >= 0) & (target < self.num_classes)
        if mask.sum() > 0:
            hist = np.bincount(
                self.num_classes * target[mask].astype(int) + pred[mask],
                minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
            self.confusion_matrix += hist
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def pixel_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc
    
    def class_accuracy(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return acc
    
    def mean_iou(self):
        intersection = np.diag(self.confusion_matrix)
        ground_truth_set = self.confusion_matrix.sum(axis=1)
        predicted_set = self.confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        
        iou = intersection / union.astype(np.float32)
        return np.nanmean(iou)
    
    def class_iou(self):
        intersection = np.diag(self.confusion_matrix)
        ground_truth_set = self.confusion_matrix.sum(axis=1)
        predicted_set = self.confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        
        iou = intersection / union.astype(np.float32)
        return iou
    
    def frequency_weighted_iou(self):
        freq = self.confusion_matrix.sum(axis=1) / self.confusion_matrix.sum()
        iu = np.diag(self.confusion_matrix) / (
                self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) -
                np.diag(self.confusion_matrix))
        
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwavacc
    
    def dice_coefficient(self):
        intersection = np.diag(self.confusion_matrix)
        ground_truth_set = self.confusion_matrix.sum(axis=1)
        predicted_set = self.confusion_matrix.sum(axis=0)
        
        dice = (2 * intersection) / (ground_truth_set + predicted_set)
        return np.nanmean(dice)

# Save checkpoint
def save_checkpoint(epoch, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved at {path}")

# Load checkpoint
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

# Training function
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    best_iou = 0.0
    
    # Keep track of metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
       
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
           
            running_loss = 0.0
            evaluator = Evaluator(2)
           
            for batch in tqdm(dataloaders[phase], desc=f"{phase}"):
                inputs, labels = batch['image'].to(device), batch['label'].to(device)
               
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                   
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
               
                running_loss += loss.item() * inputs.size(0)
                evaluator.add(outputs.argmax(dim=1).cpu().numpy(), labels.cpu().numpy())
           
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_iou = evaluator.mean_iou()
            epoch_dice = evaluator.dice_coefficient()
            
            # Save metrics
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_iou'].append(epoch_iou)
            
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f}, Dice: {epoch_dice:.4f}")
           
            # Save best model based on IoU
            if phase == 'val' and epoch_iou > best_iou:
                best_iou = epoch_iou
                best_model_wts = model.state_dict()
                save_checkpoint(epoch, model, optimizer, checkpoint_path)
                print(f"New best model saved with IoU: {best_iou:.4f}")
            
            # Also check loss for scheduler
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss

        scheduler.step(best_loss)  # Update learning rate
        print()

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    save_checkpoint(num_epochs, model, optimizer, final_path)
    
    # Plot training history
    plot_training_history(history, checkpoint_dir)
    
    return model, history

# Function to plot training history
def plot_training_history(history, save_dir):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_iou'], label='Training IoU')
    plt.plot(history['val_iou'], label='Validation IoU')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

# Testing function
def test_model(model, dataloader, criterion, device):
    model.eval()
    evaluator = Evaluator(2)
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            evaluator.add(outputs.argmax(dim=1).cpu().numpy(), labels.cpu().numpy())
    
    test_loss = running_loss / len(dataloader.dataset)
    test_iou = evaluator.mean_iou()
    test_dice = evaluator.dice_coefficient()
    class_iou = evaluator.class_iou()
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test IoU: {test_iou:.4f}")
    print(f"Test Dice coefficient: {test_dice:.4f}")
    print(f"Per-class IoU: Background: {class_iou[0]:.4f}, Kidney: {class_iou[1]:.4f}")
    
    return {
        'loss': test_loss,
        'iou': test_iou,
        'dice': test_dice,
        'class_iou': class_iou
    }

# Visualize predictions
def visualize_predictions(model, dataset, device, num_samples=5, save_dir='predictions'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            true_mask = sample['label'].numpy()
            
            pred_mask = model(image).argmax(dim=1).cpu().numpy()[0]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            # True mask
            axes[1].imshow(true_mask, cmap='viridis')
            axes[1].set_title('True Mask')
            axes[1].axis('off')
            
            # Predicted mask
            axes[2].imshow(pred_mask, cmap='viridis')
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'prediction_{i}.png'))
            plt.close()

# Main function
def main():
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Configuration
    data_path = "output"  # Path to the dataset
    img_size = (128, 128)  # Input size for the model
    batch_size = 2
    num_epochs = 50
    lr = 1e-4
    num_classes = 2  # Background and kidney
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create transforms
    train_transform = MedicalTransform(output_size=img_size, is_train=True)
    val_transform = MedicalTransform(output_size=img_size, is_train=False)
    
    # Create datasets with integrated split
    train_dataset = KidneyDataset(data_path, transform=train_transform, split='train')
    val_dataset = KidneyDataset(data_path, transform=val_transform, split='val')
    test_dataset = KidneyDataset(data_path, transform=val_transform, split='test')
    
    # Verify we have data
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("One or more dataset splits contains no samples. Check your data path and directory structure.")
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=0, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=0, pin_memory=True),
    }
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=0, pin_memory=True)
    
    # Initialize the model
    model = DoubleUNet(enc_name='efficientnet-b1', classes=num_classes, pretrain='imagenet').to(device)
    
    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                     patience=5, verbose=True)
    
    # Train the model
    print("Starting training...")
    model, history = train_model(model, dataloaders, criterion, optimizer, scheduler, 
                                device, num_epochs=num_epochs)
    
    # Test the trained model
    print("Evaluating on test set...")
    test_results = test_model(model, test_dataloader, criterion, device)
    
    # Visualize some predictions
    print("Generating prediction visualizations...")
    visualize_predictions(model, test_dataset, device)
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()