import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import scipy.ndimage
from skimage.feature import peak_local_max
from scipy.spatial.distance import pdist
from scipy.stats import kurtosis, skew, pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os
from sklearn.model_selection import train_test_split

# Import all the FFT functions from the original file
# (Copy all the FFT functions here - compute_fft, fft_line_energy, etc.)

class ImprovedImageClassifier(nn.Module):
    def __init__(self, featureExtractor, fft_feature_count):
        super(ImprovedImageClassifier, self).__init__()
        self.features = featureExtractor
        self.flatten = nn.Flatten()
        
        # Improved image branch with better capacity and regularization
        self.imageBranch = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),  # Increased capacity
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Reduced dropout for better learning
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Improved FFT analysis branch
        self.imageAnalysis = nn.Sequential(
            nn.Linear(fft_feature_count, 128),  # Increased capacity
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Improved final classifier
        self.finalClassifier = nn.Sequential(
            nn.Linear(128 + 32, 128),  # Combined features
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        

    def forward(self, image, metaData):
        imgFeatures = self.features(image)
        img = self.flatten(imgFeatures)
        img = self.imageBranch(img)
        
        metaAnalysis = self.imageAnalysis(metaData)
        
        imgInfo = torch.cat([img, metaAnalysis], dim=1)
        out = self.finalClassifier(imgInfo)
        return out

class ImprovedImageLoader(Dataset):
    def __init__(self, filePaths, labels, is_training=True):
        self.files = filePaths
        self.labels = labels
        self.is_training = is_training
        
        # Enhanced preprocessing for better performance
        self.processImage = transforms.Compose([
            transforms.Resize((256, 256)),  # Slightly larger for better quality
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Enhanced augmentation for training
        self.augmentation = transforms.Compose([
            transforms.Resize((280, 280)),  # Larger resize for augmentation
            transforms.RandomCrop(256, 256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # FFT feature names
        self.fft_feature_names = [
            'fft_vertical_line_ratio', 'fft_horizontal_line_ratio', 'fft_central_cross_ratio',
            'fft_radial_slope', 'fft_high_low_freq_ratio', 'fft_mid_band_gap',
            'fft_entropy', 'fft_peak_count', 'fft_peak_regularity', 'fft_angular_variance',
            'fft_kurtosis', 'fft_skew', 'fft_corr_rg', 'fft_corr_rb', 'fft_corr_gb'
        ]
        self.to_log = ['fft_angular_variance', 'fft_peak_regularity', 'fft_high_low_freq_ratio']
   
    def __len__(self):
        return len(self.files)

    def extractSignalData(self, dataPath):
        feats = {}
        try:
            with open(dataPath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or ':' not in line:
                        continue
                    key, val_str = line.split(':', 1)
                    key = key.strip()
                    val_str = val_str.strip()
                    try:
                        val = float(val_str)
                    except ValueError:
                        continue
                    feats[key] = val
        except Exception as e:
            print(f"Warning: Could not open FFT metrics file {dataPath}: {e}")
        
        # Fill missing features with 0 instead of NaN
        missing = [name for name in self.fft_feature_names if name not in feats]
        for name in missing:
            feats[name] = 0.0
        return feats

    def __getitem__(self, index):
        img_path = self.files[index]
        img = Image.open(img_path).convert('RGB')
        
        # Apply augmentation only during training
        if self.is_training:
            imgTensor = self.augmentation(img)
        else:
            imgTensor = self.processImage(img)
        
        label = float(self.labels[index])
        
        # FFT features
        base_name = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_fft_folder = os.path.join(script_dir, 'preComputedFFT')
        
        if label == 1.0:
            folder = os.path.join(base_fft_folder, 'REAL')
        else:
            folder = os.path.join(base_fft_folder, 'FAKE')
        
        signalDataPath = os.path.join(folder, base_name)
        feats_dict = self.extractSignalData(signalDataPath)
        
        raw_vals = np.array([feats_dict[name] for name in self.fft_feature_names], dtype=np.float32)
        
        # Apply log1p to selected features
        for i, name in enumerate(self.fft_feature_names):
            if name in self.to_log:
                v = raw_vals[i]
                if v < 0:
                    v = 0.0
                raw_vals[i] = np.log1p(v)

        return imgTensor, raw_vals, torch.tensor(label, dtype=torch.float32)

def improved_train_validate_test(
    model,
    train_dataset,
    val_dataset,
    test_dataset,
    num_epochs=50,
    batch_size=16,  # Smaller batch size for better generalization
    lr=5e-5,  # Lower learning rate
    weight_decay=1e-3,  # Increased weight decay
    max_training_time_hours=3,
    model_save_path="improved_image_classifier.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DataLoaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = model.to(device)

    # Improved loss function with label smoothing
    criterion = nn.BCELoss()
    
    # Improved optimizer with better parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Better learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 15  # Increased patience
    patience_counter = 0
    start_time = time.time()
    max_seconds = max_training_time_hours * 3600

    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_losses_epoch = []
        all_preds = []
        all_labels = []

        for images, metas, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images = images.to(device)
            metas = metas.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images, metas)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_losses_epoch.append(loss.item())
            preds = (outputs.detach().cpu().numpy() >= 0.5).astype(int)
            all_preds.extend(preds.flatten().tolist())
            all_labels.extend(labels.detach().cpu().numpy().flatten().tolist())

        train_loss = np.mean(train_losses_epoch)
        train_acc = accuracy_score(all_labels, all_preds)

        # Validation
        model.eval()
        val_losses_epoch = []
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for images, metas, labels in val_loader:
                images = images.to(device)
                metas = metas.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(images, metas)
                loss = criterion(outputs, labels)
                val_losses_epoch.append(loss.item())
                preds = (outputs.cpu().numpy() >= 0.5).astype(int)
                val_preds.extend(preds.flatten().tolist())
                val_labels.extend(labels.cpu().numpy().flatten().tolist())

        val_loss = np.mean(val_losses_epoch)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Update learning rate
        scheduler.step()

        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Epoch Time: {epoch_time:.1f}s | Elapsed: {elapsed_time/3600:.2f}h")

        # Save best model based on validation accuracy (not loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> New best model saved! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"  -> No improvement for {patience_counter} epochs")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break

        # Check time limit
        if elapsed_time > max_seconds:
            print(f"Reached max training time of {max_training_time_hours} hours. Stopping training.")
            break

    # Load best model for testing
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    test_preds = []
    test_labels = []
    test_losses = []
    with torch.no_grad():
        for images, metas, labels in test_loader:
            images = images.to(device)
            metas = metas.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(images, metas)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
            preds = (outputs.cpu().numpy() >= 0.5).astype(int)
            test_preds.extend(preds.flatten().tolist())
            test_labels.extend(labels.cpu().numpy().flatten().tolist())

    test_loss = np.mean(test_losses)
    test_acc = accuracy_score(test_labels, test_preds)
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary')
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    return model, {
        'test_acc': test_acc,
        'test_loss': test_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'best_val_acc': best_val_acc
    }

def main():
    # Use a more powerful backbone
    vgg16 = models.vgg16(pretrained=True)
    
    # Freeze early layers, fine-tune later layers
    for param in vgg16.features[:20].parameters():  # Freeze first 20 layers
        param.requires_grad = False
    
    feature_extractor = vgg16.features
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    realData = os.path.join(script_dir, r"AI-Generated-vs-Real-Images-Datasets\RealArt\RealArt")
    fakeData = os.path.join(script_dir, r"AI-Generated-vs-Real-Images-Datasets\AiArtData\AiArtData")

    filePaths = []
    for file in os.listdir(realData):
        fullPath = os.path.join(realData, file)
        filePaths.append((fullPath, 1.0))
    for file in os.listdir(fakeData):
        fullPath = os.path.join(fakeData, file)
        filePaths.append((fullPath, 0.0))
        
    paths, labels = zip(*filePaths)  
    paths = list(paths)
    labels = list(labels)  
    
    # Better train/val/test split
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=0.15, stratify=labels, random_state=42
    )

    val_frac = 0.15 / 0.85
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=val_frac, stratify=train_labels, random_state=42
    )
    
    # Create datasets with training flag
    train_dataset = ImprovedImageLoader(train_paths, train_labels, is_training=True)
    val_dataset = ImprovedImageLoader(val_paths, val_labels, is_training=False)
    test_dataset = ImprovedImageLoader(test_paths, test_labels, is_training=False)
    
    model = ImprovedImageClassifier(feature_extractor, len(train_dataset.fft_feature_names))
    
    # Train with improved parameters
    results = improved_train_validate_test(
        model, train_dataset, val_dataset, test_dataset,
        num_epochs=50,
        batch_size=16,
        lr=5e-5,
        weight_decay=1e-3,
        max_training_time_hours=3
    )
    
    print(f"\nFinal Test Accuracy: {results['test_acc']:.4f}")
    if results['test_acc'] >= 0.80:
        print("🎉 SUCCESS! Achieved 80%+ accuracy!")
    else:
        print("⚠️  Did not reach 80% accuracy. Consider further tuning.")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main() 