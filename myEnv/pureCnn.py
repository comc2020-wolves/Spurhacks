import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score
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
def compute_fft(img):
    """
    Compute the log-magnitude spectrum of the grayscale image `img`.
    `img` should be a 2D numpy array (grayscale).
    """
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)
    log_magnitude = np.log1p(magnitude_spectrum)  # log scale
    return log_magnitude

def fft_line_energy(log_mag):
    """
    Compute central vertical/horizontal line energy ratios.
    Returns (vertical_ratio, horizontal_ratio).
    """
    h, w = log_mag.shape
    vertical_energy = np.sum(log_mag[:, w // 2])
    horizontal_energy = np.sum(log_mag[h // 2, :])
    total_energy = np.sum(log_mag) + 1e-8
    return (
        vertical_energy / total_energy,
        horizontal_energy / total_energy,
    )

def fft_central_cross_ratio(log_mag):
    """
    Central cross energy ratio: sum of central row + column over total energy.
    """
    h, w = log_mag.shape
    central_row = log_mag[h // 2, :]
    central_col = log_mag[:, w // 2]
    # central pixel counted twice; subtract once
    total_energy = np.sum(log_mag) + 1e-8
    cross_energy = np.sum(central_row) + np.sum(central_col) - log_mag[h // 2, w // 2]
    return cross_energy / total_energy

def radial_profile(log_mag, nbins=100):
    """
    Compute radial profile: average of log_mag over rings.
    Returns:
      bin_centers: array of radii
      profile: array of average log_mag for each radius bin
    """
    h, w = log_mag.shape
    center = (h // 2, w // 2)
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r_flat = r.flatten()
    mag_flat = log_mag.flatten()
    # Bin radii
    max_r = np.max(r_flat)
    bins = np.linspace(0, max_r, nbins + 1)
    bin_idxs = np.digitize(r_flat, bins) - 1  # indices 0..nbins-1
    profile = np.zeros(nbins)
    counts = np.zeros(nbins)
    for i in range(len(r_flat)):
        idx = bin_idxs[i]
        if 0 <= idx < nbins:
            profile[idx] += mag_flat[i]
            counts[idx] += 1
    # Avoid division by zero
    nonzero = counts > 0
    profile[nonzero] /= counts[nonzero]
    # Bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers[nonzero], profile[nonzero]

def fft_radial_slope(log_mag, fit_range=(0.05, 0.5), nbins=200):
    """
    Fit a power-law slope on the radial profile in log-log:
    P(r) ~ r^alpha, so log P vs log r slope is alpha.
    fit_range: tuple fractions of max radius (e.g. 0.05 to 0.5 of max radius).
    Returns slope alpha.
    """
    bin_centers, profile = radial_profile(log_mag, nbins=nbins)
    # Exclude zero radius
    # Normalize radius to [0,1]
    max_r = np.max(bin_centers)
    norm_r = bin_centers / (max_r + 1e-8)
    # Select fit indices
    mask = (norm_r >= fit_range[0]) & (norm_r <= fit_range[1]) & (profile > 0)
    if np.sum(mask) < 2:
        return np.nan
    log_r = np.log(norm_r[mask])
    log_p = np.log(profile[mask])
    # Fit linear model: log_p = alpha * log_r + c
    alpha, intercept = np.polyfit(log_r, log_p, 1)
    return alpha

def fft_high_low_freq_ratio(log_mag, low_frac=0.1, high_frac=0.4):
    """
    Ratio of high-frequency energy to low-frequency energy.
    low_frac: radius fraction below which is considered low-frequency.
    high_frac: radius fraction above which is considered high-frequency.
    """
    h, w = log_mag.shape
    center = (h // 2, w // 2)
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    max_r = np.max(r)
    low_mask = r < (max_r * low_frac)
    high_mask = r > (max_r * high_frac)
    low_sum = np.sum(log_mag[low_mask])
    high_sum = np.sum(log_mag[high_mask])
    return high_sum / (low_sum + 1e-8)

def fft_mid_band_gap(log_mag, fit_range=(0.05, 0.5), mid_range=(0.15, 0.35), nbins=200):
    """
    Compute mid-band gap index: difference between expected power-law profile
    and actual in mid-frequency band.
    Returns mean relative deviation in mid band: (expected - actual) / expected.
    """
    # Get radial profile and fit slope
    bin_centers, profile = radial_profile(log_mag, nbins=nbins)
    max_r = np.max(bin_centers)
    norm_r = bin_centers / (max_r + 1e-8)
    mask_fit = (norm_r >= fit_range[0]) & (norm_r <= fit_range[1]) & (profile > 0)
    if np.sum(mask_fit) < 2:
        return np.nan
    log_r = np.log(norm_r[mask_fit])
    log_p = np.log(profile[mask_fit])
    alpha, intercept = np.polyfit(log_r, log_p, 1)
    # Expected in mid band
    mask_mid = (norm_r >= mid_range[0]) & (norm_r <= mid_range[1])
    if not np.any(mask_mid):
        return np.nan
    expected = np.exp(intercept) * (norm_r[mask_mid] ** alpha)
    actual = profile[mask_mid]
    # Compute relative gap: positive if expected > actual (deficit)
    rel_gap = (expected - actual) / (expected + 1e-8)
    return np.mean(rel_gap)

def fft_entropy(log_mag, bins=128):
    """
    Compute Shannon entropy of flattened log-magnitude spectrum.
    """
    hist, _ = np.histogram(log_mag.flatten(), bins=bins, density=True)
    hist += 1e-8
    entropy = -np.sum(hist * np.log(hist))
    return entropy

def fft_peak_features(log_mag, threshold_ratio=0.6, min_distance=10):
    """
    Detect peaks in normalized log-magnitude spectrum and compute:
    - peak_count: number of peaks
    - regularity: stddev of pairwise distances among peaks
    """
    norm_fft = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)
    peaks = peak_local_max(norm_fft, min_distance=min_distance, threshold_abs=threshold_ratio)
    peak_count = len(peaks)
    if peak_count > 1:
        dists = pdist(peaks)
        regularity = np.std(dists)
    else:
        regularity = 0.0
    return peak_count, regularity

def fft_angular_variance(log_mag, n_bins=36):
    """
    Compute angular energy variance: split 0-360 degrees into bins, sum energy in each,
    return variance.
    """
    h, w = log_mag.shape
    center = (h // 2, w // 2)
    y, x = np.indices((h, w))
    angles = np.arctan2(y - center[0], x - center[1])
    angles = (angles + np.pi) * (180 / np.pi)  # 0 to 360
    bins = np.linspace(0, 360, n_bins + 1)
    angular_energy = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (angles >= bins[i]) & (angles < bins[i+1])
        angular_energy[i] = np.sum(log_mag[mask])
    return np.var(angular_energy)

def fft_kurtosis_skew(log_mag):
    """
    Compute kurtosis and skew of log-magnitude values.
    """
    flat = log_mag.flatten()
    return kurtosis(flat), skew(flat)

def fft_rgb_cross_spectral_corr(img_color):
    """
    Compute cross-spectral correlation between RGB channels.
    img_color: HxWx3 array.
    Returns correlation coefficients between pairs (R-G, R-B, G-B).
    """
    # Compute FFT magnitude for each channel
    corrs = []
    for i in range(3):
        for j in range(i+1, 3):
            ft_i = compute_fft(img_color[..., i])
            ft_j = compute_fft(img_color[..., j])
            # Flatten and compute Pearson correlation
            flat_i = ft_i.flatten()
            flat_j = ft_j.flatten()
            corr, _ = pearsonr(flat_i, flat_j)
            corrs.append(corr)
    # Return as tuple (R-G, R-B, G-B)
    return tuple(corrs)

def extract_fft_features(image_path, applyAugmentation, save_path=None):
    """
    Read image from path, convert to grayscale and color as needed, then compute a feature vector
    containing all implemented FFT-based metrics.
    Returns a dict of feature_name: value.
    
    Args:
        image_path: Path to the input image
        applyAugmentation: Boolean to determine if augmentation should be applied
        save_path: Optional path to save the augmented image as PNG (e.g., "augmented_image.png")
    """
    # Read with cv2 to ensure consistent handling; supports many formats
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")
    # If image has alpha channel, drop it
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
        
    
    # Apply augmentation if requested
    if applyAugmentation:
        # Convert BGR to RGB for PIL
        if img.ndim == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img_rgb)
        
        # Augmentation transform without any fill - uses only transformations that don't create empty areas
        augmentation_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to target size first
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
            # Removed ColorJitter to fix linter errors
        ])
        
        #Apply augmentation
        pil_img = augmentation_transform(pil_img)
        
        # Save augmented image if save_path is provided
        if save_path:
            pil_img.save(save_path, 'PNG')
            print(f"Augmented image saved to: {save_path}")
        
        #Convert back to numpy array
        img_aug = np.array(pil_img)
        
        #Convert RGB back to BGR for cv2 processing
        if img_aug.ndim == 3:
            img = cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR)
        else:
            img = img_aug
        
        
    # Convert to float grayscale for FFT
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img.astype(np.float32)
    # Normalize grayscale to [0,1]
    gray = gray - gray.min()
    if gray.max() > 0:
        gray = gray / gray.max()
    log_mag = compute_fft(gray)
    features = {}
    # Central cross
    v_line, h_line = fft_line_energy(log_mag)
    features['fft_vertical_line_ratio'] = v_line
    features['fft_horizontal_line_ratio'] = h_line
    features['fft_central_cross_ratio'] = fft_central_cross_ratio(log_mag)
    # Radial features
    features['fft_radial_slope'] = fft_radial_slope(log_mag)
    features['fft_high_low_freq_ratio'] = fft_high_low_freq_ratio(log_mag)
    features['fft_mid_band_gap'] = fft_mid_band_gap(log_mag)
    # Spectral entropy
    features['fft_entropy'] = fft_entropy(log_mag)
    # Peak features
    peak_count, peak_reg = fft_peak_features(log_mag)
    features['fft_peak_count'] = peak_count
    features['fft_peak_regularity'] = peak_reg
    # Angular
    features['fft_angular_variance'] = fft_angular_variance(log_mag)
    # Kurtosis & skew
    k, s = fft_kurtosis_skew(log_mag)
    features['fft_kurtosis'] = k
    features['fft_skew'] = s
    # Cross-spectral correlations (if color)
    if img.ndim == 3 and img.shape[2] == 3:
        # Convert BGR (cv2) to RGB order
        img_rgb = img[..., ::-1].astype(np.float32)
        # Normalize each channel to [0,1]
        for c in range(3):
            ch = img_rgb[..., c]
            ch = ch - ch.min()
            if ch.max() > 0:
                img_rgb[..., c] = ch / ch.max()
        corr_rg, corr_rb, corr_gb = fft_rgb_cross_spectral_corr(img_rgb)
        features['fft_corr_rg'] = corr_rg
        features['fft_corr_rb'] = corr_rb
        features['fft_corr_gb'] = corr_gb
    return features


class ImageClassifier(nn.Module):
    def __init__(self, featureExtractor):
        super(ImageClassifier, self).__init__()
        self.features = featureExtractor
        self.flatten = nn.Flatten()
        
        # Pure CNN classifier - only using image features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),  # VGG16 features output size
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
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

    def forward(self, image):
        imgFeatures = self.features(image)
        img = self.flatten(imgFeatures)
        out = self.classifier(img)
        return out

#BUILD DATALOADER
class imageLoader(Dataset):
    def __init__(self, filePaths, labels):
        self.files = filePaths  # list of file paths (image paths)
        self.labels = labels    # list of labels (1.0 for real, 0.0 for fake)
        # Image preprocessing to match VGG16
        self.processImage = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        # Image loading and preprocessing
        img_path = self.files[index]
        img = Image.open(img_path).convert('RGB')
        imgTensor = self.processImage(img)
        # Label
        label = float(self.labels[index])

        return imgTensor, torch.tensor(label, dtype=torch.float32)

        



def train_validate_test(
    model,
    train_dataset,
    val_dataset,
    test_dataset,
    num_epochs=50,  # Increased epochs
    batch_size=16,  # Smaller batch size for better generalization
    lr=5e-5,  # Lower learning rate for better convergence
    weight_decay=1e-3,  # Increased weight decay for better regularization
    max_training_time_hours=3,  # Increased training time
    model_save_path="pureCnn.pt"
):
    """
    Train, validate, and test the ImageClassifier model with given datasets.
    - train_dataset, val_dataset, test_dataset: instances of imageLoader (Dataset)
    - num_epochs: maximum number of epochs
    - batch_size: DataLoader batch size
    - lr: learning rate for Adam
    - weight_decay: weight decay (L2) for optimizer
    - max_training_time_hours: training time limit in hours (e.g., 1.5 for 1.5 hours)
    - model_save_path: file path to save the best model state_dict
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DataLoaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()  # assumes model outputs sigmoid probabilities
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    
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

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_losses = []
        all_preds = []
        all_labels = []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # shape (batch,1)

            optimizer.zero_grad()
            outputs = model(images)  # (batch,1) with sigmoid
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_losses.append(loss.item())
            preds = (outputs.detach().cpu().numpy() >= 0.5).astype(int)
            all_preds.extend(preds.flatten().tolist())
            all_labels.extend(labels.detach().cpu().numpy().flatten().tolist())

        train_loss = np.mean(train_losses)
        train_acc = accuracy_score(all_labels, all_preds)

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(images,)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                preds = (outputs.cpu().numpy() >= 0.5).astype(int)
                val_preds.extend(preds.flatten().tolist())
                val_labels.extend(labels.cpu().numpy().flatten().tolist())

        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Update learning rate
        scheduler.step()

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
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
            preds = (outputs.cpu().numpy() >= 0.5).astype(int)
            test_preds.extend(preds.flatten().tolist())
            test_labels.extend(labels.cpu().numpy().flatten().tolist())

    test_loss = np.mean(test_losses)
    test_acc = accuracy_score(test_labels, test_preds)
    print(f"\n=== FINAL RESULTS ===")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    if test_acc >= 0.80:
        print("🎉 SUCCESS! Achieved 80%+ accuracy!")
    else:
        print("⚠️  Did not reach 80% accuracy. Consider further tuning.")

    return model

def main():
    
    # Use a more powerful backbone with better initialization
    vgg16 = models.vgg16(pretrained=True)

    # Freeze convolutional layers
    for param in vgg16.features[:20].parameters():  # Freeze first 20 layers
        param.requires_grad = False
    
    # Unfreeze some later layers for fine-tuning
    for param in vgg16.features[20:].parameters():
        param.requires_grad = True

    feature_extractor = vgg16.features
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    realData=os.path.join(script_dir, r"AI-Generated-vs-Real-Images-Datasets\RealArt\RealArt")
    fakeData=os.path.join(script_dir, r"AI-Generated-vs-Real-Images-Datasets\AiArtData\AiArtData")

    filePaths=[]
    for file in os.listdir(realData):#Probability that it is real
        fullPath=os.path.join(realData,file)
        filePaths.append((fullPath,1.0))
    for file in os.listdir(fakeData):
        fullPath=os.path.join(fakeData,file)
        filePaths.append((fullPath,0.0))
        
        
    paths, labels=zip(*filePaths)  
    paths=list(paths)
    labels=list(labels)  
    
    print(f"Total images: {len(paths)}")
    print(f"Real images: {sum(labels)}")
    print(f"Fake images: {len(labels) - sum(labels)}")
    
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels,
        test_size=0.15,
        stratify=labels,
        random_state=42
    )

    # Now split train into train/val. The remaining after removing test is 85% of data.
    # To get ~15% of total as validation, we take test_size = 0.15/0.85 ≈ 0.1765 on train_paths:
    val_frac = 0.15 / 0.85
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=val_frac,
        stratify=train_labels,
        random_state=42
    )
    
    print(f"Train set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    print(f"Test set: {len(test_paths)} images")
    
    train_dataset = imageLoader(train_paths, train_labels)
    val_dataset   = imageLoader(val_paths, val_labels)
    test_dataset  = imageLoader(test_paths, test_labels)
    
    model=ImageClassifier(feature_extractor)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train with improved parameters
    train_validate_test(model,train_dataset,val_dataset,test_dataset)
    
if __name__=="__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()