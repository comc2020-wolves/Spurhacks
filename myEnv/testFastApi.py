from fastapi import FastAPI
import torch
from PIL import Image
from torchvision import transforms
import cv2
from torchvision import models
import numpy as np
from skimage.feature import peak_local_max
from scipy.spatial.distance import pdist
from scipy.stats import kurtosis, skew, pearsonr
import torch.nn as nn

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

def extract_fft_features(image_path):
    """
    Read image from path, convert to grayscale and color as needed, then compute a feature vector
    containing all implemented FFT-based metrics.
    Returns a dict of feature_name: value.
    """
    # Read with cv2 to ensure consistent handling; supports many formats
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")
    # If image has alpha channel, drop it
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
        
    
        
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
    def __init__(self, featureExtractor, fft_feature_count):
        super(ImageClassifier, self).__init__()
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

def runModel(imagePath):
    img = Image.open(imagePath).convert('RGB')
    processImage=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    imgTensor = processImage(img)
    
    # Add batch dimension - this is the key fix!
    imgTensor = imgTensor.unsqueeze(0)  # Shape: (1, 3, 224, 224)
    
    signalFeatures=extract_fft_features(imagePath)
    fft_feature_names = [
            'fft_vertical_line_ratio',
            'fft_horizontal_line_ratio',
            'fft_central_cross_ratio',
            'fft_radial_slope',
            'fft_high_low_freq_ratio',
            'fft_mid_band_gap',
            'fft_entropy',
            'fft_peak_count',
            'fft_peak_regularity',
            'fft_angular_variance',
            'fft_kurtosis',
            'fft_skew',
            'fft_corr_rg',
            'fft_corr_rb',
            'fft_corr_gb'
        ]
    to_log = ['fft_angular_variance', 'fft_peak_regularity', 'fft_high_low_freq_ratio']
    
    raw_vals = np.array([signalFeatures[name] for name in fft_feature_names], dtype=np.float32)
    # Handle NaN
    if np.isnan(raw_vals).any():
        raw_vals = np.nan_to_num(raw_vals, nan=0.0)
    # Apply log1p to selected features
    for i, name in enumerate(fft_feature_names):
        if name in to_log:
            v = raw_vals[i]
            if v < 0:
                v = 0.0
            raw_vals[i] = np.log1p(v)
    
    # Convert to tensor and add batch dimension
    raw_vals = torch.tensor(raw_vals, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 15)
    
    vgg16 = models.vgg16(pretrained=True)
    feature_extractor = vgg16.features
    model=ImageClassifier(feature_extractor,15)
    model=model.to(device)
    model.load_state_dict(torch.load(r"C:\Users\Tristan\Downloads\spurhacks\myEnv\image_classifier.pt"))

    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        outputs=model(imgTensor,raw_vals)
        probability = outputs.item()
        prediction = "FAKE" if probability < 0.5 else "REAL"
        print(f"Prediction: {prediction}")
        print(f"Confidence: {probability:.4f} ({prediction})")
        print(f"Raw output: {outputs}")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
runModel(r"FirstImmigrant.jpg")
# app = FastAPI()


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}