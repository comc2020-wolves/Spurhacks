import numpy as np
import scipy.ndimage
from skimage.feature import peak_local_max
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from tqdm import tqdm
def compute_fft(img):
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)
    log_magnitude = np.log1p(magnitude_spectrum)  # log scale
    return log_magnitude

def fft_line_energy(log_mag):
    h, w = log_mag.shape
    vertical_energy = np.sum(log_mag[:, w // 2])
    horizontal_energy = np.sum(log_mag[h // 2, :])
    total_energy = np.sum(log_mag)
    return (
        vertical_energy / total_energy,
        horizontal_energy / total_energy,
    )

def fft_radial_energy_ratio(log_mag):
    h, w = log_mag.shape
    y, x = np.ogrid[:h, :w]
    center = (h // 2, w // 2)
    radius = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    max_r = radius.max()

    low_band = log_mag[radius < max_r * 0.2]
    high_band = log_mag[radius > max_r * 0.6]

    return (np.sum(high_band) / (np.sum(low_band) + 1e-8))
    

def fft_entropy(log_mag, bins=128):
    hist, _ = np.histogram(log_mag.flatten(), bins=bins, density=True)
    hist += 1e-8
    entropy = -np.sum(hist * np.log(hist))
    return entropy

def fft_peak_features(log_mag, threshold_ratio=0.6):
    norm_fft = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min())
    peaks = peak_local_max(norm_fft, min_distance=10, threshold_abs=threshold_ratio)
    peak_count = len(peaks)

    if peak_count > 1:
        dists = pdist(peaks)
        regularity = np.std(dists)
    else:
        regularity = 0.0

    return (
        peak_count,
        regularity
    )

def fft_angular_variance(log_mag, n_bins=36):
    h, w = log_mag.shape
    center = (h // 2, w // 2)
    y, x = np.indices(log_mag.shape)
    angles = np.arctan2(y - center[0], x - center[1])
    angles = (angles + np.pi) * (180 / np.pi)

    bins = np.linspace(0, 360, n_bins + 1)
    angular_energy = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (angles >= bins[i]) & (angles < bins[i+1])
        angular_energy[i] = np.sum(log_mag[mask])

    return np.log(np.var(angular_energy))

def calculateMetrics(imagePath):
    image = plt.imread(imagePath)
    image = image[:, :, :3].mean(axis=2)  # Convert to grayscale
    fftImage=compute_fft(image)
    
    print(fft_line_energy(fftImage))
    print(fft_radial_energy_ratio(fftImage))
    print(fft_angular_variance(fftImage))
    print(fft_peak_features(fftImage))
    print(fft_entropy(fftImage))
    
#First val is real, second val is fake
#This will be the average
vertLineAvg=[0,0]
horzLineAvg=[0,0]
radialAvg=[0,0]
entropy=[0,0]
peakCount=[0,0]
peakRegularity=[0,0]
angularAvg=[0,0]

import os
realImagePath=r"C:\Users\Tristan\Downloads\spurhacks\myEnv\AI-Generated-vs-Real-Images-Datasets\RealArt\RealArt"
fakeImagePath=r"C:\Users\Tristan\Downloads\spurhacks\myEnv\AI-Generated-vs-Real-Images-Datasets\AiArtData"

length=0
for file in tqdm(os.listdir(realImagePath)):
    try:
        realFile=os.path.join(realImagePath,file)
    except:
        pass
    image = plt.imread(realFile)
    image = image[:, :, :3].mean(axis=2)  # Convert to grayscale
    fftImage=compute_fft(image)
    
    lineEnergy=fft_line_energy(fftImage)
    vertLineAvg[0]+=lineEnergy[0]
    horzLineAvg[0]+=lineEnergy[1]
    radialAvg[0]+=fft_radial_energy_ratio(fftImage)
    entropy[0]+=fft_entropy(fftImage)
    peaks=fft_peak_features(fftImage)
    peakCount[0]+=peaks[0]
    peakRegularity[0]+=peaks[1]
    angularAvg[0]+=fft_angular_variance(fftImage)
    length+=1
    
vertLineAvg[0]/=length
horzLineAvg[0]/=length
radialAvg[0]/=length
entropy[0]/=length
peakCount[0]/=length
peakRegularity[0]/=length
angularAvg[0]/=length


print(f"Vert-Line: {vertLineAvg}")
print(f"Horz-Line: {horzLineAvg}")
print(f"Radial: {radialAvg}")
print(f"Entropy: {entropy}")
print(f"Peak Count: {peakCount}")
print(f"Peak Reg: {peakRegularity}")
print(f"Angular: {angularAvg}")

length=0
for file in tqdm(os.listdir(fakeImagePath)):
    try:
        fakeFile=os.path.join(fakeImagePath,file)
    except:
        pass
    image = plt.imread(fakeFile)
    image = image[:, :, :3].mean(axis=2)  # Convert to grayscale
    fftImage=compute_fft(image)
    
    lineEnergy=fft_line_energy(fftImage)
    vertLineAvg[1]+=lineEnergy[0]
    horzLineAvg[1]+=lineEnergy[1]
    radialAvg[1]+=fft_radial_energy_ratio(fftImage)
    entropy[1]+=fft_entropy(fftImage)
    peaks=fft_peak_features(fftImage)
    peakCount[1]+=peaks[0]
    peakRegularity[1]+=peaks[1]
    angularAvg[1]+=fft_angular_variance(fftImage)
    length+=1
    
vertLineAvg[1]/=length
horzLineAvg[1]/=length
radialAvg[1]/=length
entropy[1]/=length
peakCount[1]/=length
peakRegularity[1]/=length
angularAvg[1]/=length

print(f"Vert-Line: {vertLineAvg}")
print(f"Horz-Line: {horzLineAvg}")
print(f"Radial: {radialAvg}")
print(f"Entropy: {entropy}")
print(f"Peak Count: {peakCount}")
print(f"Peak Reg: {peakRegularity}")
print(f"Angular: {angularAvg}")
    
# calculateMetrics("Earth.png")
# print("")
# calculateMetrics("testPeople.jpg")
    