import os
import cv2
import torch
import random
import tempfile
import numpy as np
from PIL import Image
from statistics import mean, median
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from runModel import runModel
# ---------------------------
# User-provided image classifier runner.
# ---------------------------


# ---------------------------
# VideoDataset: samples frames_per_clip frames uniformly (or pads if shorter),
# calls runModel per frame, returns:
#   - frames tensor [T, 3, H, W]
#   - confidences tensor [T]
#   - label tensor scalar
#   - sampled_indices: numpy array of original frame indices (length T)
# ---------------------------
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, frames_per_clip=16, transform=None):
        """
        video_paths: list of strings, paths to video files.
        labels: list of floats (e.g. 1.0 for real, 0.0 for AI).
        frames_per_clip: number of frames to sample per video.
        transform: torchvision transform applied to each frame (PIL Image -> Tensor normalized).
        """
        assert len(video_paths) == len(labels), "Paths and labels must align"
        self.video_paths = video_paths
        self.labels = labels
        self.frames_per_clip = frames_per_clip
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = float(self.labels[idx])

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path!r}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            total_frames = 1  # fallback

        # Determine which frame indices to sample
        if total_frames >= self.frames_per_clip:
            # uniform sampling
            indices = np.linspace(0, total_frames - 1, num=self.frames_per_clip, dtype=int)
        else:
            # take all frames then pad by repeating last
            indices = list(range(total_frames))
            while len(indices) < self.frames_per_clip:
                indices.append(total_frames - 1)
            indices = np.array(indices, dtype=int)

        frames = []
        confidences = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                # fallback to black image
                frame_rgb = None
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Get confidence via runModel
            if frame_rgb is not None:
                # Write to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpf:
                    temp_path = tmpf.name
                # Write BGR directly
                cv2.imwrite(temp_path, frame_bgr)
                try:
                    conf = runModel(temp_path)
                    conf = float(conf)
                except Exception as e:
                    print(f"Warning: runModel failed on {temp_path}: {e}")
                    conf = 0.0
                # Remove temp file
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            else:
                conf = 0.0
            confidences.append(conf)

            # Prepare PIL Image for transform
            if frame_rgb is not None:
                pil = Image.fromarray(frame_rgb)
            else:
                # Black image of default size 224x224 (transform should resize anyway)
                pil = Image.new("RGB", (224, 224), (0, 0, 0))
            if self.transform is not None:
                frame_t = self.transform(pil)  # [3,H,W]
            else:
                frame_t = T.ToTensor()(pil)
            frames.append(frame_t)

        cap.release()

        # Stack frames: [T,3,H,W]
        frames_tensor = torch.stack(frames, dim=0)
        confidences_tensor = torch.tensor(confidences, dtype=torch.float32)  # [T]
        label_tensor = torch.tensor(label, dtype=torch.float32)

        sampled_indices = indices  # numpy array of length T

        # Return a dict so DataLoader collate will handle tensors and indices list properly.
        return {
            "frames": frames_tensor,
            "confidences": confidences_tensor,
            "label": label_tensor,
            "indices": sampled_indices,
            "video_path": video_path  # so we know which video this came from
        }

# ---------------------------
# Custom Temporal Transformer Layer that returns attention weights
# ---------------------------
class TemporalEncoderLayerWithAttn(nn.Module):
    def __init__(self, hidden_dim, nhead, dim_feedforward=None, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4
        # MultiheadAttention: input shape (seq_len, batch, embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, dropout=dropout, batch_first=False)
        # Feedforward
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        # Norms and dropouts
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        """
        src: [seq_len, batch, hidden_dim]
        Returns:
          src_out: [seq_len, batch, hidden_dim]
          attn_weights: [batch, seq_len, seq_len] (averaged over heads)
        """
        # Self-attention
        attn_output, attn_output_weights = self.self_attn(src, src, src, need_weights=True)
        # attn_output_weights: [batch_size * nhead, tgt_len, src_len]
        batch_size = src.shape[1]
        tgt_len, src_len = attn_output_weights.shape[-2], attn_output_weights.shape[-1]
        # reshape to [batch_size, nhead, tgt_len, src_len]
        attn_weights = attn_output_weights.view(batch_size, self.nhead, tgt_len, src_len)
        # average over heads
        attn_weights = attn_weights.mean(dim=1)  # [batch, seq_len, seq_len]

        # Residual + Norm
        src2 = attn_output
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feedforward
        ff = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)
        return src, attn_weights

class TemporalTransformerWithAttn(nn.Module):
    def __init__(self, num_layers, hidden_dim, nhead, dim_feedforward=None, dropout=0.1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(TemporalEncoderLayerWithAttn(hidden_dim, nhead, dim_feedforward, dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, src):
        """
        src: [seq_len, batch, hidden_dim]
        Returns:
          out: [seq_len, batch, hidden_dim]
          all_attn_weights: list of length num_layers, each [batch, seq_len, seq_len]
        """
        all_attn = []
        x = src
        for layer in self.layers:
            x, attn_w = layer(x)
            all_attn.append(attn_w)
        return x, all_attn

# ---------------------------
# Video model with CLS token, temporal attention, and exposure of attention weights
# ---------------------------
class VideoTransformerWithFrameAttention(nn.Module):
    def __init__(self, frames_per_clip=16, hidden_dim=768, nhead=8, num_layers=2, dropout=0.1):
        """
        frames_per_clip: number of sampled frames per video
        hidden_dim: embedding dimension from ViT backbone (e.g., 768 for vit_b_16)
        nhead: number of heads in temporal attention
        num_layers: number of temporal transformer layers
        """
        super().__init__()
        self.frames_per_clip = frames_per_clip
        self.hidden_dim = hidden_dim

        # 1) Pretrained ViT backbone (spatial attention inside)
        vit = torchvision.models.vit_b_16(pretrained=True)
        # Remove classification head; keep backbone up to embedding
        vit.heads = nn.Identity()
        self.frame_encoder = vit  # takes [B*T,3,224,224] → [B*T, hidden_dim]

        # 2) Confidence projection: map scalar → hidden_dim, to fuse with frame feature
        self.conf_proj = nn.Linear(1, hidden_dim)

        # 3) CLS token and positional embeddings for temporal sequence
        # CLS token: one learnable vector
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        # Positional embedding: length = frames_per_clip + 1
        self.pos_embed = nn.Parameter(torch.randn(frames_per_clip + 1, hidden_dim))

        # 4) Temporal transformer with attention capture
        self.temporal_transformer = TemporalTransformerWithAttn(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )

        # 5) Classification head from CLS output
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, frames, confidences, return_attn=False):
        """
        frames: tensor [B, T, 3, H, W], e.g. H=W=224 after transform
        confidences: tensor [B, T]
        return_attn: if True, returns (logits, attn_data), else returns logits
        """
        B, T, C, H, W = frames.shape
        assert T == self.frames_per_clip, f"Expected {self.frames_per_clip} frames, got {T}"
        device = frames.device

        # 1) Frame encoding: flatten batch/time
        frames_flat = frames.view(B * T, C, H, W)  # [B*T, 3, H, W]
        feats_flat = self.frame_encoder(frames_flat)  # [B*T, hidden_dim]
        feats = feats_flat.view(B, T, self.hidden_dim)  # [B, T, hidden_dim]

        # 2) Confidence projection and fusion
        conf = confidences.view(B, T, 1)  # [B, T, 1]
        conf_proj = self.conf_proj(conf)  # [B, T, hidden_dim]
        fused = feats + conf_proj  # simple addition fusion

        # 3) Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
        seq = torch.cat([cls_tokens, fused], dim=1)  # [B, T+1, hidden_dim]

        # 4) Add positional embedding
        seq = seq + self.pos_embed.unsqueeze(0)  # [B, T+1, hidden_dim]

        # 5) Permute for transformer: [seq_len, B, hidden_dim]
        seq = seq.permute(1, 0, 2)  # [T+1, B, hidden_dim]

        # 6) Temporal transformer
        out, all_attn_weights = self.temporal_transformer(seq)  # out: [T+1, B, hidden_dim]; attn: list of [B, T+1, T+1]

        # 7) Classification from CLS token
        out = out.permute(1, 0, 2)  # [B, T+1, hidden_dim]
        cls_out = out[:, 0, :]      # [B, hidden_dim]
        logits = self.classifier(cls_out).squeeze(-1)  # [B]

        if return_attn:
            # Return raw logits and attention data
            return logits, {"temporal_attn_weights": all_attn_weights}
        else:
            return logits

# ---------------------------
# Helper: compute attention rollout for one sample
# ---------------------------
def compute_attention_rollout(attn_weights_list):
    """
    attn_weights_list: list of tensors [seq_len, seq_len] (for one sample),
      where seq_len = T+1 including CLS.
    Returns:
      rollout: [seq_len, seq_len] final attention adjacency after rollout.
      In practice, to get frame importance, look at rollout[0, 1:].
    """
    device = attn_weights_list[0].device
    seq_len = attn_weights_list[0].shape[-1]
    result = torch.eye(seq_len, seq_len, device=device)
    for A in attn_weights_list:
        # A: [seq_len, seq_len]
        # Add identity for residual
        A_res = A + torch.eye(seq_len, seq_len, device=device)
        # Normalize rows
        A_res = A_res / A_res.sum(dim=-1, keepdim=True)
        # Multiply: note we do A_res @ result so that earlier layers are multiplied last
        result = A_res @ result
    return result  # [seq_len, seq_len]

# ---------------------------
# Training and evaluation loops
# ---------------------------
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        frames = batch["frames"].to(device)           # [B, T, 3, H, W]
        confidences = batch["confidences"].to(device) # [B, T]
        labels = batch["label"].to(device)            # [B]
        optimizer.zero_grad()
        logits = model(frames, confidences)           # [B]
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        b = frames.size(0)
        total_loss += loss.item() * b
        total_samples += b
    return total_loss / total_samples

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs = []
    all_labels = []
    for batch in dataloader:
        frames = batch["frames"].to(device)
        confidences = batch["confidences"].to(device)
        labels = batch["label"].to(device)
        logits = model(frames, confidences)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        probs = torch.sigmoid(logits)
        b = frames.size(0)
        total_loss += loss.item() * b
        total_samples += b
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
    avg_loss = total_loss / total_samples
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    preds = (all_probs >= 0.5).float()
    accuracy = (preds == all_labels).float().mean().item()
    return avg_loss, accuracy

def train_model(model, train_dataset, val_dataset,
                device, batch_size=1, lr=1e-4, num_epochs=5, num_workers=0):
    """
    Train the model, saving the best by validation loss.
    - batch_size: likely small (1 or 2) because each sample loads multiple frames.
    - num_workers: 0 for simplicity; increase if using __main__ guard and want parallel loading.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_video_detector.pth")
            print("  Saved best model.")
    print("Training complete.")

# ---------------------------
# Analysis: run inference on one (or batch of) videos, extract attention, report top frames
# ---------------------------
@torch.no_grad()
def analyze_video_sample(model, sample_dict, device, top_k=5):
    """
    Given a single-sample batch (batch_size=1) from VideoDataset (i.e., one video),
    runs forward(return_attn=True), computes both last-layer CLS attention and rollout,
    and prints top_k important frames (by original frame index).
    sample_dict: dict with keys "frames", "confidences", "label", "indices", "video_path"
    Returns a dict with probabilities and attention scores.
    """
    model.eval()
    frames = sample_dict["frames"].unsqueeze(0).to(device)         # [1, T, 3, H, W]
    confidences = sample_dict["confidences"].unsqueeze(0).to(device)  # [1, T]
    indices = sample_dict["indices"]  # numpy array length T
    video_path = sample_dict.get("video_path", None)
    label = sample_dict["label"].item()

    logits, attn_data = model(frames, confidences, return_attn=True)
    prob = torch.sigmoid(logits)[0].item()
    print(f"Video: {video_path} | True label: {label} | Predicted real-prob: {prob:.4f}")

    # Extract temporal attention weights
    # attn_data["temporal_attn_weights"]: list of length L, each [B=1, seq_len, seq_len]
    all_attn = attn_data["temporal_attn_weights"]
    seq_len = all_attn[0].shape[-1]  # = T+1
    T = seq_len - 1

    # Last-layer CLS attention (direct)
    last_attn = all_attn[-1][0]  # [seq_len, seq_len]
    cls_attn = last_attn[0, 1:]  # [T]
    if cls_attn.sum() > 0:
        cls_attn_norm = cls_attn / cls_attn.sum()
    else:
        cls_attn_norm = cls_attn

    # Rollout attention
    # Build list of [seq_len, seq_len] for this sample
    attn_list = [layer_attn[0] for layer_attn in all_attn]
    rollout = compute_attention_rollout(attn_list)  # [seq_len, seq_len]
    rollout_scores = rollout[0, 1:]  # [T]
    if rollout_scores.sum() > 0:
        rollout_norm = rollout_scores / rollout_scores.sum()
    else:
        rollout_norm = rollout_scores

    # Report top_k frames by rollout_norm
    k = min(top_k, T)
    topk = torch.topk(rollout_norm, k=k)
    print("Top frames by attention rollout:")
    for rank, idx in enumerate(topk.indices):
        frame_num = int(indices[idx])  # original frame index
        score = rollout_norm[idx].item()
        print(f"  {rank+1}: frame index {frame_num} (score {score:.4f})")
    # Also report direct last-layer attention
    topk2 = torch.topk(cls_attn_norm, k=k)
    print("Top frames by last-layer CLS attention:")
    for rank, idx in enumerate(topk2.indices):
        frame_num = int(indices[idx])
        score = cls_attn_norm[idx].item()
        print(f"  {rank+1}: frame index {frame_num} (score {score:.4f})")

    # Return data for further processing if desired
    return {
        "probability": prob,
        "cls_attn": cls_attn_norm.cpu(),
        "rollout_attn": rollout_norm.cpu(),
        "sampled_indices": indices
    }

# ---------------------------
# Utility: analyze dataset frame counts distribution (optional)
# ---------------------------
def analyze_frame_counts(video_paths):
    counts = []
    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            continue
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if cnt > 0:
            counts.append(cnt)
    if not counts:
        print("No valid videos or frames found.")
        return
    print(f"Total videos: {len(counts)}")
    print(f"Min frames: {min(counts)}, Max frames: {max(counts)}")
    print(f"Mean frames: {mean(counts):.1f}, Median frames: {median(counts)}")
    # For quick histogram, you could use matplotlib, but omitted here.

# ---------------------------
# Main: prepare datasets, model, train, and optionally analyze some videos
# ---------------------------
def main():
    # 1) Prepare your lists of video paths and labels.
    # Example: read from a CSV or define manually.
    # For demonstration, placeholders:
    
    videoPaths=[]
    for file in os.listdir(r"C:\Users\Tristan\Downloads\spurhacks\myEnv\realVideos"):
        fullPath=os.path.join(r"C:\Users\Tristan\Downloads\spurhacks\myEnv\realVideos",file)
        videoPaths.append((fullPath,1.0))
    for file in os.listdir(r"C:\Users\Tristan\Downloads\spurhacks\myEnv\aiVideos"):
        fullPath=os.path.join(r"C:\Users\Tristan\Downloads\spurhacks\myEnv\aiVideos",file)
        videoPaths.append((fullPath,0.0))
    
    paths, labels=zip(*videoPaths)  
    paths=list(paths)
    labels=list(labels)  
    
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels,
        test_size=0.15,
        stratify=labels,
        random_state=42
    )
    
    val_frac = 0.15 / 0.85
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=val_frac,
        stratify=train_labels,
        random_state=42
    )
    
    if not train_paths or not val_paths:
        print("Please fill in video_paths_train, labels_train, video_paths_val, labels_val in main().")
        return

    # 2) Hyperparameters
    frames_per_clip = 16  # try 8, 16, 32 based on dataset and resources
    batch_size = 1       # small, because each sample loads T frames
    num_epochs = 5       # adjust
    lr = 1e-4
    num_workers = 0      # set >0 if you want parallel loading and have guarded entrypoint

    # 3) Transforms for ViT backbone
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # 4) Create datasets
    train_dataset = VideoDataset(train_paths, train_labels,
                                 frames_per_clip=frames_per_clip,
                                 transform=transform)
    val_dataset = VideoDataset(val_paths, val_labels,
                               frames_per_clip=frames_per_clip,
                               transform=transform)

    # 5) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 6) Instantiate model
    model = VideoTransformerWithFrameAttention(
        frames_per_clip=frames_per_clip,
        hidden_dim=768,
        nhead=8,
        num_layers=2,
        dropout=0.1
    )

    # 7) Train
    train_model(model, train_dataset, val_dataset,
                device, batch_size=batch_size,
                lr=lr, num_epochs=num_epochs,
                num_workers=num_workers)

    # 8) After training, you can load best model for analysis:
    model.load_state_dict(torch.load("best_video_detector.pth", map_location=device))
    model.to(device)
    model.eval()

    # 9) Example: analyze a few validation videos for attention
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print("\n--- Analyzing some validation videos for frame importance ---")
    # Iterate a few samples
    for i, sample in enumerate(val_loader):
        print(f"\nSample {i+1}:")
        analyze_video_sample(model, sample, device, top_k=5)
        if i >= 4:
            break  # analyze first 5 videos only

if __name__ == "__main__":
    # For Windows multiprocessing support if num_workers>0
    from multiprocessing import freeze_support
    freeze_support()
    main()
