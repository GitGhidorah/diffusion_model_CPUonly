import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import math
from tqdm import tqdm

# ====================== 1. Infinite Data Generation Dataset ======================
class InfiniteSquareDataset(Dataset):
    def __init__(self, img_size=64, epoch_size=3000):
        """
        epoch_size: A virtual size to define the number of iterations per epoch.
        """
        self.img_size = img_size
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # Generate a random image on the fly (Infinite variations)
        img = Image.new('RGB', (self.img_size, self.img_size), color='black')
        draw = ImageDraw.Draw(img)

        size = 20
        center_x = random.randint(size//2 + 5, self.img_size - size//2 - 5)
        center_y = random.randint(size//2 + 5, self.img_size - size//2 - 5)
        angle = random.uniform(-15, 15)
        color = (255, 255, 255)

        # Calculate square vertices (with rotation)
        half = size / 2
        points = [
            (center_x - half, center_y - half),
            (center_x + half, center_y - half),
            (center_x + half, center_y + half),
            (center_x - half, center_y + half)
        ]

        # Rotation matrix
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rotated_points = []
        for px, py in points:
            dx = px - center_x
            dy = py - center_y
            new_x = center_x + dx * cos_a - dy * sin_a
            new_y = center_y + dx * sin_a + dy * cos_a
            rotated_points.append((new_x, new_y))

        draw.polygon(rotated_points, fill=color)

        # Normalize to [-1, 1]
        img_np = (np.array(img).astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        return tensor

# ====================== 2. Diffusion Utilities ======================
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

class DDPM:
    def __init__(self, timesteps=1000, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.timesteps = timesteps
        self.device = device
        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod)

    def add_noise(self, x0, t):
        """Forward process: Add noise to the original image"""
        noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    @torch.no_grad()
    def sample(self, model, batch_size=16, img_size=64):
        """Reverse process: Generate image from pure noise"""
        x = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
        model.eval()
        for t in tqdm(reversed(range(self.timesteps)), desc="Sampling", leave=False):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            predicted_noise = model(x, t_tensor)
            
            alpha = self.alphas[t]
            alpha_cumprod = self.alpha_cumprod[t]
            beta = self.betas[t]
            
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            
            # DDPM Step
            x = (1 / torch.sqrt(alpha)) * (x - ((beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)) + torch.sqrt(beta) * noise
        
        model.train()
        # Restore range from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0
        return torch.clamp(x, 0, 1)

# ====================== 3. Simple U-Net Architecture ======================
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class SimpleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_ch),
            nn.SiLU(),
            nn.Linear(out_ch, out_ch)
        )
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.relu(self.norm1(self.conv1(x)))
        t_emb = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_emb
        h = self.relu(self.norm2(self.conv2(h)))
        return h

class UNet(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        # Encoder
        self.down1 = SimpleBlock(3, 64, time_dim)
        self.down2 = SimpleBlock(64, 128, time_dim)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.mid = SimpleBlock(128, 128, time_dim)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = SimpleBlock(128 + 64, 64, time_dim)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = SimpleBlock(64 + 64, 64, time_dim)
        
        # Output (Predicting Noise)
        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        # Downsampling
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        
        # Mid
        m = self.mid(self.pool(d2), t_emb)
        
        # Upsampling
        u2 = self.up2(m)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.dec2(u2, t_emb)
        
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.dec1(u1, t_emb)
        
        return self.final(u1)

# ====================== 4. Training and Generation ======================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use Infinite Dataset
    dataset = InfiniteSquareDataset(img_size=64, epoch_size=3000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = UNet().to(device)
    ddpm = DDPM(timesteps=1000, device=device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    epochs = 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Sampling t and adding noise
            t = torch.randint(0, ddpm.timesteps, (batch.shape[0],), device=device)
            noisy_batch, noise = ddpm.add_noise(batch, t)
            
            # Predict noise and backprop
            predicted_noise = model(noisy_batch, t)
            loss = nn.MSELoss()(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Generate preview every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            samples = ddpm.sample(model, batch_size=4)
            samples = samples.cpu().permute(0, 2, 3, 1).numpy()
            fig, axs = plt.subplots(1, 4, figsize=(12, 3))
            for i in range(4):
                axs[i].imshow(samples[i])
                axs[i].axis('off')
            plt.savefig(f'samples_epoch_{epoch+1}.png')
            plt.close()

    # Save final model
    torch.save(model.state_dict(), 'square_diffusion_model.pth')
    print("Training Finished. Model saved as 'square_diffusion_model.pth'")

@torch.no_grad()
def run_generation(num_samples=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    try:
        model.load_state_dict(torch.load('square_diffusion_model.pth', map_location=device))
    except FileNotFoundError:
        print("Model file 'square_diffusion_model.pth' not found. Please train the model first.")
        return
        
    model.eval()
    ddpm = DDPM(timesteps=1000, device=device)
    
    print(f"Generating {num_samples} samples...")
    samples = ddpm.sample(model, batch_size=num_samples)
    samples = samples.cpu().permute(0, 2, 3, 1).numpy()
    
    # Visualization
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axs.flat):
        if i < num_samples:
            ax.imshow(samples[i])
            ax.set_title(f"Sample {i+1}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # To start training, leave the line below active.
    ##train()
    
    # To only generate samples from a saved model, 
    # comment out 'train()' and uncomment 'run_generation(8)' below.
    run_generation(8)