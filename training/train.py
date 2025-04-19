
# training/train.py
import torch
import torchvision
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, AutoencoderKL
from models.condition_encoder import ConditionEncoder
from datasets.image_condition_dataset import ImageConditionDataset
from utils.scheduler import cosine_with_warmup
from utils.diffusion_utils import sample_iadb, generate_evaluate
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os

def train_diffusion_model(train_loader, model, vae, encoder, device, epochs=50, nb_steps=128):
    optimizer = AdamW(list(model.parameters()) + list(encoder.parameters()), lr=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=cosine_with_warmup(500, epochs * len(train_loader)))

    for epoch in range(epochs):
        model.train()
        encoder.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            images = batch["image"].to(device)
            labels = batch["condition"].to(device)

            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            cond_embed = encoder(labels)
            latents = vae.encode(images).latent_dist.sample() * 0.18215
            x1 = latents
            x0 = torch.randn_like(x1)
            bs = x0.shape[0]
            alpha = torch.rand(bs, device=device)
            x_alpha = alpha.view(-1, 1, 1, 1) * x1 + (1 - alpha).view(-1, 1, 1, 1) * x0

            d = model(x_alpha, alpha, encoder_hidden_states=cond_embed)['sample']
            loss = torch.sum((d - (x1 - x0)) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        generate_evaluate(model, vae, encoder, device, epoch)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dummy_images = torch.randn(64, 1, 64, 64)
    dummy_conditions = torch.randint(0, 2, (64,)).float()
    dataset = ImageConditionDataset(dummy_images, dummy_conditions)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
    encoder = ConditionEncoder(embedding_dim=768).to(device)

    train_diffusion_model(train_loader, model, vae, encoder, device)
