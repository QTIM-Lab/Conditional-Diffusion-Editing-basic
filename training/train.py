# training/train.py
import torch
import argparse
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, AutoencoderKL
from models.condition_encoder import ConditionEncoder
from datasets.image_condition_dataset import ImageConditionDataset
from utils.scheduler import cosine_with_warmup
from utils.diffusion_utils import generate_evaluate
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import importlib

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Python module path to your train dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dynamically import dataset
    dataset_module = importlib.import_module(args.dataset)
    train_dataset = dataset_module.get_dataset()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
    encoder = ConditionEncoder(embedding_dim=768).to(device)

    train_diffusion_model(train_loader, model, vae, encoder, device, epochs=args.epochs)
