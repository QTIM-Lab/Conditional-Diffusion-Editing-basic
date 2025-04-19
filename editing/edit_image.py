# editing/edit_image.py
import torch
import torchvision
import os
import argparse
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, AutoencoderKL
from models.condition_encoder import ConditionEncoder
from datasets.image_condition_dataset import ImageConditionDataset
from utils.diffusion_utils import invert_diffusion, forward_diffusion

def generate_edits(image_tensor, original_label, vae, model, encoder, device, nb_steps=128):
    edits = []
    image_tensor = image_tensor.to(device)

    # Step 1: Encode image to latent using VAE
    image_tensor = image_tensor.repeat(1, 3, 1, 1) if image_tensor.shape[1] == 1 else image_tensor
    with torch.no_grad():
        orig_latent = vae.encode(image_tensor).latent_dist.sample() * 0.18215

    # Step 2: Run reverse diffusion to get noise latent
    label = torch.tensor([original_label], dtype=torch.float, device=device)
    orig_cond = encoder(label)
    noise_latent = invert_diffusion(model, orig_latent, nb_steps, encoder_hidden_states=orig_cond)

    # Step 3: Re-run forward diffusion for different conditions
    for target_label in range(2):
        target_tensor = torch.tensor([target_label], dtype=torch.float, device=device)
        new_cond = encoder(target_tensor)

        if target_label == original_label:
            edited_latent = orig_latent
        else:
            edited_latent = forward_diffusion(model, noise_latent, nb_steps, encoder_hidden_states=new_cond)

        # Step 4: Decode edited latent using VAE
        edited_image = vae.decode(edited_latent / 0.18215).sample
        edits.append((target_label, edited_image))

    return edits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="path/to/test_images.pt")
    parser.add_argument("--label_path", type=str, default="path/to/test_labels.pt")
    parser.add_argument("--model_ckpt", type=str, default="outputs/unet_epoch_50.pth")
    parser.add_argument("--encoder_ckpt", type=str, default="outputs/encoder_epoch_50.pth")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pretrained model components
    model = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
    model.to(device)

    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)

    encoder = ConditionEncoder(embedding_dim=768).to(device)
    encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location=device))

    # Load test dataset
    images = torch.load(args.image_path)
    conditions = torch.load(args.label_path)
    test_dataset = ImageConditionDataset(images, conditions)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    os.makedirs("edited_outputs", exist_ok=True)

    # Iterate through test samples and generate edits
    for i, batch in enumerate(test_loader):
        imgs = batch["image"]
        labels = batch["condition"]

        for j in range(imgs.size(0)):
            img = imgs[j].unsqueeze(0)
            label = labels[j].item()

            edits = generate_edits(img, label, vae, model, encoder, device)
            # Sort edits by label to keep original first
            edits = sorted(edits, key=lambda x: x[0])

            # Concatenate original and edited images side by side
            concat_img = torch.cat([e[1] for e in edits], dim=-1)
            torchvision.utils.save_image(
                concat_img,
                f"edited_outputs/sample_{i * args.batch_size + j}_comparison.png"
            )
        break # test only one batch 