
import torch
import torchvision
import os

@torch.no_grad()
def sample_iadb(model, x0, nb_step, encoder_hidden_states):
    x_alpha = x0
    for t in range(nb_step):
        alpha_start = (t / nb_step)
        alpha_end = ((t + 1) / nb_step)
        d = model(x_alpha, torch.tensor(alpha_start, device=x0.device), encoder_hidden_states=encoder_hidden_states)['sample']
        x_alpha = x_alpha + (alpha_end - alpha_start) * d
    return x_alpha

@torch.no_grad()
def invert_diffusion(model, latent, nb_steps, encoder_hidden_states=None):
    x_alpha = latent.clone()
    for t in reversed(range(nb_steps)):
        alpha_start = (t / nb_steps)
        alpha_end = ((t + 1) / nb_steps)
        d = model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device), encoder_hidden_states=encoder_hidden_states)['sample']
        x_alpha = x_alpha - (alpha_end - alpha_start) * d
    return x_alpha

@torch.no_grad()
def forward_diffusion(model, noise_latent, nb_steps, encoder_hidden_states):
    x_alpha = noise_latent.clone()
    for t in range(nb_steps):
        alpha_start = (t / nb_steps)
        alpha_end = ((t + 1) / nb_steps)
        d = model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device), encoder_hidden_states=encoder_hidden_states)['sample']
        x_alpha = x_alpha + (alpha_end - alpha_start) * d
    return x_alpha

@torch.no_grad()
def generate_evaluate(model, vae, encoder, device, epoch, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"unet_epoch_{epoch+1}.pth"))
    torch.save(encoder.state_dict(), os.path.join(save_dir, f"encoder_epoch_{epoch+1}.pth"))

    model.eval()
    encoder.eval()
    for cond in [0.0, 1.0]:
        cond_tensor = torch.tensor([cond], device=device).float()
        cond_embed = encoder(cond_tensor)
        latent_noise = torch.randn((1, 4, 64, 64), device=device)
        sampled_latents = sample_iadb(model, latent_noise, nb_step=128, encoder_hidden_states=cond_embed)
        generated_image = vae.decode(sampled_latents / 0.18215).sample
        torchvision.utils.save_image(generated_image, os.path.join(save_dir, f"epoch_{epoch+1}_class_{int(cond)}.png"))
