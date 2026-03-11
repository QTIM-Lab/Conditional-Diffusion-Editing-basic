import argparse
import torch

from diffusers import UNet2DConditionModel, AutoencoderKL

from inference.diffusion_engine import invert_diffusion, forward_diffusion
from inference.visualization import show_counterfactuals


device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_counterfactuals(image, label, encoder, model, prepare_latent, decode_latent, num_classes):

    latent = prepare_latent(image)

    cond_orig = encoder(torch.tensor([label]).to(device))

    z0 = invert_diffusion(model, latent, 128, encoder_hidden_states=cond_orig)

    images = []
    labels = []

    for target in range(num_classes):

        cond = encoder(torch.tensor([target]).to(device))

        edited_latent = forward_diffusion(
            model,
            z0,
            128,
            encoder_hidden_states=cond
        )

        img = decode_latent(edited_latent)

        images.append(img)
        labels.append(str(target))

    return images, labels


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--image", type=str)

    args = parser.parse_args()

    model = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet"
    ).to(device)

    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae"
    ).to(device)

    if args.dataset == "oct":

        from dataset_adapters.oct_adapter import load_sample

        image, label, encoder, prepare_latent, decode_latent, num_classes = load_sample(args.image)

    elif args.dataset == "retinamnist":

        from dataset_adapters.retinamnist_adapter import load_sample

        image, label, encoder, prepare_latent, decode_latent, num_classes = load_sample()

    elif args.dataset == "spider":

        from dataset_adapters.spider_adapter import load_sample

        image, label, encoder, prepare_latent, decode_latent, num_classes = load_sample()

    elif args.dataset == "brats":

        from dataset_adapters.brats_adapter import load_sample

        image, label, encoder, prepare_latent, decode_latent, num_classes = load_sample()

    images, labels = generate_counterfactuals(
        image,
        label,
        encoder,
        model,
        prepare_latent,
        decode_latent,
        num_classes
    )

    show_counterfactuals(images, labels)


if __name__ == "__main__":

    main()
