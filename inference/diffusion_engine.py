import torch


@torch.no_grad()
def invert_diffusion(model, latent, nb_steps, encoder_hidden_states=None, class_labels=None):

    x = latent.clone()

    for t in reversed(range(nb_steps)):

        alpha_start = t / nb_steps
        alpha_end = (t + 1) / nb_steps

        d = model(
            x,
            torch.tensor(alpha_start, device=x.device),
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels
        )["sample"]

        x = x - (alpha_end - alpha_start) * d

    return x


@torch.no_grad()
def forward_diffusion(model, latent, nb_steps, encoder_hidden_states=None, class_labels=None):

    x = latent.clone()

    for t in range(nb_steps):

        alpha_start = t / nb_steps
        alpha_end = (t + 1) / nb_steps

        d = model(
            x,
            torch.tensor(alpha_start, device=x.device),
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels
        )["sample"]

        x = x + (alpha_end - alpha_start) * d

    return x
