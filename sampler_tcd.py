import comfy.samplers
from comfy.k_diffusion.sampling import default_noise_sampler, to_d
from tqdm.auto import trange, tqdm
from itertools import product
import torch
from torch import nn

@torch.no_grad()
def sample_tcd(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, eta=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        d = to_d(x, sigmas[i], denoised)

        sigma_from = sigmas[i]
        sigma_to = sigmas[i + 1]

        t = model.inner_model.inner_model.model_sampling.timestep(sigma_from)
        # down_t = torch.floor( (1 - eta) * t )
        down_t = (1 - eta) * t
        sigma_down = model.inner_model.inner_model.model_sampling.sigma(down_t)

        if sigma_down > sigma_to:
            sigma_down = sigma_to

        sigma_up = (sigma_to ** 2 - sigma_down ** 2) ** 0.5
        
        # same as euler ancestral
        d = to_d(x, sigma_from, denoised)
        dt = sigma_down - sigma_from
        x = x + d * dt
        if sigma_to > 0:
            x = x + noise_sampler(sigma_from, sigma_to) * sigma_up
    return x

class TCDScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps):
        sigmas = comfy.samplers.calculate_sigmas_scheduler(model.model, "ddim_uniform", steps).cpu()
        return (sigmas, )

class SamplerTCD:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "eta": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01, "round": False})
                    }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta):
        sampler = comfy.samplers.KSAMPLER(sample_tcd, extra_options={"eta": eta})
        return (sampler, )

NODE_CLASS_MAPPINGS = {
    "TCDScheduler": TCDScheduler,
    "SamplerTCD": SamplerTCD,
}
