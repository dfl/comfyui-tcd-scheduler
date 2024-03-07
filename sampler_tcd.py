import comfy.samplers
from comfy.k_diffusion.sampling import default_noise_sampler
from tqdm.auto import trange, tqdm
from itertools import product
import torch
from torch import nn

@torch.no_grad()
def sample_tcd(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, eta=0.3):
    s_in = x.new_ones([x.shape[0]])
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    steps = len(sigmas)-1
    for i in trange(steps, disable=disable):  # Reverse diffusion steps
        # Adjusted Timestep Calculation (Strategic Stochastic Sampling):
        ii = int((1 - eta) * i)

        s0 = sigmas[i]
        s1 = sigmas[i + 1]

        # Generate denoised estimate from the model
        x = model(x, s0 * s_in, **extra_args)

        # Callback for progress monitoring/logging
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': s0, 'denoised': x})

        if eta > 0 and s1 > 0:
            # introduce stochasticity
            noise = noise_sampler(s0, s1)
            # sigma = std; beta = variance; alpha = 1.0 - beta = 1 - sigma**2
            a0 = 1.0 - s0**2
            a1 = 1.0 - s1**2
            a0s = 1.0 - sigmas[ii]
            # adjusted variance scaling
            v = a0*a1 / a0s # cumulative alpha product rescaled by the adjusted timestep
            deterministic = v.sqrt() * x
            stochastic = (1 - v).sqrt() * noise
            x = deterministic + stochastic

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
