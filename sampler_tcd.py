import comfy.samplers
from comfy.k_diffusion.sampling import default_noise_sampler
from tqdm.auto import trange, tqdm
from itertools import product
import torch
from torch import nn

class TCDSampler_step(torch.nn.Module):
    def __init__(self, eta=0.0):
        super(TCDSampler_step, self).__init__()
        self.eta = eta
    
    def forward(self, x, sigma, sigma_next, denoised, noise_sampler):
        # Generate additional noise based on sigma and sigma_next
        if self.eta > 0 and sigma_next > 0:
            additional_noise = noise_sampler(sigma, sigma_next)
            # Apply eta adjustment to additional noise
            x = denoised + self.eta * additional_noise
        else:
            # If eta=0 or sigma_next<=0, use the denoised estimation directly
            x = denoised
        return x

@torch.no_grad()
def sample_tcd(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, eta=0.3):
    step_function = TCDSampler_step(eta=eta)
    s_in = x.new_ones([x.shape[0]])
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    for i in trange(len(sigmas) - 1, disable=disable):
        # Forward model to generate denoising estimation
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        # Call step_function with all necessary parameters including noise_sampler
        x = step_function(x, sigmas[i], sigmas[i + 1], denoised, noise_sampler)

        # Invoke the callback with current progress data after denoised estimation
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'denoised': denoised})

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
