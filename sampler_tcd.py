import comfy.samplers
from comfy.k_diffusion.sampling import default_noise_sampler, to_d, DDPMSampler_step
from tqdm.auto import trange, tqdm
from itertools import product
import torch
from torch import nn

@torch.no_grad()
def sample_tcd(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, gamma=0.3):
    """TCD sampling with a DDPMSampler_step-like process."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        sigma_from, sigma_to = sigmas[i], sigmas[i+1]

        # TCD offset, based on gamma, and conversion between sigma and timestep
        t = model.inner_model.inner_model.model_sampling.timestep(sigma_from)
        t_s = (1 - gamma) * t 
        sigma_down = model.inner_model.inner_model.model_sampling.sigma(t_s)

        if sigma_down > sigma_to:
            sigma_down = sigma_to
        # if sigma_down < 0:
        #     sigma_down = torch.tensor(1.0)

        # x = DDPMSampler_step(x / torch.sqrt(1.0 + sigma_from ** 2.0), sigma_from, sigma_to, (x - denoised) / sigma_from, noise_sampler)
        noise_est = (x - denoised) / sigma_from
        x /= torch.sqrt(1.0 + sigma_from ** 2.0)
        alpha_cumprod = 1 / ((sigma_from * sigma_from) + 1)
        alpha_cumprod_prev = 1 / ((sigma_to * sigma_to) + 1)
        alpha_cumprod_down = 1 / ((sigma_down * sigma_down) + 1)
        alpha = (alpha_cumprod / alpha_cumprod_prev)

        x = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise_est / (1 - alpha_cumprod).sqrt())
        if sigma_to > 0:
            noise = noise_sampler(sigma_from, sigma_to)
            x += ((1 - alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * noise
            x *= torch.sqrt(1.0 + sigma_to ** 2.0)

        # beta_cumprod_t = 1 - alpha_cumprod
        # beta_cumprod_s = 1 - alpha_cumprod_down

        # # variance = ((1 - alpha_cumprod_next) / (1 - alpha_cumprod)) * (1 - alpha_cumprod / alpha_cumprod_next)

        # alpha0 = (alpha_cumprod / alpha_cumprod_next)
        # # alpha0 = (alpha_cumprod / alpha_cumprod_down)

            # pred_noised_sample = alpha_prod_s.sqrt() * pred_original_sample + beta_prod_s.sqrt() * pred_epsilon

        # # epsilon noise prediction
        # x = (1.0 / alpha0).sqrt() * (x - (1 - alpha0) * denoised / (1 - alpha_cumprod).sqrt())

        # # x = (1.0 / alpha0).sqrt() * (x - (1 - alpha0) * denoised / (1 - alpha_cumprod_down).sqrt())

        # # x = (1.0 / alpha0).sqrt() * (x - (1-alpha_cumprod).sqrt() * denoised) + (1-alpha_cumprod_down).sqrt() * denoised


        # # DDPM sampler step, with added gamma conditional
        # first_step = sigma_to <= 0
        # last_step = i == len(sigmas) - 2
        # if gamma > 0 and not (first_step or last_step):
        #     noise = noise_sampler(sigma_from, sigma_to) # sigma_down?
        #     # x += ((1 - alpha0) * (1. - alpha_cumprod_next) / (1. - alpha_cumprod)).sqrt() * noise
        #     x += ((1 - alpha0) * (1. - alpha_cumprod_next) / (1. - alpha_cumprod)).sqrt() * noise

        # print(sigmas[i], sigmas[i+1], alpha,mu,x)

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
                     "gamma": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01, "round": False})
                    }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, gamma):
        sampler = comfy.samplers.KSAMPLER(sample_tcd, extra_options={"gamma": gamma})
        return (sampler, )

NODE_CLASS_MAPPINGS = {
    "TCDScheduler": TCDScheduler,
    "SamplerTCD": SamplerTCD,
}
