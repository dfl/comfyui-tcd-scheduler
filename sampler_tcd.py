import comfy.samplers
from comfy.k_diffusion.sampling import default_noise_sampler, to_d, DDPMSampler_step
from tqdm.auto import trange, tqdm
from itertools import product
import torch
from torch import nn

@torch.no_grad()
def sample_tcd_euler_a(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, gamma=0.3):
    # TCD sampling using modified Euler Ancestral sampler. by @laksjdjf
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
        down_t = (1 - gamma) * t
        sigma_down = model.inner_model.inner_model.model_sampling.sigma(down_t)

        if sigma_down > sigma_to:
            sigma_down = sigma_to

        sigma_up = (sigma_to ** 2 - sigma_down ** 2) ** 0.5
        
        # same as euler ancestral
        d = to_d(x, sigma_from, denoised)
        dt = sigma_down - sigma_from
        x += d * dt

        if sigma_to > 0 and gamma > 0:
            x = model.inner_model.inner_model.model_sampling.noise_scaling(sigma_up, noise_sampler(sigma_from, sigma_to), x)
    return x


@torch.no_grad()
def sample_tcd(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, gamma=0.3):
    # TCD sampling using modified DDPM.
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
        sigma_to_s = model.inner_model.inner_model.model_sampling.sigma(t_s)

        # if sigma_to_s > sigma_to:
        #     sigma_to_s = sigma_to
        # if sigma_to_s < 0:
        #     sigma_to_s = torch.tensor(1.0)
        print(f"sigma_from: {sigma_from}, sigma_to: {sigma_to}, sigma_to_s: {sigma_to_s}")


        # The following is equivalent to the comfy DDPM implementation
        # x = DDPMSampler_step(x / torch.sqrt(1.0 + sigma_from ** 2.0), sigma_from, sigma_to, (x - denoised) / sigma_from, noise_sampler)

        noise_est = (x - denoised) / sigma_from
        x /= torch.sqrt(1.0 + sigma_from ** 2.0)

        alpha_cumprod = 1 / ((sigma_from * sigma_from) + 1)  # _t
        alpha_cumprod_prev = 1 / ((sigma_to * sigma_to) + 1) # _t_prev
        alpha = (alpha_cumprod / alpha_cumprod_prev)

        ## These values should approach 1.0?
        # print(f"alpha_cumprod: {alpha_cumprod}")
        # print(f"alpha_cumprod_prev: {alpha_cumprod_prev}")
        # print(f"alpha: {alpha}")


        # alpha_cumprod_down = 1 / ((sigma_to_s * sigma_to_s) + 1)  # _s
        # alpha_d = (alpha_cumprod_prev / alpha_cumprod_down)
        # alpha2 = (alpha_cumprod / alpha_cumprod_down)
        # print(f"** alpha_cumprod_down: {alpha_cumprod_down}")
        # print(f"** alpha_d: {alpha_d}, alpha2: #{alpha2}")

        # epsilon noise prediction from comfy DDPM implementation
        x = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise_est / (1 - alpha_cumprod).sqrt())
        # x = (1.0 / alpha_d).sqrt() * (x - (1 - alpha) * noise_est / (1 - alpha_cumprod).sqrt())

        first_step = sigma_to == 0
        last_step = i == len(sigmas) - 2

        if not first_step:
            if gamma > 0 and not last_step:
                noise = noise_sampler(sigma_from, sigma_to)

                # x += ((1 - alpha_d) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * noise
                variance = ((1 - alpha_cumprod_prev) / (1 - alpha_cumprod)) * (1 - alpha_cumprod / alpha_cumprod_prev)
                x += variance.sqrt() * noise # scale noise by std deviation

                # relevant diffusers code from scheduling_tcd.py
                # prev_sample = (alpha_prod_t_prev / alpha_prod_s).sqrt() * pred_noised_sample + (
                #     1 - alpha_prod_t_prev / alpha_prod_s
                # ).sqrt() * noise

            x *= torch.sqrt(1.0 + sigma_to ** 2.0)

        # beta_cumprod_t = 1 - alpha_cumprod
        # beta_cumprod_s = 1 - alpha_cumprod_down

 
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

class SamplerTCDEulerA:
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
        sampler = comfy.samplers.KSAMPLER(sample_tcd_euler_a, extra_options={"gamma": gamma})
        return (sampler, )

NODE_CLASS_MAPPINGS = {
    "TCDScheduler": TCDScheduler,
    "SamplerTCD": SamplerTCD,
    "SamplerTCD EulerA": SamplerTCDEulerA,
}
