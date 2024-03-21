# ComfyUI-TCD-scheduler

ComfyUI Custom Sampler nodes that implement Zheng et al.'s Trajectory Consistency Distillation based on https://mhh0318.github.io/tcd

This custom node repository adds `TCDScheduler`, `SamplerTCD Euler A`, and `SamplerTCD` nodes to the Custom Sampler category.

Just clone it into your custom_nodes folder and you can start using it as soon as you restart ComfyUI.

NOTE: `SamplerTCD` is a WIP and currently just operates as DDIM with no gamma parameter. Please use `SamplerTCD Euler A` for the time being.

`LCMScheduler` has one special parameter:

- `gamma`, a parameter used to control the stochasticity in every step.
  When gamma = 0, it represents deterministic sampling, whereas gamma = 1 indicates full stochastic sampling. In a way it acts as a sort of a crossfade between Karras and Euler.

  The default value is 0.3, but it is recommend using a higher value when increasing the number of inference steps.

Thanks to @laksjdjf for their help with converting from sigmas to timestep and SamplerTCDEulerA.

BTW If you're curious about learning more about samplers and schedulers, check out this article: https://www.felixsanz.dev/articles/complete-guide-to-samplers-in-stable-diffusion
