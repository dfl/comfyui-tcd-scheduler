# ComfyUI-tcd-scheduler

ComfyUI Custom Sampler nodes that adds TCD with eta

This custom node repository adds LCMScheduler to the Custom Sampler category.
Just clone it into your custom_nodes folder and you can start using it as soon as you restart ComfyUI.

LCMScheduler has one special parameter:

- `eta`, a stochastic parameter (referred to as `gamma` in the paper) used to control the stochasticity in every step.
  When eta = 0, it represents deterministic sampling, whereas eta = 1 indicates full stochastic sampling.

  The default value is 0.3, but it is recommend using a higher eta when increasing the number of inference steps.