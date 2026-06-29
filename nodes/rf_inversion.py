import warnings

import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.sample
import comfy.model_management
from tqdm.auto import trange


# ─────────────────────────────────────────────────────────────────────────────
# Sampler functions via closure — no more extra_args
# ─────────────────────────────────────────────────────────────────────────────

def make_rf_forward_fn(x_source, eta, eta_trend, start_step, end_step, steps):
    """
    Returns a sampler function with params captured by closure.
    x_source, eta, etc. are frozen at the node call time.

    Fade-in: smoothstep quadratic ramp on etas during [start_step, start_step+warmup)
    to avoid a shock when the RF term first kicks in.
    """
    if eta_trend == "constant":
        etas = [eta] * steps
    elif eta_trend == "linear_increase":
        etas = [eta * i / max(steps - 1, 1) for i in range(steps)]
    else:
        etas = [eta * (1 - i / max(steps - 1, 1)) for i in range(steps)]

    # Warmup fade-in: ramp from 0→1 over the first ~30% of the injection window
    warmup = max(int((end_step - start_step) * 0.3), 1) if end_step > start_step else 0
    fade_in = torch.ones(steps)
    if warmup > 0 and end_step > start_step:
        ramp = torch.linspace(0, 1, min(warmup, steps))
        fade_in[:warmup] = ramp * ramp  # smoothstep quadratic ease-in

    def rf_forward_fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = extra_args or {}

        for i in trange(len(sigmas) - 1, disable=disable):
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]

            # ComfyUI-wrapped model call — safe for LTX 2.3
            denoised = model(x, sigma_curr * torch.ones(x.shape[0], device=x.device),
                             **extra_args)

            # RF: denoised = direct x̂0 (x-prediction in LTX formulation)
            v = (x - denoised) / sigma_curr.clamp(min=1e-8)

            # Euler forward (ramps up noise)
            x_next = x + (sigma_next - sigma_curr) * v

            # RF inversion correction term (with fade-in)
            if start_step <= i < end_step:
                x0_est = x - sigma_curr * v
                src = x_source.to(x.device, non_blocking=True)
                # Auto-resize if source shape differs from target (spatial dims only)
                if src.shape != x_next.shape:
                    ndim       = src.ndim
                    n_target   = x_next.ndim
                    target_sp  = list(x_next.shape[-2:])  # [H, W] of target
                    if ndim != n_target:
                        warnings.warn(
                            f"RF forward: shape rank mismatch src={ndim}d vs x_next={n_target}d, "
                            "skipping RF term for this step."
                        )
                        x = x_next
                        continue
                    mode = "trilinear" if ndim == 5 else "bilinear"
                    src  = F.interpolate(src, size=target_sp, mode=mode, align_corners=False)
                x_next = x_next + etas[i] * fade_in[i] * (src - x0_est)

            if callback is not None:
                callback({"x": x_next, "denoised": denoised,
                          "i": i, "sigma": sigma_curr})

            x = x_next

        return x

    return rf_forward_fn


def make_rf_reverse_fn(inv_latents, eta, eta_trend, start_step, end_step, steps):
    """
    Returns a reverse sampler function with params captured by closure.

    Fade-in: smoothstep quadratic ramp on etas during [start_step, start_step+warmup)
    to avoid a shock when the injection term first kicks in.
    """
    if eta_trend == "constant":
        etas = [eta] * steps
    elif eta_trend == "linear_increase":
        etas = [eta * i / max(steps - 1, 1) for i in range(steps)]
    else:
        etas = [eta * (1 - i / max(steps - 1, 1)) for i in range(steps)]

    # Warmup fade-in: ramp from 0→1 over the first ~30% of the injection window
    warmup = max(int((end_step - start_step) * 0.3), 1) if end_step > start_step else 0
    fade_in = torch.ones(steps)
    if warmup > 0 and end_step > start_step:
        ramp = torch.linspace(0, 1, min(warmup, steps))
        fade_in[:warmup] = ramp * ramp  # smoothstep quadratic ease-in

    def rf_reverse_fn(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = extra_args or {}

        for i in trange(len(sigmas) - 1, disable=disable):
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]

            denoised = model(x, sigma_curr * torch.ones(x.shape[0], device=x.device),
                             **extra_args)

            if sigma_next == 0:
                x = denoised
                break

            v = (x - denoised) / sigma_curr.clamp(min=1e-8)

            # Euler reverse (descends in noise)
            x_next = x + (sigma_next - sigma_curr) * v

            # Inject inverted latents (with fade-in + shape safety)
            if inv_latents is not None and start_step <= i < end_step:
                inv  = inv_latents.to(x.device, non_blocking=True)
                if inv.shape != x_next.shape:
                    ndim       = inv.ndim
                    n_target   = x_next.ndim
                    target_sp  = list(x_next.shape[-2:])
                    if ndim != n_target:
                        warnings.warn(
                            f"RF reverse: shape rank mismatch inv={ndim}d vs x_next={n_target}d, "
                            "skipping RF term for this step."
                        )
                    else:
                        mode = "trilinear" if ndim == 5 else "bilinear"
                        inv  = F.interpolate(inv, size=target_sp, mode=mode, align_corners=False)
                x_next = x_next + etas[i] * fade_in[i] * (inv - x_next)

            if callback is not None:
                callback({"x": x_next, "denoised": denoised,
                          "i": i, "sigma": sigma_curr})

            x = x_next

        return x

    return rf_reverse_fn


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

class LTXRFForwardSampler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":      ("MODEL",),
                "latents":    ("LATENT",),
                "positive":   ("CONDITIONING",),
                "negative":   ("CONDITIONING",),
                "steps":      ("INT",   {"default": 30,  "min": 1,   "max": 200}),
                "cfg":        ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "eta":        ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.5,
                               "step": 0.01}),
                "eta_trend":  (["constant", "linear_decrease", "linear_increase"],
                               {"default": "constant"}),
                "start_step": ("INT",   {"default": 0,  "min": 0, "max": 200}),
                "end_step":   ("INT",   {"default": 30, "min": 0, "max": 200}),
                "scheduler":  (comfy.samplers.KSampler.SCHEDULERS,
                               {"default": "simple"}),
                "seed":       ("INT",   {"default": 0,  "min": 0,
                               "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("inverted_latents", "source_latents")
    FUNCTION     = "forward_sample"
    CATEGORY     = "g_raw/LTX/RFInversion"

    def forward_sample(self, model, latents, positive, negative,
                       steps, cfg, eta, eta_trend,
                       start_step, end_step, scheduler, seed):

        device   = comfy.model_management.get_torch_device()
        x_source = latents["samples"].to(device)

        # Flip sigmas (small → large for forward/denoising direction)
        model_sampling = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(
            model_sampling, scheduler, steps
        ).to(device)
        sigmas_fwd = sigmas.flip(0)

        # Replace the terminal zero with the last valid sigma
        # (avoids division by zero in the loop)
        if sigmas_fwd[-1] < 1e-6:
            sigmas_fwd[-1] = sigmas_fwd[-2]

        # Clamp step ranges to actual number of steps and validate
        n_steps_actual = len(sigmas_fwd) - 1
        start_step = max(0, min(start_step, n_steps_actual))
        end_step   = max(start_step + 1, min(end_step, n_steps_actual))

        # Build sampler via closure — x_source captured here
        sampler_fn  = make_rf_forward_fn(
            x_source  = x_source,
            eta       = eta,
            eta_trend = eta_trend,
            start_step = start_step,
            end_step  = end_step,
            steps     = n_steps_actual,
        )
        sampler = comfy.samplers.KSAMPLER(sampler_fn)

        # sample_custom — real signature, no extra_args
        result = comfy.sample.sample_custom(
            model,
            torch.zeros_like(x_source),   # noise disabled
            cfg,
            sampler,
            sigmas_fwd,
            positive,
            negative,
            x_source,                      # latent_image
            noise_mask   = None,
            callback     = None,
            disable_pbar = False,
            seed         = seed,
        )

        return (
            {"samples": result.cpu()},
            {"samples": x_source.cpu()},
        )


class LTXRFReverseSampler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":      ("MODEL",),
                "latents":    ("LATENT",),
                "positive":   ("CONDITIONING",),
                "negative":   ("CONDITIONING",),
                "steps":      ("INT",   {"default": 30,  "min": 1,   "max": 200}),
                "cfg":        ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0}),
                "scheduler":  (comfy.samplers.KSampler.SCHEDULERS,
                               {"default": "simple"}),
                "seed":       ("INT",   {"default": 0,   "min": 0,
                               "max": 0xffffffffffffffff}),
            },
            "optional": {
                "inverted_latents": ("LATENT", {}),
                "eta":        ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.5,
                               "step": 0.01}),
                "eta_trend":  (["constant", "linear_decrease", "linear_increase"],
                               {"default": "linear_decrease"}),
                "start_step": ("INT",   {"default": 0,  "min": 0, "max": 200}),
                "end_step":   ("INT",   {"default": 15, "min": 0, "max": 200}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION     = "reverse_sample"
    CATEGORY     = "g_raw/LTX/RFInversion"

    def reverse_sample(self, model, latents, positive, negative,
                       steps, cfg, scheduler, seed,
                       inverted_latents=None, eta=0.4,
                       eta_trend="linear_decrease",
                       start_step=0, end_step=15):

        device = comfy.model_management.get_torch_device()
        x_T    = latents["samples"].to(device)

        inv_lat = None
        if inverted_latents is not None:
            inv_lat = inverted_latents["samples"]

        model_sampling = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(
            model_sampling, scheduler, steps
        ).to(device)

        # Clamp step ranges to actual number of steps and validate
        n_steps_actual = len(sigmas) - 1
        start_step = max(0, min(start_step, n_steps_actual))
        end_step   = max(start_step + 1, min(end_step, n_steps_actual))

        sampler_fn = make_rf_reverse_fn(
            inv_latents = inv_lat,
            eta         = eta,
            eta_trend   = eta_trend,
            start_step  = start_step,
            end_step    = end_step,
            steps       = n_steps_actual,
        )
        sampler = comfy.samplers.KSAMPLER(sampler_fn)

        result = comfy.sample.sample_custom(
            model,
            torch.zeros_like(x_T),
            cfg,
            sampler,
            sigmas,
            positive,
            negative,
            x_T,
            noise_mask   = None,
            callback     = None,
            disable_pbar = False,
            seed         = seed,
        )

        return ({"samples": result.cpu()},)


# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "LTXRFForwardSampler": LTXRFForwardSampler,
    "LTXRFReverseSampler": LTXRFReverseSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXRFForwardSampler": "LTX RF-Inv Forward (x0→xT)",
    "LTXRFReverseSampler": "LTX RF-Inv Reverse (xT→x0)",
}