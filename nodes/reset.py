from __future__ import annotations

from ..core.model_patch import reset_all_layers


class LTXResetPatches:
    """Manual escape hatch for orphaned layers: if an intervention/capture
    node is deleted or rewired out of the graph, its last-registered
    layer on the shared diffusion_model is never explicitly unregistered
    (nothing calls unregister_layer for a node_id that stops executing)
    and would otherwise linger -- still consuming RAM and still applying
    on future runs. This node clears every registered layer and restores
    the pristine forward outright."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("clean_model",)
    FUNCTION     = "reset"
    CATEGORY     = "g_raw/LTX/Profiler"

    @classmethod
    def IS_CHANGED(cls, model):
        # This node has no widgets, only a MODEL input -- ComfyUI would
        # otherwise cache it and skip re-running reset() whenever nothing
        # upstream changes, silently turning it into a no-op on repeat
        # queues. Its entire job is the side effect (mutating the shared
        # diffusion_model's layer registry), so it must run every time
        # regardless of input identity. NaN != NaN, so this always
        # compares as "changed".
        return float("nan")

    def reset(self, model):
        patched = model.clone()
        reset_all_layers(patched.model.diffusion_model)
        print("[LTXProfiler] ResetPatches: cleared all registered layers.")
        return (patched,)
