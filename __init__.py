from .nodes.capture         import LTXAttentionCaptureSetup
from .nodes.transfer        import LTXAttentionHeadFreeze, LTXQKVTransfer
from .nodes.visualize       import (LTXAttentionQueryMap, LTXAttentionKeyMap,
                                   LTXAttentionMetricsViz, LTXAttentionGridViz)
from .nodes.evolution       import LTXAttentionTimestepEvolution
from .nodes.io              import LTXStoreDump, LTXStoreLoad
from .nodes.inspect         import LTXAttentionStoreInspect, LTXQKVStoreInspect
from .nodes.utils           import (LTXLatentDims, LTXAttentionCompareRuns,
                                   LTXAttentionHeadCandidates)
from .nodes.zone_analysis   import LTXAttentionZoneAnalysis
from .nodes.rf_inversion    import LTXRFForwardSampler, LTXRFReverseSampler

NODE_CLASS_MAPPINGS = {
    # Capture
    "LTXAttentionCaptureSetup":      LTXAttentionCaptureSetup,
    # Transfer / Intervention
    "LTXAttentionHeadFreeze":        LTXAttentionHeadFreeze,
    "LTXQKVTransfer":                LTXQKVTransfer,
    # Visualisation
    "LTXAttentionQueryMap":          LTXAttentionQueryMap,
    "LTXAttentionKeyMap":            LTXAttentionKeyMap,
    "LTXAttentionMetricsViz":        LTXAttentionMetricsViz,
    "LTXAttentionGridViz":           LTXAttentionGridViz,
    "LTXAttentionTimestepEvolution": LTXAttentionTimestepEvolution,
    "LTXAttentionZoneAnalysis":      LTXAttentionZoneAnalysis,
    # IO
    "LTXStoreDump":                  LTXStoreDump,
    "LTXStoreLoad":                  LTXStoreLoad,
    # Inspect / Debug
    "LTXAttentionStoreInspect":      LTXAttentionStoreInspect,
    "LTXQKVStoreInspect":            LTXQKVStoreInspect,
    # Utils
    "LTXLatentDims":                 LTXLatentDims,
    "LTXAttentionCompareRuns":       LTXAttentionCompareRuns,
    "LTXAttentionHeadCandidates":    LTXAttentionHeadCandidates,
    #RF Inversion
    "LTXRFForwardSampler":           LTXRFForwardSampler,
    "LTXRFReverseSampler":           LTXRFReverseSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXAttentionCaptureSetup":      "LTX Attn — Setup Capture",
    "LTXAttentionHeadFreeze":        "LTX Attn — Head Freeze",
    "LTXQKVTransfer":                "LTX Attn — QKV Transfer",
    "LTXAttentionQueryMap":          "LTX Attn — Query Map",
    "LTXAttentionKeyMap":            "LTX Attn — Key Map",
    "LTXAttentionMetricsViz":        "LTX Attn — Metrics Heatmap",
    "LTXAttentionGridViz":           "LTX Attn — Grid Viz",
    "LTXAttentionTimestepEvolution": "LTX Attn — Timestep Evolution",
    "LTXAttentionZoneAnalysis":      "LTX Attn — Zone Analysis",
    "LTXStoreDump":                  "LTX Attn — Store Dump",
    "LTXStoreLoad":                  "LTX Attn — Store Load",
    "LTXAttentionStoreInspect":      "LTX Attn — Store Inspect",
    "LTXQKVStoreInspect":            "LTX Attn — QKV Store Inspect",
    "LTXLatentDims":                 "LTX Attn — Latent Dims",
    "LTXAttentionCompareRuns":       "LTX Attn — Compare Runs",
    "LTXAttentionHeadCandidates":    "LTX Attn — Head Candidates",
    "LTXRFForwardSampler":           "LTX Attn — RF-Inv Forward (x0→xT)",
    "LTXRFReverseSampler":           "LTX Attn — RF-Inv Reverse (xT→x0)",
}