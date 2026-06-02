from .nodes.capture         import LTXAttentionCaptureSetup, LTXQKVCapture
from .nodes.transfer        import LTXAttentionHeadFreeze, LTXQKVTransfer
from .nodes.visualize       import (LTXAttentionQueryMap, LTXAttentionKeyMap,
                                   LTXAttentionMetricsViz, LTXAttentionGridViz)
from .nodes.evolution       import LTXAttentionTimestepEvolution
from .nodes.io              import (LTXAttentionStoreDump, LTXAttentionStoreLoad,
                                   LTXQKVDump, LTXQKVLoad)
from .nodes.inspect         import (LTXAttentionStoreInspect, LTXQKVStoreInspect,
                                   LTXMapStoreInspect)
from .nodes.map_store_node  import LTXAttentionMapStore
from .nodes.utils           import LTXLatentDims, LTXAttentionCompareRuns
from .nodes.zone_analysis   import LTXAttentionZoneAnalysis

NODE_CLASS_MAPPINGS = {
    # Capture
    "LTXAttentionCaptureSetup":      LTXAttentionCaptureSetup,
    "LTXQKVCapture":                 LTXQKVCapture,
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
    # Map Store
    "LTXAttentionMapStore":          LTXAttentionMapStore,
    # IO
    "LTXAttentionStoreDump":         LTXAttentionStoreDump,
    "LTXAttentionStoreLoad":         LTXAttentionStoreLoad,
    "LTXQKVDump":                    LTXQKVDump,
    "LTXQKVLoad":                    LTXQKVLoad,
    # Inspect / Debug
    "LTXAttentionStoreInspect":      LTXAttentionStoreInspect,
    "LTXQKVStoreInspect":            LTXQKVStoreInspect,
    "LTXMapStoreInspect":            LTXMapStoreInspect,
    # Utils
    "LTXLatentDims":                 LTXLatentDims,
    "LTXAttentionCompareRuns":       LTXAttentionCompareRuns,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXAttentionCaptureSetup":      "LTX Attn — Setup Capture",
    "LTXQKVCapture":                 "LTX QKV — Capture Source",
    "LTXAttentionHeadFreeze":        "LTX Attn — Head Freeze",
    "LTXQKVTransfer":                "LTX QKV — Transfer",
    "LTXAttentionQueryMap":          "LTX Attn — Query Map",
    "LTXAttentionKeyMap":            "LTX Attn — Key Map",
    "LTXAttentionMetricsViz":        "LTX Attn — Metrics Heatmap",
    "LTXAttentionGridViz":           "LTX Attn — Grid Viz",
    "LTXAttentionTimestepEvolution": "LTX Attn — Timestep Evolution",
    "LTXAttentionZoneAnalysis":      "LTX Attn — Zone Analysis",
    "LTXAttentionMapStore":          "LTX Attn — Map Store",
    "LTXAttentionStoreDump":         "LTX Attn — Store Dump",
    "LTXAttentionStoreLoad":         "LTX Attn — Store Load",
    "LTXQKVDump":                    "LTX QKV — Dump",
    "LTXQKVLoad":                    "LTX QKV — Load",
    "LTXAttentionStoreInspect":      "LTX Attn — Store Inspect",
    "LTXQKVStoreInspect":            "LTX QKV — Store Inspect",
    "LTXMapStoreInspect":            "LTX Map Store — Inspect",
    "LTXLatentDims":                 "LTX — Latent Dims",
    "LTXAttentionCompareRuns":       "LTX Attn — Compare Runs",
}