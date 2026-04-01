try:
    from .time_attention import FusedTimeSelfAttention
except (ImportError, TypeError):
    pass

try:
    from .group_attention import FusedGroupSelfAttention
except (ImportError, TypeError):
    pass

try:
    from .feedforward import FusedFeedForward
except (ImportError, TypeError):
    pass

try:
    from .output import FusedOutputHead
except (ImportError, TypeError):
    pass
