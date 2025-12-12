"""Ghost-aware RSSM variant.

This class currently reuses the base RSSM dynamics but serves as an explicit
toggle for ghost-aware world models so we can attach additional heads and loss
terms without changing the base dynamics API.
"""

from . import nets


class GhostAwareRSSM(nets.RSSM):
  """Interface-compatible RSSM with ghost-aware head support."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
