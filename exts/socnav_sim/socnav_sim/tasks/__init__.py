"""Register task metadata without loading simulation configs or task resources.

Isaac Lab resolves string config entry points when the selected task is created.
Do not recursively import task modules: they may initialize CUDA and load assets.
"""

from .socnav import cadrl
from .socnav import drl_vo
from .socnav import egonav
