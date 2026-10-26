"""Memory-bounded backward passes with one native MimicKit optimizer update."""

from learning.mp_optimizer import MPOptimizer
from util import mp_util


class MicrobatchMPOptimizer(MPOptimizer):
    def step_accumulated(self, losses):
        """Consume already weighted losses, releasing each graph immediately.

        The finalization mirrors the pinned MPOptimizer.step: distributed mean,
        gradient clipping, momentum/weight decay and the step counter happen once
        per logical batch. No model parameters change between microbatches.
        """
        self._optimizer.zero_grad()
        for loss in losses:
            if loss.requires_grad:
                loss.backward()
            del loss

        if mp_util.enable_mp():
            self._aggregate_mp_grads()
        if self._enable_grad_clip():
            self._clip_grads(self._grad_clip)
        self._optimizer.step()
        if mp_util.enable_mp() and self.get_steps() % self.CHECK_SYNC_STEPS == 0:
            assert self._check_synced(), "Network parameters desynchronized"
        self._steps += 1
