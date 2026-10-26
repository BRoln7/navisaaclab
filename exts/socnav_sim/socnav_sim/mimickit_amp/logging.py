"""Write every completed iteration without changing upstream checkpoint/eval cadence."""

import sys
import os
from torch.utils.tensorboard import SummaryWriter
from util.logger import Logger
from util.tb_logger import TBLogger


class IterationTBLogger(TBLogger):
    def __init__(self):
        super().__init__()
        self._last_written_iteration = None

    def configure_output_file(self, filename=None):
        Logger.configure_output_file(self, filename)
        if Logger.is_root():
            # This writer's flush waits for its queue. The installed
            # tensorboardX writer flushes the file before queued events drain.
            self._writer = SummaryWriter(os.path.dirname(filename), flush_secs=5)
        if self._writer is not None:
            self._writer.add_scalar("Progress/Started", 1., 0)
            self._writer.flush()

    def print_log(self):
        super().print_log()
        sys.stdout.flush()
        self.write_log()

    def write_log(self):
        iteration = self.get_current_val("Iteration")
        if iteration is not None and iteration == self._last_written_iteration:
            return
        super().write_log()
        if self._writer is not None:
            self._writer.flush()
        self._last_written_iteration = iteration
