"""Host storage and selective GPU staging for native MimicKit experience buffers."""

from contextlib import contextmanager
import torch
from learning.experience_buffer import ExperienceBuffer


class StagedExperienceBuffer(ExperienceBuffer):
    """Keep upstream ring/shuffle semantics; transfer only sampled training fields.

    Large demonstration/replay arrays live in ordinary CPU memory. Keeping them
    unpinned avoids reserving several GiB of page-locked RAM on the workstation.
    """

    def __init__(self, buffer_length, batch_size, device, *, host_keys=(), sample_device=None):
        self._host_keys = frozenset(host_keys)
        self._sample_device = torch.device(sample_device or device)
        self._sample_fields = None
        self._sample_plan = None
        self._sample_plan_head = 0
        super().__init__(buffer_length, batch_size, device)

    def add_buffer(self, name, data_shape, dtype):
        if name not in self._host_keys:
            return super().add_buffer(name, data_shape, dtype)
        assert name not in self._buffers
        shape = (self._buffer_length, self._batch_size, *data_shape)
        data = torch.zeros(shape, dtype=dtype, device="cpu")
        self._buffers[name] = data
        self._flat_buffers[name] = data.view(self.get_capacity(), *data_shape)

    @contextmanager
    def sampling_fields(self, fields):
        """Upstream optimizers can sample only the tensors their loss consumes."""
        previous = self._sample_fields
        self._sample_fields = frozenset(fields)
        try:
            yield
        finally:
            self._sample_fields = previous

    def sample_indices(self, n):
        """Reserve one logical batch with native shuffle/wraparound semantics."""
        # Native indices can alias its shuffle buffer; own them until all chunks
        # have been consumed, even if another sampling call reshuffles the ring.
        return self._sample_rand_idx(n).clone()

    @contextmanager
    def sampling_plan(self, n):
        """Let native discriminator code consume a reserved replay batch in chunks."""
        if self._sample_plan is not None:
            raise RuntimeError("Replay sampling plans cannot be nested")
        self._sample_plan = self.sample_indices(n)
        self._sample_plan_head = 0
        try:
            yield
            if self._sample_plan_head != n:
                raise RuntimeError("Incomplete logical replay batch")
        finally:
            self._sample_plan = None
            self._sample_plan_head = 0

    def sample(self, n):
        if self._sample_plan is None:
            indices = self.sample_indices(n)
        else:
            end = self._sample_plan_head + n
            if end > len(self._sample_plan):
                raise RuntimeError("Replay sampling exceeded its logical batch")
            indices = self._sample_plan[self._sample_plan_head:end]
            self._sample_plan_head = end
        return self.gather(indices)

    def gather(self, indices):
        """Transfer only selected fields for this microbatch, including host demo."""
        device_indices = {indices.device: indices}
        result = {}
        for key, data in self._flat_buffers.items():
            if self._sample_fields is not None and key not in self._sample_fields:
                continue
            if data.device not in device_indices:
                device_indices[data.device] = indices.to(data.device)
            result[key] = data[device_indices[data.device]].to(self._sample_device)
        return result
