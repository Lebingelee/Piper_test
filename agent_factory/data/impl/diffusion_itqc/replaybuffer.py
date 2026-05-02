from agent_factory.data.impl.replaybuffer import (
    ClassicReplayBuffer as _BaseClassicReplayBuffer,
    FileReplayBuffer as _BaseFileReplayBuffer,
)


class DiffusionITQCReplayBuffer(_BaseFileReplayBuffer):
    """
    File replay buffer implementation for Diffusion_ITQC.

    The inherited behavior returns the `action` field expected by ITQC critic
    and actor mixins while keeping compatibility with legacy H5 files that use
    `actions` internally.
    """


class FileReplayBuffer(DiffusionITQCReplayBuffer):
    pass


class ClassicReplayBuffer(_BaseClassicReplayBuffer):
    pass


__all__ = [
    "DiffusionITQCReplayBuffer",
    "FileReplayBuffer",
    "ClassicReplayBuffer",
]
