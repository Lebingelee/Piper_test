"""Remote environment clients and their versioned v2 wire primitives."""

from .legacy import SocketEnv as LegacySocketEnv
from .protocol import ProtocolError, RemoteProtocolError, decode_ndarray_tree, encode_ndarray_tree
from .v2 import SocketEnvV2

# The public client is descriptor-first v2.  v1 is available only by its
# explicit legacy name, so an endpoint version is never guessed.
SocketEnv = SocketEnvV2

__all__ = [
    "SocketEnv",
    "SocketEnvV2",
    "LegacySocketEnv",
    "ProtocolError",
    "RemoteProtocolError",
    "decode_ndarray_tree",
    "encode_ndarray_tree",
]
