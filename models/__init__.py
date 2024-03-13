import os

from track import Neptune

_USE_HALFLIFE = os.environ.get("_USE_HALFLIFE", "0")
_USE_TF = os.environ.get("_USE_TF", "0")
_USE_SEQ = os.environ.get("_USE_SEQ", "0")

tracker_cls = Neptune
