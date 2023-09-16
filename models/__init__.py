import os

from track import Neptune

_USE_HALFLIFE = "1"
_USE_TF = "0"
_USE_SEQ = "1"

tracker_cls = Neptune

os.environ["_USE_HALFLIFE"] = _USE_HALFLIFE
os.environ["_USE_TF"] = _USE_TF
os.environ["_USE_SEQ"] = _USE_SEQ