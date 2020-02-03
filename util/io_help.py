import pickle
import os
import logging
import numpy as np
from functools import singledispatch

logger = logging.getLogger(__name__)


def cached_load(cached_file, load_function, should_save: bool = True):
    try:
        with open(cached_file, "rb") as reader:
            data = pickle.load(reader)
    except:
        data = load_function()
        if cached_file and should_save:
            try:
                os.makedirs(os.path.split(cached_file)[0], exist_ok=True)
                with open(cached_file, "wb") as writer:
                    pickle.dump(data, writer)
                # CONSIDER: write to cached_file+'__temp__' then move, so if we try reading we will always read from a complete file
            except:
                logger.warning(f"WARNING: Could not create cached file: {cached_file}")
    return data


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float16)
@to_serializable.register(np.float32)
def ts_npfloat(val):
    return float(val)


@to_serializable.register(np.int32)
@to_serializable.register(np.int16)
def ts_npint(val):
    return int(val)
