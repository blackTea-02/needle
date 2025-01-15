"""Logic for backend selection"""
import os


BACKEND = os.environ.get("NEEDLE_BACKEND", "nd")


if BACKEND == "nd":
    print("Using a backend")
    from needle import backend_ndarray as array_api

    NDArray = array_api.NDArray
elif BACKEND == "np":
    print("Using numpy backend")
    import numpy as array_api

    NDArray = array_api.ndarray
else:
    raise RuntimeError("Unknown a array backend %s" % BACKEND)
