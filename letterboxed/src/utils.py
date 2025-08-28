# src/utils.py

import time
from contextlib import contextmanager

@contextmanager
def timed(label="Timer"):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    #print(f"{label} took {end - start:.4f} seconds")
