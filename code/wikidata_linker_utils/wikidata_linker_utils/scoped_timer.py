from contextlib import contextmanager
import time

timings = {}

@contextmanager
def scoped_timer(name):
    t0 = time.time()
    yield
    if name not in timings:
        timings[name] = 0.0
    timings[name] += time.time() - t0


def scoped_timer_summarize():
    res = ""
    if len(timings) > 0:
        max_key_len = max(map(len, timings.keys())) 
        for key, value in sorted(timings.items(), key=lambda x: x[1], reverse=True):
            res += key.ljust(max_key_len) + " : " + str(value) + "\n"
        timings.clear()
    return res
