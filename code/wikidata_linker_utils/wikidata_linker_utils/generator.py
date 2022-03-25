import queue
import threading
from contextlib import contextmanager
from threading import Event


def _prefetch_generator_internal(generator, running, to_fetch=10):
    q = queue.Queue(maxsize=to_fetch)
    some_exception = [None]

    def thread_worker(queue, gen):
        try:
            for val in gen:
                if running.is_set():
                    queue.put(val)
                else:
                    break
            queue.put(None)
        except Exception as e:
            some_exception[0] = e

    t = threading.Thread(target=thread_worker, args=(q, generator), daemon=True)
    t.start()
    while running.is_set():
        try:
            job = q.get(block=True, timeout=0.01)
            if job is None:
                t.join()
                break
            yield job
        except queue.Empty:
            if some_exception[0] is not None:
                t.join()
                raise some_exception[0]
    del t


def _flush_generator(generator, running):
    running.clear()
    while True:
        try:
            next(generator)
        except StopIteration:
            break


@contextmanager
def prefetch_generator(generator, to_fetch=10):
    running = Event()
    running.set()
    try:
        yield _prefetch_generator_internal(generator, running=running)
    finally:
        _flush_generator(generator, running)
