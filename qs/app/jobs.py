"""Background CPU jobs. No GL objects leave the main thread."""
from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal


class JobSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)


class FnJob(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        # Keep Python ownership until queued signals have been delivered on the main thread.
        self.setAutoDelete(False)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = JobSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


_pool = None
_ACTIVE_JOBS: set[FnJob] = set()


def pool() -> QThreadPool:
    global _pool
    if _pool is None:
        _pool = QThreadPool.globalInstance()
        _pool.setMaxThreadCount(max(2, _pool.maxThreadCount() // 2 or 2))
    return _pool


def submit(fn, *args, on_ok=None, on_err=None, **kwargs):
    job = FnJob(fn, *args, **kwargs)
    _ACTIVE_JOBS.add(job)

    def _finish_ok(res, j=job, cb=on_ok):
        _ACTIVE_JOBS.discard(j)
        if cb is not None:
            cb(res)

    def _finish_err(err, j=job, cb=on_err):
        _ACTIVE_JOBS.discard(j)
        if cb is not None:
            cb(err)

    job.signals.finished.connect(_finish_ok, Qt.ConnectionType.QueuedConnection)
    job.signals.failed.connect(_finish_err, Qt.ConnectionType.QueuedConnection)
    pool().start(job)
    return job
