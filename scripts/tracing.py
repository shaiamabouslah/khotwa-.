import os
from contextlib import nullcontext


class _Noop:
    def set_attributes(self, *_):
        pass


def span(name, kind="CHAIN"):
    if not os.environ.get("KHOTWA_TRACE"):
        return nullcontext(_Noop())
    import mlflow
    return mlflow.start_span(name=name, span_type=kind)
