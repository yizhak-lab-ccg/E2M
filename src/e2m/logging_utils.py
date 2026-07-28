"""Progress logging. Silent by default; opt in with :func:`set_verbose`.

The package logs step-by-step progress (downloads, cross-validation folds, training)
through the ``e2m`` logger. Following library convention it is silent unless the caller
asks for it::

    import e2m
    e2m.set_verbose()            # print progress to stderr
    e2m.set_verbose(False)       # silence it again
"""

from __future__ import annotations

import logging

logger = logging.getLogger("e2m")
logger.addHandler(logging.NullHandler())


def set_verbose(enabled: bool = True, level: int = logging.INFO) -> None:
    """Turn e2m progress messages on or off.

    When enabled, e2m writes progress lines (prefixed ``[e2m]``) to stderr. This is
    handy in notebooks and scripts where the workflow calls otherwise run silently.
    """
    for handler in list(logger.handlers):
        if getattr(handler, "_e2m_stream", False):
            logger.removeHandler(handler)
    if enabled:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[e2m] %(message)s"))
        handler._e2m_stream = True  # tag so a later call can find and replace it
        logger.addHandler(handler)
        logger.setLevel(level)
    logger.propagate = False
