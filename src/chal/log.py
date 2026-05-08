"""
log.py

Centralized logger for the CHAL framework.

All modules should import ``logger`` from here rather than creating their
own loggers.  Logging is configured once at CLI startup (see cli/main.py),
using ``format="%(message)s"`` so output looks identical to the previous
print()-based interface.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("chal")
