"""Centralized logging configuration for Scribyte.

All loggers in the application should use hierarchical names under
"scribyte" so they inherit this configuration. For example:
    import logging
    logger = logging.getLogger("scribyte.transcriber")

The LOGGING_CONFIG dict below is passed to
`uvicorn.run()` to ensure *all* loggers (uvicorn's own + scribyte's)
use a consistent, structured formatter with timestamps and severity levels.
"""

from typing import Any


LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": True,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(asctime)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": True,
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        "uvicorn.access_log": {"handlers": ["access"], "level": "INFO", "propagate": False},
        # Application loggers — all scribyte.* loggers inherit this config.
        "scribyte": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
