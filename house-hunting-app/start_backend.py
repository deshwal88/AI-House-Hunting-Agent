import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

# Custom logging config passed directly to uvicorn so it doesn't override our handlers.
# All agent.* loggers (agent.ingest, agent.fetch, etc.) are routed here automatically
# because they are children of the "agent" logger.
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(asctime)s  %(levelprefix)s %(message)s",
            "datefmt": "%H:%M:%S",
            "use_colors": True,
        },
        "plain": {
            "format": "%(asctime)s  %(levelname)-8s  %(message)s",
            "datefmt": "%H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "plain",
        },
        "uvicorn_console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "formatter": "default",
        },
    },
    "loggers": {
        # Uvicorn's own loggers
        "uvicorn":        {"handlers": ["uvicorn_console"], "level": "INFO", "propagate": False},
        "uvicorn.error":  {"handlers": ["uvicorn_console"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["uvicorn_console"], "level": "INFO", "propagate": False},
        # All our agent.* loggers (agent.main, agent.ingest, agent.fetch, etc.)
        "agent":          {"handlers": ["console"], "level": "INFO", "propagate": False},
        # LangChain / Google GenAI — set to WARNING to suppress verbose internal HTTP logs
        "langchain":      {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "httpx":          {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "httpcore":       {"handlers": ["console"], "level": "WARNING", "propagate": False},
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:api",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=LOGGING_CONFIG,
    )
