import logging
import logging.config
import os

# Define color codes for log levels
class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[95m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            levelname_color = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            record.levelname = levelname_color
        return super().format(record)

def setup_logging(logger_name="llm_serv"):
    """
    Set up and configure the logging system with colored output
    
    Args:
        logger_name (str): Name of the logger to return
        
    Returns:
        logging.Logger: Configured logger instance
    """
    log_level = os.getenv("LOG_LEVEL", "INFO")
    use_colors = os.getenv("LOG_COLORS", "true").lower() == "true"
    
    # Standard format string
    format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": format_str
            },
            "colored": {
                "()": ColorFormatter,
                "format": format_str
            }
        },
        "handlers": {
            "default": {
                "level": log_level,
                "formatter": "colored" if use_colors else "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["default"],
                "level": log_level,
                "propagate": True
            },
            "uvicorn": {
                "handlers": ["default"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["default"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["default"],
                "level": log_level,
                "propagate": False,
            },
            "llm_serv": {
                "handlers": ["default"],
                "level": log_level,
                "propagate": False,
            },
        }
    }
    
    logging.config.dictConfig(logging_config)
    return logging.getLogger(logger_name)

# Create a default logger instance that can be imported directly
logger = setup_logging() 