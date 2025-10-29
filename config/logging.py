import logging
import sys


def setup_logging():
    """Configures the root logger for the application.

    This should be called once at the very beginning of the app startup.
    It sets the logging level to INFO and defines a clear log format.
    """
    # Check if handlers are already added to avoid duplication in Streamlit reruns
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        # If handlers are already set, just ensure the level is correct
        root_logger.setLevel(logging.INFO)
        return

    # If no handlers, set them up
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)-8s] [%(name)s:%(lineno)d] — %(message)s",
        stream=sys.stdout,  # Log to standard output, which Docker/Streamlit captures
    )

    # You can set different levels for noisy libraries here if needed
    # logging.getLogger("httpx").setLevel(logging.WARNING)
    # logging.getLogger("urllib3").setLevel(logging.WARNING)


# This allows you to just import 'logger' from this file in other modules
logger = logging.getLogger(__name__)
