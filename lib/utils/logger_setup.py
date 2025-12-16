"""
Logging Utilities

Unified logging configuration for the framework.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    name: str = "contextaware_testbed",
    level: str = "INFO",
    log_file: str | Path | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Custom format string

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger or create new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """
    Specialized logger for experiments with automatic log file creation.
    """

    def __init__(
        self, experiment_name: str, output_dir: str | Path | None = None, level: str = "INFO"
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            output_dir: Output directory for logs
            level: Logging level
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.level = level

        # Create log file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / "logs" / f"{experiment_name}_{timestamp}.log"

        # Setup logger
        self.logger = setup_logging(
            name=f"experiment.{experiment_name}", level=level, log_file=self.log_file
        )

    def log_experiment_start(self, config: dict):
        """
        Log experiment start with configuration.

        Args:
            config: Experiment configuration
        """
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info(f"Configuration: {config}")

    def log_experiment_end(self, results_summary: dict):
        """
        Log experiment end with results summary.

        Args:
            results_summary: Summary of results
        """
        self.logger.info(f"Completed experiment: {self.experiment_name}")
        self.logger.info(f"Results summary: {results_summary}")

    def log_metric(self, metric_name: str, value: float, step: int | None = None):
        """
        Log a metric value.

        Args:
            metric_name: Name of metric
            value: Metric value
            step: Optional step number
        """
        if step is not None:
            self.logger.info(f"Step {step} - {metric_name}: {value}")
        else:
            self.logger.info(f"{metric_name}: {value}")

    def log_error(self, error: Exception, context: str = ""):
        """
        Log an error with context.

        Args:
            error: Exception that occurred
            context: Additional context
        """
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)

    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)

    def log_info(self, message: str):
        """Log an info message."""
        self.logger.info(message)

    def log_debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)


# Default logger instance
default_logger = setup_logging()
