"""
Defines and loads logging-related configuration.
"""

from pathlib import Path
from typing import Dict

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    """
    Logging settings loaded from environment variables and .env.
    """

    log_dir: Path = Field(
        default=Path("logs"), description="Base directory for all log files."
    )

    test_log_file: Path = Field(
        default=Path("logs/test.log"), description="Log file path for 'test' component."
    )

    model_config = SettingsConfigDict(
        env_prefix="LOGGING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def log_files(self) -> Dict[str, Path]:
        """
        Returns a mapping from component name to its log file Path,
        picking up both defaults and any LOGGING_* overrides.
        """
        files: Dict[str, Path] = {}
        data = self.model_dump()  # includes defaults + env overrides
        for key, val in data.items():
            if key.endswith("_log_file"):
                comp = key[: -len("_log_file")]
                files[comp] = val
        return files


# instantiate once, to be imported by other modules
logging_settings = LoggingSettings()
