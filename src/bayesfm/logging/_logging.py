import json
import logging
import logging.config
from pathlib import Path
from tqdm import tqdm
import time


class TqdmLoggingHandler(logging.StreamHandler):
    """Overwrite logging.StreamHandler.emit to use tqdm.write."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def _override_formatter_using_tqdm(config: dict) -> dict:
    if "console" in config.get("handlers", {}):
        orig = config["handlers"]["console"]
        config["handlers"]["console"] = {
            "()": TqdmLoggingHandler,
            "formatter": orig.get("formatter", "colored"),
        }
    return config


def _setup_logging() -> None:
    with open(Path(__file__).parent / "logging.config.json", "r") as f:
        config = json.load(f)
    config = _override_formatter_using_tqdm(config)
    logging.config.dictConfig(config)


def set_verbosity(level: int) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(level)


_setup_logging()


def get_logger() -> logging.Logger:
    return logging.getLogger()


if __name__ == "__main__":
    logger = get_logger()

    for i in tqdm(range(5), desc="Processing"):
        logger.debug(f"処理中のステップ {i}")
        logger.info(f"処理中のステップ {i}")
        logger.warning(f"処理中のステップ {i}")
        logger.error(f"処理中のステップ {i}")
        logger.critical(f"処理中のステップ {i}")
        time.sleep(0.1)
