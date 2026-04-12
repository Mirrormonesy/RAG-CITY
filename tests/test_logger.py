import logging
from src.utils.logger import get_logger

def test_get_logger_returns_logger(tmp_path):
    log_file = tmp_path / "test.log"
    logger = get_logger("test_logger_unique_name", str(log_file))
    logger.info("hello")
    assert log_file.exists()
    assert "hello" in log_file.read_text(encoding="utf-8")
    assert isinstance(logger, logging.Logger)
