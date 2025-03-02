import sys

from loguru import logger

from entrag.config.load_config import load_eval_config


logger.remove()  # Remove the default logger
logger.add(
    sys.stderr,
    level="INFO",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
)


def main() -> None:
    """
    Main entry point for the evaluation script.
    """
    # config = load_eval_config("default")
    # logger.info(f"Loaded evaluation configuration {config}")
    print("OK")


if __name__ == "__main__":
    main()
