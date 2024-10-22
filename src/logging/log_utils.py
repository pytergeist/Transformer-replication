# log_utils.py

from src.logging.log_config import logger


def log_tensor_shape(tensor, tensor_name):
    logger.info(f"{tensor_name} shape: {tensor.shape}")


def log_debug_message(message):
    logger.debug(message)


def log_error_message(message):
    logger.error(message)
