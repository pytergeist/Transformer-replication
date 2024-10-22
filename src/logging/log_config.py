# log_config.py

import logging
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

info_handler = logging.FileHandler(os.path.join(log_dir, "transformer_info.log"))
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(os.path.join(log_dir, "transformer_error.log"))
error_handler.setLevel(logging.ERROR)

debug_handler = logging.FileHandler(os.path.join(log_dir, "transformer_debug.log"))
debug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)

logger.addHandler(info_handler)
logger.addHandler(error_handler)
logger.addHandler(debug_handler)
