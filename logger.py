"""
日志工具模块

统一日志格式，替代 print 语句。
支持控制台 + 文件输出，支持 --verbose / --log-file 参数。
"""

import logging
import sys


def setup_logger(name="tracker", level=logging.INFO, log_file=None):
    """创建并配置 logger

    Args:
        name: logger 名称
        level: 日志级别 (默认 INFO)
        log_file: 日志文件路径 (None=仅控制台)

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                            datefmt="%H:%M:%S")

    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件（可选）
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
