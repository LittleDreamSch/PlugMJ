"""
日志
"""

from loguru import logger
import sys


class Log:
    def __init__(self, log_path=""):
        """
        初始化日志

        Args:
            log_path: 日志文件路径，为空则不保存
        """
        self.log_path = log_path
        self.logger = self.init_log()
        self.add_level()

    def add_level(self):
        """
        添加自定义日志等级
        """
        self.logger.level("MOSEK", no=20, color="<blue>")

    def mosek(self, msg):
        """
        输出MOSEK日志

        Args:
            msg: 日志信息
        """
        self.logger.log("MOSEK", msg)

    def init_log(self):
        """
        设置日志格式
        """
        # 清空日志
        logger.remove()

        # 设置命令行输出格式
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD @ HH:mm:ss}</green> | <level>{level.name:<7}</level> | {message}",
            level="DEBUG",
            # enqueue=True,
            backtrace=True,
        )

        # 设置文件输出格式
        if self.log_path != "":
            logger.add(
                self.log_path,
                format="{time:YYYY-MM-DD @ HH:mm:ss} | {level.name:<7} | {message}",
                level="DEBUG",
                enqueue=True,
                backtrace=True,
            )

        return logger

    def __getattr__(self, level: str):
        return getattr(self.logger, level)
