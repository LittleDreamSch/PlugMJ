import argparse

from data.task_loader import TaskLoader
from interface.mosek_interface import MosekInterface
from data.data_saver import DataSaver
from utils.log import Log

cow = """ __________________________ 
< PlugMJ Beta 1.0.0 @Dream >
 -------------------------- 
        \\   ^__^
         \\  (OO)\\_______
            (__)\\       )\\/\\
                ||----w |
                ||     ||
"""

desc = (
    cow
    + """Execute SDP Json exported from mathematica. 

Example: 
    PlugMJ -t Task.json -o output.csv -n NAME
                : Run Task.json and save the result as output.csv 
                  with task name = NAME by using cvxpy as interface
"""
)


def print_cow(logger):
    """
    打印牛
    """
    [logger.info(_) for _ in cow.split("\n")]  # cow


def generate_mosek_options(args):
    """
    生成 Mosek 参数

    Args:
        args: 命令行参数
    Returns:
        Mosek 参数字典
    """
    return {
        # 线程数
        "MSK_IPAR_NUM_THREADS": int(args.threads),
        # 日志衰减
        "MSK_IPAR_LOG_CUT_SECOND_OPT": 0,
    }


def parser_handler():
    """
    解析命令行参数

    Returns:
        parser
        args: 命令行参数
    """
    parser = argparse.ArgumentParser(
        prog="PlugMJ", description=desc, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-t", "--task", help="Path of the task file.")
    parser.add_argument(
        "-o", "--output", default="output.csv", help="Path of the output file."
    )
    parser.add_argument(
        "-i", "--interface", default="cvxpy", help="Interface to use. cvxpy as default."
    )
    parser.add_argument("-n", "--name", default="SDP_TASK", help="Name of the task.")
    parser.add_argument("-l", "--log", default="", help="Path of the log file.")
    parser.add_argument("-T", "--threads", default=0, help="Number of threads.")
    return parser, parser.parse_args()


def build_solver(args):
    """
    构建求解器

    Args:
        args: 命令行参数
    """
    # 配置 Mosek 参数
    MOSEK_OPTIONS = generate_mosek_options(args)

    logger = Log(args.log)
    print_cow(logger)

    # 求解
    saver = DataSaver(args.output, logger)
    data = TaskLoader(args.task, logger, args.name)
    task = MosekInterface(data, logger, saver, **MOSEK_OPTIONS)
    task.optimize()


def main():
    # 解析命令行参数
    parser, args = parser_handler()

    # 判断 --task 是否为空
    if args.task is None:
        parser.print_help()
        exit(1)

    build_solver(args)


if __name__ == "__main__":
    main()
