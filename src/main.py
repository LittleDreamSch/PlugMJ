import argparse

from data.task_loader import TaskLoader
from interface.mosek_interface import MosekInterface
from interface.cvxpy_interface import CvxpyInterface
from data.data_saver import DataSaver
from utils.log import Log

cow = """ __________________________ 
< PlugMJ Beta 1.2.3 @Dream >
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
    PlugMJ -t Task.json -o output.csv -d min
                : Run Task.json and minimize the objective and save the result 
                  as output.csv by using cvxpy as default interface
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
        # 求解对偶问题
        # "MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_DUAL",
    }


def parser_handler():
    """
    解析命令行参数

    Returns:
        parser: 命令行参数解析器
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
        "-d",
        "--direction",
        default="min",
        help="Direction of the task. min or max. (minimize as default)",
    )
    parser.add_argument(
        "-T", "--threads", default=0, help="Number of threads. (0 as default)"
    )
    parser.add_argument("-e", "--eps", type=float, help="Tolerence of the solver.")
    parser.add_argument(
        "-i",
        "--interface",
        default="cvxpy",
        help="Interface to use, cvxpy or original. (cvxpy as default)",
    )
    parser.add_argument(
        "-l", "--log", default="", help="Save log to path of the given file."
    )
    parser.add_argument(
        "-n", "--name", default="SDP_TASK", help="Name of the task (disable in cvxpy)."
    )
    return parser, parser.parse_args()


def build_solver_options(args):
    """
    构建求解器参数

    Args:
        args: 命令行参数

    Returns:
        solver_options: 字典，包含求解器参数
    """
    # Mosek 参数
    MOSEK_OPTIONS = generate_mosek_options(args)
    solver_options = {"MOSEK_OPTIONS": MOSEK_OPTIONS}

    # eps
    if args.eps is not None:
        solver_options["eps"] = args.eps

    return solver_options


def build_solver(args):
    """
    构建求解器

    Args:
        args: 命令行参数
    """

    logger = Log(False, args.log)
    print_cow(logger)

    saver = DataSaver(args.output, logger)
    data = TaskLoader(args.task, logger, args.name)
    task = None

    solver_options = build_solver_options(args)

    if args.interface == "original":  # 原始接口
        task = MosekInterface(data, logger, saver, args.direction, **solver_options)
    elif args.interface == "cvxpy":  # cvxpy 接口
        task = CvxpyInterface(data, logger, saver, args.direction, **solver_options)

    return task, logger


def main():
    # 解析命令行参数
    parser, args = parser_handler()

    # 判断 --task 是否为空
    if args.task is None:
        parser.print_help()
        exit(1)

    task, logger = build_solver(args)

    if task is None:
        parser.print_help()
        logger.error(f"Interface {args.interface} is not supported.")
        exit(1)
    else:
        task.optimize()


if __name__ == "__main__":
    main()
