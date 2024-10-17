import argparse

from core.task_loader import TaskLoader
from core.mosek_interface import MosekInterface
from core.data_saver import DataSaver
from core.log import Log

cow = """ __________________________ 
< PlugMJ Beta 1.0.0 @Dream >
 -------------------------- 
        \   ^__^
         \  (OO)\_______
            (__)\       )\/\\
                ||----w |
                ||     ||
"""

desc = (
    cow
    + """Execute SDP Json exported from mathematica. 

Example: 
    PlugMJ -t Task.json -o output.csv -n NAME
                : Run Task.json and save the result as output.csv 
                  with task name = NAME
"""
)


def main():
    parser = argparse.ArgumentParser(
        prog="PlugMJ", description=desc, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-t", "--task", help="Path of the task file.")
    parser.add_argument(
        "-o", "--output", default="output.csv", help="Path of the output file."
    )
    parser.add_argument("-l", "--log", default="", help="Path of the log file.")
    args = parser.parse_args()

    # 判断 --task 是否为空
    if args.task is None:
        parser.print_help()
        exit(1)

    # TODO: Add thread

    logger = Log(args.log)
    # cow
    [logger.info(_) for _ in cow.split("\n")]

    # 求解
    saver = DataSaver(args.output, logger)
    data = TaskLoader(args.task, logger)
    task = MosekInterface(data, logger, saver, g=1)
    task.optimize()


if __name__ == "__main__":
    main()
