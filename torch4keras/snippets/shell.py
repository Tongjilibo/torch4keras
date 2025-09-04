'''可供命令行直接使用的一些shell脚本
'''
from argparse import ArgumentParser
from enum import Enum, unique
from typing import Literal
import os
import subprocess
import time


def get_pid_using_port(port):
    """获取使用指定端口的进程ID"""
    try:
        # 使用lsof命令查找监听指定端口的进程ID
        result = subprocess.check_output(
            f"lsof -i:{port} -s TCP:LISTEN -t",
            shell=True,
            text=True
        )
        # 可能返回多个PID，取第一个
        pids = result.strip().split('\n')
        return pids[0] if pids and pids[0] else None
    except subprocess.CalledProcessError:
        # 没有找到使用该端口的进程
        return None


def get_child_pids(parent_pid):
    """获取指定进程的所有直接子进程ID"""
    try:
        # 使用pgrep命令查找父进程的子进程
        result = subprocess.check_output(
            f"pgrep -P {parent_pid}",
            shell=True,
            text=True
        )
        return [pid.strip() for pid in result.strip().split('\n') if pid.strip()]
    except subprocess.CalledProcessError:
        # 没有找到子进程
        return []

def terminate_children_and_self(parent_pid):
    """递归终止子进程，最后终止父进程"""
    # 先获取所有子进程
    children = get_child_pids(parent_pid)
    
    # 递归终止每个子进程
    for child in children:
        print(f"Terminating child process >>> {child}")
        terminate_children_and_self(child)
    
    # 终止父进程
    try:
        os.kill(int(parent_pid), 9)  # 9表示SIGKILL信号
        print(f"Terminating parent process >>> {parent_pid}")
    except OSError as e:
        print(f"Failed to terminate process {parent_pid}: {e}")


def kill_processes_on_port(port):
    """终止使用指定端口的进程及其所有子进程"""
    pid = get_pid_using_port(port)
    
    if not pid:
        print(f"No process found using port {port}")
        return
    
    print(f"Found process using port {port}: {pid}")
    terminate_children_and_self(pid)
    
    print("Please wait for processes to terminate...")
    time.sleep(5)
    print("Operation completed")


@unique
class Command(str, Enum):
    KILL_PORT_PID = "kill_port_pid"
    KILL_PID = "kill_pid"


def get_args_parser() -> ArgumentParser:
    """Helper function parsing the command line options."""

    parser = ArgumentParser(description="Torch4Keras Shell Script")

    parser.add_argument("--script", type=str, choices=[cmd.value for cmd in Command], default=None, help="Specify the script command to execute")
    parser.add_argument("--port", type=int, help="The port number to target")
    parser.add_argument("--pid", type=int, help="The process ID to target")

    args = parser.parse_args()

    return args


def run_shell_script():
    '''命令行运行shell脚本'''
    args = get_args_parser()
    script = args.script

    if script == Command.KILL_PORT_PID:
        kill_processes_on_port(args.port)
    elif script == Command.KILL_PID:
        terminate_children_and_self(args.pid)