# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from enum import Enum, unique
from .snippets.shell import run_shell_script

VERSION = "v0.3.2"

line_length = 70

def create_bordered_text(lines, total_length):
    total_length = max(total_length, max(len(line) for line in lines) + 4)
    border = "-" * total_length
    # 每行格式：| + 内容左对齐 + 空格填充 + |
    formatted_lines = [f"| {line.ljust(total_length - 4)} |" for line in lines]
    return f"{border}\n" + "\n".join(formatted_lines) + f"\n{border}"

# 定义内容行（无需关心空格，只需写核心内容）
usage_lines = [
    "Usage:",
    "  torch4keras shell: execute a shell script",
    "  torch4keras version: show version info"
]
USAGE = create_bordered_text(usage_lines, line_length)

# WELCOME单独处理（总长度58）
welcome_lines = [
    f"Welcome to torch4keras, version {VERSION}",
    "",  # 空行
    "Project page: https://github.com/Tongjilibo/torch4keras"
]
WELCOME = create_bordered_text(welcome_lines, line_length)


@unique
class Command(str, Enum):
    SHEEL = "shell"
    VER = "version"
    HELP = "help"


def main():
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
    if command == Command.SHEEL:
        run_shell_script()
    elif command == Command.VER:
        print(WELCOME)
    elif command == Command.HELP:
        print(USAGE)
    else:
        print(f"Unknown command: {command}.\n{USAGE}")


if __name__ == "__main__":
    main()
