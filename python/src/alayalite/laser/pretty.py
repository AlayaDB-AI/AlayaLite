"""Terminal pretty-print helpers with automatic color detection."""

import builtins
import functools
import sys

# Force all print() calls to flush immediately,
# so Python output stays in sync with C++ stdout.
builtins.print = functools.partial(builtins.print, flush=True)

DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

if not sys.stdout.isatty():
    DIM = BOLD = GREEN = CYAN = YELLOW = RED = RESET = ""


def header(text, width=60):
    print(f"\n{BOLD}{CYAN}{'=' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * width}{RESET}")


def step_header(name, width=40):
    print(f"\n{BOLD}>> {name.upper()}{RESET}")
    print(f"{DIM}{'─' * width}{RESET}")


def info(tag, msg):
    print(f"  {DIM}[{tag}]{RESET} {msg}")


def success(tag, msg):
    print(f"  {GREEN}[{tag}]{RESET} {msg}")


def warn(tag, msg):
    print(f"  {YELLOW}[{tag}]{RESET} {msg}")


def separator(width=56):
    print(f"  {DIM}{'─' * width}{RESET}")
