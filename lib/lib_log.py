import os
import re
from typing import Optional, Union

from rich import print as rprint
from rich.text import Text


VERBOSITY_LEVELS = {
    "quiet": 0,
    "warning": 0,
    "warnings": 0,
    "normal": 1,
    "log": 1,
    "verbose": 2,
    "info": 2,
    "debug": 2,
}

GLOBAL_VERBOSE = 0
GLOBAL_MAX_LOG_LINES = 300


def _parse_verbose(verbose: Optional[Union[int, str]]) -> int:
    if verbose is None:
        return GLOBAL_VERBOSE
    if isinstance(verbose, int):
        return max(0, min(2, verbose))
    return VERBOSITY_LEVELS.get(str(verbose).strip().lower(), GLOBAL_VERBOSE)


def configure_global_logging(
    verbose: Optional[Union[int, str]] = None,
    max_lines: Optional[int] = None,
) -> None:
    """Configure global logging defaults for workflow output buffering."""
    global GLOBAL_VERBOSE, GLOBAL_MAX_LOG_LINES

    if verbose is not None:
        GLOBAL_VERBOSE = _parse_verbose(verbose)

    if max_lines is not None:
        GLOBAL_MAX_LOG_LINES = max(1, int(max_lines))


def get_global_logging_config() -> dict[str, int]:
    return {
        "verbose": GLOBAL_VERBOSE,
        "max_lines": GLOBAL_MAX_LOG_LINES,
    }


def init_global_logging_from_env() -> None:
    verbose_env = os.getenv("SOLAR_VERBOSE")
    max_lines_env = os.getenv("SOLAR_MAX_LOG_LINES")

    if verbose_env is not None:
        configure_global_logging(verbose=verbose_env)

    if max_lines_env is not None:
        try:
            configure_global_logging(max_lines=int(max_lines_env))
        except ValueError:
            pass


class WorkflowLogBuffer:
    """String-like workflow log collector with verbosity filtering and line capping."""

    def __init__(
        self,
        verbose: Optional[Union[int, str]] = None,
        max_lines: Optional[int] = None,
    ) -> None:
        self.verbose = _parse_verbose(verbose)
        self.max_lines = GLOBAL_MAX_LOG_LINES if max_lines is None else max(1, int(max_lines))
        self._lines: list[str] = []
        self._seen: set[str] = set()
        self._dropped_by_verbosity = 0
        self._dropped_by_limit = 0

    @staticmethod
    def _strip_rich_markup(message: str) -> str:
        return re.sub(r"\[[^\]]*\]", "", message)

    @classmethod
    def _infer_level(cls, message: str) -> int:
        plain = cls._strip_rich_markup(message).upper()
        if "ERROR" in plain or "WARNING" in plain:
            return 0
        if "[LOG]" in message.upper() or " DONE" in plain or "COMPUT" in plain:
            return 1
        if "INFO" in plain:
            return 2
        return 2

    def _append_line(self, line: str) -> None:
        text = line.rstrip()
        if not text:
            return

        self._seen.add(text)
        level = self._infer_level(text)
        if level > self.verbose:
            self._dropped_by_verbosity += 1
            return

        self._lines.append(text)
        if len(self._lines) > self.max_lines:
            overflow = len(self._lines) - self.max_lines
            self._lines = self._lines[overflow:]
            self._dropped_by_limit += overflow

    def __iadd__(self, other):
        if other is None:
            return self

        if isinstance(other, WorkflowLogBuffer):
            for line in other._lines:
                self._append_line(line)
            self._dropped_by_verbosity += other._dropped_by_verbosity
            self._dropped_by_limit += other._dropped_by_limit
            self._seen.update(other._seen)
            return self

        text = str(other)
        for line in text.splitlines():
            self._append_line(line)
        return self

    def __contains__(self, item: object) -> bool:
        return str(item) in self._seen

    def __len__(self) -> int:
        return len(self.render(include_footer=False))

    def __getitem__(self, key):
        # Support legacy string-style indexing/slicing, e.g. output[:-3].
        return self.render(include_footer=False)[key]

    def render(self, include_footer: bool = True) -> str:
        lines = list(self._lines)

        if include_footer:
            if self._dropped_by_verbosity > 0:
                lines.append(
                    f"[yellow][WARNING][/yellow] {self._dropped_by_verbosity} log lines hidden by verbosity={self.verbose}."
                )
            if self._dropped_by_limit > 0:
                lines.append(
                    f"[yellow][WARNING][/yellow] {self._dropped_by_limit} older log lines omitted (max_lines={self.max_lines})."
                )

        if not lines:
            return ""

        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.render()

    def __rich__(self) -> Text:
        """Provide a Rich-native renderable so rprint(buffer) applies markup colors."""
        rendered = self.render(include_footer=True)
        if not rendered:
            return Text()
        return Text.from_markup(rendered.rstrip("\n"))

    def rich_print(self, include_footer: bool = True) -> None:
        """Print using Rich so markup tags are rendered as terminal colors."""
        rendered = self.render(include_footer=include_footer)
        if rendered:
            rprint(rendered, end="")

    def __add__(self, other):
        return self.render(include_footer=True) + str(other)

    def __radd__(self, other):
        return str(other) + self.render(include_footer=True)


def create_workflow_log(
    verbose: Optional[Union[int, str]] = None,
    max_lines: Optional[int] = None,
) -> WorkflowLogBuffer:
    return WorkflowLogBuffer(verbose=verbose, max_lines=max_lines)


def print_workflow_log(log: Union[WorkflowLogBuffer, str], include_footer: bool = True) -> None:
    """Print workflow logs with Rich markup support."""
    if isinstance(log, WorkflowLogBuffer):
        log.rich_print(include_footer=include_footer)
        return

    rendered = str(log)
    if rendered:
        rprint(rendered, end="" if rendered.endswith("\n") else "\n")


# Initialize defaults from environment once on import.
init_global_logging_from_env()
