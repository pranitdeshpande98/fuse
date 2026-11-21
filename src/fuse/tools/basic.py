"""Simple read-only tools that need no external account."""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import BaseTool, tool


def make_basic_tools(default_timezone: str) -> list[BaseTool]:
    @tool
    def get_current_time(timezone: str = default_timezone) -> str:
        """Get the current date and time in an IANA timezone such as 'America/Chicago'."""
        try:
            now = datetime.now(ZoneInfo(timezone))
        except ZoneInfoNotFoundError:
            return f"Unknown timezone: {timezone}"
        return now.strftime("%A %Y-%m-%d %H:%M %Z")

    return [get_current_time]
