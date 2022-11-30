import enum
import re

__all__ = ["StrEnum", "DictEnum"]


class StrEnum(str, enum.Enum):
    def __str__(self) -> str:
        return str(self.value)

    def _generate_next_value_(name: str, *_) -> str:
        return name


class RegexEnum(StrEnum):
    """A class that allows for regex matching of enum values"""

    def escape(self) -> str:
        return re.escape(self)

    def compile(self, flags=re.VERBOSE, escape: bool = False) -> re.Pattern[str]:
        if escape:
            return re.compile(self.escape(), flags=flags)
        return re.compile(self, flags)

    @property
    def re(self) -> re.Pattern[str]:
        return self.compile()


class DictEnum(dict, enum.Enum):
    def __str__(self) -> str:
        return self.name
