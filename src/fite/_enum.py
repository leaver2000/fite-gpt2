import enum
import re
from typing import Generic, TypeVar

__all__ = ["StrEnum", "RegexEnum", "DictEnum"]
_VT = TypeVar("_VT")


class EnumMember(Generic[_VT]):
    name: str
    value: _VT
    _member_type_: type
    _member_names_: list[str]
    _member_map_: dict[str, _VT]


class EnumBase(Generic[_VT], enum.Enum):
    value: _VT

    @classmethod
    def list_members(cls) -> list[str]:
        return cls._member_names_

    @classmethod
    def list_values(cls) -> list[_VT]:
        return [member.value for member in cls]


class StrEnum(str, EnumBase):
    value: str

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


class DictEnum(dict, EnumBase):
    def __str__(self) -> str:
        return self.name
