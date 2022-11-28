import enum

__all__ = ["StrEnum"]


class StrEnum(str, enum.Enum):
    value: str

    def __str__(self) -> str:
        return self.value

    def _generate_next_value_(name: str, *_: int | tuple[int]) -> str:
        return name
