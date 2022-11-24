from enum import Enum, auto


class StrEnum(str, Enum):
    value: str
    _member_map_: dict[str, str]

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return str(self)

    def _generate_next_value_(name: str, *_: tuple[int, int, tuple[int, ...]]) -> str:
        return name


class IntensityOrProximity(StrEnum):
    LIGHT = "-"
    HEAVY = "+"
    VC = auto()


class Descriptor(StrEnum):
    MI = auto()
    "Shallow"
    PR = auto()
    "Partial"
    BC = auto()
    "Patches"
    DR = auto()
    "Low Drifting"
    BL = auto()
    "Blowing"
    SH = auto()
    "Showers"
    TS = auto()
    "Thunderstorm"
    FZ = auto()
    "Freezing"


class Precipitation(StrEnum):
    DZ = auto()
    "Drizzle"
    RA = auto()
    "Rain"
    SN = auto()
    "Snow"
    SG = auto()
    "Snow Grains"
    IC = auto()
    "Ice Crystals"
    PL = auto()
    "Ice Pellets"
    GR = auto()
    "Hail"
    GS = auto()
    "Small Hail"


class Obscuration(StrEnum):
    BR = auto()
    "Mist"
    FG = auto()
    "Fog"
    FU = auto()
    "Smoke"
    VA = auto()
    "Volcanic Ash"
    DU = auto()
    "Widespread Dust"
    SA = auto()
    "Sand"
    HZ = auto()
    "Haze"
    PY = auto()
    "Spray"


class Other(StrEnum):
    SQ = auto()
    "Squalls"
    FC = auto()
    "Funnel Cloud"
    SS = auto()
    "Sandstorm"
    DS = auto()
    "Dust-storm"


WEATHER_GROUP_TOKENS = (
    set(IntensityOrProximity)
    .union(Descriptor)
    .union(Precipitation)
    .union(Obscuration)
    .union(Other)
)
