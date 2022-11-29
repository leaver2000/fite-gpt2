import re

TOKEN_PATTERN = re.compile(
    r"(?<=\s\d{3})(?=\d{2,3})|(?=KT)|(?=G\d{2}KT)|(?=G\d{3}KT)|(?<=FEW|SCT|BKN|OVC)|(?<=(FEW|SCT|BKN|OVC)\d{3})(?=CB)"
)


class RegexPatterns:
    # split-winds: 23015G25KT -> 230 15 G 25 KT
    WIND_GUST = r"""
    (?<=\s(\d{3}|VRB))(?=\d{2,3}(KT|G\d{2,3}KT)) # wind direction and speed
    |(?=G\d{2,3}KT\s) # before Gust
    |(?<=G)(?=\d{2,3}KT\s) # after Gust
    |(?=KT\s) # before KT
    """
    # split-clouds: SCT250 -> SCT 250MODEL_PATH
    CLOUD_COVER = r"""
    (?<=FEW|SCT|BKN|OVC)(?=\d{3}) # after cloud cover
    |(?<=(FEW|SCT|BKN|OVC)\d{3})(?=CB) # before CB
    """
    TOKEN_PATTERN = re.compile("|".join([WIND_GUST, CLOUD_COVER]), re.VERBOSE)
    TEMPERATURE_GROUP = r"\sTX(M)?\d{2}\/\d{4}Z\sTN(M)?\d{2}\/\d{4}Z"
    sub = TOKEN_PATTERN.sub
