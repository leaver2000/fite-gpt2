import re

from ._enum import RegexEnum

TOKEN_PATTERN = re.compile(
    r"(?<=\s\d{3})(?=\d{2,3})|(?=KT)|(?=G\d{2}KT)|(?=G\d{3}KT)|(?<=FEW|SCT|BKN|OVC)|(?<=(FEW|SCT|BKN|OVC)\d{3})(?=CB)"
)


class TAF(RegexEnum):
    # split-winds: 23015G25KT -> 230 15 G 25 KT
    wind = r"""
    (?<=\s(\d{3}|VRB))(?=\d{2,3}(KT|G\d{2,3}KT))# wind direction and speed
    |(?=G\d{2,3}KT\s)                           # before Gust
    |(?<=G)(?=\d{2,3}KT\s)                      # after Gust
    |(?=KT\s)                                   # before KT
    """
    """
    split-winds: 23015G25KT -> 230 15 G 25 KT
    >>> TAF.wind.compile.split("23015G25KT")
    ['230', '15', 'G', '25', 'KT']
    """
    cloud_cover = r"""
    (?<=FEW|SCT|BKN|OVC)(?=\d{3})       # after cloud cover
    |(?<=(FEW|SCT|BKN|OVC)\d{3})(?=CB)  # before CB
    """
    """
    split-clouds: SCT250 -> SCT 250MODEL_PATH
    >>> TAF.cloud_cover.compile.split("SCT250")
    ['SCT', '250']
    """
    all = re.compile("|".join([wind, cloud_cover]), re.VERBOSE)
    TEMPERATURE_GROUP = r"\sTX(M)?\d{2}\/\d{4}Z\sTN(M)?\d{2}\/\d{4}Z"
    sub = TOKEN_PATTERN.sub
