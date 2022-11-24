import numpy as np
import pandas as pd
from .util import SpecialTokens


def get_additional_special_tokens() -> list[str]:

    header_tokens = [
        "\u0120COR",
        "\u0120AMD",
        "BECMG",
        "TEMPO",
    ]
    wind_tokens = [
        f"VRB{SpecialTokens.unk_token}",
        f"{SpecialTokens.unk_token}G",
        f"{SpecialTokens.unk_token}KT",
    ] + pd.Series(np.linspace(10, 360, 36, dtype=int)).astype(str).str.zfill(3).tolist()

    icao_tokens = [
        "PASY",
    ]

    return (
        header_tokens
        + icao_tokens
        + wind_tokens
        + [
            # WIND
            # VISIBILITY
            "5000",
            "8000",
            "9999",
            "NSW",
            # PRECIPITATION
            "TS",
            "TSRA",
            "TSRAGR",
            "SHRA",
            "SH##",
            "SHSN",
            "RA",
            "SN",
            # OBSCURATION
            "BLSN",
            "FG",
            "BR",
            "FU",
            "HZ",
            "DU",
            "SA",
            "SS",
            "DS",
            "PO",
            # OTHER
            "FC",
            # CLOUD COVER
            "FEW",
            "SCT",
            "BKN",
            "OVC",
            "##CB",
            # TURBULENCE
            # "5",
            # ICING
            # "6",
            # ALTIMETER
            "QNH",
            "##INS",
            # TEMPERATURES
            "TX",
            "TN",
            # REMARKS
            "\u0120WND",
            "\u0120AFT",
            "\u0120LAST",
            "\u0120NO",
            "\u0120AMDS",
            # MISC
            "Z",
        ]
    )  # type: ignore
