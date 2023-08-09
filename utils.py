from typing import Tuple

import extra_data as xd
import extra_geom as xg
import numpy as np

pnccdpos_src_key = {
    "pnccdUp": ("SQS_NQS_PNCCD/MOTOR/PNCCD_UP", "actualPosition.value"),
    "pnccdDown": ("SQS_NQS_PNCCD/MOTOR/PNCCD_DOWN", "actualPosition.value"),
    "pnccdLeft": ("SQS_NQS_PNCCD/MOTOR/PNCCD_RIGHT", "actualPosition.value"),
    "pnccdRight": ("SQS_NQS_PNCCD/MOTOR/PNCCD_LEFT", "actualPosition.value"),
}


pnccdoffsets = {
    "gapAdd": 1.3,
    "moveAdd": 0.0,
    "closedPnccdUp": 16.32,
    "closedPnccdDown": 18.56,
    "rightStart": 15.98414063,
    "leftStart": 21.10546875,
}


maskparam = {"maskAvg": 10000, "maskStd": 5000, "maskMinStd": 4, "maskAvgToStd": 3000}


def get_mask(data: np.ndarray):
    """
    Compute mask from the first chunk of mask data trains.
    raw data without geom
    """
    images = np.array(data, dtype=float)
    avg = images.mean(axis=0)
    std = images.std(axis=0)
    avgToStd = avg / std
    mask = np.ones_like(avg)
    mask[avg > maskparam["maskAvg"]] = np.NAN
    mask[std > maskparam["maskStd"]] = np.NAN
    mask[std <= maskparam["maskMinStd"]] = np.NAN
    mask[avgToStd > maskparam["maskAvgToStd"]] = np.NAN
    mask[490:534, 481:543] = np.NAN
    return mask


def get_geom(
    proposal: int = 4396, run: int = 11, num_data_for_mask: int = 100
) -> Tuple[xg.PNCCDGeometry, np.ndarray]:
    """
    Get the detector geometry for a given run. The mask is determined from the first
    `num_data_for_mask` images in the run.
    """
    run = xd.open_run(proposal, run, data="raw")
    pnccdpos = {}
    for key in pnccdpos_src_key.keys():
        pnccdpos[key] = run[pnccdpos_src_key[key]].train_from_index(0)[1]

    gapInMm = (
        pnccdpos["pnccdUp"]
        + pnccdpos["pnccdDown"]
        - pnccdoffsets["closedPnccdUp"]
        - pnccdoffsets["closedPnccdDown"]
        + pnccdoffsets["gapAdd"]
    )

    gap_config = {
        "gap": gapInMm * 10 ** (-3),
        "top_offset": (
            (pnccdpos["pnccdRight"] - pnccdoffsets["rightStart"]) * 10 ** (-3),
            0,
            0.0,
        ),
        "bottom_offset": (
            -(
                pnccdpos["pnccdLeft"]
                - pnccdoffsets["leftStart"]
                - pnccdoffsets["moveAdd"]
            )
            * 10 ** (-3),
            0,
            0.0,
        ),
    }
    geom = xg.PNCCDGeometry.from_relative_positions(**gap_config)
    return geom, get_mask(
        run["SQS_NQS_PNCCD1MP/CAL/PNCCD_FMT-0:output", "data.image"][:num_data_for_mask]
        .xarray()
        .data
    )


def test_get_geom():
    geom, det = get_geom()
    run = xd.open_run(4396, 11, data="proc")
    img = run["SQS_NQS_PNCCD1MP/CAL/PNCCD_FMT-0:output", "data.image"][0].xarray()
    img.trainId
    geom.position_modules(img.data)  # render data with geom
