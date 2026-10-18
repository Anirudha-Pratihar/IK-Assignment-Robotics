"""Stress presets used to provoke (and then analyse) failures.
   reach  : table 2.2x bigger -> part of the cube-spawn area is outside the dexterous workspace  (ik_infeasible / ik_limits)
   tray   : cubes may spawn right next to the tray walls                                         (collision / ik)
   limits : arm joint ranges shrunk to 55 %                                                       (ik_limits)
   all    : reach + tray + limits"""
import copy
from .config import Config


def apply_stress(cfg: Config, design: dict, mode: str):
    cfg, design = copy.deepcopy(cfg), copy.deepcopy(design)
    if mode in ("none", "", None): return cfg, design
    modes = ("reach", "tray", "limits") if mode == "all" else (mode,)
    outer = cfg.tray_inner[1] + 2 * cfg.tray_wall_t
    tray_y = design["tray_center"][1]
    for m in modes:
        if m == "reach":
            k = 2.2
            design["L"] *= k; design["W"] *= k
            design["xmin"] = design["cx"] - design["L"] / 2; design["xmax"] = design["cx"] + design["L"] / 2
            design["ymin"] = design["cy"] - design["W"] / 2; design["ymax"] = design["cy"] + design["W"] / 2
            sp = design["spawn"]
            sp.update(xmin=design["xmin"] + 0.04, xmax=design["xmax"] - 0.04, ymin=design["ymin"] + 0.04,
                      ymax=tray_y - outer / 2 - cfg.spawn_tray_margin)
            design["x_shift_note"] = "table scaled for reach stress"
        elif m == "tray":
            design["spawn"]["ymax"] = tray_y - outer / 2 - 0.5 * cfg.spawn_tray_margin
        elif m == "limits":
            cfg.limit_scale = 0.55
    return cfg, design
