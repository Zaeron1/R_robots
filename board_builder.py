import json
import os
import random
from typing import Optional, Tuple

DIRS = ["U", "R", "D", "L"]
DIR_TO_I = {d: i for i, d in enumerate(DIRS)}
I_TO_DIR = {i: d for i, d in enumerate(DIRS)}

MOVE = {"U": (0, -1), "R": (1, 0), "D": (0, 1), "L": (-1, 0)}
OPP  = {"U": "D", "D": "U", "L": "R", "R": "L"}

ALLOWED_COLORS  = {"RED", "BLUE", "GREEN", "YELLOW"}
ALLOWED_SYMBOLS = {"CIRCLE", "TRI", "SQUARE", "DIAMOND"}

# rotation clockwise qui envoie (0,0) vers le coin du quadrant collé au centre noir
SLOT_BASE_ROT = {
    "SE": 0,    # centre = top-left
    "SW": 90,   # centre = top-right
    "NE": 270,  # centre = bottom-left
    "NW": 180,  # centre = bottom-right
}

SLOTS = ["NW", "NE", "SW", "SE"]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def rotate_dir(d: str, rot_deg: int) -> str:
    steps = (rot_deg // 90) % 4
    return I_TO_DIR[(DIR_TO_I[d] + steps) % 4]


def rotate_xy(x: int, y: int, size: int, rot_deg: int) -> Tuple[int, int]:
    # rotation clockwise sur une grille size x size (0..size-1)
    if rot_deg == 0:
        return x, y
    if rot_deg == 90:
        return size - 1 - y, x
    if rot_deg == 180:
        return size - 1 - x, size - 1 - y
    if rot_deg == 270:
        return y, size - 1 - x
    raise ValueError("rot_deg must be 0/90/180/270")


def in_bounds(x: int, y: int, W: int, H: int) -> bool:
    return 0 <= x < W and 0 <= y < H


def add_wall_sym(full_walls: set, x: int, y: int, d: str, W: int, H: int):
    """
    Ajoute mur (x,y,d) et son opposé sur la case voisine
    SAUF si le voisin est hors de la grille.
    Les murs de bord sont donc affichés normalement.
    """
    full_walls.add((x, y, d))
    dx, dy = MOVE[d]
    nx, ny = x + dx, y + dy
    if 0 <= nx < W and 0 <= ny < H:
        full_walls.add((nx, ny, OPP[d]))



def assemble_board(
    base_dir: str = "boards",
    set_path: str = "boards/sets/default_set.json",
    out_path: str = "boards/out/board_full.json",
    seed: Optional[int] = None,
):
    rng = random.Random(seed)
    cfg = load_json(set_path)

    grid_size = int(cfg.get("grid_size", 16))
    qsize     = int(cfg.get("quadrant_size", 8))
    if grid_size != 2 * qsize:
        raise ValueError("grid_size doit être 2 * quadrant_size (ex: 16 et 8).")

    # --- 1) choisir 1 face par groupe (A: a1 ou a2, etc)
    groups = cfg["groups"]  # {"A":["a1","a2"], ...}
    chosen = {g: rng.choice(opts) for g, opts in groups.items()}

    # --- 2) placer les 4 groupes dans les 4 slots (aléatoire ou fixe)
    slot_policy = cfg.get("slot_policy", "fixed")  # "fixed" ou "random"

    if "slots" in cfg and slot_policy == "fixed":
        # ex: {"NW":"A","NE":"B","SW":"C","SE":"D"}
        slots_map = cfg["slots"]
        slot_to_quad = {slot: chosen[slots_map[slot]] for slot in SLOTS}
    else:
        # random: permutation des groupes sur les slots
        gnames = list(groups.keys())
        rng.shuffle(gnames)
        slot_to_quad = {slot: chosen[g] for slot, g in zip(SLOTS, gnames)}

    # offsets dans la grille 16x16
    offsets = {
        "NW": (0, 0),
        "NE": (qsize, 0),
        "SW": (0, qsize),
        "SE": (qsize, qsize),
    }

    full_walls = set()
    full_targets = []

    # centre 2x2 (classique)
    cx = grid_size // 2 - 1
    cy = grid_size // 2 - 1
    blocked = {(cx, cy), (cx+1, cy), (cx, cy+1), (cx+1, cy+1)}

    # --- 3) assembler chaque quadrant
    picks = {}
    for slot in SLOTS:
        qname = slot_to_quad[slot]
        rotdeg = SLOT_BASE_ROT[slot]  # rotation FORCÉE pour que (0,0) aille au centre noir
        picks[slot] = {"quad": qname, "rot": rotdeg}

        qpath = os.path.join(base_dir, "quadrants", f"{qname}.json")
        q = load_json(qpath)

        if int(q.get("size", qsize)) != qsize:
            raise ValueError(f"{qname}: size != {qsize}")

        ox, oy = offsets[slot]

        # walls (coords JSON: (0,0) top-left du quadrant)
        for w in q.get("walls", []):
            x, y, d = int(w["x"]), int(w["y"]), str(w["dir"]).upper()
            if not (0 <= x < qsize and 0 <= y < qsize):
                raise ValueError(f"{qname}: wall out of bounds ({x},{y})")
            if d not in DIR_TO_I:
                raise ValueError(f"{qname}: bad dir {d}")

            rx, ry = rotate_xy(x, y, qsize, rotdeg)
            rd = rotate_dir(d, rotdeg)
            add_wall_sym(full_walls, ox + rx, oy + ry, rd, grid_size, grid_size)

        # targets
        for t in q.get("targets", []):
            x, y = int(t["x"]), int(t["y"])
            color = str(t["color"]).upper()
            sym   = str(t["symbol"]).upper()
            if color not in ALLOWED_COLORS:
                raise ValueError(f"{qname}: bad target color {color}")
            if sym not in ALLOWED_SYMBOLS:
                raise ValueError(f"{qname}: bad target symbol {sym}")
            if not (0 <= x < qsize and 0 <= y < qsize):
                raise ValueError(f"{qname}: target out of bounds ({x},{y})")

            rx, ry = rotate_xy(x, y, qsize, rotdeg)
            full_targets.append({"x": ox + rx, "y": oy + ry, "color": color, "symbol": sym})

    if not full_targets:
        # Debug safety: ensure au moins une cible pour éviter un jeu sans cible.
        for y in range(grid_size):
            for x in range(grid_size):
                if (x, y) not in blocked:
                    full_targets.append({"x": x, "y": y, "color": "RED", "symbol": "CIRCLE"})
                    break
            if full_targets:
                break

    # --- 4) bordures extérieures
    for x in range(grid_size):
        add_wall_sym(full_walls, x, 0, "U", grid_size, grid_size)
        add_wall_sym(full_walls, x, grid_size - 1, "D", grid_size, grid_size)
    for y in range(grid_size):
        add_wall_sym(full_walls, 0, y, "L", grid_size, grid_size)
        add_wall_sym(full_walls, grid_size - 1, y, "R", grid_size, grid_size)

    out = {
        "grid_size": grid_size,
        "blocked": [{"x": x, "y": y} for (x, y) in sorted(blocked)],
        "walls": [{"x": x, "y": y, "dir": d} for (x, y, d) in sorted(full_walls)],
        "targets": full_targets,
        "meta": {
            "seed": seed,
            "picks": picks,
            "rule": "Chaque quadrant JSON a (0,0) top-left; on force une rotation par slot pour que (0,0) soit collé au centre noir."
        }
    }

    save_json(out_path, out)
    return out


# compat avec ton import actuel: from board_builder import build_board
def build_board(seed=None, write_file=True):
    return assemble_board(
        base_dir="boards",
        set_path="boards/sets/default_set.json",
        out_path="boards/out/board_full.json",
        seed=seed
    )
