import os
import sys
import time
import random
import heapq
from collections import deque

import pygame

from board_builder import build_board  # <-- IMPORTANT


# =========================
#        PARAMÈTRES UI
# =========================
FPS = 60

TILE = 40
MARGIN = 16
TOPBAR_H = 128

ASSETS_DIR = "assets"
ROBOTS_DIR = os.path.join(ASSETS_DIR, "robots")
TARGETS_DIR = os.path.join(ASSETS_DIR, "targets")

ROBOT_COLORS = [
    ("RED", (220, 60, 60)),
    ("BLUE", (70, 140, 235)),
    ("GREEN", (60, 200, 120)),
    ("YELLOW", (235, 205, 70)),
    ("WHITE", (240, 240, 245)),
]

TARGET_COLORS = ["RED", "BLUE", "GREEN", "YELLOW"]
SYMBOLS = ["CIRCLE", "TRI", "SQUARE", "DIAMOND"]
SYMBOL_TO_FILE = {"CIRCLE": "circle", "TRI": "tri", "SQUARE": "square", "DIAMOND": "diamond"}

DIRS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # U R D L

# solveur incrémental (pas de freeze)
SOLVER_MAX_NODES = 1_200_000
SOLVER_BUDGET_MS_IDLE = 1
SOLVER_BUDGET_MS_HOVER = 12


def safe_load_png(path: str):
    if not os.path.isfile(path):
        return None
    try:
        img = pygame.image.load(path)
        if pygame.display.get_surface():
            return img.convert_alpha()
        return img
    except Exception:
        return None


# =========================
#           BOARD
# =========================
class Board:
    """
    Board construit depuis board_builder (16x16).
    walls[idx] = bitmask 4 bits (U,R,D,L).
    blocked[idx] = True pour les cases du bloc central.
    targets = list of {"idx": int, "color_i": 0..3, "symbol": str}
    """
    def __init__(self, grid_size: int):
        self.W = grid_size
        self.H = grid_size
        self.N = self.W * self.H
        self.walls = [0] * self.N
        self.blocked = [False] * self.N
        self.targets = []
        self.center_block_cells = []
        self.meta = {}

        self.idx_to_xy = [(i % self.W, i // self.W) for i in range(self.N)]

    def xy_to_idx(self, x, y):
        return y * self.W + x

    def in_bounds(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H

    def has_wall(self, idx, d):
        return (self.walls[idx] >> d) & 1

    @classmethod
    def from_full_dict(cls, data: dict):
        grid = int(data["grid_size"])
        b = cls(grid)
        b.meta = data.get("meta", {})

        DIR_TO_BIT = {"U": 0, "R": 1, "D": 2, "L": 3}
        COLOR_TO_I = {"RED": 0, "BLUE": 1, "GREEN": 2, "YELLOW": 3}

        # blocked
        for bb in data.get("blocked", []):
            x, y = int(bb["x"]), int(bb["y"])
            idx = b.xy_to_idx(x, y)
            b.blocked[idx] = True
            b.center_block_cells.append(idx)

        # walls
        for w in data.get("walls", []):
            x, y = int(w["x"]), int(w["y"])
            d = str(w["dir"]).upper()
            idx = b.xy_to_idx(x, y)
            b.walls[idx] |= (1 << DIR_TO_BIT[d])

        # targets
        for t in data.get("targets", []):
            x, y = int(t["x"]), int(t["y"])
            color = str(t["color"]).upper()
            sym = str(t["symbol"]).upper()
            if color not in COLOR_TO_I:
                continue
            b.targets.append({"idx": b.xy_to_idx(x, y), "color_i": COLOR_TO_I[color], "symbol": sym})

        return b

    def slide(self, state, robot_i, d):
        cur = state[robot_i]
        x, y = self.idx_to_xy[cur]
        dx, dy = DIRS[d]

        while True:
            if self.has_wall(cur, d):
                break

            nx, ny = x + dx, y + dy
            if not self.in_bounds(nx, ny):
                break
            nidx = self.xy_to_idx(nx, ny)

            if self.blocked[nidx]:
                break

            for j, rp in enumerate(state):
                if j != robot_i and rp == nidx:
                    return cur

            cur = nidx
            x, y = nx, ny

        return cur


# =========================
#   HEURISTIQUE (relaxée)
# =========================
def relaxed_dist_table(board: Board, target_idx: int):
    INF = 10**9
    dist = [INF] * board.N
    dist[target_idx] = 0
    q = deque([target_idx])

    while q:
        cur = q.popleft()
        cd = dist[cur]
        x, y = board.idx_to_xy[cur]

        for d in range(4):
            dx, dy = DIRS[d]
            tx, ty = x, y
            while True:
                if board.has_wall(board.xy_to_idx(tx, ty), d):
                    break
                px, py = tx + dx, ty + dy
                if not board.in_bounds(px, py):
                    break
                pidx = board.xy_to_idx(px, py)
                if board.blocked[pidx]:
                    break
                tx, ty = px, py
                if dist[pidx] > cd + 1:
                    dist[pidx] = cd + 1
                    q.append(pidx)

    return dist


class IncrementalAStar:
    def __init__(self, board: Board, start_state, target_robot_i, target_idx, h_table):
        self.board = board
        self.start = start_state
        self.target_robot_i = target_robot_i
        self.target_idx = target_idx
        self.h = h_table

        self.open = []
        self.g = {start_state: 0}
        self.closed = set()
        self.nodes = 0
        self.done = False
        self.result = None

        heapq.heappush(self.open, (self._heuristic(start_state), 0, start_state))

    def _heuristic(self, state):
        pos = state[self.target_robot_i]
        v = self.h[pos]
        return 0 if v >= 10**9 else v

    def _is_goal(self, state):
        return state[self.target_robot_i] == self.target_idx

    def step(self, budget_ms):
        if self.done:
            return

        t0 = pygame.time.get_ticks()
        while self.open:
            if self.nodes >= SOLVER_MAX_NODES:
                self.done = True
                self.result = None
                return
            if pygame.time.get_ticks() - t0 >= budget_ms:
                return

            f, gcur, state = heapq.heappop(self.open)
            if state in self.closed:
                continue
            self.closed.add(state)
            self.nodes += 1

            if self._is_goal(state):
                self.done = True
                self.result = gcur
                return

            for ri in range(len(state)):
                for d in range(4):
                    nxt = self.board.slide(state, ri, d)
                    if nxt == state[ri]:
                        continue
                    new_state = list(state)
                    new_state[ri] = nxt
                    new_state = tuple(new_state)

                    ng = gcur + 1
                    old = self.g.get(new_state)
                    if old is None or ng < old:
                        self.g[new_state] = ng
                        nf = ng + self._heuristic(new_state)
                        heapq.heappush(self.open, (nf, ng, new_state))

        self.done = True
        self.result = None


# =========================
#            UI
# =========================
class Button:
    def __init__(self, rect, text):
        self.rect = pygame.Rect(rect)
        self.text = text

    def hit(self, pos):
        return self.rect.collidepoint(pos)

    def draw(self, surf, font, hover=False):
        color = (70, 70, 80) if not hover else (95, 95, 110)
        pygame.draw.rect(surf, color, self.rect, border_radius=10)
        pygame.draw.rect(surf, (130, 130, 150), self.rect, width=2, border_radius=10)
        label = font.render(self.text, True, (240, 240, 245))
        surf.blit(label, label.get_rect(center=self.rect.center))


def draw_symbol(surf, sym, center, size, color):
    x, y = center
    s = size
    if sym == "CIRCLE":
        pygame.draw.circle(surf, color, center, s)
    elif sym == "SQUARE":
        r = pygame.Rect(x - s, y - s, 2*s, 2*s)
        pygame.draw.rect(surf, color, r)
    elif sym == "DIAMOND":
        pts = [(x, y - s), (x + s, y), (x, y + s), (x - s, y)]
        pygame.draw.polygon(surf, color, pts)
    elif sym == "TRI":
        pts = [(x, y - s), (x + s, y + s), (x - s, y + s)]
        pygame.draw.polygon(surf, color, pts)


# =========================
#           GAME
# =========================
class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Ricochet Robots (Solo) - Quadrants 16x16")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Arial", 20, bold=True)
        self.small = pygame.font.SysFont("Arial", 16)
        self.big = pygame.font.SysFont("Arial", 24, bold=True)
        self.quad_font = pygame.font.SysFont("Arial", 64, bold=True)

        # assets
        self.robot_imgs = {}
        self.target_imgs = {}
        self._load_assets()

        # state
        self.board = None
        self.W = 16
        self.H = 16

        self.robots = [0] * 5
        self.selected_robot = 0
        self.moves = 0
        self.manche_start = None

        self.remaining_targets = []
        self.current_target = None

        # solver cache
        self.h_tables = {}
        self.hint_cache = {}
        self.solver = None
        self.solver_key = None
        self.hint_value = None
        self.anim_move = None

        # manche status / overlay
        self.round_won = False
        self.round_result = None
        self.modal_panel = None
        self.btn_modal_retry = None
        self.btn_modal_next = None

        # UI rects (créés après init board)
        self.btn_backup = None
        self.btn_next = None
        self.btn_new = None
        self.target_rect = None
        self.moves_rect = None
        self.hint_rect = None

        self._new_game(first_time=True)

    def _load_assets(self):
        # robots
        for i, (name, _) in enumerate(ROBOT_COLORS):
            self.robot_imgs[i] = safe_load_png(os.path.join(ROBOTS_DIR, f"{name.lower()}.png"))

        # targets (4 couleurs possibles)
        for ci, cname in enumerate(TARGET_COLORS):
            for sym in SYMBOLS:
                file_sym = SYMBOL_TO_FILE.get(sym, sym.lower())
                self.target_imgs[(ci, sym)] = safe_load_png(os.path.join(TARGETS_DIR, f"{cname.lower()}_{file_sym}.png"))

    # ---------- init board ----------
    def _generate_board_dict(self):
        # seed différent à chaque new game
        seed = int(time.time() * 1000) ^ random.getrandbits(32)
        return build_board(seed=seed, write_file=True)  # écrit board_full.json aussi (debug)

    def _new_game(self, first_time=False):
        data = self._generate_board_dict()
        self.board = Board.from_full_dict(data)
        self.W, self.H = self.board.W, self.board.H
        if not self.board.targets:
            # Safety: avoid recursion if quadrants contain no targets (debug boards).
            for idx in range(self.board.N):
                if not self.board.blocked[idx]:
                    self.board.targets.append({"idx": idx, "color_i": 0, "symbol": "CIRCLE"})
                    break

        # create window once (size depends on grid)
        w = MARGIN * 2 + self.W * TILE
        h = MARGIN * 2 + TOPBAR_H + self.H * TILE
        if first_time:
            self.screen = pygame.display.set_mode((w, h))
        else:
            # recreate if needed
            self.screen = pygame.display.set_mode((w, h))

        # UI layout (no overlap)
        self.btn_backup = Button((MARGIN, MARGIN, 130, 40), "BACKUP")
        self.btn_next = Button((MARGIN + 140, MARGIN, 130, 40), "NEXT")
        self.btn_new = Button((MARGIN + 280, MARGIN, 130, 40), "NEW GAME")

        self.target_rect = pygame.Rect(MARGIN, MARGIN + 56, 54, 54)
        self.moves_rect = pygame.Rect(MARGIN + 70, MARGIN + 56, 210, 54)
        self.hint_rect = pygame.Rect(w - MARGIN - 200, MARGIN + 66, 200, 40)
        self._setup_modal(w, h)

        # targets
        self.remaining_targets = list(self.board.targets)
        random.shuffle(self.remaining_targets)

        # robots positions
        self.robots = self._random_robot_positions()
        self._start_new_manche(keep_positions=True)

    def _setup_modal(self, w, h):
        panel_w, panel_h = 420, 220
        self.modal_panel = pygame.Rect(0, 0, panel_w, panel_h)
        self.modal_panel.center = (w // 2, MARGIN + TOPBAR_H + (self.H * TILE) // 2)

        self.btn_modal_retry = Button((0, 0, 170, 44), "BACKUP (rejouer)")
        self.btn_modal_next = Button((0, 0, 170, 44), "NEXT MANCHE")

        y = self.modal_panel.bottom - 60
        cx = self.modal_panel.centerx
        self.btn_modal_retry.rect.topleft = (cx - 180, y)
        self.btn_modal_next.rect.topleft = (cx + 10, y)

    def _random_robot_positions(self):
        used = set()
        res = []
        for _ in range(5):
            for _ in range(8000):
                idx = random.randrange(self.board.N)
                if self.board.blocked[idx]:
                    continue
                if idx in used:
                    continue
                used.add(idx)
                res.append(idx)
                break
        return res

    def _start_new_manche(self, keep_positions=True):
        if not self.remaining_targets:
            # recycle targets on the same board without touching robot positions
            self.remaining_targets = list(self.board.targets)
            random.shuffle(self.remaining_targets)

        self.round_won = False
        self.round_result = None

        self.current_target = self.remaining_targets.pop()
        self.selected_robot = self.current_target["color_i"]  # 0..3
        self.manche_start = list(self.robots)
        self.moves = 0

        self._start_or_update_solver()

    def _backup(self):
        self.robots = list(self.manche_start)
        self.moves = 0
        self._start_or_update_solver()

    def _on_round_won(self):
        # freeze solver and display overlay
        self.round_won = True
        self.round_result = {
            "moves": self.moves,
            "target": self.current_target,
        }
        self.solver = None

    def _on_modal_next(self):
        self.round_won = False
        self.round_result = None
        self._start_new_manche(keep_positions=True)

    def _on_modal_backup(self):
        self.round_won = False
        self.round_result = None
        self._backup()

    # ---------- solver ----------
    def _solver_key_now(self):
        t = self.current_target
        return (tuple(self.robots), (t["idx"], t["color_i"], t["symbol"]))

    def _get_h_table(self, target_idx):
        ht = self.h_tables.get(target_idx)
        if ht is None:
            ht = relaxed_dist_table(self.board, target_idx)
            self.h_tables[target_idx] = ht
        return ht

    def _start_or_update_solver(self):
        key = self._solver_key_now()
        self.solver_key = key

        if key in self.hint_cache:
            self.hint_value = self.hint_cache[key]
            self.solver = None
            return

        t = self.current_target
        start_state = tuple(self.robots)
        htab = self._get_h_table(t["idx"])
        self.solver = IncrementalAStar(self.board, start_state, t["color_i"], t["idx"], htab)
        self.hint_value = None

    def _solver_tick(self, budget_ms):
        if self.solver is None:
            return
        if self.solver_key != self._solver_key_now():
            self._start_or_update_solver()
            return
        self.solver.step(budget_ms)
        if self.solver.done:
            self.hint_value = self.solver.result
            self.hint_cache[self.solver_key] = self.hint_value
            self.solver = None

    # ---------- coords ----------
    def board_origin(self):
        return (MARGIN, MARGIN + TOPBAR_H)

    def cell_rect(self, x, y):
        ox, oy = self.board_origin()
        return pygame.Rect(ox + x * TILE, oy + y * TILE, TILE, TILE)

    def pos_to_cell(self, pos):
        ox, oy = self.board_origin()
        px, py = pos[0] - ox, pos[1] - oy
        if px < 0 or py < 0:
            return None
        x = px // TILE
        y = py // TILE
        if 0 <= x < self.W and 0 <= y < self.H:
            return int(x), int(y)
        return None

    def _blit_scaled_center(self, surf, img, rect, pad=6):
        if img is None:
            return False
        w = rect.width - 2 * pad
        h = rect.height - 2 * pad
        if w <= 2 or h <= 2:
            return False
        try:
            scaled = pygame.transform.smoothscale(img, (w, h))
            surf.blit(scaled, scaled.get_rect(center=rect.center))
            return True
        except Exception:
            return False

    # ---------- moves ----------
    def _valid_stops_for_selected(self):
        state = tuple(self.robots)
        ri = self.selected_robot
        stops = []
        for d in range(4):
            s = self.board.slide(state, ri, d)
            if s != state[ri]:
                stops.append(s)
        return stops

    def _try_click_move_to(self, dest_idx):
        if self.anim_move:
            return  # ignore inputs while animating
        state = tuple(self.robots)
        src = state[self.selected_robot]
        if dest_idx == src:
            return

        sx, sy = self.board.idx_to_xy[src]
        dx, dy = self.board.idx_to_xy[dest_idx]
        if sx != dx and sy != dy:
            return

        if sx == dx:
            d = 2 if dy > sy else 0
        else:
            d = 1 if dx > sx else 3

        stop = self.board.slide(state, self.selected_robot, d)
        if stop == dest_idx and stop != src:
            # start animation before committing
            sx, sy = self.board.idx_to_xy[src]
            dx, dy = self.board.idx_to_xy[stop]
            sc = self.cell_rect(sx, sy).center
            dc = self.cell_rect(dx, dy).center
            self.robots[self.selected_robot] = stop
            self.moves += 1
            self.anim_move = {
                "robot": self.selected_robot,
                "start": sc,
                "end": dc,
                "t0": pygame.time.get_ticks(),
                "dur": 180,  # ms
            }
            self._start_or_update_solver()

            t = self.current_target
            if self.robots[t["color_i"]] == t["idx"]:
                self._on_round_won()

    # ---------- draw ----------
    def _draw_center_block(self):
        if not self.board.center_block_cells:
            return
        xs = [self.board.idx_to_xy[i][0] for i in self.board.center_block_cells]
        ys = [self.board.idx_to_xy[i][1] for i in self.board.center_block_cells]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        tl = self.cell_rect(minx, miny)
        big = pygame.Rect(tl.x, tl.y, (maxx - minx + 1) * TILE, (maxy - miny + 1) * TILE)
        pygame.draw.rect(self.screen, (10, 10, 12), big)

    def _draw_quadrant_labels(self):
        picks = self.board.meta.get("picks")
        if not picks:
            return
        qsize = self.board.W // 2
        ox, oy = self.board_origin()
        slots = {
            "NW": (0, 0),
            "NE": (qsize, 0),
            "SW": (0, qsize),
            "SE": (qsize, qsize),
        }
        for slot, (sx, sy) in slots.items():
            pick = picks.get(slot)
            if not pick:
                continue
            label = str(pick.get("quad", ""))
            if not label:
                continue
            cx = ox + (sx + qsize / 2) * TILE
            cy = oy + (sy + qsize / 2) * TILE
            surf = self.quad_font.render(label, True, (220, 220, 235))
            surf.set_alpha(70)
            self.screen.blit(surf, surf.get_rect(center=(cx, cy)))

    def _draw_win_modal(self):
        if not self.round_won or self.modal_panel is None:
            return

        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        pygame.draw.rect(self.screen, (28, 28, 36), self.modal_panel, border_radius=14)
        pygame.draw.rect(self.screen, (80, 80, 110), self.modal_panel, width=2, border_radius=14)

        cx, cy = self.modal_panel.center
        title = self.big.render("Manche gagnée !", True, (240, 240, 245))
        self.screen.blit(title, title.get_rect(center=(cx, self.modal_panel.top + 36)))

        moves = self.round_result["moves"] if self.round_result else self.moves
        info = self.font.render(f"Coups joués : {moves}", True, (225, 225, 235))
        self.screen.blit(info, info.get_rect(center=(cx, self.modal_panel.top + 84)))

        mouse = pygame.mouse.get_pos()
        self.btn_modal_retry.draw(self.screen, self.font, hover=self.btn_modal_retry.hit(mouse))
        self.btn_modal_next.draw(self.screen, self.font, hover=self.btn_modal_next.hit(mouse))

    def draw(self):
        self.screen.fill((18, 18, 22))
        pygame.draw.rect(self.screen, (25, 25, 32), (0, 0, self.screen.get_width(), TOPBAR_H))

        mouse = pygame.mouse.get_pos()

        # buttons
        self.btn_backup.draw(self.screen, self.font, hover=self.btn_backup.hit(mouse))
        self.btn_next.draw(self.screen, self.font, hover=self.btn_next.hit(mouse))
        self.btn_new.draw(self.screen, self.font, hover=self.btn_new.hit(mouse))

        # target preview (image only)
        pygame.draw.rect(self.screen, (40, 40, 52), self.target_rect, border_radius=10)
        pygame.draw.rect(self.screen, (120, 120, 150), self.target_rect, width=2, border_radius=10)
        t = self.current_target
        img = self.target_imgs.get((t["color_i"], t["symbol"]))
        if not self._blit_scaled_center(self.screen, img, self.target_rect, pad=6):
            col = ROBOT_COLORS[t["color_i"]][1]
            draw_symbol(self.screen, t["symbol"], self.target_rect.center, 16, col)

        # moves
        pygame.draw.rect(self.screen, (40, 40, 52), self.moves_rect, border_radius=10)
        pygame.draw.rect(self.screen, (120, 120, 150), self.moves_rect, width=2, border_radius=10)
        mv = self.big.render(f"MOVES: {self.moves}", True, (240, 240, 245))
        self.screen.blit(mv, mv.get_rect(center=self.moves_rect.center))

        # hint
        hover_hint = self.hint_rect.collidepoint(mouse)
        pygame.draw.rect(self.screen, (40, 40, 52), self.hint_rect, border_radius=10)
        pygame.draw.rect(self.screen, (120, 120, 150), self.hint_rect, width=2, border_radius=10)
        if hover_hint:
            if self.solver is not None and self.hint_value is None:
                txt = "Min: ..."
            else:
                txt = "Min: ?" if self.hint_value is None else f"Min: {self.hint_value}"
        else:
            txt = "Hint (hover)"
        label = self.font.render(txt, True, (235, 235, 245))
        self.screen.blit(label, label.get_rect(center=self.hint_rect.center))

        # grid (skip blocked cells, drawn as one block)
        for idx in range(self.board.N):
            x, y = self.board.idx_to_xy[idx]
            r = self.cell_rect(x, y)
            if self.board.blocked[idx]:
                continue
            pygame.draw.rect(self.screen, (22, 22, 28), r)
            pygame.draw.rect(self.screen, (30, 30, 38), r, width=1)

        self._draw_center_block()
        self._draw_quadrant_labels()

        # highlight valid stops for selected robot (super utile)
        for s in self._valid_stops_for_selected():
            x, y = self.board.idx_to_xy[s]
            r = self.cell_rect(x, y)
            pygame.draw.rect(self.screen, (60, 60, 80), r, width=3)

        # targets on board
        for tgt in self.board.targets:
            if self.board.blocked[tgt["idx"]]:
                continue
            x, y = self.board.idx_to_xy[tgt["idx"]]
            r = self.cell_rect(x, y)
            img = self.target_imgs.get((tgt["color_i"], tgt["symbol"]))
            if not self._blit_scaled_center(self.screen, img, r, pad=10):
                col = ROBOT_COLORS[tgt["color_i"]][1]
                draw_symbol(self.screen, tgt["symbol"], r.center, TILE // 2 - 14, col)

        # walls (white) + remove internal walls inside blocked block
        wall_col = (245, 245, 245)
        thick = 4
        for idx in range(self.board.N):
            x, y = self.board.idx_to_xy[idx]
            r = self.cell_rect(x, y)
            mask = self.board.walls[idx]

            def neighbor_blocked(d):
                dx, dy = DIRS[d]
                nx, ny = x + dx, y + dy
                if not self.board.in_bounds(nx, ny):
                    return False
                nidx = self.board.xy_to_idx(nx, ny)
                return self.board.blocked[idx] and self.board.blocked[nidx]

            # U
            if (mask & 1) and not neighbor_blocked(0):
                pygame.draw.line(self.screen, wall_col, r.topleft, r.topright, thick)
            # R
            if (mask & 2) and not neighbor_blocked(1):
                pygame.draw.line(self.screen, wall_col, r.topright, r.bottomright, thick)
            # D
            if (mask & 4) and not neighbor_blocked(2):
                pygame.draw.line(self.screen, wall_col, r.bottomleft, r.bottomright, thick)
            # L
            if (mask & 8) and not neighbor_blocked(3):
                pygame.draw.line(self.screen, wall_col, r.topleft, r.bottomleft, thick)

        # robots
        for i, (name, col) in enumerate(ROBOT_COLORS):
            idx = self.robots[i]
            x, y = self.board.idx_to_xy[idx]
            r = self.cell_rect(x, y)
            center = r.center
            anim = self.anim_move
            if anim and anim.get("robot") == i:
                t = (pygame.time.get_ticks() - anim["t0"]) / anim["dur"]
                if t < 0:
                    t = 0
                if t >= 1.0:
                    t = 1.0
                sx, sy = anim["start"]
                ex, ey = anim["end"]
                center = (int(sx + (ex - sx) * t), int(sy + (ey - sy) * t))
                if t >= 1.0:
                    self.anim_move = None
            img = self.robot_imgs.get(i)
            r_for_draw = r.copy()
            r_for_draw.center = center
            if not self._blit_scaled_center(self.screen, img, r_for_draw, pad=6):
                pygame.draw.circle(self.screen, col, center, TILE // 2 - 7)

            if i == self.selected_robot:
                pygame.draw.rect(self.screen, (250, 250, 250), r.inflate(-6, -6), width=3, border_radius=10)

        self._draw_win_modal()

        info = self.small.render("Click robot to select. Click highlighted stop cell (or exact stop) to move.", True, (200, 200, 210))
        self.screen.blit(info, (MARGIN, self.screen.get_height() - 26))

    # ---------- loop ----------
    def run(self):
        while True:
            self.clock.tick(FPS)
            mouse = pygame.mouse.get_pos()
            hover_hint = self.hint_rect.collidepoint(mouse)

            if not self.round_won:
                self._solver_tick(SOLVER_BUDGET_MS_HOVER if hover_hint else SOLVER_BUDGET_MS_IDLE)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.round_won:
                        if self.btn_modal_retry and self.btn_modal_retry.hit(mouse):
                            self._on_modal_backup()
                            continue
                        if self.btn_modal_next and self.btn_modal_next.hit(mouse):
                            self._on_modal_next()
                            continue
                        if self.btn_new.hit(mouse):
                            self._new_game()
                            continue
                        # ignore clicks elsewhere during modal
                        continue

                    if self.btn_backup.hit(mouse):
                        self._backup()
                        continue
                    if self.btn_next.hit(mouse):
                        self._start_new_manche(keep_positions=True)
                        continue
                    if self.btn_new.hit(mouse):
                        self._new_game()
                        continue

                    cell = self.pos_to_cell(mouse)
                    if cell is not None:
                        x, y = cell
                        idx = self.board.xy_to_idx(x, y)

                        # click robot -> select
                        for i, rp in enumerate(self.robots):
                            if rp == idx:
                                self.selected_robot = i
                                break
                        else:
                            if not self.board.blocked[idx]:
                                self._try_click_move_to(idx)

                if event.type == pygame.KEYDOWN:
                    if self.round_won:
                        if event.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_n):
                            self._on_modal_next()
                        elif event.key in (pygame.K_r, pygame.K_b):
                            self._on_modal_backup()
                        elif event.key == pygame.K_ESCAPE:
                            # escape cancels modal but keeps same manche state
                            self.round_won = False
                        continue

                    if event.key == pygame.K_r:
                        self._backup()
                    elif event.key == pygame.K_n:
                        self._start_new_manche(keep_positions=True)

            self.draw()
            pygame.display.flip()


if __name__ == "__main__":
    Game().run()
