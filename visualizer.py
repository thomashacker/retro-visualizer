import os, sys, time, math, random
import numpy as np
import pygame

# ============================
# Global Look & Overlay Params
# ============================
SCANLINE_STRENGTH = 0.6
SCANLINE_SPEED = 0.8
RGB_SPLIT_PIXELS = 1.6
JITTER_AMOUNT = 1.6
JITTER_SPEED = 2.2
NOISE_STRENGTH = 0.05
VIGNETTE_STRENGTH = 0.35
BLOOM_STRENGTH = 0.45
BLOOM_DOWNSCALE = 4
NEON_GLOW = 1.6
SATURATION_BOOST = 1.25
BG_COLOR = (8, 10, 12)
FULLSCREEN = False
MONITOR_INDEX = 0
AUTO_CYCLE_SECS = 40
TARGET_FPS = 60


# ============================
# Util: color & display helpers
# ============================
def hsv_to_rgb(h, s, v):
    i = (h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = np.mod(i, 6)
    out = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.float32)
    mask = i_mod == 0
    out[mask] = np.stack([v[mask], t[mask], p[mask]], -1)
    mask = i_mod == 1
    out[mask] = np.stack([q[mask], v[mask], p[mask]], -1)
    mask = i_mod == 2
    out[mask] = np.stack([p[mask], v[mask], t[mask]], -1)
    mask = i_mod == 3
    out[mask] = np.stack([p[mask], q[mask], v[mask]], -1)
    mask = i_mod == 4
    out[mask] = np.stack([t[mask], p[mask], v[mask]], -1)
    mask = i_mod == 5
    out[mask] = np.stack([v[mask], p[mask], q[mask]], -1)
    return out


def boost_neon(rgb):
    if SATURATION_BOOST > 1.0:
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[
            ..., None
        ]
        rgb = gray + SATURATION_BOOST * (rgb - gray)
    if NEON_GLOW > 1.0:
        rgb *= NEON_GLOW
    return np.clip(rgb, 0, 1)


def surface_from_numpy(arr_u8):
    return pygame.image.frombuffer(
        arr_u8.tobytes(), (arr_u8.shape[1], arr_u8.shape[0]), "RGB"
    )


# ============================
# Overlays: CRT + VHS bundle
# ============================
def apply_overlays(rgb_u8, t):
    H, W, _ = rgb_u8.shape
    rgb = rgb_u8.astype(np.float32)

    # Scanlines
    y = np.arange(H, dtype=np.float32)
    line_mask = 0.5 + 0.5 * np.sign(np.sin((y + t * SCANLINE_SPEED * 120.0) * math.pi))
    line_mask = 1.0 - SCANLINE_STRENGTH * (1.0 - line_mask)
    rgb *= line_mask[:, None, None]

    # global wobble (fast, no per-scanline loop)
    wobble = int(JITTER_AMOUNT * np.sin(t * JITTER_SPEED + np.sin(t * 1.3) * 0.5))
    if wobble != 0:
        rgb = np.roll(rgb, wobble, axis=1)

    # RGB split
    shift = int(RGB_SPLIT_PIXELS * (1.0 + 0.3 * math.sin(t * 1.3)))
    r = np.roll(rgb[..., 0], shift, axis=1)
    g = rgb[..., 1]
    b = np.roll(rgb[..., 2], -shift, axis=1)
    out = np.stack([r, g, b], axis=-1)

    # Grain
    if NOISE_STRENGTH > 0:
        noise = np.random.randn(H, W, 1).astype(np.float32) * 255.0 * NOISE_STRENGTH
        out += noise

    # (optional tiny vignette for speed; feel free to re-add bloom if you want)
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================
# Preset Generators (vectorized)
# ============================
class PresetBase:
    name = "Base"

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.yy, self.xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.xn = (self.xx / W) * 2 - 1
        self.yn = (self.yy / H) * 2 - 1

    def frame(self, t):
        raise NotImplementedError


class Plasma(PresetBase):
    name = "Plasma"

    def frame(self, t):
        a = self.xn * np.cos(t * 0.3) - self.yn * np.sin(t * 0.3)
        b = self.xn * np.sin(t * 0.3) + self.yn * np.cos(t * 0.3)
        v = (
            np.sin(a * 3.0 + t * 1.4)
            + np.sin(b * 4.0 - t * 1.1)
            + np.sin((a + b) * 2.5 + t * 0.7)
        ) / 3.0
        v = (v + 1) / 2.0
        h = (v + 0.15 * np.sin(t * 0.2)) % 1.0
        s = np.clip(0.85 + 0.15 * np.sin(t * 0.13), 0.7, 1.0)
        v2 = 0.85 + 0.15 * np.sin(t * 0.11 + v * 2.0)
        rgb = hsv_to_rgb(h, s, v2)
        rgb = boost_neon(rgb)
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)


class RibbonWaves(PresetBase):
    name = "Ribbon Waves"

    def frame(self, t):
        r = np.sqrt(self.xn**2 + self.yn**2)
        ang = np.arctan2(self.yn, self.xn)
        bands = np.sin(10 * r - t * 2.2) * 0.5 + 0.5
        hue = (ang / (2 * np.pi) + 0.5 + 0.1 * np.sin(t * 0.5)) % 1.0
        val = 0.6 + 0.4 * bands
        sat = 0.8 - 0.25 * np.cos(ang * 3 + t * 1.5)
        rgb = hsv_to_rgb(hue, np.clip(sat, 0, 1), np.clip(val, 0, 1))
        rgb = boost_neon(rgb)
        return (rgb * 255).astype(np.uint8)


class Metaballs(PresetBase):
    name = "Metaballs"

    def __init__(self, W, H, n=6):
        super().__init__(W, H)
        self.n = n
        rng = np.random.RandomState(7)
        self.ph = rng.rand(n) * 6.283
        self.sp = 0.4 + rng.rand(n) * 0.9
        self.rad = 0.15 + rng.rand(n) * 0.25
        self.dir = rng.rand(n) * 6.283

    def frame(self, t):
        f = np.zeros_like(self.xn)
        for i in range(self.n):
            cx = 0.6 * np.cos(self.dir[i]) * np.cos(t * self.sp[i] + self.ph[i])
            cy = (
                0.6
                * np.sin(self.dir[i])
                * np.sin(t * self.sp[i] * 1.1 + self.ph[i] * 0.7)
            )
            dx = self.xn - cx
            dy = self.yn - cy
            d2 = dx * dx + dy * dy
            f += (self.rad[i] ** 2) / (d2 + 1e-4)
        f = np.clip(f, 0, 2.0)
        h = (f * 0.35 + 0.15 * np.sin(t * 0.4)) % 1.0
        s = np.clip(0.6 + 0.4 * np.sin(f * 2.0 + t * 0.8), 0.5, 1.0)
        v = np.clip(0.5 + 0.5 * np.tanh((f - 0.6) * 2.2), 0.0, 1.0)
        rgb = hsv_to_rgb(h, s, v)
        rgb = boost_neon(rgb)
        return (rgb * 255).astype(np.uint8)


class FlowField(PresetBase):
    name = "Flow Field"

    def __init__(self, W, H, count=3500):
        super().__init__(W, H)
        self.count = count
        rng = np.random.RandomState(1337)
        self.px = rng.rand(count) * W
        self.py = rng.rand(count) * H
        self.h = rng.rand(count)
        self.trails = np.zeros((H, W, 3), dtype=np.float32)

    def vector(self, x, y, t):
        xn = (x / self.W) * 2 - 1
        yn = (y / self.H) * 2 - 1
        a = np.sin(yn * 3.0 + t * 0.7) + np.cos(xn * 2.0 - t * 0.9)
        b = np.cos(xn * 3.7 + t * 0.6) - np.sin(yn * 2.4 - t * 1.1)
        return a * 0.6, b * 0.6

    def frame(self, t):
        self.trails *= 0.92
        for i in range(self.count):
            vx, vy = self.vector(self.px[i], self.py[i], t)
            self.px[i] = (self.px[i] + vx * 2.2) % self.W
            self.py[i] = (self.py[i] + vy * 2.2) % self.H
            c = hsv_to_rgb(
                np.array([[self.h[i]]]) * 1.0, np.array([[0.8]]), np.array([[0.9]])
            )[0, 0]
            x = int(self.px[i])
            y = int(self.py[i])
            if 0 <= x < self.W and 0 <= y < self.H:
                self.trails[y, x, :] = np.maximum(self.trails[y, x, :], c)
        rgb = boost_neon(self.trails)
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)


# ------------ NEW: LissajousNeon with Shapes -------------
class LissajousNeon(PresetBase):
    name = "Lissajous / Shapes"
    SHAPES = ["lissajous", "circle", "heart", "star", "polygon", "spiral", "rose"]

    def __init__(self, W, H, points=3000):
        super().__init__(W, H)
        self.points = points
        self.buffer = np.zeros((H, W, 3), dtype=np.float32)
        self.shape_index = 6  # start with lissajous
        self.params = {
            "star_points": 5,
            "poly_sides": 6,
            "rose_k": 5,  # petals (odd k => k petals, even k => 2k petals)
            "spiral_turns": 2.5,
        }

    # ---- shape parameterizations (normalized to 0..1) ----
    def shape_lissajous(self, u, t):
        a = 3 + int(2 * np.sin(t * 0.3))
        b = 4 + int(2 * np.cos(t * 0.27))
        d = t * 0.25
        x = (np.sin(a * u + d) * 0.7) * 0.48 + 0.5
        y = (np.sin(b * u) * 0.7) * 0.48 + 0.5
        return x, y

    def shape_circle(self, u, t):
        r = 0.42 + 0.03 * np.sin(t * 0.9)
        x = np.cos(u) * r * 1.0 * 0.95 + 0.5
        y = np.sin(u) * r * 1.0 * 0.95 + 0.5
        return x, y

    def shape_heart(self, u, t):
        # classic heart param, normalized
        # x = 16 sin^3 t, y = 13 cos t - 5 cos 2t - 2 cos 3t - cos 4t
        xu = np.sin(u)
        x = 16 * (xu**3)
        y = 13 * np.cos(u) - 5 * np.cos(2 * u) - 2 * np.cos(3 * u) - np.cos(4 * u)
        # normalize to [-1,1] roughly then to [0,1]
        x /= 18.0
        y /= 18.0
        s = 0.45 + 0.03 * np.sin(t * 0.6)
        x = x * s + 0.5
        y = -y * s + 0.5
        return x, y

    def shape_star(self, u, t):
        n = max(3, int(self.params["star_points"]))
        # inner/outer radii
        r1 = 0.18 + 0.02 * np.sin(t * 1.3)
        r2 = 0.42 + 0.03 * np.cos(t * 0.9)
        # map param to piecewise radial polygon
        k = n * 2
        idx = (u / (2 * np.pi) * k) % 1.0
        angles = u
        radii = np.where(idx < 0.5, r2, r1)
        x = np.cos(angles) * radii + 0.5
        y = np.sin(angles) * radii + 0.5
        return x, y

    def shape_polygon(self, u, t):
        n = max(3, int(self.params["poly_sides"]))
        # polygon via rounding angle to nearest vertex direction
        angle = u
        # snap to edges by making radius piecewise-constant between vertex angles
        # use superellipse-style smoothing to keep it pretty
        r = 0.45 + 0.02 * np.sin(t * 0.7)
        k = n
        # superformula-lite: |cos(m u/4)/a|^n + |sin(m u/4)/b|^n
        m = float(k) * 4.0
        a = b = 1.0
        n1 = n2 = 24.0
        s = (
            np.abs(np.cos(m * angle / 4) / a) ** n1
            + np.abs(np.sin(m * angle / 4) / b) ** n2
        ) ** (-1 / float(24))
        rr = r * s
        x = np.cos(angle) * rr + 0.5
        y = np.sin(angle) * rr + 0.5
        return x, y

    def shape_spiral(self, u, t):
        # Archimedean spiral: r = a + b*u  (in revolutions)
        turns = float(self.params["spiral_turns"])
        u2 = u * turns / (2 * np.pi)  # [0,turns)
        a = 0.04
        b = 0.42 / (turns + 0.5)
        r = a + b * u2
        rot = t * 0.3
        x = np.cos(u + rot) * r + 0.5
        y = np.sin(u + rot) * r + 0.5
        return x, y

    def shape_rose(self, u, t):
        # rhodonea: r = cos(k * theta)
        k = max(1, int(self.params["rose_k"]))
        theta = u + 0.3 * np.sin(t * 0.6)
        r = 0.45 * np.abs(np.cos(k * theta))
        x = np.cos(theta) * r + 0.5
        y = np.sin(theta) * r + 0.5
        return x, y

    def _sample_shape(self, t):
        u = np.linspace(0, 2 * np.pi, self.points, endpoint=False)
        shape = self.SHAPES[self.shape_index]
        if shape == "lissajous":
            x, y = self.shape_lissajous(u, t)
        elif shape == "circle":
            x, y = self.shape_circle(u, t)
        elif shape == "heart":
            x, y = self.shape_heart(u, t)
        elif shape == "star":
            x, y = self.shape_star(u, t)
        elif shape == "polygon":
            x, y = self.shape_polygon(u, t)
        elif shape == "spiral":
            x, y = self.shape_spiral(u, t)
        elif shape == "rose":
            x, y = self.shape_rose(u, t)
        else:
            x, y = self.shape_lissajous(u, t)
        return x, y, u

    def randomize(self):
        self.params["star_points"] = random.randint(5, 9)
        self.params["poly_sides"] = random.randint(3, 10)
        self.params["rose_k"] = random.randint(2, 9)
        self.params["spiral_turns"] = random.choice([1.5, 2.0, 2.5, 3.0, 4.0])

    def next_shape(self, d=1):
        self.shape_index = (self.shape_index + d) % len(self.SHAPES)

    def frame(self, t):
        # neon trail buffer
        self.buffer *= 0.90

        x, y, u = self._sample_shape(t)
        X = np.clip((x * self.W).astype(int), 0, self.W - 1)
        Y = np.clip((y * self.H).astype(int), 0, self.H - 1)

        # hue along path, slowly shifting over time
        hue = (u / (2 * np.pi) + 0.08 * np.sin(t * 0.3)) % 1.0
        col = hsv_to_rgb(
            hue[:, None],
            np.full((self.points, 1), 0.95),
            np.full((self.points, 1), 1.0),
        )[:, 0, :]

        # splat to buffer (max for glow accumulation)
        self.buffer[Y, X, :] = np.maximum(self.buffer[Y, X, :], col)

        rgb = boost_neon(self.buffer)
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)


class AuroraBands(PresetBase):
    name = "Aurora Bands"

    def frame(self, t):
        x = self.xn * 1.0 + 0.1 * np.sin(self.yn * 3.0 + t * 0.7)
        v = 0.5 + 0.5 * np.sin(
            6.0 * x + 1.7 * np.sin(t * 0.4) + 0.7 * np.cos(self.yn * 2.0 - t * 0.6)
        )
        h = (
            0.4
            + 0.2 * np.sin(x * 2.0 + t * 0.2)
            + 0.1 * np.sin(self.yn * 4.0 + t * 0.5)
        ) % 1.0
        s = 0.75 + 0.2 * np.sin(t * 0.33)
        rgb = hsv_to_rgb(h, np.clip(s, 0, 1), v)
        rgb = boost_neon(rgb)
        return (rgb * 255).astype(np.uint8)


PRESETS = [Plasma, RibbonWaves, Metaballs, FlowField, LissajousNeon, AuroraBands]


# ============================
# Main
# ============================
def pick_display_and_open():
    pygame.init()
    try:
        displays = pygame.display.get_desktop_sizes()
    except AttributeError:
        info = pygame.display.Info()
        displays = [(info.current_w, info.current_h)]
    idx = MONITOR_INDEX if MONITOR_INDEX < len(displays) else 0
    x_offset = sum(displays[i][0] for i in range(idx))
    if idx > 0:
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x_offset},0"
    flags = pygame.FULLSCREEN if FULLSCREEN else 0
    if FULLSCREEN:
        screen = pygame.display.set_mode((0, 0), flags)
    else:
        screen = pygame.display.set_mode(displays[idx], flags)
    pygame.display.set_caption("Retro Abstract Visuals (XP Vibes) — Shapes")
    return screen


def main():
    screen = pick_display_and_open()
    clock = pygame.time.Clock()
    W, H = screen.get_size()

    base = np.zeros((H, W, 3), dtype=np.uint8)
    base[..., 0] = BG_COLOR[0]
    base[..., 1] = BG_COLOR[1]
    base[..., 2] = BG_COLOR[2]

    gens = [
        (
            P(W, H)
            if P not in (Metaballs, FlowField, LissajousNeon)
            else (
                Metaballs(W, H, 6)
                if P is Metaballs
                else (
                    FlowField(W, H, 3500)
                    if P is FlowField
                    else LissajousNeon(W, H, points=3000)
                )
            )
        )
        for P in PRESETS
    ]
    gi = 4  # start at Lissajous/Shapes
    current = gens[gi]

    t0 = time.time()
    next_auto = t0 + AUTO_CYCLE_SECS
    auto = True

    running = True
    while running:
        now = time.time()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                elif e.key == pygame.K_SPACE:
                    auto = not auto
                elif pygame.K_1 <= e.key <= pygame.K_6:
                    gi = e.key - pygame.K_1
                    current = gens[gi]
                    next_auto = now + AUTO_CYCLE_SECS
                elif gi == 4:  # shape controls only when LissajousNeon is active
                    ln = current  # type: LissajousNeon
                    if e.key == pygame.K_j:
                        ln.next_shape(-1)
                    elif e.key == pygame.K_k:
                        ln.next_shape(+1)
                    elif e.key == pygame.K_r:
                        ln.randomize()
                    elif e.key == pygame.K_UP:
                        ln.points = min(8000, ln.points + 500)
                    elif e.key == pygame.K_DOWN:
                        ln.points = max(500, ln.points - 500)

        if auto and now >= next_auto:
            gi = (gi + 1) % len(gens)
            current = gens[gi]
            next_auto = now + AUTO_CYCLE_SECS

        t = now - t0

        rgb = current.frame(t)
        frame = np.maximum(rgb, base)
        frame = apply_overlays(frame, t)

        surf = surface_from_numpy(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(TARGET_FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
