import sys, os, time, math
import numpy as np
import pygame
from svgpathtools import svg2paths2, Path

# ============================
# Visual & Overlay Settings
# ============================
TARGET_FPS = 60
BG_COLOR = (8, 10, 12)
FULLSCREEN = False
WINDOW_SIZE = None  # None -> desktop; or (1920,1080)
MONITOR_INDEX = 0  # 0 = primary, 1 = secondary, etc.
INSET_FRACTION = 0.88  # fit logo within this fraction of screen (padding)

# Lissajous-on-outline (adjustable at runtime)
POINTS_ON_PATH = 5000  # total samples across all contours (lower on slow PCs)
GLOW_POINTS = 16  # number of moving “heads”
GLOW_WIDTH = 10  # +/- pixels along index around each head
DECAY = 0.90  # trail persistence
A_FREQ = 3  # Lissajous A
B_FREQ = 4  # Lissajous B
SPEED = 0.25  # animation speed multiplier

# Lightweight CRT-y overlay (fast)
SCANLINE_STRENGTH = 0.55
SCANLINE_SPEED = 0.80
RGB_SPLIT_PIXELS = 1.2


# ============================
# Small helpers
# ============================
def surface_from_numpy_rgb(arr_u8):
    return pygame.image.frombuffer(
        arr_u8.tobytes(), (arr_u8.shape[1], arr_u8.shape[0]), "RGB"
    )


def hsv_to_rgb(h, s, v):
    i = (h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = np.mod(i, 6)
    out = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.float32)
    m = i_mod == 0
    out[m] = np.stack([v[m], t[m], p[m]], -1)
    m = i_mod == 1
    out[m] = np.stack([q[m], v[m], p[m]], -1)
    m = i_mod == 2
    out[m] = np.stack([p[m], v[m], t[m]], -1)
    m = i_mod == 3
    out[m] = np.stack([p[m], q[m], v[m]], -1)
    m = i_mod == 4
    out[m] = np.stack([t[m], p[m], v[m]], -1)
    m = i_mod == 5
    out[m] = np.stack([v[m], p[m], q[m]], -1)
    return out


def apply_light_overlays(rgb_u8, t):
    H, W, _ = rgb_u8.shape
    rgb = rgb_u8.astype(np.float32)
    # scanlines
    y = np.arange(H, dtype=np.float32)
    line_mask = 0.5 + 0.5 * np.sign(np.sin((y + t * SCANLINE_SPEED * 120.0) * math.pi))
    line_mask = 1.0 - SCANLINE_STRENGTH * (1.0 - line_mask)
    rgb *= line_mask[:, None, None]
    # tiny RGB split
    shift = int(RGB_SPLIT_PIXELS * (1.0 + 0.3 * math.sin(t * 1.3)))
    r = np.roll(rgb[..., 0], shift, axis=1)
    g = rgb[..., 1]
    b = np.roll(rgb[..., 2], -shift, axis=1)
    out = np.stack([r, g, b], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================
# SVG path sampling
# ============================
def load_svg_paths(svg_path):
    # svg2paths2 returns (paths, attributes, svg_attributes)
    paths, attrs, svg_attr = svg2paths2(svg_path)
    # Filter out empty paths
    paths = [p for p in paths if isinstance(p, Path) and len(p) > 0]
    if not paths:
        raise ValueError("No <path> elements found in SVG.")
    return paths, svg_attr


def bbox_of_paths(paths):
    xmin = ymin = +1e18
    xmax = ymax = -1e18
    for p in paths:
        bx0, bx1, by0, by1 = p.bbox()  # (xmin, xmax, ymin, ymax)
        xmin = min(xmin, bx0)
        xmax = max(xmax, bx1)
        ymin = min(ymin, by0)
        ymax = max(ymax, by1)
    return xmin, xmax, ymin, ymax


def sample_paths_uniform(paths, total_samples):
    """Sample all paths by arc length proportionally to each path length."""
    lengths = np.array([p.length(error=1e-4) for p in paths], dtype=np.float64)
    total_len = float(np.sum(lengths))
    if total_len <= 1e-9:
        # degenerate — sample first path uniformly in parameter
        p = paths[0]
        ts = np.linspace(0, 1, total_samples, endpoint=False)
        pts = np.array([p.point(t) for t in ts], dtype=np.complex128)
        return pts

    samples_per = np.maximum(
        1, np.round(total_samples * (lengths / total_len)).astype(int)
    )
    # adjust to exact total count
    diff = total_samples - int(samples_per.sum())
    if diff != 0:
        # give/take the remainder to the longest path(s)
        order = np.argsort(-lengths)
        for i in range(abs(diff)):
            samples_per[order[i % len(order)]] += 1 if diff > 0 else -1

    pts = []
    for p, n in zip(paths, samples_per):
        L = p.length(error=1e-4)
        if n <= 0 or L <= 1e-12:
            continue
        ss = np.linspace(0, L, n, endpoint=False)
        # try arclength inverse if available
        try:
            ts = [p.ilength(s) for s in ss]
        except AttributeError:
            # fallback: uniform t (ok for short segments)
            ts = np.linspace(0, 1, n, endpoint=False)
        pts.extend([p.point(t) for t in ts])
    return np.array(pts, dtype=np.complex128)


def normalize_to_screen(
    pts_complex, screen_w, screen_h, inset=INSET_FRACTION, bbox=None
):
    """Map SVG coords to screen with aspect-correct scale and centering."""
    xs = pts_complex.real
    ys = pts_complex.imag
    if bbox is None:
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
    else:
        xmin, xmax, ymin, ymax = bbox
    w = max(1e-6, xmax - xmin)
    h = max(1e-6, ymax - ymin)

    target_w = screen_w * inset
    target_h = screen_h * inset
    s = min(target_w / w, target_h / h)  # uniform scale

    # center
    X = (xs - xmin) * s + (screen_w - w * s) * 0.5
    Y = (ys - ymin) * s + (screen_h - h * s) * 0.5
    # SVG and Pygame both use y-down, so no flip needed
    return np.stack([X, Y], axis=1).astype(np.float32)


# ============================
# Visualizer
# ============================
class LineVisualizer:
    def __init__(self, svg_path, W, H):
        self.W, self.H = W, H
        # parse & sample
        paths, _svgattr = load_svg_paths(svg_path)
        bbox = bbox_of_paths(paths)
        pts = sample_paths_uniform(paths, POINTS_ON_PATH)
        XY = normalize_to_screen(pts, W, H, inset=INSET_FRACTION, bbox=bbox)

        # store as integers for fast indexing; keep float copy for future
        self.path_xy_f = XY
        self.path_x = np.clip(np.round(XY[:, 0]).astype(int), 0, W - 1)
        self.path_y = np.clip(np.round(XY[:, 1]).astype(int), 0, H - 1)
        self.N = len(self.path_x)

        # neon buffer
        self.buf = np.zeros((H, W, 3), dtype=np.float32)

    def resample(self, svg_path):
        # allows re-load if needed (kept simple)
        self.__init__(svg_path, self.W, self.H)

    def frame(self, t):
        self.buf *= DECAY

        # heads with staggered phases
        k = np.arange(GLOW_POINTS, dtype=np.float32)
        phase = (k / max(1, GLOW_POINTS)) * 2 * np.pi

        u = (
            np.sin(A_FREQ * (t * SPEED) + phase) * 0.5
            + 0.5
            + 0.12 * np.sin(B_FREQ * (t * SPEED * 1.11) + phase * 1.7)
        )
        base_idx = (u * self.N).astype(int) % self.N

        band = np.arange(-GLOW_WIDTH, GLOW_WIDTH + 1, dtype=int)[None, :]
        idxs = (base_idx[:, None] + band) % self.N
        idxs = idxs.reshape(-1)

        X = self.path_x[idxs]
        Y = self.path_y[idxs]

        # color gradient per head along hue
        hue = ((k[:, None] / max(1, GLOW_POINTS)) + 0.12 * np.sin(t * 0.2))[..., None]
        hue = np.repeat(hue, band.shape[1], axis=1).reshape(-1, 1)
        S = np.full_like(hue, 0.95)
        V = np.full_like(hue, 1.0)
        col = hsv_to_rgb(hue, S, V)[..., 0, :]  # (n,3)

        self.buf[Y, X, :] = np.maximum(self.buf[Y, X, :], col)

        out = np.clip(self.buf * 255.0, 0, 255).astype(np.uint8)
        out = np.maximum(out, np.array(BG_COLOR, dtype=np.uint8)[None, None, :])
        return out


# ============================
# Pygame bootstrap
# ============================
def list_monitors():
    """Show available monitors/displays"""
    pygame.init()

    print(f"\n=== Display Information ===")

    # Try to get all displays (pygame 2.0+)
    try:
        displays = pygame.display.get_desktop_sizes()
        print(f"Total monitors detected: {len(displays)}")
        for i, (w, h) in enumerate(displays):
            print(f"  Monitor {i}: {w}x{h}")
    except AttributeError:
        info = pygame.display.Info()
        print(f"Monitor detection not available (old pygame version)")
        print(f"Primary monitor: {info.current_w}x{info.current_h}")

    print(f"\nCurrent Settings:")
    print(f"  MONITOR_INDEX: {MONITOR_INDEX}")
    print(f"  WINDOW_SIZE: {WINDOW_SIZE}")
    print(f"  FULLSCREEN: {FULLSCREEN}\n")

    pygame.quit()


def main():
    if len(sys.argv) < 2:
        print("Usage: python line_visualizer.py path/to/image.svg")
        print("       python line_visualizer.py --monitors  (show monitor info)")
        sys.exit(1)

    # Show monitor info if requested
    if sys.argv[1] == "--monitors":
        list_monitors()
        sys.exit(0)

    svg_path = sys.argv[1]

    pygame.init()

    # Monitor Selection (for multi-monitor setups)
    displays = []
    try:
        # Get all display sizes (pygame 2.0+)
        displays = pygame.display.get_desktop_sizes()
        print(f"Detected {len(displays)} monitor(s): {displays}")
    except AttributeError:
        # Fallback for older pygame versions
        info = pygame.display.Info()
        displays = [(info.current_w, info.current_h)]
        print(f"Single monitor detected: {displays[0]}")

    # Calculate position for target monitor
    if MONITOR_INDEX >= len(displays):
        print(
            f"Warning: MONITOR_INDEX={MONITOR_INDEX} exceeds available monitors ({len(displays)}). Using monitor 0."
        )
        monitor_idx = 0
    else:
        monitor_idx = MONITOR_INDEX

    # Calculate X offset by summing widths of previous monitors
    x_offset = sum(displays[i][0] for i in range(monitor_idx))

    # Set window position before creating the window
    if monitor_idx > 0:
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x_offset},0"
        print(f"Positioning window on monitor {monitor_idx} at x={x_offset}")

    # Window flags
    flags = pygame.FULLSCREEN if FULLSCREEN else 0

    # Window size
    if WINDOW_SIZE is None:
        # Auto-detect: use target monitor's size
        if FULLSCREEN:
            screen = pygame.display.set_mode((0, 0), flags)
        else:
            # Use the target monitor's dimensions
            target_size = displays[monitor_idx]
            screen = pygame.display.set_mode(target_size, flags)
    else:
        screen = pygame.display.set_mode(WINDOW_SIZE, flags)

    pygame.display.set_caption("Line Visualizer — SVG Path Lissajous")
    clock = pygame.time.Clock()
    W, H = screen.get_size()

    viz = LineVisualizer(svg_path, W, H)

    global GLOW_WIDTH, GLOW_POINTS, A_FREQ, B_FREQ
    t0 = time.time()
    running = True
    while running:
        now = time.time()
        t = now - t0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                elif e.key == pygame.K_LEFT:
                    GLOW_WIDTH = max(1, GLOW_WIDTH - 1)
                elif e.key == pygame.K_RIGHT:
                    GLOW_WIDTH = min(24, GLOW_WIDTH + 1)
                elif e.key == pygame.K_UP:
                    GLOW_POINTS = min(32, GLOW_POINTS + 1)
                elif e.key == pygame.K_DOWN:
                    GLOW_POINTS = max(1, GLOW_POINTS - 1)
                elif e.key == pygame.K_LEFTBRACKET:  # '['
                    A_FREQ = max(1, A_FREQ - 1)
                elif e.key == pygame.K_RIGHTBRACKET:  # ']'
                    A_FREQ = min(16, A_FREQ + 1)
                elif e.key == pygame.K_SEMICOLON:  # ';'
                    B_FREQ = max(1, B_FREQ - 1)
                elif e.key == pygame.K_QUOTE:  # '\''
                    B_FREQ = min(16, B_FREQ + 1)
                elif e.key == pygame.K_s:
                    pygame.image.save(screen, f"screenshot_{int(now)}.png")
                elif e.key == pygame.K_r:
                    viz.resample(svg_path)

        frame = viz.frame(t)
        frame = apply_light_overlays(frame, t)

        surf = surface_from_numpy_rgb(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(TARGET_FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
