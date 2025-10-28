import sys, io, time, math, random, os
import numpy as np
import pygame
from PIL import Image, ImageSequence
import cairosvg

# ---------------- Tweakbare Parameter ----------------
SCANLINE_STRENGTH = 0.8  # 0..1
SCANLINE_SPEED = 0.9  # Hz
RGB_SPLIT_PIXELS = 1.8  # Pixelversatz (animiert)
JITTER_AMOUNT = 2.0  # max horizontales Zittern (px)
JITTER_SPEED = 2.5  # Hz
NOISE_STRENGTH = 0.1  # 0..1 (additiv)
VIGNETTE_STRENGTH = 0.4  # 0..1
BLOOM_STRENGTH = 0.6  # 0..1
BLOOM_DOWNSCALE = 4  # 2..8 (größer = weicher/schneller)
NEON_GLOW = 2.0  # 1.0-2.0 (brightness boost for neon effect)
SATURATION_BOOST = 1.3  # 1.0-2.0 (color intensity)
BASE_SCALE = 0.8  # Motiv etwas kleiner für CRT-Ränder
BG_COLOR = (8, 10, 12)  # „Gehäuse-Schwarz"
FIT_MODE = "contain"  # "contain" oder "cover"
# Window Settings
WINDOW_SIZE = None  # None = auto-detect, or tuple like (1920, 1080)
FULLSCREEN = True  # True = start in fullscreen mode
MONITOR_INDEX = 1  # 0 = primary, 1 = secondary, etc.
# -----------------------------------------------------

SUPPORTED_EXT = {".svg", ".png", ".jpg", ".jpeg", ".gif"}


def numpy_from_pygame_surface(surf, with_alpha=True):
    arr_rgb = pygame.surfarray.pixels3d(surf).copy()  # (W,H,3)
    arr_rgb = np.transpose(arr_rgb, (1, 0, 2))  # -> (H,W,3)
    if with_alpha:
        try:
            a = pygame.surfarray.pixels_alpha(surf).copy()
        except:
            a = np.full((surf.get_width(), surf.get_height()), 255, dtype=np.uint8)
        a = np.transpose(a, (1, 0))  # -> (H,W)
        out = np.zeros((arr_rgb.shape[0], arr_rgb.shape[1], 4), dtype=np.uint8)
        out[..., :3] = arr_rgb
        out[..., 3] = a
        return out
    return arr_rgb


def surface_from_numpy(arr):
    # arr: HxWx3 oder HxWx4
    fmt = "RGBA" if arr.shape[2] == 4 else "RGB"
    return pygame.image.frombuffer(
        arr.tobytes(), (arr.shape[1], arr.shape[0]), fmt
    ).convert_alpha()


def svg_to_pil(svg_path, target_w, target_h):
    with open(svg_path, "rb") as f:
        svg_bytes = f.read()
    # Render zunächst groß (Zielgröße), Aspect Ratio übernimmt CairoSVG aus der ViewBox
    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes, output_width=target_w, output_height=target_h
    )
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def pil_to_numpy_rgba(img: Image.Image):
    return np.array(img.convert("RGBA"), dtype=np.uint8)


def fit_size(src_w, src_h, dst_w, dst_h, mode="contain"):
    ar_s, ar_d = src_w / src_h, dst_w / dst_h
    if mode == "cover":
        if ar_s > ar_d:
            # Quelle breiter -> Höhe füllen
            new_h = dst_h
            new_w = int(new_h * ar_s)
        else:
            new_w = dst_w
            new_h = int(new_w / ar_s)
    else:  # contain
        if ar_s > ar_d:
            new_w = dst_w
            new_h = int(new_w / ar_s)
        else:
            new_h = dst_h
            new_w = int(new_h * ar_s)
    return new_w, new_h


class MediaSource:
    """Lädt eine Datei (SVG/PNG/JPG/GIF). Für GIF: Frames + Dauer; für andere: ein einzelner Frame."""

    def __init__(self, path, screen_w, screen_h):
        self.path = path
        self.ext = os.path.splitext(path)[1].lower()
        self.frames = []  # Liste np.uint8 (H,W,4)
        self.durations = []  # ms pro Frame (GIF); bei statisch: [∞]
        self._load(screen_w, screen_h)
        self.total_frames = len(self.frames)
        self.is_animated = self.total_frames > 1

    def _load(self, screen_w, screen_h):
        if self.ext not in SUPPORTED_EXT:
            raise ValueError(f"Nicht unterstütztes Format: {self.ext}")

        if self.ext == ".svg":
            img = svg_to_pil(self.path, screen_w, screen_h)
            self.frames = [pil_to_numpy_rgba(img)]
            self.durations = [10_000_000]  # „unendlich“ groß
            return

        # Rasterformate via PIL
        img = Image.open(self.path)

        if self.ext == ".gif" and getattr(img, "is_animated", False):
            # Alle Frames extrahieren mit Dauer
            for frame in ImageSequence.Iterator(img):
                fr = frame.convert("RGBA")
                self.frames.append(pil_to_numpy_rgba(fr))
                # Dauer in ms; fallback 100ms
                dur = frame.info.get("duration", 100)
                if dur <= 0:
                    dur = 100
                self.durations.append(dur)
        else:
            self.frames = [pil_to_numpy_rgba(img)]
            self.durations = [10_000_000]

    def get_frame(self, index):
        return self.frames[index % self.total_frames]

    def get_duration_ms(self, index):
        return self.durations[index % self.total_frames]


def place_on_canvas(base_rgba, canvas_w, canvas_h):
    """Skaliert das Motiv gemäß BASE_SCALE und FIT_MODE und legt es zentriert auf BG."""
    H0, W0, _ = base_rgba.shape
    target_w = int(canvas_w * BASE_SCALE)
    target_h = int(canvas_h * BASE_SCALE)
    new_w, new_h = fit_size(W0, H0, target_w, target_h, mode=FIT_MODE)

    surf_src = surface_from_numpy(base_rgba)
    surf_scaled = pygame.transform.smoothscale(surf_src, (new_w, new_h))
    rgb_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    rgb_canvas[..., 0] = BG_COLOR[0]
    rgb_canvas[..., 1] = BG_COLOR[1]
    rgb_canvas[..., 2] = BG_COLOR[2]

    tmp_rgb = numpy_from_pygame_surface(surf_scaled, with_alpha=True)
    x0 = (canvas_w - new_w) // 2
    y0 = (canvas_h - new_h) // 2

    sub = rgb_canvas[y0 : y0 + new_h, x0 : x0 + new_w, :]
    a = tmp_rgb[..., 3:4].astype(np.float32) / 255.0
    sub[:] = (
        tmp_rgb[..., :3].astype(np.float32) * a + sub.astype(np.float32) * (1.0 - a)
    ).astype(np.uint8)

    out = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    out[..., :3] = rgb_canvas
    out[..., 3] = 255
    return out


def apply_effects(rgba_img, t, out_size):
    """CRT-Scanlines, VHS-Jitter, RGB-Split, Vignette, Neon Glow, Bloom, Grain"""
    H0, W0, _ = rgba_img.shape
    W, H = out_size

    # 1) Auf Canvas platzieren (bereits BG + Alpha=255)
    frame = rgba_img if (W0 == W and H0 == H) else place_on_canvas(rgba_img, W, H)
    rgb = frame[..., :3]

    # 2) CRT Scanlines
    y = np.arange(H, dtype=np.float32)
    line_mask = 0.5 + 0.5 * np.sign(
        np.sin((y + t * SCANLINE_SPEED * 120.0) * math.pi)
    )  # 0/1
    line_mask = 1.0 - SCANLINE_STRENGTH * (1.0 - line_mask)
    rgb = (
        (rgb.astype(np.float32) * line_mask[:, None, None])
        .clip(0, 255)
        .astype(np.uint8)
    )

    # 3) VHS-Jitter
    jitter_rgb = rgb.copy()
    rnd = np.random.RandomState(int(t * 10))  # langsames Update
    jit = np.sin(2 * math.pi * (y / H) * JITTER_SPEED + t * 2.3) * 0.5 + 0.5
    jit += 0.25 * rnd.randn(H).astype(np.float32)
    jit = (jit - jit.mean()) * JITTER_AMOUNT
    for yi in range(H):
        s = int(jit[yi])
        if s != 0:
            jitter_rgb[yi] = np.roll(jitter_rgb[yi], s, axis=0)

    # 4) RGB Split
    r = jitter_rgb[..., 0]
    g = jitter_rgb[..., 1]
    b = jitter_rgb[..., 2]
    shift = int(RGB_SPLIT_PIXELS * (1.0 + 0.3 * math.sin(t * 1.3)))
    r2 = np.roll(r, shift, axis=1)
    b2 = np.roll(b, -shift, axis=1)
    rgb2 = np.stack([r2, g, b2], axis=-1)

    # 5) Vignette
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2.0, H / 2.0
    rr = np.sqrt(((xx - cx) / (W * 0.5)) ** 2 + ((yy - cy) / (H * 0.5)) ** 2)
    vign = 1.0 - VIGNETTE_STRENGTH * (rr**1.5)
    vign = np.clip(vign, 0.0, 1.0)
    rgb3 = (rgb2.astype(np.float32) * vign[..., None]).clip(0, 255).astype(np.uint8)

    # 5.5) Neon Glow Effect (saturation + brightness boost)
    if NEON_GLOW > 1.0 or SATURATION_BOOST > 1.0:
        rgb_float = rgb3.astype(np.float32)

        # Boost saturation
        if SATURATION_BOOST > 1.0:
            # Convert to HSV-like approach: boost color relative to grayscale
            gray = (
                0.299 * rgb_float[..., 0]
                + 0.587 * rgb_float[..., 1]
                + 0.114 * rgb_float[..., 2]
            )
            rgb_float = gray[..., None] + SATURATION_BOOST * (
                rgb_float - gray[..., None]
            )

        # Brightness boost for neon effect
        if NEON_GLOW > 1.0:
            rgb_float = rgb_float * NEON_GLOW

        rgb3 = np.clip(rgb_float, 0, 255).astype(np.uint8)

    # 6) Bloom
    if BLOOM_STRENGTH > 0:
        surf_tmp = pygame.image.frombuffer(rgb3.tobytes(), (W, H), "RGB")
        dw, dh = max(1, W // BLOOM_DOWNSCALE), max(1, H // BLOOM_DOWNSCALE)
        small = pygame.transform.smoothscale(surf_tmp, (dw, dh))
        bloom = pygame.transform.smoothscale(small, (W, H))
        bloom_np = pygame.surfarray.pixels3d(bloom).copy()
        bloom_np = np.transpose(bloom_np, (1, 0, 2))  # (W,H,3) -> (H,W,3)
        out = (
            rgb3.astype(np.float32) * (1.0 - BLOOM_STRENGTH)
            + bloom_np.astype(np.float32) * BLOOM_STRENGTH
        )
        out = out.clip(0, 255).astype(np.uint8)
    else:
        out = rgb3

    # 7) Grain
    if NOISE_STRENGTH > 0:
        noise = np.random.randn(H, W, 1).astype(np.float32) * 255.0 * NOISE_STRENGTH
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[..., :3] = out
    rgba[..., 3] = 255
    return rgba


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
        print("Nutzung: python script.py <pfad.svg|png|jpg|jpeg|gif>")
        print("         python script.py --monitors  (show monitor info)")
        sys.exit(1)

    # Show monitor info if requested
    if sys.argv[1] == "--monitors":
        list_monitors()
        sys.exit(0)
    path = sys.argv[1]
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXT:
        print(f"Format nicht unterstützt: {ext}")
        sys.exit(1)

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

    pygame.display.set_caption("CRT + VHS Media")
    clock = pygame.time.Clock()
    W, H = screen.get_size()

    media = MediaSource(path, W, H)

    # Frames ggf. vorab auf Canvas-Größe projizieren (beschleunigt die Laufzeit)
    preprocessed = []
    for f in media.frames:
        preprocessed.append(place_on_canvas(f, W, H))

    running = True
    t0 = time.time()
    frame_idx = 0
    next_switch = 0.0  # Zeitpunkt (Sekunden seit Start), wann nächster GIF-Frame kommt

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

        # GIF-Framewechsel steuern
        if media.is_animated:
            if now >= next_switch:
                frame_idx = (frame_idx + 1) % media.total_frames
                dur_ms = media.get_duration_ms(frame_idx)
                next_switch = now + max(1, dur_ms) / 1000.0
        else:
            frame_idx = 0  # statisch

        base = preprocessed[frame_idx]
        t = now - t0
        frame_rgba = apply_effects(base, t, (W, H))
        surf = surface_from_numpy(frame_rgba)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
