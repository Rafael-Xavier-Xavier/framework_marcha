"""
GAIT_PREVIEW.py — Pré-visualização animada da marcha (Tkinter).
"""
import math
import time
import tkinter as tk
from tkinter import ttk

LEG_PHASE = {"TE": 0.00, "DE": 0.25, "TD": 0.50, "DD": 0.75}
FRONT_LEGS = ("DE", "DD")
NEAR_LEGS = ("TE", "DE")
L1 = 46.0
L2 = 46.0
STAND_H = 78.0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _norm(preset, bounds, key):
    try:
        lo, hi = bounds[key]
        v = float(preset.get(key, (lo + hi) / 2.0))
        return 0.5 if hi == lo else clamp((v - lo) / (hi - lo), 0.0, 1.0)
    except Exception:
        return 0.5

def gait_geometry(preset, bounds):
    n = lambda k: _norm(preset, bounds, k)
    return {
        "duty": clamp(float(preset.get("support_fraction", 0.75)), 0.35, 0.95),
        "stride": (20.0 + 60.0 * n("shoulder_amp")) * (0.7 + 0.6 * n("cycle_scale")),
        "lift_front": 10.0 + 50.0 * n("front_knee_amp"),
        "lift_rear": 10.0 + 50.0 * n("rear_knee_amp"),
        "rear_shift": -0.4 * float(preset.get("abertura_traseira", 0.0)),
        "period": max(0.2, float(preset.get("tempo_ciclo", 1.0))),
        "height": float(preset.get("height_scale", 0.0)) * 1.5,
    }

def foot_trajectory(phase, geo, front):
    t = phase % 1.0
    duty, S = geo["duty"], geo["stride"]
    shift = 0.0 if front else geo["rear_shift"]
    lift = geo["lift_front"] if front else geo["lift_rear"]
    if t < duty:
        u = t / duty
        return S / 2 - S * u + shift, 0.0, True
    u = (t - duty) / (1 - duty)
    return -S / 2 + S * u + shift, lift * math.sin(math.pi * u), False

def inverse_kinematics(x, y, knee_back):
    d = math.hypot(x, y)
    d = max(min(d, L1 + L2 - 0.5), abs(L1 - L2) + 0.5)
    a = math.acos(clamp((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1))
    base = math.atan2(y, x)
    b = base + a if knee_back else base - a
    return L1 * math.cos(b), L1 * math.sin(b)

def leg_points(leg, phase, geo):
    front = leg in FRONT_LEGS
    fx, fz, contact = foot_trajectory(phase + LEG_PHASE[leg], geo, front)
    fy = STAND_H + geo["height"] - fz
    kx, ky = inverse_kinematics(fx, fy, knee_back=front)
    return (kx, ky), (fx, fy), contact

class GaitPreview(ttk.Frame):
    BG = "#0A1018"
    FG = "#EAF2F8"
    FAR = "#5B6F84"
    ACC = "#64D2FF"
    COLORS = {"TE": "#FF8A5B", "DE": "#64D2FF", "TD": "#C79BFF", "DD": "#6EE7A8"}

    def __init__(self, parent, bounds, width=480, height=250):
        super().__init__(parent)
        self.bounds = bounds
        self.W, self.H = width, height
        self.preset = {}
        self.geo = gait_geometry({}, bounds)
        self.phase = 0.0
        self.scroll = 0.0
        self.playing = True
        self.speed = 1.0
        self._last = time.monotonic()
        self._setting = False
        self._drag_x = None
        self._after_id = None

        self.canvas = tk.Canvas(self, width=width, height=height, bg=self.BG,
                                highlightthickness=0, cursor="fleur")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)

        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", pady=(4, 0))
        self.btn = ttk.Button(ctrl, text="Pausar", width=9, command=self.toggle)
        self.btn.pack(side="left")
        self.phase_var = tk.DoubleVar(value=0.0)
        self.scale = ttk.Scale(ctrl, from_=0.0, to=1.0, variable=self.phase_var,
                               orient="horizontal", command=self._on_scale)
        self.scale.pack(side="left", fill="x", expand=True, padx=6)
        self.speed_var = tk.StringVar(value="1x")
        combo = ttk.Combobox(ctrl, textvariable=self.speed_var, width=6,
                             state="readonly", values=["0.25x", "0.5x", "1x", "2x"])
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", self._on_speed)

        self.bind("<Destroy>", self._on_destroy)
        self._draw()
        self._tick()

    def set_preset(self, preset):
        self.preset = dict(preset)
        self.geo = gait_geometry(self.preset, self.bounds)
        self._draw()

    def toggle(self):
        self.playing = not self.playing
        self.btn.config(text="Pausar" if self.playing else "Reproduzir")

    def _on_speed(self, _e=None):
        try: self.speed = float(self.speed_var.get().rstrip("x"))
        except ValueError: self.speed = 1.0

    def _on_scale(self, value):
        if self._setting: return
        if self.playing: self.toggle()
        self._set_phase(math.floor(self.phase) + float(value))

    def _on_press(self, e):
        self._drag_x = e.x
        if self.playing: self.toggle()

    def _on_drag(self, e):
        if self._drag_x is None: return
        px_per_cycle = max(1.0, self.geo["stride"] / self.geo["duty"])
        self._set_phase(self.phase - (e.x - self._drag_x) / px_per_cycle)
        self._drag_x = e.x

    def _on_destroy(self, e):
        if e.widget is self and self._after_id is not None:
            try: self.after_cancel(self._after_id)
            except Exception: pass
            self._after_id = None

    def _tick(self):
        try:
            if not self.winfo_exists(): return
        except tk.TclError: return
        now = time.monotonic()
        dt = min(now - self._last, 0.1)
        self._last = now
        if self.playing:
            self._set_phase(self.phase + dt * self.speed / self.geo["period"])
        self._after_id = self.after(33, self._tick)

    def _set_phase(self, new_phase):
        dphase = new_phase - self.phase
        if abs(dphase) < 0.25:
            self.scroll -= dphase * self.geo["stride"] / self.geo["duty"]
        self.phase = new_phase
        self._setting = True
        self.phase_var.set(new_phase % 1.0)
        self._setting = False
        self._draw()

    def _draw(self):
        c = self.canvas
        c.delete("all")
        W = self.W
        ground = self.H - 40 - self.geo["height"]
        cx = W / 2
        hip_y = ground - STAND_H
        body_l = 190.0
        hips = {False: cx - body_l / 2 + 26, True: cx + body_l / 2 - 26}

        c.create_line(0, ground, W, ground, fill="#2B3B4C", width=3)
        step = 40
        x0 = self.scroll % step
        for i in range(-1, int(W / step) + 2):
            x = x0 + i * step
            c.create_line(x, ground + 10, x - 12, ground + 22, fill="#1E2C3A", width=2)

        legs = {n: leg_points(n, self.phase, self.geo) for n in LEG_PHASE}

        def draw_leg(n):
            (kx, ky), (fx, fy), contact = legs[n]
            hx = hips[n in FRONT_LEGS]
            near = n in NEAR_LEGS
            col = self.FG if near else self.FAR
            w = 7 if near else 6
            c.create_line(hx, hip_y, hx + kx, hip_y + ky, hx + fx, hip_y + fy,
                          fill=col, width=w, capstyle="round", joinstyle="round")
            c.create_oval(hx + kx - 4, hip_y + ky - 4, hx + kx + 4, hip_y + ky + 4,
                          fill=self.ACC, outline="")
            r = 6
            c.create_oval(hx + fx - r, hip_y + fy - r, hx + fx + r, hip_y + fy + r,
                          fill=self.COLORS[n] if contact else self.BG,
                          outline=self.COLORS[n], width=2)

        for n in LEG_PHASE:
            if n not in NEAR_LEGS: draw_leg(n)

        sway = math.sin(self.phase * 4 * math.pi) * 6
        c.create_line(cx - body_l / 2 + 4, hip_y - 16, cx - body_l / 2 - 26, hip_y - 30 + sway,
                      cx - body_l / 2 - 36, hip_y - 62 + sway, smooth=True,
                      fill=self.FG, width=6, capstyle="round")
        c.create_rectangle(cx - body_l / 2, hip_y - 32, cx + body_l / 2, hip_y + 4,
                           fill="#25384B", outline=self.FG, width=2)
        hx_, hy_ = cx + body_l / 2 + 16, hip_y - 26
        c.create_oval(hx_ - 24, hy_ - 20, hx_ + 24, hy_ + 20, fill="#25384B",
                      outline=self.FG, width=2)
        c.create_polygon(hx_ - 14, hy_ - 16, hx_ - 8, hy_ - 36, hx_ + 2, hy_ - 18,
                         fill=self.FG, outline="")
        c.create_polygon(hx_ + 6, hy_ - 18, hx_ + 16, hy_ - 36, hx_ + 22, hy_ - 14,
                         fill=self.FG, outline="")
        c.create_oval(hx_ + 8, hy_ - 6, hx_ + 16, hy_ + 2, fill=self.ACC, outline="")

        for n in LEG_PHASE:
            if n in NEAR_LEGS: draw_leg(n)

        for i, n in enumerate(LEG_PHASE):
            x = 18 + i * 56
            contact = legs[n][2]
            c.create_oval(x - 7, 16, x + 7, 30, outline=self.COLORS[n], width=2,
                          fill=self.COLORS[n] if contact else self.BG)
            c.create_text(x + 12, 23, text=n, anchor="w", fill="#9FB0C0", font=("Consolas", 9))
        g = self.geo
        c.create_text(14, self.H - 10, anchor="w", fill="#6F8294", font=("Consolas", 8),
                      text=(f"ciclo {g['period']:.2f}s  apoio {g['duty']*100:.0f}%  "
                            f"passada {g['stride']:.0f}px  (ilustrativo)"))