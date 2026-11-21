"""
GAIT_PRESET.py — parâmetros da marcha, limites e presets.

Dois presets independentes (pasta modelos/configuracoes/):
    manual.json          -> ajuste manual
    iterativo_best.json  -> melhor configuração da busca iterativa

Compatibilidade: se manual.json ainda não existe, o preset antigo
gait_experiments/gait_preset.json é usado como "manual".
"""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRESETS_DIR = ROOT / "modelos" / "configuracoes"
LEGACY_PRESET_JSON = ROOT / "gait_experiments" / "gait_preset.json"

PRESET_KINDS = ("manual", "iterativo")
PRESET_FILES = {
    "manual": PRESETS_DIR / "manual.json",
    "iterativo": PRESETS_DIR / "iterativo_best.json",
}
PRESET_JSON = PRESET_FILES["manual"]          # compatibilidade

PARAMETER_BOUNDS = {
    "cycle_scale":      (0.50, 2.00),
    "support_fraction": (0.35, 0.90),
    "shoulder_amp":     (5.0, 60.0),
    "front_knee_amp":   (5.0, 70.0),
    "rear_knee_amp":    (2.0, 60.0),
    "height_scale":     (-25.0, 25.0),
}
PARAMETER_NAMES = list(PARAMETER_BOUNDS.keys())


PARAMETER_DEFAULTS = {
    "cycle_scale": 1.3405146459036668,
    "support_fraction": 0.90,
    "shoulder_amp": 47.45059614720662,
    "front_knee_amp": 23.370641594324823,
    "rear_knee_amp": 38.207837028929475,
    "height_scale": 24.871142039031362,
}


TEMPO_CICLO_BOUNDS = (0.3, 2.5)
ABERTURA_TRASEIRA_BOUNDS = (0.0, 20.0)

TEMPO_CICLO_DEFAULT = 0.5540676847949711
ABERTURA_TRASEIRA_DEFAULT = 20.0


ALL_BOUNDS = dict(PARAMETER_BOUNDS)
ALL_BOUNDS["tempo_ciclo"] = TEMPO_CICLO_BOUNDS
ALL_BOUNDS["abertura_traseira"] = ABERTURA_TRASEIRA_BOUNDS
ALL_NAMES = list(ALL_BOUNDS.keys())

ALL_DEFAULTS = dict(PARAMETER_DEFAULTS)
ALL_DEFAULTS["tempo_ciclo"] = TEMPO_CICLO_DEFAULT
ALL_DEFAULTS["abertura_traseira"] = ABERTURA_TRASEIRA_DEFAULT

# Parâmetros que o controlador atual realmente usa
EFFECTIVE_NAMES = ["shoulder_amp", "front_knee_amp", "rear_knee_amp",
                   "height_scale", "tempo_ciclo", "abertura_traseira"]

PARAM_LABELS = {
    "cycle_scale": "cycle_scale (reservado)",
    "support_fraction": "support_fraction (reservado)",
    "shoulder_amp": "Amplitude do ombro",
    "front_knee_amp": "Amplitude joelho dianteiro",
    "rear_knee_amp": "Amplitude joelho traseiro",
    "height_scale": "Altura do corpo (flexão)",
    "tempo_ciclo": "Tempo do ciclo (s)",
    "abertura_traseira": "Abertura traseira",
}


def clamp(v, low, high):
    return max(low, min(high, v))


def default_preset():
    return dict(ALL_DEFAULTS)


def sanitize_preset(raw):
    if not isinstance(raw, dict):
        return default_preset()
    out = {}
    for name in ALL_NAMES:
        low, high = ALL_BOUNDS[name]
        try:
            v = float(raw.get(name, ALL_DEFAULTS[name]))
        except (TypeError, ValueError):
            v = ALL_DEFAULTS[name]
        out[name] = clamp(v, low, high)
    return out


def _extract_params(raw):
    if isinstance(raw, dict) and isinstance(raw.get("parameters"), dict):
        merged = dict(raw["parameters"])
        for k in ("tempo_ciclo", "abertura_traseira"):
            if k in raw:
                merged[k] = raw[k]
        return merged
    return raw


def preset_path(kind="manual"):
    if kind not in PRESET_FILES:
        raise ValueError(f"Tipo de preset inválido: {kind}")
    return PRESET_FILES[kind]


def preset_exists(kind="manual"):
    if preset_path(kind).exists():
        return True
    return kind == "manual" and LEGACY_PRESET_JSON.exists()


def load_preset(kind="manual"):
    """manual: nunca falha. iterativo: None se ainda não existe."""
    path = preset_path(kind)
    if not path.exists():
        if kind == "manual" and LEGACY_PRESET_JSON.exists():
            path = LEGACY_PRESET_JSON
        elif kind == "manual":
            return default_preset()
        else:
            return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_preset() if kind == "manual" else None
    return sanitize_preset(_extract_params(raw))


def load_preset_meta(kind="manual"):
    try:
        raw = json.loads(preset_path(kind).read_text(encoding="utf-8"))
        meta = raw.get("meta", {})
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def save_preset(preset, kind="manual", meta=None):
    """
    Salva o preset. Agora aceita o argumento 'meta' para compatibilidade
    com o init_testing.py que envia os scores da busca iterativa.
    """
    p = sanitize_preset(preset)
    path = preset_path(kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    full_meta = {"origem": kind,
                 "salvo_em": datetime.now().isoformat(timespec="seconds")}
    if meta:
        full_meta.update(meta)
    data = {
        "format_version": 4,
        "parameters": {n: p[n] for n in PARAMETER_NAMES},
        "tempo_ciclo": p["tempo_ciclo"],
        "abertura_traseira": p["abertura_traseira"],
        "parameter_bounds": {n: list(PARAMETER_BOUNDS[n]) for n in PARAMETER_NAMES},
        "tempo_ciclo_bounds": list(TEMPO_CICLO_BOUNDS),
        "abertura_traseira_bounds": list(ABERTURA_TRASEIRA_BOUNDS),
        "meta": full_meta,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path