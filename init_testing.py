"""
SUPERVISOR
"""
from controller import Supervisor

from PIL import Image, ImageTk

import csv
import json
import math
import random
import statistics
import threading
import time
import traceback
import zipfile
import urllib.request

from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np

try:
    from GAIT_PRESET import (
        PRESET_JSON, PARAMETER_BOUNDS, PARAMETER_NAMES, PARAMETER_DEFAULTS,
        TEMPO_CICLO_BOUNDS, ABERTURA_TRASEIRA_BOUNDS,
        TEMPO_CICLO_DEFAULT, ABERTURA_TRASEIRA_DEFAULT,
        default_preset, load_preset, save_preset, sanitize_preset,
    )
except ImportError:
    raise RuntimeError("Não encontrei GAIT_PRESET.py na mesma pasta.")

try:
    from GAIT_PREVIEW import GaitPreview
except ImportError:
    raise RuntimeError("Não encontrei GAIT_PREVIEW.py na mesma pasta.")


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================
ROBOT_DEF = "GATO"
FORWARD_AXIS = "x"
FORWARD_SIGN = -1

TRIAL_DURATION = 12.0
SETTLE_STEPS = 180

SEARCH_TRIALS = 60
INITIAL_RANDOM_TRIALS = 12
ELITE_POOL_SIZE = 5
RANDOM_RESTART_PROBABILITY = 0.15
INITIAL_MUTATION_STRENGTH = 0.15
MIN_MUTATION_STRENGTH = 0.04

NEURAL_UPDATE_INTERVAL = 0.5
DELTA_FRACTION = 0.10
OBS_DIM = 20
ACTION_DIM = 4
PARAMS_NEURAL = ["shoulder_amp", "front_knee_amp", "rear_knee_amp", "height_scale"]

SENSOR_SEND_INTERVAL_SIM = 0.1

ADAPT_PPO_EPISODES = 100
ADAPT_PPO_LEARNING_RATE = 1e-4
ADAPT_PPO_N_STEPS = 512
ADAPT_PPO_BATCH_SIZE = 128
ADAPT_PPO_EPOCHS = 10
ADAPT_PPO_GAMMA = 0.99
ADAPT_PPO_NET_ARCH = [128, 128]
ADAPT_PPO_ENT_COEF = 0.01

FALL_HEIGHT_DROP = 0.10
MAX_TILT_DEG = 50.0
RANDOM_SEED = 42
rng = random.Random(RANDOM_SEED)

SENSOR_STALE_TIMEOUT = 0.5

OUTPUT_DIR = Path("gait_experiments")
MODELS_DIR = Path("models")
EXPERIMENTS_DIR = Path("experiments")
PACKAGES_DIR = Path("packages")
LOGS_DIR = Path("logs")
RESULTS_DIR = Path("results")

for d in (OUTPUT_DIR, MODELS_DIR, EXPERIMENTS_DIR, PACKAGES_DIR, LOGS_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True)

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
STATUS_JSON = OUTPUT_DIR / "latest_status.json"
EXP_CONDITIONS_JSON = OUTPUT_DIR / "experimental_conditions.json"

BASE_BODY_MASS_KG = 0.32  # Petoi Nybble

YAW_LIMITATION_NOTE = (
    "Yaw no robô real deriva da integração do gyro Z (MPU6050 sem magnetômetro) "
    "e acumula drift ao longo do tempo. Yaw no Webots é absoluto. "
    "Comparações diretas de yaw sim × real devem considerar essa limitação; "
    "para o robô real recomenda-se também avaliar gz (velocidade angular Z), "
    "que não depende de integração."
)

NOISE_PRESETS = {
    "disabled": {"imu_sigma_deg": 0.0, "cam_sigma": 0.0},
    "low":      {"imu_sigma_deg": 1.0, "cam_sigma": 0.05},
    "medium":   {"imu_sigma_deg": 2.5, "cam_sigma": 0.10},
    "high":     {"imu_sigma_deg": 5.0, "cam_sigma": 0.20},
}

IMU_NOISE_SCALES = {
    "roll_deg": 1.0, "pitch_deg": 1.0, "yaw_deg": 1.0,
    "gx": 0.01, "gy": 0.01, "gz": 0.01,
    "ax": 0.10, "ay": 0.10, "az": 0.10,
}
CAM_NOISE_SCALES = {
    "flow_x_mean": 0.05, "flow_y_mean": 0.05,
    "flow_mag_mean": 2.0, "flow_dir_deg": 30.0,
    "flow_points": 100.0,
}

# ------------------------------------------------------------
# Helpers de path (por tag de modelo)
# ------------------------------------------------------------
def _sanitize_tag(tag):
    t = (tag or "a").strip().lower()
    if not t: t = "a"
    return "".join(c for c in t if c.isalnum() or c in "_-")[:16]

def _ppo_paths(tag):
    t = _sanitize_tag(tag)
    return {
        "tag": t,
        "model": MODELS_DIR / f"imu_visual_ppo_{t}.zip",
        "model_base": MODELS_DIR / f"imu_visual_ppo_{t}",
        "vecnorm": MODELS_DIR / f"vecnormalize_{t}.pkl",
        "ckpt_dir": MODELS_DIR / f"checkpoints_{t}",
    }

def _legacy_ppo_paths():
    return {
        "model": MODELS_DIR / "imu_visual_ppo.zip",
        "model_base": MODELS_DIR / "imu_visual_ppo",
        "vecnorm": MODELS_DIR / "vecnormalize.pkl",
        "ckpt_dir": MODELS_DIR / "checkpoints",
    }


# ============================================================
# WEBOTS SETUP
# ============================================================
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

robot_node = supervisor.getFromDef(ROBOT_DEF)
if robot_node is None:
    raise RuntimeError(f"Não encontrei DEF {ROBOT_DEF}.")

translation_field = robot_node.getField("translation")
rotation_field = robot_node.getField("rotation")
initial_translation = translation_field.getSFVec3f()
initial_rotation = rotation_field.getSFRotation()

emitter = supervisor.getDevice("emitter")
receiver = supervisor.getDevice("receiver")
receiver.enable(timestep)

_chassis_mass_field = None
_chassis_physics_node = None
_wbt_chassis_mass_kg = None

def _resolve_chassis_mass_field():
    global _chassis_mass_field, _chassis_physics_node, _wbt_chassis_mass_kg
    try:
        children = robot_node.getField("children")
        if children is None: return
        for i in range(children.getCount()):
            child = children.getMFNode(i)
            if child is None: continue
            phys = child.getField("physics")
            if phys is None: continue
            phys_node = phys.getSFNode()
            if phys_node is None: continue
            m = phys_node.getField("mass")
            if m is not None:
                _chassis_mass_field = m
                _chassis_physics_node = phys_node
                try: _wbt_chassis_mass_kg = float(m.getSFFloat())
                except Exception: _wbt_chassis_mass_kg = None
                print(f"[MASS] Chassi idx {i}. wbt={_wbt_chassis_mass_kg} kg. Base runtime={BASE_BODY_MASS_KG} kg.", flush=True)
                return
        print("[MASS] AVISO: nenhum physics.mass encontrado.", flush=True)
    except Exception as e:
        print(f"[MASS] Erro: {e}", flush=True)

_resolve_chassis_mass_field()

def apply_total_mass(total_kg):
    if _chassis_mass_field is None: return False
    try:
        _chassis_mass_field.setSFFloat(float(total_kg))
        return True
    except Exception as e:
        print(f"[MASS] Falha: {e}", flush=True)
        return False

# ============================================================
# SENSOR RECEIVER
# ============================================================
sensor_state = {
    "t": 0.0, "trial": -1, "cmd": "hold", "direction": FORWARD_SIGN,
    "gx": 0.0, "gy": 0.0, "gz": 0.0, "ax": 0.0, "ay": 0.0, "az": 0.0,
    "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0,
    "flow_x_mean": 0.0, "flow_y_mean": 0.0, "flow_mag_mean": 0.0, "flow_dir_deg": 0.0,
    "flow_points": 0, "flow_var": 0.0, "_last_wall": 0.0,
}
lock_sensor = threading.Lock()

def sensor_receiver_thread():
    global sensor_state
    while True:
        try:
            if receiver.getQueueLength() > 0:
                raw = receiver.getString()
                receiver.nextPacket()
                if not raw: continue
                try: msg = json.loads(raw)
                except Exception: continue
                if isinstance(msg, dict) and msg.get("tipo") == "sen":
                    with lock_sensor:
                        sensor_state.update(msg)
                        sensor_state["_last_wall"] = time.monotonic()
            else:
                time.sleep(timestep / 1000.0)
        except Exception as e:
            print(f"[COM] Erro receptor: {e}", flush=True)
            time.sleep(0.01)

threading.Thread(target=sensor_receiver_thread, daemon=True).start()

def get_latest_sensors():
    with lock_sensor: s = dict(sensor_state)
    if time.monotonic() - s["_last_wall"] > SENSOR_STALE_TIMEOUT: return None
    return s

def send_command(payload):
    try: emitter.send(json.dumps(payload))
    except Exception as e: print(f"[COM] Falha envio: {e}", flush=True)

# ============================================================
# CONDIÇÕES EXPERIMENTAIS
# ============================================================
def default_exp_conditions():
    return {
        "payload_ratio_percent": 0.0,
        "camera_noise_level": "disabled",
        "camera_noise_sigma_custom": 0.10,
        "imu_noise_level": "disabled",
        "imu_noise_sigma_custom": 2.5,
        "servo_delay_ms": 0.0,
        "inference_delay_ms": 0.0,
        "search_discard_falls": True,
        "ppo_model_tag": "a",
        "ppo_start_fresh": True,
        "ppo_reward_disp_gain": 10.0,
        "ppo_reward_tilt_enabled": True,
        "ppo_reward_tilt_threshold_deg": 5.0,
        "ppo_reward_tilt_coef": 0.50,
        "ppo_reward_gz_enabled": True,
        "ppo_reward_gz_coef": 0.05,
        "ppo_reward_final_disp_bonus": 0.5,
        "ppo_penalty_enabled": False,
        "ppo_penalty_value": -5.0,
        "stab_iter_roll_enabled": False, "stab_iter_roll_max_deg": 20.0,
        "stab_iter_pitch_enabled": False, "stab_iter_pitch_max_deg": 20.0,
        "stab_iter_yaw_enabled": False, "stab_iter_yaw_max_deg": 30.0,
        "stab_ppo_roll_enabled": False, "stab_ppo_roll_target_deg": 0.0,
        "stab_ppo_roll_weight": 1.0, "stab_ppo_roll_scale_deg": 45.0,
        "stab_ppo_pitch_enabled": False, "stab_ppo_pitch_target_deg": 0.0,
        "stab_ppo_pitch_weight": 1.0, "stab_ppo_pitch_scale_deg": 45.0,
        "stab_ppo_yaw_enabled": False, "stab_ppo_yaw_target_deg": 0.0,
        "stab_ppo_yaw_weight": 1.0, "stab_ppo_yaw_scale_deg": 180.0,
    }

def load_exp_conditions():
    if not EXP_CONDITIONS_JSON.exists():
        return default_exp_conditions()
    try:
        d = json.loads(EXP_CONDITIONS_JSON.read_text(encoding="utf-8"))
        base = default_exp_conditions()
        base.update(d or {})
        return base
    except Exception:
        return default_exp_conditions()

def save_exp_conditions_dict(cond):
    try:
        EXP_CONDITIONS_JSON.write_text(json.dumps(cond, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception as e:
        print(f"[EXP] Erro: {e}", flush=True)
        return False

# ============================================================
# OBS / AÇÃO / SCORE
# ============================================================
def clamp(v, low, high): return max(low, min(high, v))

def _apply_sensor_noise(sensors):
    if sensors is None: return sensors
    u = globals().get("ui")
    if u is None: return sensors
    s = dict(sensors)
    imu_sigma = u.get_effective_imu_sigma()
    if imu_sigma and imu_sigma > 0:
        for k, scale in IMU_NOISE_SCALES.items():
            if s.get(k) is not None:
                try: s[k] = float(s[k]) + rng.gauss(0.0, imu_sigma * scale)
                except Exception: pass
    cam_sigma = u.get_effective_camera_sigma()
    if cam_sigma and cam_sigma > 0:
        for k, scale in CAM_NOISE_SCALES.items():
            if s.get(k) is not None:
                try: s[k] = float(s[k]) + rng.gauss(0.0, cam_sigma * scale)
                except Exception: pass
    return s

def build_observation(sensors, params):
    if sensors is None: sensors = {}
    sensors = _apply_sensor_noise(sensors)
    vec = [
        float(sensors.get("roll_deg", 0.0)) / 45.0, float(sensors.get("pitch_deg", 0.0)) / 45.0,
        float(sensors.get("yaw_deg", 0.0)) / 180.0, float(sensors.get("gx", 0.0)) / 2.0,
        float(sensors.get("gy", 0.0)) / 2.0, float(sensors.get("gz", 0.0)) / 2.0,
        float(sensors.get("ax", 0.0)) / 10.0, float(sensors.get("ay", 0.0)) / 10.0,
        float(sensors.get("az", 0.0)) / 10.0, float(sensors.get("flow_x_mean", 0.0)) / 10.0,
        float(sensors.get("flow_y_mean", 0.0)) / 10.0, float(sensors.get("flow_mag_mean", 0.0)) / 10.0,
        float(sensors.get("flow_dir_deg", 0.0)) / 180.0, float(sensors.get("flow_points", 0.0)) / 500.0,
    ]
    for n in PARAMETER_NAMES:
        low, high = PARAMETER_BOUNDS[n]
        v = float(params.get(n, PARAMETER_DEFAULTS[n]))
        vec.append(2.0 * (v - low) / (high - low) - 1.0)
    return vec

def apply_delta_action(action, baseline):
    result = dict(baseline)
    for i, n in enumerate(PARAMS_NEURAL):
        low, high = PARAMETER_BOUNDS[n]
        a = max(-1.0, min(1.0, float(action[i])))
        base = clamp(float(baseline.get(n, PARAMETER_DEFAULTS[n])), low, high)
        delta = a * DELTA_FRACTION * (high - low)
        result[n] = clamp(base + delta, low, high)
    return result

def calculate_score(displacement, max_tilt, fallen):
    s = displacement
    s -= max(0.0, max_tilt - 5.0) * 0.02
    if fallen: s -= 2.0
    return s

# ============================================================
# PROTOCOLO
# ============================================================
def build_test_protocol(metodo, duracao, freq_hz, baseline, metrics, exp_cond=None, model_tag="a"):
    try:
        import torch, stable_baselines3, gymnasium
        versoes = {"torch": torch.__version__, "sb3": stable_baselines3.__version__,
                   "gymnasium": gymnasium.__version__, "numpy": np.__version__}
    except Exception:
        versoes = {"numpy": np.__version__}
    if exp_cond is None: exp_cond = {}
    p = _ppo_paths(model_tag)
    return {
        "protocol_version": "1.9",
        "metodo": metodo, "duracao_s": float(duracao), "freq_hz": float(freq_hz),
        "dt_sample_alvo_s": 1.0 / max(0.1, float(freq_hz)),
        "obs_dim": OBS_DIM, "action_dim": ACTION_DIM,
        "params_neural": list(PARAMS_NEURAL), "delta_fraction": DELTA_FRACTION,
        "neural_update_interval_s": NEURAL_UPDATE_INTERVAL,
        "sensor_send_interval_s": SENSOR_SEND_INTERVAL_SIM,
        "sensor_stale_timeout_s": SENSOR_STALE_TIMEOUT,
        "baseline": {k: float(v) for k, v in baseline.items()},
        "metricas": dict(metrics), "forward_sign": FORWARD_SIGN,
        "random_seed": RANDOM_SEED, "experimental_conditions": exp_cond,
        "yaw_limitation_note": YAW_LIMITATION_NOTE,
        "modelo": {
            "path": str(p["model"]) if metodo == "nn_imu" else None,
            "vecnormalize_path": str(p["vecnorm"]) if metodo == "nn_imu" else None,
            "tag": _sanitize_tag(model_tag) if metodo == "nn_imu" else None,
        },
        "obs_spec": {
            "sensor_keys": ["roll_deg", "pitch_deg", "yaw_deg", "gx", "gy", "gz",
                            "ax", "ay", "az", "flow_x_mean", "flow_y_mean",
                            "flow_mag_mean", "flow_dir_deg", "flow_points"],
            "sensor_dividers": [45.0, 45.0, 180.0, 2.0, 2.0, 2.0,
                                10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 180.0, 500.0],
            "params_normalized": list(PARAMETER_NAMES),
            "params_bounds": {n: list(PARAMETER_BOUNDS[n]) for n in PARAMETER_NAMES},
            "params_defaults": {n: float(PARAMETER_DEFAULTS[n]) for n in PARAMETER_NAMES},
            "param_norm_formula": "2*(v-low)/(high-low) - 1",
        },
        "action_spec": {
            "params": list(PARAMS_NEURAL),
            "formula": "clamp(base + a*DELTA_FRACTION*(high-low), low, high)",
            "delta_fraction": DELTA_FRACTION, "action_clip": [-1.0, 1.0],
        },
        "versoes": versoes, "criado_em": datetime.now().isoformat(),
    }

# ============================================================
# ARQUIVOS / EXPERIMENTOS
# ============================================================
def create_experiment_folder(exp_id, metadata):
    folder = EXPERIMENTS_DIR / exp_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return folder

def save_experiment_sensors(folder, timeseries):
    if not timeseries: return
    all_cols, seen = [], set()
    for row in timeseries:
        for k in row.keys():
            if k not in seen:
                seen.add(k); all_cols.append(k)
    with open(folder / "sensors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        w.writeheader(); w.writerows(timeseries)

def save_experiment_gait_params(folder, timeseries):
    if not timeseries: return
    cols = [c for c in (["time_s"] + PARAMETER_NAMES + ["tempo_ciclo", "abertura_traseira"]) if c in timeseries[0]]
    with open(folder / "gait_parameters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in timeseries: w.writerow({k: row.get(k, "") for k in cols})

def save_experiment_performance(folder, results):
    if not results: return
    clean = [{k: v for k, v in r.items() if k != "timeseries"} for r in results]
    all_cols, seen = [], set()
    for r in clean:
        for k in r.keys():
            if k not in seen:
                seen.add(k); all_cols.append(k)
    with open(folder / "performance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        w.writeheader(); w.writerows(clean)

def save_experiment_protocol(folder, protocol):
    """Grava test_protocol.json dentro da pasta do experimento."""
    try:
        (Path(folder) / "test_protocol.json").write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return True
    except Exception as e:
        print(f"[PROTO] Falha ao salvar protocolo: {e}", flush=True)
        return False


def _ensure_test_protocol(folder, default_method="unknown"):
    """
    Se test_protocol.json não existir na pasta, gera a partir do metadata.json
    e salva (retroativo). Usado como fallback na hora do zip de deploy.
    """
    folder = Path(folder)
    tp = folder / "test_protocol.json"
    if tp.exists():
        return True
    meta = {}
    try:
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    baseline = {}
    for n in PARAMETER_NAMES:
        try:
            baseline[n] = float(meta.get("baseline", {}).get(n, PARAMETER_DEFAULTS[n]))
        except Exception:
            baseline[n] = float(PARAMETER_DEFAULTS[n])
    protocol = build_test_protocol(
        metodo=meta.get("method", meta.get("metodo", default_method)),
        duracao=float(meta.get("duration_s", TRIAL_DURATION)),
        freq_hz=float(meta.get("freq_hz", 10.0)),
        baseline=baseline,
        metrics=meta.get("metrics", {"imu": True, "flow": True, "disp": True, "tilt": True}),
        exp_cond=meta.get("experimental_conditions", {}),
        model_tag=str(meta.get("model_tag", "a")),
    )
    return save_experiment_protocol(folder, protocol)


def load_experiment(folder):
    folder = Path(folder)
    meta, sensors, params, perf, proto = {}, [], [], [], {}
    try: meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    except Exception: pass
    try:
        with open(folder / "sensors.csv", "r", encoding="utf-8") as f: sensors = list(csv.DictReader(f))
    except Exception: pass
    try:
        with open(folder / "gait_parameters.csv", "r", encoding="utf-8") as f: params = list(csv.DictReader(f))
    except Exception: pass
    try:
        with open(folder / "performance.csv", "r", encoding="utf-8") as f: perf = list(csv.DictReader(f))
    except Exception: pass
    try: proto = json.loads((folder / "test_protocol.json").read_text(encoding="utf-8"))
    except Exception: pass
    return {"meta": meta, "sensors": sensors, "params": params, "perf": perf, "protocol": proto}

def list_experiments():
    if not EXPERIMENTS_DIR.exists(): return []
    return sorted([p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir()])

def extrair_arquitetura_modelo(model_path):
    info = {"ok": False, "erro": None, "algoritmo": "?", "arquitetura": "?", "entradas": 0,
            "camadas_ocultas": [], "saidas_actor": 0, "saidas_critic": 0, "tipo_saida": "?"}
    try:
        import torch
        from stable_baselines3 import PPO
    except ImportError as exc:
        info["erro"] = f"Dependências: {exc}"; return info
    mf = Path(model_path if str(model_path).endswith(".zip") else str(model_path) + ".zip")
    if not mf.exists(): info["erro"] = f"Não encontrado: {mf}"; return info
    try: model = PPO.load(str(mf.with_suffix("")), device="cpu")
    except Exception as exc: info["erro"] = f"Falha: {exc}"; return info
    try:
        info["algoritmo"] = "PPO"
        policy = model.policy
        if hasattr(policy, "mlp_extractor"): info["arquitetura"] = "MLP"
        if hasattr(model.observation_space, "shape") and model.observation_space.shape:
            info["entradas"] = int(model.observation_space.shape[0])
        def lineares(module):
            return [m.out_features for m in module.modules() if isinstance(m, torch.nn.Linear)]
        if hasattr(policy, "mlp_extractor") and hasattr(policy.mlp_extractor, "policy_net"):
            info["camadas_ocultas"] = lineares(policy.mlp_extractor.policy_net)
        if hasattr(policy, "action_net") and isinstance(policy.action_net, torch.nn.Linear):
            info["saidas_actor"] = int(policy.action_net.out_features)
        if hasattr(policy, "value_net") and isinstance(policy.value_net, torch.nn.Linear):
            info["saidas_critic"] = int(policy.value_net.out_features)
        tipo = type(model.action_space).__name__
        info["tipo_saida"] = "Contínua" if "Box" in tipo else ("Discreta" if "Discrete" in tipo else tipo)
        info["ok"] = True
    except Exception as exc: info["erro"] = f"Extração: {exc}"
    return info

# ============================================================
# FÍSICA HELPERS
# ============================================================
def read_position(): return robot_node.getPosition()

def body_tilt_degrees():
    ax, ay, az, angle = rotation_field.getSFRotation()
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-9 or abs(angle) < 1e-9: return 0.0
    uy = ay / norm
    vy = math.cos(angle) + uy * uy * (1.0 - math.cos(angle))
    vy = max(-1.0, min(1.0, vy))
    return math.degrees(math.acos(vy))

def forward_coordinate(position):
    if FORWARD_AXIS.lower() == "x": return position[0] * FORWARD_SIGN
    if FORWARD_AXIS.lower() == "z": return position[2] * FORWARD_SIGN
    raise ValueError("FORWARD_AXIS deve ser x ou z.")

def hold_robot(trial=-1): send_command({"command": "hold", "trial": trial})

def reset_robot():
    hold_robot()
    translation_field.setSFVec3f(initial_translation)
    rotation_field.setSFRotation(initial_rotation)
    robot_node.resetPhysics()
    for _ in range(SETTLE_STEPS):
        if supervisor.step(timestep) == -1: return False
        ui.update()
        if ui.closed: return False
    return True

def _update_stab_maxima(s, maxima):
    if s is None: return
    try:
        r = abs(float(s.get("roll_deg", 0.0)))
        p = abs(float(s.get("pitch_deg", 0.0)))
        y = abs(float(s.get("yaw_deg", 0.0)))
        if r > maxima["max_roll_abs"]: maxima["max_roll_abs"] = r
        if p > maxima["max_pitch_abs"]: maxima["max_pitch_abs"] = p
        if y > maxima["max_yaw_abs"]: maxima["max_yaw_abs"] = y
    except Exception:
        pass

# ============================================================
# TRIAL CORE
# ============================================================
def run_trial_fixed(trial_index, parameters, mode_name, duration, parent_trial=None,
                    collect_timeseries=False, freq_hz=10.0, metrics=None):
    if not reset_robot(): return None
    start_position = read_position()
    start_forward = forward_coordinate(start_position)
    start_height = start_position[1]
    ui_ref = globals().get("ui")
    servo_delay_ms = float(ui_ref.servo_delay_ms.get()) if ui_ref else 0.0

    send_command({
        "command": "run", "trial": trial_index, "direction": FORWARD_SIGN,
        "servo_delay_ms": servo_delay_ms,
        "tempo_ciclo": parameters.get("tempo_ciclo", TEMPO_CICLO_DEFAULT),
        "abertura_traseira": parameters.get("abertura_traseira", ABERTURA_TRASEIRA_DEFAULT),
        **{n: parameters[n] for n in PARAMETER_NAMES},
    })

    start_time = supervisor.getTime()
    max_tilt, min_height, fallen = 0.0, start_height, False
    fall_time_s = None
    if metrics is None: metrics = {"imu": True, "flow": True, "disp": True, "tilt": True}
    dt_sample = 1.0 / max(0.1, float(freq_hz))
    next_sample = start_time + dt_sample
    timeseries, sensor_ages = [], []
    elapsed = 0.0
    stab_max = {"max_roll_abs": 0.0, "max_pitch_abs": 0.0, "max_yaw_abs": 0.0}

    def _build_row(position, tilt, fall_detected):
        row = {"time_s": round(elapsed, 4), "trial": trial_index + 1, "mode": mode_name,
               "direction": FORWARD_SIGN, "fallen": 1 if fall_detected else 0,
               "x_pos": round(position[0], 4), "y_pos": round(position[1], 4), "z_pos": round(position[2], 4)}
        s = get_latest_sensors()
        if s is not None:
            age_ms = round((time.monotonic() - s.get("_last_wall", 0.0)) * 1000.0, 2)
            row["sensor_age_ms"] = age_ms
            sensor_ages.append(age_ms)
            if metrics.get("imu", True):
                row.update({"roll_deg": round(s["roll_deg"], 3), "pitch_deg": round(s["pitch_deg"], 3), "yaw_deg": round(s["yaw_deg"], 3),
                            "gx": round(s["gx"], 5), "gy": round(s["gy"], 5), "gz": round(s["gz"], 5),
                            "ax": round(s["ax"], 4), "ay": round(s["ay"], 4), "az": round(s["az"], 4)})
            if metrics.get("flow", True):
                row.update({"flow_x_mean": round(s["flow_x_mean"], 5), "flow_y_mean": round(s["flow_y_mean"], 5),
                            "flow_mag_mean": round(s["flow_mag_mean"], 5), "flow_dir_deg": round(s["flow_dir_deg"], 2),
                            "flow_points": int(s["flow_points"]), "flow_var": round(s["flow_var"], 4)})
        if metrics.get("disp", True): row["displacement_m"] = round(forward_coordinate(position) - start_forward, 4)
        if metrics.get("tilt", True): row["tilt_deg"] = round(tilt, 3)
        row.update({k: round(v, 6) for k, v in parameters.items() if isinstance(v, (int, float))})
        return row

    while True:
        if supervisor.step(timestep) == -1: return None
        if not pump_ui():
            hold_robot(trial_index); return None
        _update_stab_maxima(get_latest_sensors(), stab_max)
        now = supervisor.getTime()
        elapsed = now - start_time
        position = read_position()
        min_height = min(min_height, position[1])
        tilt = body_tilt_degrees()
        max_tilt = max(max_tilt, tilt)
        fall_detected = (start_height - min_height > FALL_HEIGHT_DROP) or (tilt > MAX_TILT_DEG)
        sampled = False
        if collect_timeseries and now >= next_sample:
            timeseries.append(_build_row(position, tilt, fall_detected))
            next_sample += dt_sample
            sampled = True
        if fall_detected:
            fallen = True; fall_time_s = round(elapsed, 4)
            if collect_timeseries and not sampled:
                timeseries.append(_build_row(position, tilt, True))
            break
        if elapsed >= duration: break

    end_position = read_position()
    displacement = forward_coordinate(end_position) - start_forward
    score = calculate_score(displacement, max_tilt, fallen)
    result = {
        "trial": trial_index + 1, "mode": mode_name, "direction": FORWARD_SIGN,
        "parent_trial": "" if parent_trial is None else parent_trial + 1,
        **{n: parameters[n] for n in PARAMETER_NAMES},
        "tempo_ciclo": parameters.get("tempo_ciclo", TEMPO_CICLO_DEFAULT),
        "abertura_traseira": parameters.get("abertura_traseira", ABERTURA_TRASEIRA_DEFAULT),
        "duration_s": duration, "elapsed_s": round(elapsed, 4),
        "displacement_m": displacement, "max_tilt_deg": max_tilt,
        "fallen": fallen, "fall_time_s": fall_time_s if fall_time_s is not None else "",
        "score": score,
        "max_roll_deg": round(stab_max["max_roll_abs"], 3),
        "max_pitch_deg": round(stab_max["max_pitch_abs"], 3),
        "max_yaw_deg": round(stab_max["max_yaw_abs"], 3),
    }
    if sensor_ages: result["sensor_age_mean_ms"] = round(statistics.mean(sensor_ages), 2)
    if collect_timeseries: result["timeseries"] = timeseries
    hold_robot(trial_index)
    return result


def run_trial_adaptive(trial_index, model, vecnorm, baseline_params, mode_name, duration,
                       collect_timeseries=True, freq_hz=10.0):
    if not reset_robot(): return None
    hold_robot(trial_index)
    for _ in range(SETTLE_STEPS):
        if supervisor.step(timestep) == -1: return None
        if not pump_ui():
            hold_robot(trial_index); return None
    start_position = read_position()
    start_forward = forward_coordinate(start_position)
    start_height = start_position[1]
    ui_ref = globals().get("ui")
    servo_delay_ms = float(ui_ref.servo_delay_ms.get()) if ui_ref else 0.0
    inference_delay_ms = float(ui_ref.inference_delay_ms.get()) if ui_ref else 0.0

    current_params = dict(baseline_params)
    send_command({"command": "run", "trial": trial_index, "direction": FORWARD_SIGN,
                  "servo_delay_ms": servo_delay_ms,
                  "tempo_ciclo": current_params.get("tempo_ciclo", TEMPO_CICLO_DEFAULT),
                  "abertura_traseira": current_params.get("abertura_traseira", ABERTURA_TRASEIRA_DEFAULT),
                  **{n: current_params[n] for n in PARAMETER_NAMES}})

    start_time = supervisor.getTime()
    max_tilt, min_height, fallen, total_reward, n_updates = 0.0, start_height, False, 0.0, 0
    fall_time_s = None
    dt_sample = 1.0 / max(0.1, float(freq_hz))
    next_sample = start_time + dt_sample
    next_neural_update = start_time + NEURAL_UPDATE_INTERVAL
    timeseries, sensor_ages, latency_samples_ms = [], [], []
    last_pos = start_position
    last_obs_raw = last_obs_norm = last_action = None
    last_sample_time = start_time; elapsed = 0.0
    stab_max = {"max_roll_abs": 0.0, "max_pitch_abs": 0.0, "max_yaw_abs": 0.0}

    def _build_row(position, tilt, fall_detected):
        row = {"time_s": round(elapsed, 4), "trial": trial_index + 1, "mode": mode_name,
               "direction": FORWARD_SIGN, "neural_updates": n_updates,
               "dt_actual_s": round(now - last_sample_time, 4),
               "fallen": 1 if fall_detected else 0,
               "x_pos": round(position[0], 4), "y_pos": round(position[1], 4), "z_pos": round(position[2], 4)}
        s = get_latest_sensors()
        if s is not None:
            age_ms = round((time.monotonic() - s.get("_last_wall", 0.0)) * 1000.0, 2)
            row["sensor_age_ms"] = age_ms
            sensor_ages.append(age_ms)
            row.update({"roll_deg": round(s["roll_deg"], 3), "pitch_deg": round(s["pitch_deg"], 3), "yaw_deg": round(s["yaw_deg"], 3),
                        "gx": round(s["gx"], 5), "gy": round(s["gy"], 5), "gz": round(s["gz"], 5),
                        "ax": round(s["ax"], 4), "ay": round(s["ay"], 4), "az": round(s["az"], 4),
                        "flow_x_mean": round(s["flow_x_mean"], 5), "flow_y_mean": round(s["flow_y_mean"], 5),
                        "flow_mag_mean": round(s["flow_mag_mean"], 5), "flow_dir_deg": round(s["flow_dir_deg"], 2),
                        "flow_points": int(s["flow_points"]), "flow_var": round(s["flow_var"], 4)})
        row["displacement_m"] = round(forward_coordinate(position) - start_forward, 4)
        row["tilt_deg"] = round(tilt, 3)
        row.update({k: round(v, 6) for k, v in current_params.items() if isinstance(v, (int, float))})
        if last_obs_raw is not None:
            for i, v in enumerate(last_obs_raw): row[f"obs_{i}"] = round(v, 6)
        if last_obs_norm is not None:
            for i, v in enumerate(last_obs_norm): row[f"obs_norm_{i}"] = round(v, 6)
        if last_action is not None:
            for i, v in enumerate(last_action): row[f"action_{i}"] = round(v, 6)
        return row

    while True:
        if supervisor.step(timestep) == -1: return None
        if not pump_ui():
            hold_robot(trial_index); return None
        _update_stab_maxima(get_latest_sensors(), stab_max)
        now = supervisor.getTime()
        elapsed = now - start_time
        position = read_position()
        min_height = min(min_height, position[1])
        tilt = body_tilt_degrees()
        max_tilt = max(max_tilt, tilt)
        fall_detected = (start_height - min_height > FALL_HEIGHT_DROP) or (tilt > MAX_TILT_DEG)
        if fall_detected:
            fallen = True; fall_time_s = round(elapsed, 4); total_reward -= 5.0
            if collect_timeseries:
                timeseries.append(_build_row(position, tilt, True))
            break
        if now >= next_neural_update:
            sensors = get_latest_sensors()
            obs_arr = np.asarray(build_observation(sensors, current_params), dtype=np.float32)
            obs_norm_arr = None
            t_pred_start = time.monotonic()
            try:
                if vecnorm is not None:
                    obs_norm_arr = vecnorm.normalize_obs(obs_arr.reshape(1, -1)).astype(np.float32)
                    action, _ = model.predict(obs_norm_arr, deterministic=True); action = action[0]
                else:
                    action, _ = model.predict(obs_arr, deterministic=True)
            except Exception:
                action = [0.0] * ACTION_DIM
            actual_ms = (time.monotonic() - t_pred_start) * 1000.0
            if inference_delay_ms > actual_ms:
                time.sleep((inference_delay_ms - actual_ms) / 1000.0)
            latency_samples_ms.append(actual_ms)

            new_params = apply_delta_action(action, baseline_params)
            send_command({"command": "run", "trial": trial_index, "direction": FORWARD_SIGN,
                          "servo_delay_ms": servo_delay_ms,
                          "tempo_ciclo": new_params.get("tempo_ciclo", TEMPO_CICLO_DEFAULT),
                          "abertura_traseira": new_params.get("abertura_traseira", ABERTURA_TRASEIRA_DEFAULT),
                          **{n: new_params[n] for n in PARAMETER_NAMES}})
            current_params = new_params
            n_updates += 1
            delta_disp = forward_coordinate(position) - forward_coordinate(last_pos)
            total_reward += (10.0 * delta_disp - 0.50 * max(0.0, tilt - 5.0) - 0.05 * abs(float((get_latest_sensors() or {}).get("gz", 0.0))))
            last_pos = position
            next_neural_update = now + NEURAL_UPDATE_INTERVAL
            last_obs_raw = obs_arr.tolist()
            last_obs_norm = obs_norm_arr[0].tolist() if obs_norm_arr is not None else None
            last_action = [float(a) for a in action]
        if collect_timeseries and now >= next_sample:
            timeseries.append(_build_row(position, tilt, False))
            last_sample_time = now
            next_sample += dt_sample
        if elapsed >= duration: break

    end_position = read_position()
    displacement = forward_coordinate(end_position) - start_forward
    total_reward += 0.5 * displacement
    score = calculate_score(displacement, max_tilt, fallen)
    result = {
        "trial": trial_index + 1, "mode": mode_name, "direction": FORWARD_SIGN, "parent_trial": "",
        **{n: current_params[n] for n in PARAMETER_NAMES},
        "tempo_ciclo": current_params.get("tempo_ciclo", TEMPO_CICLO_DEFAULT),
        "abertura_traseira": current_params.get("abertura_traseira", ABERTURA_TRASEIRA_DEFAULT),
        "duration_s": duration, "elapsed_s": round(elapsed, 4),
        "displacement_m": displacement, "max_tilt_deg": max_tilt,
        "fallen": fallen, "fall_time_s": fall_time_s if fall_time_s is not None else "",
        "score": score, "neural_updates": n_updates, "total_reward": total_reward,
        "max_roll_deg": round(stab_max["max_roll_abs"], 3),
        "max_pitch_deg": round(stab_max["max_pitch_abs"], 3),
        "max_yaw_deg": round(stab_max["max_yaw_abs"], 3),
    }
    if sensor_ages: result["sensor_age_mean_ms"] = round(statistics.mean(sensor_ages), 2)
    if latency_samples_ms: result["latencia_politica_ms"] = round(statistics.mean(latency_samples_ms), 3)
    if collect_timeseries: result["timeseries"] = timeseries
    hold_robot(trial_index)
    return result


# ============================================================
# ANÁLISE REAL × SIM (inline)
# ============================================================
def _an_float(v, default=None):
    try: return float(v)
    except Exception: return default

def _an_bool(v):
    if v is None or v == "": return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"): return True
    if s in ("false", "0", "no"): return False
    return None

def _an_load_folder(folder):
    data = load_experiment(folder)
    if not data["meta"] and not data["sensors"] and not data["perf"]:
        return None
    return {
        "folder": str(folder), "name": Path(folder).name,
        "sensors": data["sensors"], "gait_params": data["params"],
        "performance": data["perf"], "metadata": data["meta"], "protocol": data["protocol"],
    }

def _an_standardize(exp, environment):
    if exp is None: return None
    meta = exp["metadata"]
    method = meta.get("method") or meta.get("metodo") or "?"
    ts = []
    for row in exp["sensors"]:
        t = _an_float(row.get("time_s"))
        if t is None: continue
        ts.append({
            "t": t, "roll": _an_float(row.get("roll_deg")),
            "pitch": _an_float(row.get("pitch_deg")), "yaw": _an_float(row.get("yaw_deg")),
            "tilt": _an_float(row.get("tilt_deg")),
            "x_m": _an_float(row.get("x_pos")), "y_m": _an_float(row.get("y_pos")),
            "z_m": _an_float(row.get("z_pos")),
            "displacement_m": _an_float(row.get("displacement_m")),
            "flow_mag": _an_float(row.get("flow_mag_mean")),
        })
    has_disp = any(e["displacement_m"] is not None for e in ts)
    params = {}
    if exp["gait_params"]:
        keys = set()
        for r in exp["gait_params"]: keys.update(r.keys())
        keys.discard("time_s")
        for k in keys:
            vals = [_an_float(r.get(k)) for r in exp["gait_params"] if _an_float(r.get(k)) is not None]
            if vals: params[k] = sum(vals)/len(vals)
    falls = []
    for row in exp["performance"]:
        f = _an_bool(row.get("fallen"))
        ft = _an_float(row.get("fall_time_s"))
        if f is not None or ft is not None:
            falls.append({"fallen": f, "fall_time_s": ft})
    acq = {}
    if exp["performance"]:
        p = exp["performance"][0]
        acq["n_samples"] = int(_an_float(p.get("n_samples"), 0))
        acq["samples_expected"] = int(_an_float(p.get("samples_expected"), 0))
        acq["samples_lost"] = int(_an_float(p.get("samples_lost"), 0))
        acq["sensor_age_mean_ms"] = _an_float(p.get("sensor_age_mean_ms"))
        acq["control_loop_mean_ms"] = _an_float(p.get("control_loop_mean_ms"))
        acq["control_freq_hz_effective"] = _an_float(p.get("control_freq_hz_effective"))
        acq["sample_freq_hz_effective"] = _an_float(p.get("sample_freq_hz_effective"))
        acq["latencia_politica_ms"] = _an_float(p.get("latencia_politica_ms"))
        acq["duration_s"] = _an_float(p.get("duration_s"))
        acq["elapsed_s"] = _an_float(p.get("elapsed_s"))
    return {
        "name": exp["name"], "folder": exp["folder"], "environment": environment,
        "method": method, "timeseries": ts, "params": params, "falls": falls,
        "acquisition": acq, "metadata": meta, "protocol": exp["protocol"],
        "displacement_available": has_disp,
    }

def _an_metrics(std):
    m = {"name": std["name"], "environment": std["environment"],
         "method": std["method"], "displacement_available": std.get("displacement_available", False)}
    ts = std["timeseries"]
    if ts:
        rolls = [e["roll"] for e in ts if e["roll"] is not None]
        pitches = [e["pitch"] for e in ts if e["pitch"] is not None]
        tilts = [e["tilt"] for e in ts if e["tilt"] is not None]
        yaws = [e["yaw"] for e in ts if e["yaw"] is not None]
        m["max_roll_deg"] = max((abs(r) for r in rolls), default=None)
        m["max_pitch_deg"] = max((abs(p) for p in pitches), default=None)
        m["max_tilt_deg"] = max(tilts, default=None)
        m["max_yaw_deg"] = max((abs(y) for y in yaws), default=None)
        m["tilt_rms_deg"] = math.sqrt(sum(t*t for t in tilts)/len(tilts)) if tilts else None
        m["tilt_std_deg"] = statistics.pstdev(tilts) if len(tilts) > 1 else None
        m["tilt_mean_deg"] = statistics.mean(tilts) if tilts else None
    if std.get("displacement_available"):
        disps = [e["displacement_m"] for e in ts if e["displacement_m"] is not None]
        if disps:
            m["displacement_total_m"] = disps[-1] - disps[0] if len(disps) > 1 else disps[-1]
            if len(ts) > 1:
                dur = ts[-1]["t"] - ts[0]["t"]
                m["velocity_mean_m_s"] = m["displacement_total_m"] / dur if dur > 0 else None
            else: m["velocity_mean_m_s"] = None
    else:
        m["displacement_total_m"] = None; m["velocity_mean_m_s"] = None
    falls = std["falls"]
    if falls:
        n_fallen = sum(1 for f in falls if f["fallen"] is True)
        ft = [f["fall_time_s"] for f in falls if f["fall_time_s"] is not None]
        m["fall_count"] = n_fallen; m["fall_times"] = ft
        m["fall_first_s"] = min(ft) if ft else None; m["fall_recorded"] = True
    else:
        m["fall_count"] = None; m["fall_times"] = []; m["fall_first_s"] = None
        m["fall_recorded"] = False
    for k, v in std["params"].items(): m[f"param_{k}"] = v
    for k, v in std["acquisition"].items(): m[f"acq_{k}"] = v
    return m

def _an_compare(real_exps, sim_exps):
    groups = {}
    for exp, env in [(e, "real") for e in real_exps] + [(e, "sim") for e in sim_exps]:
        if exp is None: continue
        groups.setdefault((exp["method"], env), []).append(exp)
    rows = []
    for (method, env), exps in groups.items():
        disp, tilt, fall, yaw = [], [], [], []
        for e in exps:
            m = e.get("_metrics", {})
            if m.get("displacement_total_m") is not None: disp.append(m["displacement_total_m"])
            if m.get("tilt_rms_deg") is not None: tilt.append(m["tilt_rms_deg"])
            if m.get("fall_count") is not None: fall.append(m["fall_count"])
            if m.get("max_yaw_deg") is not None: yaw.append(m["max_yaw_deg"])
        rows.append({
            "method": method, "environment": env, "n_experiments": len(exps),
            "disp_mean_m": statistics.mean(disp) if disp else None,
            "disp_std_m": statistics.stdev(disp) if len(disp) > 1 else None,
            "tilt_rms_mean_deg": statistics.mean(tilt) if tilt else None,
            "tilt_rms_std_deg": statistics.stdev(tilt) if len(tilt) > 1 else None,
            "yaw_max_mean_deg": statistics.mean(yaw) if yaw else None,
            "fall_rate": (sum(fall)/len(fall)) if fall else None,
            "success_rate": (sum(1 for f in fall if f == 0)/len(fall)) if fall else None,
        })
    return rows

def _an_write_csv(rows, path):
    if not rows: return
    all_keys = set()
    for r in rows: all_keys.update(r.keys())
    cols = sorted(all_keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow({k: (r.get(k) if r.get(k) is not None else "") for k in cols})

def _an_plot_series(real_std, sim_std, field, ylabel, title, path):
    fig = Figure(figsize=(10, 4), dpi=100); ax = fig.add_subplot(111)
    def _plot(std, label, color):
        if std is None: return
        pts = [(e["t"], e.get(field)) for e in std["timeseries"] if e.get(field) is not None]
        if pts: ax.plot([t for t,_ in pts], [v for _,v in pts], label=label, color=color, linewidth=1)
    if real_std: _plot(real_std, f"{real_std['name'][:30]} (real)", "#FB923C")
    if sim_std: _plot(sim_std, f"{sim_std['name'][:30]} (sim)", "#3FA9F5")
    ax.set_xlabel("Tempo (s)"); ax.set_ylabel(ylabel); ax.set_title(title)
    if real_std or sim_std: ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")

def _an_plot_bar(pairs, extractor, ylabel, title, path):
    fig = Figure(figsize=(10, 4), dpi=100); ax = fig.add_subplot(111)
    labels, vals, colors = [], [], []
    for pair in pairs:
        for env, color in [("real", "#FB923C"), ("sim", "#3FA9F5")]:
            std = pair.get(env)
            if std is None: continue
            v = extractor(std.get("_metrics", {}))
            if v is None: continue
            labels.append(f"{pair['method']}-{env}"); vals.append(v); colors.append(color)
    if not vals: return
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=100, bbox_inches="tight")

def _an_full_report(selection, output_dir, progress_cb=None):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)
    def _log(msg):
        if progress_cb: progress_cb(msg)
    real_exps, sim_exps = [], []
    for method, paths in selection.items():
        for env_key in ("real", "sim"):
            for path in paths.get(env_key, []):
                _log(f"Carregando {env_key}: {Path(path).name}")
                raw = _an_load_folder(path)
                if raw is None: continue
                std = _an_standardize(raw, environment=env_key)
                if std is None: continue
                if not std.get("method") or std["method"] == "?": std["method"] = method
                std["_metrics"] = _an_metrics(std)
                (real_exps if env_key == "real" else sim_exps).append(std)
    methods = set()
    for s in real_exps + sim_exps: methods.add(s["method"])
    pairs = []
    for method in sorted(methods):
        real_match = next((s for s in real_exps if s["method"] == method), None)
        sim_match = next((s for s in sim_exps if s["method"] == method), None)
        pairs.append({"method": method, "real": real_match, "sim": sim_match})
    metrics_rows = [s["_metrics"] for s in real_exps + sim_exps]
    comparison_rows = _an_compare(real_exps, sim_exps)
    _an_write_csv(metrics_rows, output_dir / "metrics.csv")
    _an_write_csv(comparison_rows, output_dir / "comparison.csv")
    for pair in pairs:
        _log(f"Gráficos: {pair['method']}")
        for field, ylabel in [("roll", "Roll (°)"), ("pitch", "Pitch (°)"),
                              ("tilt", "Tilt (°)"), ("yaw", "Yaw (°)")]:
            _an_plot_series(pair.get("real"), pair.get("sim"), field, ylabel,
                            f"{field.capitalize()} × tempo — {pair['method']}",
                            plots_dir / f"{pair['method']}_{field}.png")
        if (pair.get("real") and pair["real"].get("displacement_available")) or \
           (pair.get("sim") and pair["sim"].get("displacement_available")):
            _an_plot_series(pair.get("real"), pair.get("sim"), "displacement_m", "Deslocamento (m)",
                            f"Deslocamento × tempo — {pair['method']}",
                            plots_dir / f"{pair['method']}_displacement.png")
    _an_plot_bar(pairs, lambda m: m.get("tilt_rms_deg"), "Tilt RMS (°)",
                 "Tilt RMS por método × ambiente", plots_dir / "bar_tilt_rms.png")
    _an_plot_bar(pairs, lambda m: m.get("max_tilt_deg"), "Tilt máx (°)",
                 "Tilt máx por método × ambiente", plots_dir / "bar_max_tilt.png")
    _an_plot_bar(pairs, lambda m: m.get("max_yaw_deg"), "Yaw máx (°)",
                 "Yaw máx por método × ambiente", plots_dir / "bar_max_yaw.png")
    _an_plot_bar(pairs, lambda m: m.get("fall_count"), "Quedas",
                 "Quedas por método × ambiente", plots_dir / "bar_falls.png")
    _an_plot_bar(pairs, lambda m: m.get("displacement_total_m"), "Deslocamento (m)",
                 "Deslocamento total por método × ambiente", plots_dir / "bar_displacement.png")
    summary = {
        "generated_at": datetime.now().isoformat(),
        "selection": {k: {e: [str(p) for p in v.get(e, [])] for e in ("real", "sim")}
                       for k, v in selection.items()},
        "n_real_experiments": len(real_exps), "n_sim_experiments": len(sim_exps),
        "methods": sorted(methods),
        "displacement_pending": [s["name"] for s in real_exps if not s.get("displacement_available")],
        "yaw_limitation_note": YAW_LIMITATION_NOTE,
        "files_generated": {
            "metrics_csv": str(output_dir / "metrics.csv"),
            "comparison_csv": str(output_dir / "comparison.csv"),
            "plots_dir": str(plots_dir),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


# ============================================================
# UI
# ============================================================
FONT = "Segoe UI"
BG = "#0B1118"
PANEL = "#111C27"
PANEL2 = "#1A2938"
FG = "#EAF2F8"
MUTED = "#8FA1B3"
ACCENT = "#3FA9F5"

class Card(tk.Canvas):
    def __init__(self, parent, title, desc, icon, color, command, height=110):
        super().__init__(parent, height=height, bg=BG, highlightthickness=0, cursor="hand2")
        self.title, self.desc, self.icon, self.color, self.command, self.hover = title, desc, icon, color, command, False
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<Button-1>", lambda e: self.command())
    def _set_hover(self, v): self.hover = v; self.redraw()
    def redraw(self):
        self.delete("all")
        w, h = max(self.winfo_width(), 60), max(self.winfo_height(), 60)
        outline = self.color if self.hover else "#1C2A38"
        self.create_polygon([16,0,w-16,0,w,16,w,h-16,w-16,h,16,h,0,h-16,0,16],
                            smooth=True, fill=PANEL, outline=outline, width=2 if self.hover else 1)
        self.create_text(20, 26, text=self.icon, font=("Segoe UI Emoji", 22), fill=self.color, anchor="w")
        self.create_text(20, h-46, anchor="w", text=self.title, fill=FG, font=(FONT, 12, "bold"))
        self.create_text(20, h-22, anchor="w", text=self.desc, fill=MUTED, font=(FONT, 9))


class GaitInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GATO ROBÔ | Framework")
        self.root.geometry("1300x1000")
        self.root.configure(bg=BG)
        self.closed = False
        self.pending_action = None
        self.stop_requested = False
        self.last_fig = None
        self.last_comparison_data = None

        self.exp_method = tk.StringVar(value="manual")
        self.exp_name = tk.StringVar(value="exp_sim")
        self.exp_duration = tk.DoubleVar(value=TRIAL_DURATION)
        self.exp_repetitions = tk.IntVar(value=1)
        self.exp_freq_hz = tk.DoubleVar(value=10.0)
        self.metric_imu = tk.BooleanVar(value=True)
        self.metric_flow = tk.BooleanVar(value=True)
        self.metric_disp = tk.BooleanVar(value=True)
        self.metric_tilt = tk.BooleanVar(value=True)
        self.imu_episodes = tk.IntVar(value=ADAPT_PPO_EPISODES)
        self.search_iterations = tk.IntVar(value=SEARCH_TRIALS)
        self.target_ip = tk.StringVar(value="192.168.0.10")
        self.target_port = tk.IntVar(value=8000)

        ec = load_exp_conditions()
        self.body_mass_kg = tk.DoubleVar(value=BASE_BODY_MASS_KG)
        self.payload_ratio_pct = tk.DoubleVar(value=float(ec.get("payload_ratio_percent", 0.0)))
        self.payload_mass_kg = tk.DoubleVar(value=0.0)
        self.total_mass_kg = tk.DoubleVar(value=BASE_BODY_MASS_KG)
        self.camera_noise_level = tk.StringVar(value=ec.get("camera_noise_level", "disabled"))
        self.camera_noise_sigma_custom = tk.DoubleVar(value=float(ec.get("camera_noise_sigma_custom", 0.10)))
        self.imu_noise_level = tk.StringVar(value=ec.get("imu_noise_level", "disabled"))
        self.imu_noise_sigma_custom = tk.DoubleVar(value=float(ec.get("imu_noise_sigma_custom", 2.5)))
        self.servo_delay_ms = tk.DoubleVar(value=float(ec.get("servo_delay_ms", 0.0)))
        self.inference_delay_ms = tk.DoubleVar(value=float(ec.get("inference_delay_ms", 0.0)))
        self.search_discard_falls = tk.BooleanVar(value=bool(ec.get("search_discard_falls", True)))

        self.ppo_model_tag = tk.StringVar(value=str(ec.get("ppo_model_tag", "a")))
        self.ppo_start_fresh = tk.BooleanVar(value=bool(ec.get("ppo_start_fresh", True)))
        self.ppo_path_label = tk.StringVar()
        self._update_ppo_path_label()

        self.ppo_reward_disp_gain = tk.DoubleVar(value=float(ec.get("ppo_reward_disp_gain", 10.0)))
        self.ppo_reward_tilt_enabled = tk.BooleanVar(value=bool(ec.get("ppo_reward_tilt_enabled", True)))
        self.ppo_reward_tilt_threshold_deg = tk.DoubleVar(value=float(ec.get("ppo_reward_tilt_threshold_deg", 5.0)))
        self.ppo_reward_tilt_coef = tk.DoubleVar(value=float(ec.get("ppo_reward_tilt_coef", 0.50)))
        self.ppo_reward_gz_enabled = tk.BooleanVar(value=bool(ec.get("ppo_reward_gz_enabled", True)))
        self.ppo_reward_gz_coef = tk.DoubleVar(value=float(ec.get("ppo_reward_gz_coef", 0.05)))
        self.ppo_reward_final_disp_bonus = tk.DoubleVar(value=float(ec.get("ppo_reward_final_disp_bonus", 0.5)))

        self.ppo_penalty_enabled = tk.BooleanVar(value=bool(ec.get("ppo_penalty_enabled", False)))
        self.ppo_penalty_value = tk.DoubleVar(value=float(ec.get("ppo_penalty_value", -5.0)))

        self.stab_iter_roll_enabled = tk.BooleanVar(value=bool(ec.get("stab_iter_roll_enabled", False)))
        self.stab_iter_roll_max_deg = tk.DoubleVar(value=float(ec.get("stab_iter_roll_max_deg", 20.0)))
        self.stab_iter_pitch_enabled = tk.BooleanVar(value=bool(ec.get("stab_iter_pitch_enabled", False)))
        self.stab_iter_pitch_max_deg = tk.DoubleVar(value=float(ec.get("stab_iter_pitch_max_deg", 20.0)))
        self.stab_iter_yaw_enabled = tk.BooleanVar(value=bool(ec.get("stab_iter_yaw_enabled", False)))
        self.stab_iter_yaw_max_deg = tk.DoubleVar(value=float(ec.get("stab_iter_yaw_max_deg", 30.0)))

        self.stab_ppo_roll_enabled = tk.BooleanVar(value=bool(ec.get("stab_ppo_roll_enabled", False)))
        self.stab_ppo_roll_target_deg = tk.DoubleVar(value=float(ec.get("stab_ppo_roll_target_deg", 0.0)))
        self.stab_ppo_roll_weight = tk.DoubleVar(value=float(ec.get("stab_ppo_roll_weight", 1.0)))
        self.stab_ppo_roll_scale_deg = tk.DoubleVar(value=float(ec.get("stab_ppo_roll_scale_deg", 45.0)))
        self.stab_ppo_pitch_enabled = tk.BooleanVar(value=bool(ec.get("stab_ppo_pitch_enabled", False)))
        self.stab_ppo_pitch_target_deg = tk.DoubleVar(value=float(ec.get("stab_ppo_pitch_target_deg", 0.0)))
        self.stab_ppo_pitch_weight = tk.DoubleVar(value=float(ec.get("stab_ppo_pitch_weight", 1.0)))
        self.stab_ppo_pitch_scale_deg = tk.DoubleVar(value=float(ec.get("stab_ppo_pitch_scale_deg", 45.0)))
        self.stab_ppo_yaw_enabled = tk.BooleanVar(value=bool(ec.get("stab_ppo_yaw_enabled", False)))
        self.stab_ppo_yaw_target_deg = tk.DoubleVar(value=float(ec.get("stab_ppo_yaw_target_deg", 0.0)))
        self.stab_ppo_yaw_weight = tk.DoubleVar(value=float(ec.get("stab_ppo_yaw_weight", 1.0)))
        self.stab_ppo_yaw_scale_deg = tk.DoubleVar(value=float(ec.get("stab_ppo_yaw_scale_deg", 180.0)))

        self.ppo_model_tag.trace_add("write", lambda *a: self._update_ppo_path_label())

        self.ajustador_vars = {}
        preset = load_preset()
        for n in PARAMETER_NAMES: self.ajustador_vars[n] = tk.DoubleVar(value=preset[n])
        self.ajustador_vars["tempo_ciclo"] = tk.DoubleVar(value=preset["tempo_ciclo"])
        self.ajustador_vars["abertura_traseira"] = tk.DoubleVar(value=preset["abertura_traseira"])
        self._recalculate_mass()

        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background=BG)
        self.style.configure("TLabel", background=BG, foreground=FG, font=(FONT, 10))
        self.style.configure("Section.TLabel", font=(FONT, 14, "bold"), foreground=ACCENT, background=BG)
        self.style.configure("TButton", font=(FONT, 10, "bold"), padding=6)

        self.main_area = ttk.Frame(self.root, padding=20)
        self.main_area.pack(fill="both", expand=True)
        log_frame = ttk.Frame(self.root, padding=10)
        log_frame.pack(side="bottom", fill="x")
        self.log = tk.Text(log_frame, bg=PANEL2, fg=FG, font=("Consolas", 9), height=5, relief="flat")
        self.log.pack(fill="x")
        self.status_var = tk.StringVar(value="Pronto.")
        ttk.Label(log_frame, textvariable=self.status_var, foreground=MUTED).pack(anchor="w", pady=4)

        self.show_home()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ---------- helpers base ----------
    def write(self, msg):
        try:
            self.log.insert("end", str(msg) + "\n")
            self.log.see("end")
        except Exception:
            pass

    def request(self, action):
        self.pending_action = action
        self.stop_requested = False

    def pump_ui(self):
        try:
            self.root.update()
        except tk.TclError:
            self.closed = True
            return False
        return not self.closed

    def update(self):
        return self.pump_ui()

    def close(self):
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def _update_ppo_path_label(self):
        try:
            p = _ppo_paths(self.ppo_model_tag.get())
            self.ppo_path_label.set(f"model: {p['model'].name}  |  vecnorm: {p['vecnorm'].name}")
        except Exception:
            self.ppo_path_label.set("")

    def _recalculate_mass(self):
        try:
            base = float(self.body_mass_kg.get()); ratio = float(self.payload_ratio_pct.get())
            payload = base * ratio / 100.0
            self.payload_mass_kg.set(payload)
            self.total_mass_kg.set(base + payload)
        except Exception: pass

    def _apply_mass_before_trial(self):
        total = float(self.total_mass_kg.get())
        if _chassis_mass_field is None:
            self.write("[MASS] AVISO: chassi não encontrado."); return
        if _wbt_chassis_mass_kg is not None and abs(_wbt_chassis_mass_kg - BASE_BODY_MASS_KG) > 0.05:
            self.write(f"[MASS] wbt={_wbt_chassis_mass_kg:.2f} kg (base={BASE_BODY_MASS_KG:.2f}). Edite o .wbt se necessário.")
        if apply_total_mass(total):
            self.write(f"[MASS] Aplicado {total:.3f} kg.")

    def get_effective_imu_sigma(self):
        lvl = str(self.imu_noise_level.get())
        if lvl == "disabled": return 0.0
        if lvl == "custom":
            try: return float(self.imu_noise_sigma_custom.get())
            except Exception: return 0.0
        return float(NOISE_PRESETS.get(lvl, {}).get("imu_sigma_deg", 0.0))

    def get_effective_camera_sigma(self):
        lvl = str(self.camera_noise_level.get())
        if lvl == "disabled": return 0.0
        if lvl == "custom":
            try: return float(self.camera_noise_sigma_custom.get())
            except Exception: return 0.0
        return float(NOISE_PRESETS.get(lvl, {}).get("cam_sigma", 0.0))

    def _current_exp_conditions(self):
        return {
            "payload_ratio_percent": float(self.payload_ratio_pct.get()),
            "camera_noise_level": str(self.camera_noise_level.get()),
            "camera_noise_sigma_custom": float(self.camera_noise_sigma_custom.get()),
            "imu_noise_level": str(self.imu_noise_level.get()),
            "imu_noise_sigma_custom": float(self.imu_noise_sigma_custom.get()),
            "servo_delay_ms": float(self.servo_delay_ms.get()),
            "inference_delay_ms": float(self.inference_delay_ms.get()),
            "search_discard_falls": bool(self.search_discard_falls.get()),
            "ppo_model_tag": _sanitize_tag(self.ppo_model_tag.get()),
            "ppo_start_fresh": bool(self.ppo_start_fresh.get()),
            "ppo_reward_disp_gain": float(self.ppo_reward_disp_gain.get()),
            "ppo_reward_tilt_enabled": bool(self.ppo_reward_tilt_enabled.get()),
            "ppo_reward_tilt_threshold_deg": float(self.ppo_reward_tilt_threshold_deg.get()),
            "ppo_reward_tilt_coef": float(self.ppo_reward_tilt_coef.get()),
            "ppo_reward_gz_enabled": bool(self.ppo_reward_gz_enabled.get()),
            "ppo_reward_gz_coef": float(self.ppo_reward_gz_coef.get()),
            "ppo_reward_final_disp_bonus": float(self.ppo_reward_final_disp_bonus.get()),
            "ppo_penalty_enabled": bool(self.ppo_penalty_enabled.get()),
            "ppo_penalty_value": float(self.ppo_penalty_value.get()),
            "stab_iter_roll_enabled": bool(self.stab_iter_roll_enabled.get()),
            "stab_iter_roll_max_deg": float(self.stab_iter_roll_max_deg.get()),
            "stab_iter_pitch_enabled": bool(self.stab_iter_pitch_enabled.get()),
            "stab_iter_pitch_max_deg": float(self.stab_iter_pitch_max_deg.get()),
            "stab_iter_yaw_enabled": bool(self.stab_iter_yaw_enabled.get()),
            "stab_iter_yaw_max_deg": float(self.stab_iter_yaw_max_deg.get()),
            "stab_ppo_roll_enabled": bool(self.stab_ppo_roll_enabled.get()),
            "stab_ppo_roll_target_deg": float(self.stab_ppo_roll_target_deg.get()),
            "stab_ppo_roll_weight": float(self.stab_ppo_roll_weight.get()),
            "stab_ppo_roll_scale_deg": float(self.stab_ppo_roll_scale_deg.get()),
            "stab_ppo_pitch_enabled": bool(self.stab_ppo_pitch_enabled.get()),
            "stab_ppo_pitch_target_deg": float(self.stab_ppo_pitch_target_deg.get()),
            "stab_ppo_pitch_weight": float(self.stab_ppo_pitch_weight.get()),
            "stab_ppo_pitch_scale_deg": float(self.stab_ppo_pitch_scale_deg.get()),
            "stab_ppo_yaw_enabled": bool(self.stab_ppo_yaw_enabled.get()),
            "stab_ppo_yaw_target_deg": float(self.stab_ppo_yaw_target_deg.get()),
            "stab_ppo_yaw_weight": float(self.stab_ppo_yaw_weight.get()),
            "stab_ppo_yaw_scale_deg": float(self.stab_ppo_yaw_scale_deg.get()),
        }

    def _save_exp_conditions_ui(self):
        if save_exp_conditions_dict(self._current_exp_conditions()):
            messagebox.showinfo("OK", f"Condições salvas em {EXP_CONDITIONS_JSON}")
        else: messagebox.showerror("Erro", "Falha ao salvar.")

    def _reload_exp_conditions_ui(self):
        ec = load_exp_conditions()
        self.payload_ratio_pct.set(float(ec.get("payload_ratio_percent", 0.0)))
        self.camera_noise_level.set(ec.get("camera_noise_level", "disabled"))
        self.camera_noise_sigma_custom.set(float(ec.get("camera_noise_sigma_custom", 0.10)))
        self.imu_noise_level.set(ec.get("imu_noise_level", "disabled"))
        self.imu_noise_sigma_custom.set(float(ec.get("imu_noise_sigma_custom", 2.5)))
        self.servo_delay_ms.set(float(ec.get("servo_delay_ms", 0.0)))
        self.inference_delay_ms.set(float(ec.get("inference_delay_ms", 0.0)))
        self.search_discard_falls.set(bool(ec.get("search_discard_falls", True)))
        self.ppo_model_tag.set(str(ec.get("ppo_model_tag", "a")))
        self.ppo_start_fresh.set(bool(ec.get("ppo_start_fresh", True)))
        self.ppo_reward_disp_gain.set(float(ec.get("ppo_reward_disp_gain", 10.0)))
        self.ppo_reward_tilt_enabled.set(bool(ec.get("ppo_reward_tilt_enabled", True)))
        self.ppo_reward_tilt_threshold_deg.set(float(ec.get("ppo_reward_tilt_threshold_deg", 5.0)))
        self.ppo_reward_tilt_coef.set(float(ec.get("ppo_reward_tilt_coef", 0.50)))
        self.ppo_reward_gz_enabled.set(bool(ec.get("ppo_reward_gz_enabled", True)))
        self.ppo_reward_gz_coef.set(float(ec.get("ppo_reward_gz_coef", 0.05)))
        self.ppo_reward_final_disp_bonus.set(float(ec.get("ppo_reward_final_disp_bonus", 0.5)))
        self.ppo_penalty_enabled.set(bool(ec.get("ppo_penalty_enabled", False)))
        self.ppo_penalty_value.set(float(ec.get("ppo_penalty_value", -5.0)))
        self.stab_iter_roll_enabled.set(bool(ec.get("stab_iter_roll_enabled", False)))
        self.stab_iter_roll_max_deg.set(float(ec.get("stab_iter_roll_max_deg", 20.0)))
        self.stab_iter_pitch_enabled.set(bool(ec.get("stab_iter_pitch_enabled", False)))
        self.stab_iter_pitch_max_deg.set(float(ec.get("stab_iter_pitch_max_deg", 20.0)))
        self.stab_iter_yaw_enabled.set(bool(ec.get("stab_iter_yaw_enabled", False)))
        self.stab_iter_yaw_max_deg.set(float(ec.get("stab_iter_yaw_max_deg", 30.0)))
        self.stab_ppo_roll_enabled.set(bool(ec.get("stab_ppo_roll_enabled", False)))
        self.stab_ppo_roll_target_deg.set(float(ec.get("stab_ppo_roll_target_deg", 0.0)))
        self.stab_ppo_roll_weight.set(float(ec.get("stab_ppo_roll_weight", 1.0)))
        self.stab_ppo_roll_scale_deg.set(float(ec.get("stab_ppo_roll_scale_deg", 45.0)))
        self.stab_ppo_pitch_enabled.set(bool(ec.get("stab_ppo_pitch_enabled", False)))
        self.stab_ppo_pitch_target_deg.set(float(ec.get("stab_ppo_pitch_target_deg", 0.0)))
        self.stab_ppo_pitch_weight.set(float(ec.get("stab_ppo_pitch_weight", 1.0)))
        self.stab_ppo_pitch_scale_deg.set(float(ec.get("stab_ppo_pitch_scale_deg", 45.0)))
        self.stab_ppo_yaw_enabled.set(bool(ec.get("stab_ppo_yaw_enabled", False)))
        self.stab_ppo_yaw_target_deg.set(float(ec.get("stab_ppo_yaw_target_deg", 0.0)))
        self.stab_ppo_yaw_weight.set(float(ec.get("stab_ppo_yaw_weight", 1.0)))
        self.stab_ppo_yaw_scale_deg.set(float(ec.get("stab_ppo_yaw_scale_deg", 180.0)))
        self._recalculate_mass()
        self._update_ppo_path_label()
        self.write("Condições recarregadas.")

    def _apply_ppo_preset_A(self):
        self.ppo_reward_disp_gain.set(10.0)
        self.ppo_reward_tilt_enabled.set(False)
        self.ppo_reward_tilt_threshold_deg.set(5.0)
        self.ppo_reward_tilt_coef.set(0.0)
        self.ppo_reward_gz_enabled.set(False)
        self.ppo_reward_gz_coef.set(0.0)
        self.ppo_reward_final_disp_bonus.set(0.5)
        self.ppo_penalty_enabled.set(False)
        self.ppo_penalty_value.set(-5.0)
        self.stab_ppo_roll_enabled.set(False)
        self.stab_ppo_pitch_enabled.set(False)
        self.stab_ppo_yaw_enabled.set(False)
        self.ppo_model_tag.set("a")
        self.ppo_start_fresh.set(True)
        self.write("[PPO] Preset A (Desempenho puro) aplicado — reward = disp_gain*Δd + final_bonus*D_total")

    def _apply_ppo_preset_B(self):
        self.ppo_reward_disp_gain.set(10.0)
        self.ppo_reward_tilt_enabled.set(True)
        self.ppo_reward_tilt_threshold_deg.set(5.0)
        self.ppo_reward_tilt_coef.set(0.50)
        self.ppo_reward_gz_enabled.set(True)
        self.ppo_reward_gz_coef.set(0.05)
        self.ppo_reward_final_disp_bonus.set(0.5)
        self.ppo_penalty_enabled.set(True)
        self.ppo_penalty_value.set(-5.0)
        self.stab_ppo_roll_enabled.set(True)
        self.stab_ppo_roll_target_deg.set(0.0)
        self.stab_ppo_roll_weight.set(1.0)
        self.stab_ppo_roll_scale_deg.set(45.0)
        self.stab_ppo_pitch_enabled.set(True)
        self.stab_ppo_pitch_target_deg.set(0.0)
        self.stab_ppo_pitch_weight.set(1.0)
        self.stab_ppo_pitch_scale_deg.set(45.0)
        self.stab_ppo_yaw_enabled.set(False)
        self.ppo_model_tag.set("b")
        self.ppo_start_fresh.set(True)
        self.write("[PPO] Preset B (Estabilidade) aplicado — reward = disp_gain*Δd - tilt - gz - P_roll - P_pitch - fall")

    def _clear_main(self):
        for w in self.main_area.winfo_children(): w.destroy()

    def show_home(self):
        self._clear_main()
        header = ttk.Frame(self.main_area); header.pack(fill="x", pady=(0, 15))
        ttk.Label(header, text="Framework de experimentação e controle de marcha", style="Section.TLabel").pack(anchor="w")
        content = ttk.Frame(self.main_area); content.pack(fill="both", expand=True)
        left = ttk.Frame(content); left.pack(side="left", fill="both", expand=True, padx=(0, 20))
        img_canvas = tk.Canvas(left, bg=PANEL, highlightthickness=0); img_canvas.pack(fill="both", expand=True)
        try:
            img_pil = Image.open("capa_robo.png").resize((560, 600), Image.Resampling.LANCZOS)
            self.cover_img = ImageTk.PhotoImage(img_pil)
            img_canvas.create_image(260,200, image=self.cover_img)
        except Exception as e:
            img_canvas.create_text(300, 300, text="[ Imagem não encontrada ]", fill=MUTED, font=(FONT, 14))
            print(f"Erro imagem: {e}")
        right = ttk.Frame(content); right.pack(side="right", fill="both")
        cards = [
            ("1 — Ensaios", "Executar experimentos.", "🧪", "#3FA9F5", self._menu_ensaios),
            ("2 — Métodos", "Manual, iterativo, PPO.", "⚙️", "#8B5CF6", self._menu_metodos),
            ("3 — Análise Local", "Gráficos do supervisor.", "📊", "#34D399", self._menu_analise),
            ("4 — Comunicação", "Enviar ao robô físico.", "🔌", "#FB923C", self._menu_comunicacao),
            ("5 — Condições Experimentais", "Massa, ruído, atrasos.", "⚖️", "#F59E0B", self._menu_condicoes),
            ("6 — Análise Real × Sim", "Comparar Webots e real.", "🔬", "#EC4899", self._menu_analise_rv_sim),
        ]
        for i, (t, d, ic, c, cmd) in enumerate(cards):
            r, cc = i // 2, i % 2
            Card(right, t, d, ic, c, cmd, height=110).grid(row=r, column=cc, sticky="nsew", padx=4, pady=4)
        right.columnconfigure(0, weight=1); right.columnconfigure(1, weight=1)
        for r in range(3): right.rowconfigure(r, weight=1)

    def _topbar(self, title, color=ACCENT):
        top = ttk.Frame(self.main_area); top.pack(fill="x", pady=(0, 15))
        ttk.Button(top, text="← Voltar", command=self.show_home).pack(side="left")
        lbl = ttk.Label(top, text=title, style="Section.TLabel"); lbl.configure(foreground=color)
        lbl.pack(side="left", padx=20)

    def _menu_ensaios(self):
        self._clear_main(); self._topbar("1 — Ensaios", "#3FA9F5")
        box = ttk.Frame(self.main_area); box.pack(fill="both", expand=True)
        ttk.Label(box, text="Configuração do Experimento", font=(FONT, 11, "bold")).pack(anchor="w", pady=(0, 10))
        frm = ttk.Frame(box); frm.pack(fill="x", pady=5)
        ttk.Label(frm, text="Nome:", width=15).grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.exp_name, width=30).grid(row=0, column=1, sticky="w")
        ttk.Label(frm, text="Duração (s):", width=15).grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.exp_duration, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(frm, text="Repetições:", width=15).grid(row=2, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.exp_repetitions, width=10).grid(row=2, column=1, sticky="w")
        ttk.Label(frm, text="Freq. (Hz):", width=15).grid(row=3, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.exp_freq_hz, width=10).grid(row=3, column=1, sticky="w")

        ttk.Label(box, text="Método:", font=(FONT, 11, "bold")).pack(anchor="w", pady=(15, 5))
        for text, val in [("Ajustador (Manual)", "manual"), ("Busca Iterativa", "iterativo"), ("PPO (nn_imu)", "nn_imu")]:
            ttk.Radiobutton(box, text=text, value=val, variable=self.exp_method).pack(anchor="w", pady=2)

        frm_tag = ttk.Frame(box); frm_tag.pack(anchor="w", pady=(8, 5))
        ttk.Label(frm_tag, text="Modelo PPO (tag):").pack(side="left")
        ttk.Entry(frm_tag, textvariable=self.ppo_model_tag, width=8).pack(side="left", padx=5)
        ttk.Label(frm_tag, text="(usado apenas se método = PPO; ex: 'a', 'b')",
                  foreground=MUTED, font=(FONT, 8)).pack(side="left")

        ttk.Label(box, text="Métricas:", font=(FONT, 11, "bold")).pack(anchor="w", pady=(15, 5))
        mfr = ttk.Frame(box); mfr.pack(anchor="w")
        ttk.Checkbutton(mfr, text="IMU", variable=self.metric_imu).pack(side="left", padx=5)
        ttk.Checkbutton(mfr, text="Fluxo", variable=self.metric_flow).pack(side="left", padx=5)
        ttk.Checkbutton(mfr, text="Deslocamento", variable=self.metric_disp).pack(side="left", padx=5)
        ttk.Checkbutton(mfr, text="Tilt", variable=self.metric_tilt).pack(side="left", padx=5)
        ttk.Button(box, text="▶ Iniciar Experimento", command=lambda: self.request("run_experiment")).pack(anchor="w", pady=20)

    def _menu_metodos(self):
        self._clear_main(); self._topbar("2 — Métodos de Marcha", "#8B5CF6")
        nb = ttk.Notebook(self.main_area); nb.pack(fill="both", expand=True)
        tab_manual = ttk.Frame(nb, padding=10); nb.add(tab_manual, text="Ajuste Manual")
        tab_iter = ttk.Frame(nb, padding=10); nb.add(tab_iter, text="Busca Iterativa")
        tab_ppo = ttk.Frame(nb, padding=10); nb.add(tab_ppo, text="Treinamento PPO")

        body = ttk.Frame(tab_manual); body.pack(fill="both", expand=True)
        left = ttk.Frame(body); left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(body); right.pack(side="right", fill="y", padx=(10, 0))
        for k, var in self.ajustador_vars.items():
            row = ttk.Frame(left); row.pack(fill="x", pady=4)
            ttk.Label(row, text=k, width=15).pack(side="left")
            bounds = PARAMETER_BOUNDS.get(k, TEMPO_CICLO_BOUNDS if k == "tempo_ciclo" else ABERTURA_TRASEIRA_BOUNDS)
            ttk.Label(row, text=f"{bounds[0]:.2f}", width=5, foreground=MUTED).pack(side="left")
            ttk.Scale(row, from_=bounds[0], to=bounds[1], variable=var, orient="horizontal").pack(side="left", fill="x", expand=True, padx=5)
            ttk.Label(row, text=f"{bounds[1]:.2f}", width=5, foreground=MUTED).pack(side="left")
            val = ttk.Label(row, text=f"{var.get():.3f}", width=8, font=("Consolas", 10, "bold"), foreground=ACCENT)
            val.pack(side="left", padx=(10, 0))
            def _cb(v=var, lbl=val):
                def c(*a): lbl.config(text=f"{v.get():.3f}"); self._update_preview()
                return c
            var.trace_add("write", _cb())
        self.preview = GaitPreview(right, PARAMETER_BOUNDS, width=400, height=300)
        self.preview.pack(); self._update_preview()
        btns = ttk.Frame(left); btns.pack(fill="x", pady=15)
        ttk.Button(btns, text="Testar (1 trial)", command=lambda: self.request("tuner_test")).pack(side="left", padx=5)
        ttk.Button(btns, text="Salvar no Ajustador", command=self._save_preset_from_ui).pack(side="left", padx=5)
        ttk.Button(btns, text="Recarregar", command=self._reload_preset_ui).pack(side="left", padx=5)

        ttk.Label(tab_iter, text="Busca iterativa — otimização automática", font=(FONT, 11, "bold")).pack(anchor="w", pady=5)
        fi = ttk.Frame(tab_iter); fi.pack(anchor="w", pady=5)
        ttk.Label(fi, text="Iterações:").pack(side="left")
        ttk.Entry(fi, textvariable=self.search_iterations, width=10).pack(side="left", padx=5)
        ttk.Checkbutton(tab_iter, text="Descartar iterações com queda",
                        variable=self.search_discard_falls).pack(anchor="w", pady=2)

        stab_it = ttk.LabelFrame(tab_iter, text="Critérios de estabilidade (limites máximos)", padding=8)
        stab_it.pack(fill="x", pady=8)
        for axis, lbl, en_var, max_var in [
            ("roll", "Roll", self.stab_iter_roll_enabled, self.stab_iter_roll_max_deg),
            ("pitch", "Pitch", self.stab_iter_pitch_enabled, self.stab_iter_pitch_max_deg),
            ("yaw", "Yaw", self.stab_iter_yaw_enabled, self.stab_iter_yaw_max_deg),
        ]:
            row = ttk.Frame(stab_it); row.pack(fill="x", pady=2)
            ttk.Checkbutton(row, text=f"{lbl} (máx):", variable=en_var, width=12).pack(side="left")
            ttk.Entry(row, textvariable=max_var, width=8).pack(side="left", padx=(4, 0))
            ttk.Label(row, text="°").pack(side="left")
        ttk.Label(stab_it, text="Iterações que ultrapassarem algum limite ativo serão descartadas "
                                "(registradas com discard_reason).",
                  foreground=MUTED, font=(FONT, 8)).pack(anchor="w", pady=(4, 0))
        ttk.Label(stab_it, text="⚠ Yaw real (MPU6050 sem magnetômetro) acumula drift; sim é absoluto.",
                  foreground="#FB923C", font=(FONT, 8)).pack(anchor="w", pady=(2, 0))
        ttk.Button(tab_iter, text="▶ Iniciar Busca", command=lambda: self.request("search")).pack(anchor="w", pady=10)

        ttk.Label(tab_ppo, text="Treinamento da Rede Neural (PPO)", font=(FONT, 11, "bold")).pack(anchor="w", pady=5)
        ppo_body = ttk.Frame(tab_ppo)
        ppo_body.pack(fill="both", expand=True)

        ppo_left = ttk.Frame(ppo_body)
        ppo_left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        model_frame = ttk.LabelFrame(ppo_left, text="Modelo (isolamento entre PPO-A e PPO-B)", padding=8)
        model_frame.pack(fill="x", pady=6)
        r = ttk.Frame(model_frame); r.pack(fill="x", pady=3)
        ttk.Label(r, text="Tag do modelo:", width=20).pack(side="left")
        ttk.Entry(r, textvariable=self.ppo_model_tag, width=10).pack(side="left")
        ttk.Label(r, text="  (ex: 'a', 'b')", foreground=MUTED, font=(FONT, 8)).pack(side="left")
        r = ttk.Frame(model_frame); r.pack(fill="x", pady=3)
        ttk.Label(r, textvariable=self.ppo_path_label, foreground=MUTED, font=("Consolas", 8)).pack(anchor="w")
        r = ttk.Frame(model_frame); r.pack(fill="x", pady=3)
        ttk.Checkbutton(r, text="Iniciar do zero (não carrega modelo)", variable=self.ppo_start_fresh).pack(side="left")

        presets_frame = ttk.LabelFrame(ppo_left, text="Presets rápidos", padding=8)
        presets_frame.pack(fill="x", pady=6)
        r = ttk.Frame(presets_frame); r.pack(fill="x", pady=2)
        ttk.Button(r, text="Preset PPO-A (Desempenho)", command=self._apply_ppo_preset_A).pack(side="left", padx=2)
        ttk.Button(r, text="Preset PPO-B (Estabilidade)", command=self._apply_ppo_preset_B).pack(side="left", padx=2)
        ttk.Label(presets_frame, text="A: apenas deslocamento. B: deslocamento + estabilidade.", foreground=MUTED, font=(FONT, 8)).pack(anchor="w", pady=(3, 0))

        base_frame = ttk.LabelFrame(ppo_left, text="Recompensa base (aplicada a cada passo)", padding=8)
        base_frame.pack(fill="x", pady=6)
        r = ttk.Frame(base_frame); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Episódios:", width=20).pack(side="left")
        ttk.Entry(r, textvariable=self.imu_episodes, width=10).pack(side="left")
        r = ttk.Frame(base_frame); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Ganho por deslocamento:", width=20).pack(side="left")
        ttk.Entry(r, textvariable=self.ppo_reward_disp_gain, width=10).pack(side="left")
        r = ttk.Frame(base_frame); r.pack(fill="x", pady=2)
        ttk.Checkbutton(r, text="Penalização de tilt", variable=self.ppo_reward_tilt_enabled).pack(side="left")
        ttk.Label(r, text="  limiar:").pack(side="left")
        ttk.Entry(r, textvariable=self.ppo_reward_tilt_threshold_deg, width=5).pack(side="left")
        ttk.Label(r, text="° coef:").pack(side="left")
        ttk.Entry(r, textvariable=self.ppo_reward_tilt_coef, width=5).pack(side="left")
        r = ttk.Frame(base_frame); r.pack(fill="x", pady=2)
        ttk.Checkbutton(r, text="Penalização de gz (yaw rate)", variable=self.ppo_reward_gz_enabled).pack(side="left")
        ttk.Label(r, text="  coef:").pack(side="left")
        ttk.Entry(r, textvariable=self.ppo_reward_gz_coef, width=5).pack(side="left")
        r = ttk.Frame(base_frame); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Bônus final por deslocamento:", width=25).pack(side="left")
        ttk.Entry(r, textvariable=self.ppo_reward_final_disp_bonus, width=8).pack(side="left")

        fall_frame = ttk.LabelFrame(ppo_left, text="Penalidade por queda (1× por episódio)", padding=8)
        fall_frame.pack(fill="x", pady=6)
        r = ttk.Frame(fall_frame); r.pack(fill="x", pady=2)
        ttk.Checkbutton(r, text="Aplicar penalidade", variable=self.ppo_penalty_enabled).pack(side="left")
        ttk.Label(r, text="  Valor:").pack(side="left", padx=(10, 2))
        ttk.Entry(r, textvariable=self.ppo_penalty_value, width=8).pack(side="left")

        stab_ppo = ttk.LabelFrame(ppo_left, text="Penalização de estabilidade por eixo", padding=8)
        stab_ppo.pack(fill="x", pady=6)
        header = ttk.Frame(stab_ppo); header.pack(fill="x", pady=(0, 4))
        for txt, w in [("Ativar", 8), ("Alvo (°)", 10), ("Peso", 8), ("Escala (°)", 10)]:
            ttk.Label(header, text=txt, width=w, font=(FONT, 8, "bold"), foreground=MUTED).pack(side="left", padx=2)
        for axis, lbl, en_var, tgt_var, w_var, sc_var in [
            ("roll", "Roll", self.stab_ppo_roll_enabled, self.stab_ppo_roll_target_deg,
             self.stab_ppo_roll_weight, self.stab_ppo_roll_scale_deg),
            ("pitch", "Pitch", self.stab_ppo_pitch_enabled, self.stab_ppo_pitch_target_deg,
             self.stab_ppo_pitch_weight, self.stab_ppo_pitch_scale_deg),
            ("yaw", "Yaw", self.stab_ppo_yaw_enabled, self.stab_ppo_yaw_target_deg,
             self.stab_ppo_yaw_weight, self.stab_ppo_yaw_scale_deg),
        ]:
            row = ttk.Frame(stab_ppo); row.pack(fill="x", pady=2)
            ttk.Checkbutton(row, text=lbl, variable=en_var, width=8).pack(side="left")
            ttk.Entry(row, textvariable=tgt_var, width=10).pack(side="left", padx=2)
            ttk.Entry(row, textvariable=w_var, width=8).pack(side="left", padx=2)
            ttk.Entry(row, textvariable=sc_var, width=10).pack(side="left", padx=2)

        ppo_right = ttk.Frame(ppo_body)
        ppo_right.pack(side="right", fill="both", expand=True, padx=(10, 0))

        ttk.Label(ppo_right, text="Arquitetura da Rede (após treinar)", font=(FONT, 10, "bold")).pack(anchor="w", pady=(5, 0))
        self.arch_canvas = tk.Canvas(ppo_right, bg=PANEL2, highlightthickness=0)
        self.arch_canvas.pack(fill="both", expand=True, pady=10)

        btns_ppo = ttk.Frame(ppo_right); btns_ppo.pack(fill="x", pady=10)
        ttk.Button(btns_ppo, text="▶ Treinar Rede", command=lambda: self.request("imu_train")).pack(side="left", padx=4)
        ttk.Button(btns_ppo, text="Desenhar Arquitetura", command=self._draw_network).pack(side="left", padx=4)

        btns_save = ttk.Frame(ppo_right); btns_save.pack(fill="x", pady=5)
        ttk.Button(btns_save, text="💾 Salvar condições", command=self._save_exp_conditions_ui).pack(side="left", padx=4)
        ttk.Button(btns_save, text="🔄 Recarregar", command=self._reload_exp_conditions_ui).pack(side="left", padx=4)

    def _update_preview(self):
        self.preview.set_preset({k: float(v.get()) for k, v in self.ajustador_vars.items()})

    def _draw_network(self):
        self.arch_canvas.delete("all")
        p = _ppo_paths(self.ppo_model_tag.get())
        candidates = [p["model_base"], _legacy_ppo_paths()["model_base"]]
        model_base = None
        for c in candidates:
            if Path(str(c) + ".zip").exists():
                model_base = str(c); break
        if model_base is None:
            self.arch_canvas.create_text(250, 80,
                text=f"Modelo não encontrado (tag '{_sanitize_tag(self.ppo_model_tag.get())}')",
                fill="#FF5A5A", font=(FONT, 10))
            return
        info = extrair_arquitetura_modelo(model_base)
        if not info["ok"]:
            self.arch_canvas.create_text(250, 80, text=f"Erro: {info['erro']}", fill="#FF5A5A", font=(FONT, 10)); return
        cols = [{"t": f"Entrada ({info['entradas']})", "n": info["entradas"], "c": "#3FA9F5"}]
        for i, n in enumerate(info["camadas_ocultas"]):
            cols.append({"t": f"Oculta {i+1} ({n})", "n": n, "c": "#8B5CF6"})
        cols.append({"t": f"Actor ({info['saidas_actor']})", "n": info["saidas_actor"], "c": "#34D399"})
        w = self.arch_canvas.winfo_width() or 400; h = self.arch_canvas.winfo_height() or 160
        dx = w / (len(cols) + 1)
        for idx, col in enumerate(cols):
            x = (idx + 1) * dx; qtd = min(col["n"], 12); dy = h / (qtd + 1)
            for j in range(qtd):
                y = (j + 1) * dy
                self.arch_canvas.create_oval(x-5, y-5, x+5, y+5, fill=col["c"], outline=BG)
                if idx < len(cols) - 1:
                    nq = min(cols[idx+1]["n"], 12); nx = (idx + 2) * dx; ndy = h / (nq + 1)
                    for k in range(nq):
                        self.arch_canvas.create_line(x, y, nx, (k + 1) * ndy, fill="#2A3B4C", width=1)
            self.arch_canvas.create_text(x, 10, text=col["t"], fill=FG, font=(FONT, 8, "bold"))

    def _save_preset_from_ui(self):
        p = sanitize_preset({k: float(v.get()) for k, v in self.ajustador_vars.items()})
        path = save_preset(p)
        self.write(f"Preset salvo em {path}")
        messagebox.showinfo("OK", f"Preset em {path}")

    def _reload_preset_ui(self):
        p = load_preset()
        for n in PARAMETER_NAMES: self.ajustador_vars[n].set(p[n])
        self.ajustador_vars["tempo_ciclo"].set(p["tempo_ciclo"])
        self.ajustador_vars["abertura_traseira"].set(p["abertura_traseira"])
        self._update_preview()

    def _menu_analise(self):
        self._clear_main(); self._topbar("3 — Análise de Dados", "#34D399")
        box = ttk.Frame(self.main_area); box.pack(fill="both", expand=True)
        ttk.Label(box, text="Comparação de Experimentos", font=(FONT, 12, "bold")).pack(anchor="w", pady=10)
        frm = ttk.Frame(box); frm.pack(fill="x", pady=5)
        ttk.Label(frm, text="Selecione:", width=20).grid(row=0, column=0, sticky="w")
        self.lb_exp = tk.Listbox(frm, selectmode="extended", height=8, bg=PANEL2, fg=FG, selectbackground="#34D399", font=("Consolas", 9))
        self.lb_exp.grid(row=0, column=1, sticky="ew", pady=5)
        for e in [p.name for p in list_experiments()]: self.lb_exp.insert("end", e)
        ttk.Label(box, text="Métrica:", font=(FONT, 12, "bold")).pack(anchor="w", pady=(15, 5))
        self.metrica_var = tk.StringVar(value="1")
        for text, val in [("1. Deslocamento (m)", "1"), ("2. Velocidade (m/s)", "2"),
                          ("3. Tilt Máx (°)", "3"), ("4. RMS Tilt (°)", "4"),
                          ("5. Quedas", "5"), ("6. Latência (ms)", "6")]:
            ttk.Radiobutton(box, text=text, value=val, variable=self.metrica_var).pack(anchor="w", pady=2)
        btns = ttk.Frame(box); btns.pack(anchor="w", pady=20)
        ttk.Button(btns, text="▶ Gráfico Comparativo", command=self._generate_comparison).pack(side="left", padx=5)
        ttk.Button(btns, text="▶ Gráfico Temporal", command=self._generate_temporal).pack(side="left", padx=5)
        ttk.Button(btns, text="💾 Exportar PNG", command=self._export_current_fig).pack(side="left", padx=5)
        ttk.Button(btns, text="💾 Exportar CSV", command=self._export_current_table).pack(side="left", padx=5)
        self.plot_frame = ttk.Frame(box); self.plot_frame.pack(fill="both", expand=True, pady=10)
        self.table_frame = ttk.Frame(box); self.table_frame.pack(fill="x", pady=5)

    def _generate_comparison(self):
        idxs = self.lb_exp.curselection()
        if not idxs: messagebox.showwarning("Aviso", "Selecione experimentos."); return
        selecionados = [self.lb_exp.get(i) for i in idxs]
        metrica = self.metrica_var.get()
        dados = []
        for nome in selecionados:
            data = load_experiment(EXPERIMENTS_DIR / nome)
            perf = data["perf"]
            if not perf: continue
            displs, tilts_max, elapsed, falls, lat = [], [], [], [], []
            for row in perf:
                try: displs.append(abs(float(row.get("displacement_m", 0.0))))
                except: pass
                try: tilts_max.append(float(row.get("max_tilt_deg", 0.0)))
                except: pass
                try: elapsed.append(float(row.get("elapsed_s", row.get("duration_s", 12.0))))
                except: elapsed.append(12.0)
                falls.append(1 if str(row.get("fallen","")).lower()=="true" else 0)
                if row.get("latencia_politica_ms","") != "":
                    try: lat.append(float(row["latencia_politica_ms"]))
                    except: pass
            sens = data["sensors"]
            tilt_rms = 0.0
            if sens:
                all_t = [float(r["tilt_deg"]) for r in sens if r.get("tilt_deg","") != ""]
                if len(all_t) > 1: tilt_rms = statistics.pstdev(all_t)
            n = len(displs)
            dados.append({
                "nome": nome[-20:], "n_rep": n,
                "deslocamento": statistics.mean(displs) if displs else 0.0,
                "deslocamento_std": statistics.stdev(displs) if n > 1 else 0.0,
                "velocidade": (statistics.mean(displs)/statistics.mean(elapsed)) if displs and elapsed else 0.0,
                "tilt_max": statistics.mean(tilts_max) if tilts_max else 0.0,
                "tilt_max_std": statistics.stdev(tilts_max) if len(tilts_max) > 1 else 0.0,
                "tilt_rms": tilt_rms, "quedas": sum(falls),
                "latencia": statistics.mean(lat) if lat else 0.0,
            })
        for w in self.plot_frame.winfo_children(): w.destroy()
        for w in self.table_frame.winfo_children(): w.destroy()
        if not dados: return
        fig = Figure(figsize=(10, 4), dpi=90); ax = fig.add_subplot(111)
        nomes = [d["nome"] for d in dados]; x = range(len(nomes))
        y, err, label = [], [], "?"
        if metrica == "1": y = [d["deslocamento"] for d in dados]; err = [d["deslocamento_std"] for d in dados]; label = "Deslocamento (m)"
        elif metrica == "2": y = [d["velocidade"] for d in dados]; err = [0]*len(dados); label = "Velocidade (m/s)"
        elif metrica == "3": y = [d["tilt_max"] for d in dados]; err = [d["tilt_max_std"] for d in dados]; label = "Tilt Máx (°)"
        elif metrica == "4": y = [d["tilt_rms"] for d in dados]; err = [0]*len(dados); label = "RMS Tilt (°)"
        elif metrica == "5": y = [d["quedas"] for d in dados]; err = [0]*len(dados); label = "Quedas"
        elif metrica == "6": y = [d["latencia"] for d in dados]; err = [0]*len(dados); label = "Latência (ms)"
        ax.bar(x, y, yerr=err, color="#34D399", capsize=5)
        ax.set_xticks(x); ax.set_xticklabels(nomes, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(label); ax.grid(True, alpha=0.3); fig.tight_layout()
        self.last_fig = fig; self.last_comparison_data = dados
        c = FigureCanvasTkAgg(fig, master=self.plot_frame); c.draw(); c.get_tk_widget().pack(fill="both", expand=True)
        headers = ["Exp", "Rep", "Dist μ±σ", "Vel", "Tilt μ±σ", "RMS", "Quedas"]
        for j, h in enumerate(headers):
            ttk.Label(self.table_frame, text=h, font=(FONT, 9, "bold"), foreground="#34D399").grid(row=0, column=j, padx=5)
        for i, d in enumerate(dados, 1):
            ttk.Label(self.table_frame, text=d["nome"]).grid(row=i, column=0, padx=5)
            ttk.Label(self.table_frame, text=str(d["n_rep"])).grid(row=i, column=1)
            ttk.Label(self.table_frame, text=f"{d['deslocamento']:.2f}±{d['deslocamento_std']:.2f}").grid(row=i, column=2)
            ttk.Label(self.table_frame, text=f"{d['velocidade']:.2f}").grid(row=i, column=3)
            ttk.Label(self.table_frame, text=f"{d['tilt_max']:.1f}±{d['tilt_max_std']:.1f}").grid(row=i, column=4)
            ttk.Label(self.table_frame, text=f"{d['tilt_rms']:.1f}").grid(row=i, column=5)
            ttk.Label(self.table_frame, text=str(d["quedas"])).grid(row=i, column=6)

    def _generate_temporal(self):
        idxs = self.lb_exp.curselection()
        if len(idxs) != 1: messagebox.showwarning("Aviso", "Selecione 1 experimento."); return
        nome = self.lb_exp.get(idxs[0])
        sens = load_experiment(EXPERIMENTS_DIR / nome)["sensors"]
        if not sens: messagebox.showwarning("Aviso", "Sem sensors.csv."); return
        def col(name):
            out = []
            for r in sens:
                v, t = r.get(name,""), r.get("time_s","")
                if v == "" or t == "": continue
                try: out.append((float(t), float(v)))
                except: pass
            return out
        pitch, roll, disp, tilts, yaw = col("pitch_deg"), col("roll_deg"), col("displacement_m"), col("tilt_deg"), col("yaw_deg")
        falls = []
        for r in sens:
            if str(r.get("fallen","0"))=="1":
                try: falls.append(float(r["time_s"]))
                except: pass
        for w in self.plot_frame.winfo_children(): w.destroy()
        for w in self.table_frame.winfo_children(): w.destroy()
        fig = Figure(figsize=(10, 8), dpi=90)
        ax1 = fig.add_subplot(411)
        if pitch: ax1.plot([t for t,_ in pitch], [v for _,v in pitch], label="pitch", color="#3FA9F5")
        if roll: ax1.plot([t for t,_ in roll], [v for _,v in roll], label="roll", color="#FB923C")
        for ft in falls: ax1.axvline(ft, color="red", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Roll/Pitch (°)"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax4 = fig.add_subplot(412)
        if yaw: ax4.plot([t for t,_ in yaw], [v for _,v in yaw], color="#F59E0B")
        for ft in falls: ax4.axvline(ft, color="red", linestyle="--", alpha=0.5)
        ax4.set_ylabel("Yaw (°)"); ax4.grid(True, alpha=0.3)
        ax2 = fig.add_subplot(413)
        if disp: ax2.plot([t for t,_ in disp], [v for _,v in disp], color="#34D399")
        for ft in falls: ax2.axvline(ft, color="red", linestyle="--", alpha=0.5)
        ax2.set_ylabel("Deslocamento (m)"); ax2.grid(True, alpha=0.3)
        ax3 = fig.add_subplot(414)
        if tilts: ax3.plot([t for t,_ in tilts], [v for _,v in tilts], color="#8B5CF6")
        for ft in falls: ax3.axvline(ft, color="red", linestyle="--", alpha=0.5)
        ax3.set_xlabel("Tempo (s)"); ax3.set_ylabel("Tilt (°)"); ax3.grid(True, alpha=0.3)
        fig.tight_layout(); self.last_fig = fig; self.last_comparison_data = None
        c = FigureCanvasTkAgg(fig, master=self.plot_frame); c.draw(); c.get_tk_widget().pack(fill="both", expand=True)

    def _export_current_fig(self):
        if self.last_fig is None: messagebox.showwarning("Aviso", "Gere um gráfico."); return
        out = RESULTS_DIR / f"grafico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.last_fig.savefig(out, dpi=150, bbox_inches="tight")
        messagebox.showinfo("OK", f"Salvo em {out}")

    def _export_current_table(self):
        if not self.last_comparison_data: messagebox.showwarning("Aviso", "Gere uma tabela."); return
        out = RESULTS_DIR / f"tabela_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.last_comparison_data[0].keys()))
            w.writeheader(); w.writerows(self.last_comparison_data)
        messagebox.showinfo("OK", f"Salvo em {out}")

    # ======================================================
    # [CORRIGIDO v1.9] _menu_comunicacao — ZIP para o robô real
    # ======================================================
    def _menu_comunicacao(self):
        self._clear_main(); self._topbar("4 — Comunicação", "#FB923C")
        box = ttk.Frame(self.main_area); box.pack(fill="both", expand=True)
        ttk.Label(box, text="Enviar ao Robô Real", font=(FONT, 12, "bold")).pack(anchor="w", pady=10)

        frm = ttk.Frame(box); frm.pack(fill="x", pady=5)
        ttk.Label(frm, text="IP:").pack(side="left")
        ttk.Entry(frm, textvariable=self.target_ip, width=20).pack(side="left", padx=5)
        ttk.Label(frm, text="Porta:").pack(side="left")
        ttk.Entry(frm, textvariable=self.target_port, width=10).pack(side="left", padx=5)
        ttk.Button(frm, text="📡 Testar Ping", command=self._test_ping).pack(side="left", padx=5)

        # Seleção do experimento para o pacote de deploy
        ttk.Label(box, text="Experimento para empacotar:",
                  font=(FONT, 11, "bold")).pack(anchor="w", pady=(15, 5))
        self.robot_lb = tk.Listbox(box, selectmode="browse", height=8,
                                    bg=PANEL2, fg=FG, selectbackground="#FB923C",
                                    font=("Consolas", 9))
        self.robot_lb.pack(fill="x", pady=5)
        for e in [p.name for p in list_experiments()]:
            self.robot_lb.insert("end", e)

        self.robot_include_model = tk.BooleanVar(value=True)
        ttk.Checkbutton(box,
                        text="Incluir modelo PPO treinado (apenas se método = nn_imu)",
                        variable=self.robot_include_model).pack(anchor="w", pady=2)

        info = ttk.LabelFrame(box, text="O que vai no pacote", padding=8)
        info.pack(fill="x", pady=10)
        ttk.Label(info, text=(
            "• test_protocol.json       ← SEMPRE (obrigatório — é o que faltava)\n"
            "• metadata.json\n"
            "• performance.csv, gait_parameters.csv\n"
            "• gait_preset.json         ← baseline de marcha\n"
            "• experimental_conditions.json  ← ruído/atrasos\n"
            "• model/imu_visual_ppo_<tag>.zip + vecnormalize_<tag>.pkl (se PPO)\n"
            "• README.txt (instruções para o firmware do robô)"
        ), foreground=MUTED, font=("Consolas", 8), justify="left").pack(anchor="w")

        btns = ttk.Frame(box); btns.pack(anchor="w", pady=15)
        ttk.Button(btns, text="📦 Gerar pacote para o robô real",
                   command=self._create_robot_package).pack(side="left", padx=4)

    def _test_ping(self):
        ip = self.target_ip.get(); port = int(self.target_port.get())
        self.write(f"[COM] Ping → {ip}:{port}")
        try:
            urllib.request.urlopen(f"http://{ip}:{port}/ping", timeout=1.0)
            self.write("[COM] OK: alvo respondeu.")
        except Exception as e:
            self.write(f"[COM] Sem resposta (esperado em simulação): {e}")

    def _create_robot_package(self):
        """Cria o ZIP de deploy para o robô real a partir do experimento selecionado."""
        idxs = self.robot_lb.curselection()
        if not idxs:
            messagebox.showwarning("Aviso", "Selecione um experimento para empacotar.")
            return
        name = self.robot_lb.get(idxs[0])
        folder = EXPERIMENTS_DIR / name
        if not folder.is_dir():
            messagebox.showerror("Erro", f"Pasta não encontrada: {folder}")
            return

        # GARANTE test_protocol.json (gera retroativamente se faltar)
        _ensure_test_protocol(folder)
        if not (folder / "test_protocol.json").exists():
            messagebox.showerror("Erro", "Não foi possível gerar test_protocol.json.")
            return

        # Lê o metadata para saber o método
        try:
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        method = meta.get("method") or meta.get("metodo") or "manual"

        out_path = filedialog.asksaveasfilename(
            defaultextension=".zip", filetypes=[("ZIP", "*.zip")],
            initialdir=str(PACKAGES_DIR),
            initialfile=f"robot_package_{name}.zip",
        )
        if not out_path:
            return
        out_path = Path(out_path)

        manifest = {
            "experiment": name,
            "method": method,
            "created_at": datetime.now().isoformat(),
            "files": [],
            "yaw_limitation_note": YAW_LIMITATION_NOTE,
        }

        try:
            with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # ---------- OBRIGATÓRIO: test_protocol.json ----------
                zf.write(folder / "test_protocol.json", "test_protocol.json")
                manifest["files"].append("test_protocol.json")

                # ---------- Metadados e dados ----------
                for fname in ("metadata.json", "performance.csv", "gait_parameters.csv"):
                    src = folder / fname
                    if src.exists():
                        zf.write(src, fname)
                        manifest["files"].append(fname)

                # ---------- Condições experimentais ----------
                if EXP_CONDITIONS_JSON.exists():
                    zf.write(EXP_CONDITIONS_JSON, "experimental_conditions.json")
                    manifest["files"].append("experimental_conditions.json")

                # ---------- Preset de marcha ----------
                if PRESET_JSON.exists():
                    zf.write(PRESET_JSON, "gait_preset.json")
                    manifest["files"].append("gait_preset.json")

                # ---------- Modelo PPO (se aplicável) ----------
                if method == "nn_imu" and self.robot_include_model.get():
                    tag = _sanitize_tag(
                        meta.get("model_tag",
                                 meta.get("experimental_conditions", {}).get("ppo_model_tag", "a"))
                    )
                    p = _ppo_paths(tag)
                    if p["model"].exists():
                        zf.write(p["model"], f"model/{p['model'].name}")
                        manifest["files"].append(f"model/{p['model'].name}")
                    if p["vecnorm"].exists():
                        zf.write(p["vecnorm"], f"model/{p['vecnorm'].name}")
                        manifest["files"].append(f"model/{p['vecnorm'].name}")

                # ---------- README de instruções ----------
                readme = self._build_robot_readme(name, method, manifest)
                zf.writestr("README.txt", readme)
                manifest["files"].append("README.txt")

                # ---------- Manifest ----------
                zf.writestr("manifest.json",
                            json.dumps(manifest, indent=2, ensure_ascii=False))
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Erro", f"Falha ao gerar pacote:\n{e}")
            return

        self.write(f"[ROBOT] Pacote criado: {out_path}")
        messagebox.showinfo("OK",
                            f"Pacote para o robô real criado:\n{out_path}\n\n"
                            f"{len(manifest['files'])} arquivos, "
                            f"test_protocol.json SEMPRE incluído.")

    def _build_robot_readme(self, exp_name, method, manifest):
        return (
            f"PACOTE DE DEPLOY — EXPERIMENTO: {exp_name}\n"
            f"Método: {method}\n"
            f"Criado em: {datetime.now().isoformat()}\n"
            f"Protocolo: v1.9\n"
            f"\n"
            f"ARQUIVOS INCLUÍDOS:\n"
            + "\n".join(f"  - {f}" for f in manifest["files"]) + "\n"
            f"\n"
            f"INSTRUÇÕES:\n"
            f"1. test_protocol.json é o arquivo PRINCIPAL do pacote.\n"
            f"   Ele contém baseline, obs_spec, action_spec e condições.\n"
            f"   Configure o firmware do robô real a partir dele.\n"
            f"2. Se método = nn_imu, os arquivos em model/ contêm o PPO treinado\n"
            f"   e o VecNormalize. Use-os no pipeline de inferência do robô real.\n"
            f"3. gait_preset.json contém os parâmetros de marcha de partida.\n"
            f"4. experimental_conditions.json contém ruído/atrasos a replicar.\n"
            f"5. O robô real deve consumir test_protocol.json e reproduzir as ações.\n"
            f"\n"
            f"LIMITAÇÕES:\n"
            f"- Yaw no real (MPU6050 sem magnetômetro) acumula drift; sim é absoluto.\n"
            f"  Para comparações sim × real, considere também avaliar gz.\n"
            f"\n"
            f"NOTA: este pacote foi gerado pelo supervisor v1.9.\n"
        )
    # ======================================================
    # [FIM da seção corrigida]
    # ======================================================

    def _menu_condicoes(self):
        self._clear_main(); self._topbar("5 — Condições Experimentais", "#F59E0B")
        box = ttk.Frame(self.main_area); box.pack(fill="both", expand=True)

        mass = ttk.LabelFrame(box, text="Massa e payload", padding=8)
        mass.pack(fill="x", pady=6)
        r = ttk.Frame(mass); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Massa base (kg):", width=22).pack(side="left")
        ttk.Entry(r, textvariable=self.body_mass_kg, width=10).pack(side="left")
        r = ttk.Frame(mass); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Payload (%):", width=22).pack(side="left")
        ttk.Entry(r, textvariable=self.payload_ratio_pct, width=10).pack(side="left")
        ttk.Button(r, text="Calcular", command=self._recalculate_mass).pack(side="left", padx=5)
        r = ttk.Frame(mass); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Payload (kg):", width=22).pack(side="left")
        ttk.Label(r, textvariable=self.payload_mass_kg, foreground=ACCENT).pack(side="left")
        r = ttk.Frame(mass); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Total (kg):", width=22).pack(side="left")
        ttk.Label(r, textvariable=self.total_mass_kg, foreground=ACCENT).pack(side="left")

        noise = ttk.LabelFrame(box, text="Ruído de sensores", padding=8)
        noise.pack(fill="x", pady=6)
        r = ttk.Frame(noise); r.pack(fill="x", pady=2)
        ttk.Label(r, text="IMU:", width=22).pack(side="left")
        ttk.Combobox(r, textvariable=self.imu_noise_level,
                     values=["disabled","low","medium","high","custom"], width=12).pack(side="left")
        ttk.Label(r, text="  sigma custom:").pack(side="left")
        ttk.Entry(r, textvariable=self.imu_noise_sigma_custom, width=8).pack(side="left")
        r = ttk.Frame(noise); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Câmera:", width=22).pack(side="left")
        ttk.Combobox(r, textvariable=self.camera_noise_level,
                     values=["disabled","low","medium","high","custom"], width=12).pack(side="left")
        ttk.Label(r, text="  sigma custom:").pack(side="left")
        ttk.Entry(r, textvariable=self.camera_noise_sigma_custom, width=8).pack(side="left")

        delays = ttk.LabelFrame(box, text="Atrasos", padding=8)
        delays.pack(fill="x", pady=6)
        r = ttk.Frame(delays); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Servo delay (ms):", width=22).pack(side="left")
        ttk.Entry(r, textvariable=self.servo_delay_ms, width=10).pack(side="left")
        r = ttk.Frame(delays); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Inference delay (ms):", width=22).pack(side="left")
        ttk.Entry(r, textvariable=self.inference_delay_ms, width=10).pack(side="left")

        btns = ttk.Frame(box); btns.pack(anchor="w", pady=12)
        ttk.Button(btns, text="💾 Salvar condições", command=self._save_exp_conditions_ui).pack(side="left", padx=4)
        ttk.Button(btns, text="🔄 Recarregar", command=self._reload_exp_conditions_ui).pack(side="left", padx=4)

    def _menu_analise_rv_sim(self):
        self._clear_main(); self._topbar("6 — Análise Real × Sim", "#EC4899")
        box = ttk.Frame(self.main_area); box.pack(fill="both", expand=True)
        ttk.Label(box, text="Seleção de experimentos para comparação",
                  font=(FONT, 11, "bold")).pack(anchor="w", pady=(0, 8))

        for method in ["manual", "iterativo", "nn_imu"]:
            grp = ttk.LabelFrame(box, text=f"Método: {method}", padding=6)
            grp.pack(fill="x", pady=4)
            row = ttk.Frame(grp); row.pack(fill="x", pady=2)
            ttk.Label(row, text="Real:", width=8).pack(side="left")
            lb_real = tk.Listbox(row, selectmode="extended", height=3, bg=PANEL2, fg=FG,
                                 selectbackground="#EC4899", font=("Consolas", 9))
            lb_real.pack(side="left", fill="x", expand=True)
            row2 = ttk.Frame(grp); row2.pack(fill="x", pady=2)
            ttk.Label(row2, text="Sim:", width=8).pack(side="left")
            lb_sim = tk.Listbox(row2, selectmode="extended", height=3, bg=PANEL2, fg=FG,
                                selectbackground="#EC4899", font=("Consolas", 9))
            lb_sim.pack(side="left", fill="x", expand=True)
            for e in [p.name for p in list_experiments()]:
                try:
                    meta = json.loads((EXPERIMENTS_DIR / e / "metadata.json").read_text(encoding="utf-8"))
                    m = meta.get("method") or meta.get("metodo") or "manual"
                except Exception:
                    m = "manual"
                if m == method:
                    lb_real.insert("end", e); lb_sim.insert("end", e)
            if not hasattr(self, "_an_sel"): self._an_sel = {}
            self._an_sel[method] = {"real_lb": lb_real, "sim_lb": lb_sim}

        btns = ttk.Frame(box); btns.pack(anchor="w", pady=15)
        ttk.Button(btns, text="▶ Gerar relatório", command=self._generate_real_vs_sim).pack(side="left", padx=4)
        self.rvsim_frame = ttk.Frame(box); self.rvsim_frame.pack(fill="both", expand=True, pady=8)

    def _generate_real_vs_sim(self):
        if not hasattr(self, "_an_sel") or not self._an_sel:
            messagebox.showwarning("Aviso", "Nada selecionado."); return
        selection = {}
        for method, d in self._an_sel.items():
            real = [d["real_lb"].get(i) for i in d["real_lb"].curselection()]
            sim = [d["sim_lb"].get(i) for i in d["sim_lb"].curselection()]
            if real or sim:
                selection[method] = {
                    "real": [str(EXPERIMENTS_DIR / x) for x in real],
                    "sim":  [str(EXPERIMENTS_DIR / x) for x in sim],
                }
        if not selection:
            messagebox.showwarning("Aviso", "Selecione pelo menos 1 experimento."); return
        out_dir = RESULTS_DIR / f"rvsim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            summary = _an_full_report(selection, out_dir, progress_cb=self.write)
            self.write(f"[RVSim] Relatório em {out_dir}")
            messagebox.showinfo("OK", f"Relatório em:\n{out_dir}\n\n"
                                      f"Real: {summary['n_real_experiments']}\n"
                                      f"Sim:  {summary['n_sim_experiments']}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Erro", str(e))

    # ---------- dispatcher + rotinas de execução ----------
    def _dispatch(self, action):
        try:
            if action == "run_experiment":
                self._run_experiment_action()
            elif action == "tuner_test":
                self._run_tuner_test()
            elif action == "search":
                self._run_search()
            elif action == "imu_train":
                self._run_ppo_training()
            else:
                self.write(f"[DISPATCH] ação desconhecida: {action}")
        except Exception as e:
            traceback.print_exc()
            self.write(f"[ERRO] {action}: {e}")

    def _run(self):
        self.write(f"[SESSION] {SESSION_ID}")
        while not self.closed:
            try:
                self.root.update()
            except tk.TclError:
                break
            if self.closed:
                break
            if self.pending_action:
                action = self.pending_action
                self.pending_action = None
                self._dispatch(action)
            time.sleep(0.01)

    def _get_baseline_params(self):
        return {k: float(v.get()) for k, v in self.ajustador_vars.items()}

    def _get_metrics(self):
        return {
            "imu": bool(self.metric_imu.get()),
            "flow": bool(self.metric_flow.get()),
            "disp": bool(self.metric_disp.get()),
            "tilt": bool(self.metric_tilt.get()),
        }

    def _run_experiment_action(self):
        method = self.exp_method.get()
        if method == "manual":
            self._run_manual_experiment()
        elif method == "iterativo":
            self._run_search()
        elif method == "nn_imu":
            self._run_ppo_eval()

    def _run_manual_experiment(self):
        name = self.exp_name.get() or f"exp_{SESSION_ID}"
        duration = float(self.exp_duration.get())
        reps = int(self.exp_repetitions.get())
        freq = float(self.exp_freq_hz.get())
        metrics = self._get_metrics()
        baseline = self._get_baseline_params()

        exp_id = f"{SESSION_ID}_{name}_manual"
        metadata = {
            "session_id": SESSION_ID, "method": "manual",
            "duration_s": duration, "freq_hz": freq, "repetitions": reps,
            "baseline": baseline, "metrics": metrics,
            "experimental_conditions": self._current_exp_conditions(),
            "created_at": datetime.now().isoformat(),
        }
        folder = create_experiment_folder(exp_id, metadata)
        # >>> test_protocol.json SEMPRE gerado junto:
        proto = build_test_protocol("manual", duration, freq, baseline, metrics,
                                    self._current_exp_conditions(),
                                    self.ppo_model_tag.get())
        save_experiment_protocol(folder, proto)

        self._apply_mass_before_trial()
        results = []
        for rep in range(reps):
            if self.closed: break
            self.write(f"[EXP] Rep {rep+1}/{reps} — {name}")
            r = run_trial_fixed(rep, baseline, "manual", duration,
                                collect_timeseries=True, freq_hz=freq, metrics=metrics)
            if r is None:
                self.write("[EXP] Trial retornou None (abortado?)"); break
            results.append(r)
            self.write(f"[EXP]   score={r['score']:.3f}  disp={r['displacement_m']:.3f}  fallen={r['fallen']}")

        save_experiment_sensors(folder, [row for r in results for row in r.get("timeseries", [])])
        save_experiment_gait_params(folder, [row for r in results for row in r.get("timeseries", [])])
        save_experiment_performance(folder, results)
        self.write(f"[EXP] Salvo em {folder}")

    def _run_tuner_test(self):
        baseline = self._get_baseline_params()
        duration = float(self.exp_duration.get())
        self._apply_mass_before_trial()
        r = run_trial_fixed(0, baseline, "tuner", duration,
                            collect_timeseries=True,
                            freq_hz=float(self.exp_freq_hz.get()),
                            metrics=self._get_metrics())
        if r is None:
            self.write("[TUNER] Trial abortado."); return
        self.write(f"[TUNER] score={r['score']:.3f}  disp={r['displacement_m']:.3f}  "
                   f"tilt={r['max_tilt_deg']:.2f}°  fallen={r['fallen']}")

    def _run_search(self):
        n_iter = int(self.search_iterations.get())
        duration = float(self.exp_duration.get())
        freq = float(self.exp_freq_hz.get())
        discard = bool(self.search_discard_falls.get())
        metrics = self._get_metrics()
        baseline = self._get_baseline_params()

        exp_id = f"{SESSION_ID}_{self.exp_name.get() or 'search'}_iter"
        metadata = {
            "session_id": SESSION_ID, "method": "iterativo",
            "duration_s": duration, "freq_hz": freq, "iterations": n_iter,
            "baseline": baseline, "metrics": metrics,
            "experimental_conditions": self._current_exp_conditions(),
            "created_at": datetime.now().isoformat(),
        }
        folder = create_experiment_folder(exp_id, metadata)
        proto = build_test_protocol("iterativo", duration, freq, baseline, metrics,
                                    self._current_exp_conditions(),
                                    self.ppo_model_tag.get())
        save_experiment_protocol(folder, proto)
        self._apply_mass_before_trial()

        best = dict(baseline); best_score = -1e9
        results = []; history = []
        for i in range(n_iter):
            if self.closed: break
            trial = dict(best)
            for n in PARAMETER_NAMES:
                low, high = PARAMETER_BOUNDS[n]
                amp = INITIAL_MUTATION_STRENGTH * (high - low) * (1.0 - i / max(1, n_iter))
                trial[n] = clamp(trial[n] + rng.uniform(-amp, amp), low, high)
            r = run_trial_fixed(i, trial, "iter", duration,
                                collect_timeseries=True, freq_hz=freq, metrics=metrics)
            if r is None: break
            discard_reason = ""
            if discard and r["fallen"]:
                discard_reason = "fall"
            if (self.stab_iter_roll_enabled.get() and
                r.get("max_roll_deg", 0) > float(self.stab_iter_roll_max_deg.get())):
                discard_reason = "roll"
            if (self.stab_iter_pitch_enabled.get() and
                r.get("max_pitch_deg", 0) > float(self.stab_iter_pitch_max_deg.get())):
                discard_reason = "pitch"
            if (self.stab_iter_yaw_enabled.get() and
                r.get("max_yaw_deg", 0) > float(self.stab_iter_yaw_max_deg.get())):
                discard_reason = "yaw"
            if discard_reason:
                r["discard_reason"] = discard_reason
            else:
                if r["score"] > best_score:
                    best_score = r["score"]; best = trial
            results.append(r)
            history.append({"iter": i, "score": r["score"], "best_score": best_score,
                            "discard_reason": discard_reason})
            self.write(f"[SEARCH] i={i} score={r['score']:.3f} "
                       f"{'DISCARD('+discard_reason+')' if discard_reason else ''}")

        save_experiment_sensors(folder, [row for r in results for row in r.get("timeseries", [])])
        save_experiment_gait_params(folder, [row for r in results for row in r.get("timeseries", [])])
        save_experiment_performance(folder, results)
        (folder / "search_history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        self.write(f"[SEARCH] Melhor score={best_score:.3f}. Salvo em {folder}")
        for n in PARAMETER_NAMES:
            if n in self.ajustador_vars: self.ajustador_vars[n].set(float(best[n]))
        if "tempo_ciclo" in self.ajustador_vars: self.ajustador_vars["tempo_ciclo"].set(float(best.get("tempo_ciclo", TEMPO_CICLO_DEFAULT)))
        if "abertura_traseira" in self.ajustador_vars: self.ajustador_vars["abertura_traseira"].set(float(best.get("abertura_traseira", ABERTURA_TRASEIRA_DEFAULT)))
        self._update_preview()

    def _run_ppo_training(self):
        try:
            import torch
            import gymnasium as gym
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError as e:
            messagebox.showerror("Erro", f"Dependências ausentes: {e}")
            return

        tag = _sanitize_tag(self.ppo_model_tag.get())
        p = _ppo_paths(tag)
        episodes = int(self.imu_episodes.get())
        baseline = self._get_baseline_params()
        duration = float(self.exp_duration.get())
        exp_cond = self._current_exp_conditions()

        reward_cfg = {
            "disp_gain": float(self.ppo_reward_disp_gain.get()),
            "tilt_enabled": bool(self.ppo_reward_tilt_enabled.get()),
            "tilt_threshold": float(self.ppo_reward_tilt_threshold_deg.get()),
            "tilt_coef": float(self.ppo_reward_tilt_coef.get()),
            "gz_enabled": bool(self.ppo_reward_gz_enabled.get()),
            "gz_coef": float(self.ppo_reward_gz_coef.get()),
            "final_disp_bonus": float(self.ppo_reward_final_disp_bonus.get()),
            "penalty_enabled": bool(self.ppo_penalty_enabled.get()),
            "penalty_value": float(self.ppo_penalty_value.get()),
            "stab_roll": (float(self.stab_ppo_roll_target_deg.get()),
                          float(self.stab_ppo_roll_weight.get()),
                          float(self.stab_ppo_roll_scale_deg.get()),
                          bool(self.stab_ppo_roll_enabled.get())),
            "stab_pitch": (float(self.stab_ppo_pitch_target_deg.get()),
                           float(self.stab_ppo_pitch_weight.get()),
                           float(self.stab_ppo_pitch_scale_deg.get()),
                           bool(self.stab_ppo_pitch_enabled.get())),
            "stab_yaw": (float(self.stab_ppo_yaw_target_deg.get()),
                         float(self.stab_ppo_yaw_weight.get()),
                         float(self.stab_ppo_yaw_scale_deg.get()),
                         bool(self.stab_ppo_yaw_enabled.get())),
        }

        parent = self

        class WebotsEnv(gym.Env):
            metadata = {"render_modes": []}
            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Box(-1.0, 1.0, (OBS_DIM,), dtype=np.float32)
                self.action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)
                self.current_params = dict(baseline)
                self.ep_steps = 0
                self.last_pos = None

            def _sensor_obs(self):
                s = get_latest_sensors()
                obs = build_observation(s, self.current_params)
                return np.clip(np.asarray(obs, dtype=np.float32), -1.0, 1.0)

            def reset(self, seed=None, options=None):
                reset_robot()
                self.current_params = dict(baseline)
                self.ep_steps = 0
                self.last_pos = read_position()
                send_command({"command": "run", "trial": -1, "direction": FORWARD_SIGN,
                              "servo_delay_ms": float(parent.servo_delay_ms.get()),
                              "tempo_ciclo": baseline.get("tempo_ciclo", TEMPO_CICLO_DEFAULT),
                              "abertura_traseira": baseline.get("abertura_traseira", ABERTURA_TRASEIRA_DEFAULT),
                              **{n: baseline[n] for n in PARAMETER_NAMES}})
                return self._sensor_obs(), {}

            def step(self, action):
                action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
                new_params = apply_delta_action(action, baseline)
                send_command({"command": "run", "trial": -1, "direction": FORWARD_SIGN,
                              "servo_delay_ms": float(parent.servo_delay_ms.get()),
                              "tempo_ciclo": new_params.get("tempo_ciclo", TEMPO_CICLO_DEFAULT),
                              "abertura_traseira": new_params.get("abertura_traseira", ABERTURA_TRASEIRA_DEFAULT),
                              **{n: new_params[n] for n in PARAMETER_NAMES}})
                self.current_params = new_params
                t0 = supervisor.getTime()
                while supervisor.getTime() - t0 < 0.5:
                    if supervisor.step(timestep) == -1:
                        return self._sensor_obs(), 0.0, True, False, {}
                    if not parent.pump_ui():
                        return self._sensor_obs(), 0.0, True, False, {}
                pos = read_position()
                delta_disp = forward_coordinate(pos) - forward_coordinate(self.last_pos)
                self.last_pos = pos
                s = get_latest_sensors() or {}
                tilt = body_tilt_degrees()
                gz = abs(float(s.get("gz", 0.0)))
                r = reward_cfg["disp_gain"] * delta_disp
                if reward_cfg["tilt_enabled"]:
                    r -= reward_cfg["tilt_coef"] * max(0.0, tilt - reward_cfg["tilt_threshold"])
                if reward_cfg["gz_enabled"]:
                    r -= reward_cfg["gz_coef"] * gz
                for key, val in (("stab_roll", s.get("roll_deg", 0.0)),
                                 ("stab_pitch", s.get("pitch_deg", 0.0)),
                                 ("stab_yaw", s.get("yaw_deg", 0.0))):
                    tgt, w, sc, en = reward_cfg[key]
                    if en and sc > 1e-6:
                        r -= w * ((float(val) - tgt) / sc) ** 2
                fallen = (tilt > MAX_TILT_DEG)
                if fallen and reward_cfg["penalty_enabled"]:
                    r += reward_cfg["penalty_value"]
                self.ep_steps += 1
                terminated = bool(fallen)
                truncated = (self.ep_steps * 0.5) >= duration
                if truncated or terminated:
                    r += reward_cfg["final_disp_bonus"] * delta_disp
                return self._sensor_obs(), float(r), terminated, truncated, {}

        self.write(f"[PPO] Iniciando treino tag={tag} episodes={episodes}")
        try:
            env = DummyVecEnv([lambda: WebotsEnv()])
            if not self.ppo_start_fresh.get() and p["model"].exists():
                self.write(f"[PPO] Carregando modelo existente: {p['model']}")
                model = PPO.load(str(p["model_base"]), env=env, device="auto")
            else:
                self.write("[PPO] Começando do zero")
                model = PPO("MlpPolicy", env,
                            learning_rate=ADAPT_PPO_LEARNING_RATE,
                            n_steps=ADAPT_PPO_N_STEPS,
                            batch_size=ADAPT_PPO_BATCH_SIZE,
                            n_epochs=ADAPT_PPO_EPOCHS,
                            gamma=ADAPT_PPO_GAMMA,
                            ent_coef=ADAPT_PPO_ENT_COEF,
                            policy_kwargs={"net_arch": ADAPT_PPO_NET_ARCH},
                            verbose=1)
            total_steps = max(1024, episodes * ADAPT_PPO_N_STEPS)
            model.learn(total_timesteps=total_steps, progress_bar=False)
            p["model_base"].parent.mkdir(exist_ok=True, parents=True)
            model.save(str(p["model_base"]))
            self.write(f"[PPO] Modelo salvo em {p['model_base']}.zip")
            messagebox.showinfo("OK", f"Modelo salvo em:\n{p['model_base']}.zip")
        except Exception as e:
            traceback.print_exc()
            self.write(f"[PPO] Erro: {e}")
            messagebox.showerror("Erro no treino", str(e))

    def _run_ppo_eval(self):
        try:
            from stable_baselines3 import PPO
        except ImportError as e:
            messagebox.showerror("Erro", f"Dependências: {e}"); return
        tag = _sanitize_tag(self.ppo_model_tag.get())
        p = _ppo_paths(tag)
        model_base = p["model_base"]
        if not Path(str(model_base) + ".zip").exists():
            legacy = _legacy_ppo_paths()["model_base"]
            if Path(str(legacy) + ".zip").exists():
                model_base = legacy
            else:
                messagebox.showerror("Erro", f"Modelo {tag} não encontrado."); return
        try:
            model = PPO.load(str(model_base), device="auto")
        except Exception as e:
            messagebox.showerror("Erro", str(e)); return

        name = self.exp_name.get() or f"exp_{SESSION_ID}"
        duration = float(self.exp_duration.get())
        reps = int(self.exp_repetitions.get())
        freq = float(self.exp_freq_hz.get())
        baseline = self._get_baseline_params()

        exp_id = f"{SESSION_ID}_{name}_ppo_{tag}"
        metadata = {
            "session_id": SESSION_ID, "method": "nn_imu",
            "duration_s": duration, "freq_hz": freq, "repetitions": reps,
            "baseline": baseline, "model_tag": tag,
            "metrics": {"imu": True, "flow": True, "disp": True, "tilt": True},
            "experimental_conditions": self._current_exp_conditions(),
            "created_at": datetime.now().isoformat(),
        }
        folder = create_experiment_folder(exp_id, metadata)
        proto = build_test_protocol("nn_imu", duration, freq, baseline,
                                    metadata["metrics"],
                                    self._current_exp_conditions(), tag)
        save_experiment_protocol(folder, proto)
        self._apply_mass_before_trial()

        results = []
        for rep in range(reps):
            if self.closed: break
            self.write(f"[PPO-EVAL] Rep {rep+1}/{reps}")
            r = run_trial_adaptive(rep, model, None, baseline, "ppo", duration,
                                   collect_timeseries=True, freq_hz=freq)
            if r is None: break
            results.append(r)
            self.write(f"[PPO-EVAL]   score={r['score']:.3f}  disp={r['displacement_m']:.3f}")
        save_experiment_sensors(folder, [row for r in results for row in r.get("timeseries", [])])
        save_experiment_gait_params(folder, [row for r in results for row in r.get("timeseries", [])])
        save_experiment_performance(folder, results)
        self.write(f"[PPO-EVAL] Salvo em {folder}")


# ============================================================
# MAIN
# ============================================================
ui = GaitInterface()

def pump_ui():
    return ui.pump_ui()

if __name__ == "__main__":
    try:
        ui._run()
    except Exception:
        traceback.print_exc()
    finally:
        ui.close()