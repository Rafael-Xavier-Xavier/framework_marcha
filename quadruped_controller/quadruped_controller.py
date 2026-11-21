"""
QUADRUPED_CONTROLLER
"""
from controller import Robot
import json
import math
import cv2
import numpy as np
from collections import deque

CAMERA_NAME = "camera"
CAMERA_WINDOW = "GATO ROBÔ | Câmera + Fluxo"
CAMERA_SCALE = 2.0
SHOW_CAMERA = True
SENSOR_SEND_INTERVAL = 0.1
FLOW_POINT_THRESHOLD = 0.5

GYRO_NAME = "gyro"
ACCELEROMETER_NAME = "accelerometer"
INERTIAL_UNIT_NAME = "inertial unit"

MOTOR_NAMES = [
    "hip0x", "hip0z", "knee0", "hip1x", "hip1z", "knee1",
    "hip2x", "hip2z", "knee2", "hip3x", "hip3z", "knee3",
]
DIRECTIONS = [1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1]
LEG_NEUTRAL = (0.0, 16.0, 36.0)
BASE_POSE = list(LEG_NEUTRAL) * 4
MOTOR_VELOCITY = 10.0
STEP_LENGTH = 0.0000

FRAMES_BASE = [
    {"TE": (16, 36), "DE": (16, 36), "TD": (16, 36), "DD": (25, 50)},
    {"TE": (16, 36), "DE": (16, 36), "TD": (16, 36), "DD": (5, 45)},
    {"TE": (16, 36), "DE": (16, 36), "TD": (25, 50), "DD": (16, 36)},
    {"TE": (16, 36), "DE": (16, 36), "TD": (5, 45),  "DD": (16, 36)},
    {"TE": (16, 36), "DE": (25, 50), "TD": (16, 36), "DD": (16, 36)},
    {"TE": (16, 36), "DE": (5, 45),  "TD": (16, 36), "DD": (16, 36)},
    {"TE": (25, 50), "DE": (16, 36), "TD": (16, 36), "DD": (16, 36)},
    {"TE": (5, 45),  "DE": (16, 36), "TD": (16, 36), "DD": (16, 36)}
]
FRAME_LEGS_ORDER = ["DD", "DE", "TD", "TE"]

# --- Servo lag buffer ---
SERVO_DELAY_S = 0.0
_pose_buffer = deque()  # [(apply_at_sim_time, pose)]

def apply_pose_with_delay(pose, sim_time):
    """Aplica pose com atraso configurável para simular resposta de servo.
    Se SERVO_DELAY_S <= 0, aplica imediatamente (comportamento original).
    """
    if SERVO_DELAY_S <= 0:
        apply_pose(pose)
        return
    _pose_buffer.append((sim_time + SERVO_DELAY_S, pose))
    while _pose_buffer and _pose_buffer[0][0] <= sim_time:
        apply_pose(_pose_buffer.popleft()[1])

def clear_pose_buffer():
    _pose_buffer.clear()

# --- Setup ---
robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0

motors = []
for name in MOTOR_NAMES:
    motor = robot.getDevice(name)
    if motor is None: raise RuntimeError(f"Motor não encontrado: {name}")
    motor.setVelocity(MOTOR_VELOCITY)
    motors.append(motor)

def get_required_device(name, desc):
    d = robot.getDevice(name)
    if d is None: raise RuntimeError(f"{desc} não encontrado.")
    return d

gyro = get_required_device(GYRO_NAME, "Gyro")
accelerometer = get_required_device(ACCELEROMETER_NAME, "Accelerometer")
inertial_unit = get_required_device(INERTIAL_UNIT_NAME, "InertialUnit")
gyro.enable(timestep)
accelerometer.enable(timestep)
inertial_unit.enable(timestep)

camera = None
prev_gray = None
flow_metrics = {"flow_x_mean": 0.0, "flow_y_mean": 0.0, "flow_mag_mean": 0.0, "flow_dir_deg": 0.0, "flow_points": 0, "flow_var": 0.0}

if SHOW_CAMERA:
    camera = robot.getDevice(CAMERA_NAME)
    if camera:
        camera.enable(timestep)
        cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
    else:
        SHOW_CAMERA = False

def compute_optical_flow(frame_bgr):
    global prev_gray, flow_metrics
    if frame_bgr is None: return
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None and prev_gray.shape == gray.shape:
        try:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            fx, fy = flow[..., 0], flow[..., 1]
            mag = np.sqrt(fx*fx + fy*fy)
            flow_metrics = {
                "flow_x_mean": float(np.mean(fx)), "flow_y_mean": float(np.mean(fy)),
                "flow_mag_mean": float(np.mean(mag)),
                "flow_dir_deg": float(math.degrees(math.atan2(np.mean(fy), np.mean(fx)))) if np.mean(mag) > 0.05 else 0.0,
                "flow_points": int(np.count_nonzero(mag > FLOW_POINT_THRESHOLD)),
                "flow_var": float(np.var(mag)),
            }
        except Exception: pass
    prev_gray = gray

emitter = robot.getDevice("emitter")
receiver = robot.getDevice("receiver")
receiver.enable(timestep)

current_command = "hold"
current_trial = None
trial_start_time = 0.0
last_sensor_send_time = -SENSOR_SEND_INTERVAL
current_params = None
current_tempo_ciclo = 1.0

def apply_pose(pose):
    for i, motor in enumerate(motors):
        motor.setPosition(math.radians(float(pose[i]) * DIRECTIONS[i]))

def get_frame_pose(frame_index, params):
    frame = FRAMES_BASE[frame_index]
    pose = BASE_POSE[:]

    height_offset = float(params.get("height_scale", 0.0))
    sh_amp = float(params.get("shoulder_amp", 24.0)) / 24.0
    fr_amp = float(params.get("front_knee_amp", 28.8)) / 28.8
    rr_amp = float(params.get("rear_knee_amp", 9.6)) / 9.6

    for i, leg_name in enumerate(FRAME_LEGS_ORDER):
        shoulder, knee = frame[leg_name]

        shoulder_diff = 16.0 - shoulder
        shoulder = 16.0 + (shoulder_diff * sh_amp)

        if leg_name in ["DD", "DE"]:
            knee_diff = knee - 36.0
            knee = 36.0 + (knee_diff * fr_amp) + height_offset
        else:
            knee_diff = knee - 36.0
            knee = 36.0 + (knee_diff * rr_amp) + height_offset

        if leg_name in ["TD", "TE"]:
            shoulder += float(params.get("abertura_traseira", 4.0))

        idx = i * 3
        pose[idx + 1] = shoulder
        pose[idx + 2] = knee
    return pose

def build_animated_pose(global_phase, params):
    scaled_phase = ((-global_phase) % 1.0) * 8.0

    idx_atual = int(math.floor(scaled_phase)) % 8
    idx_proximo = (idx_atual + 1) % 8
    t = scaled_phase - math.floor(scaled_phase)

    pose_atual = get_frame_pose(idx_atual, params)
    pose_proximo = get_frame_pose(idx_proximo, params)

    return [
        pose_atual[i] + (pose_proximo[i] - pose_atual[i]) * t
        for i in range(12)
    ]

def send_physics_move(delta_x):
    try: emitter.send(json.dumps({"tipo": "move_world", "dx": delta_x}))
    except Exception: pass

def process_incoming_commands():
    global current_command, current_trial, trial_start_time, current_params, current_tempo_ciclo, SERVO_DELAY_S

    while receiver.getQueueLength() > 0:
        raw = receiver.getString()
        receiver.nextPacket()
        if not raw: continue
        try: msg = json.loads(raw)
        except: continue

        cmd = msg.get("command", "hold")
        trial = msg.get("trial", None)

        if cmd == "run":
            if current_command != "run" or trial != current_trial:
                trial_start_time = robot.getTime()
                clear_pose_buffer()  # limpa poses pendentes de trial anterior
            current_command = "run"
            current_trial = trial

            # Atualiza o atraso de servo
            try: SERVO_DELAY_S = max(0.0, float(msg.get("servo_delay_ms", 0.0)) / 1000.0)
            except Exception: SERVO_DELAY_S = 0.0

            current_params = {}
            for k in ["cycle_scale", "support_fraction", "shoulder_amp", "front_knee_amp", "rear_knee_amp", "height_scale", "abertura_traseira"]:
                try: current_params[k] = float(msg.get(k, 0.0))
                except: current_params[k] = 0.0

            try: current_tempo_ciclo = float(msg.get("tempo_ciclo", 1.0))
            except: current_tempo_ciclo = 1.0
        else:
            current_command = "hold"
            current_params = None
            SERVO_DELAY_S = 0.0
            clear_pose_buffer()
            apply_pose(BASE_POSE)

def send_sensors(sim_time):
    gyro_xyz, accel_xyz = gyro.getValues(), accelerometer.getValues()
    rpy = inertial_unit.getRollPitchYaw()
    f = flow_metrics
    msg = {
        "tipo": "sen", "t": round(sim_time, 4), "trial": current_trial or -1, "cmd": current_command,
        "tempo_ciclo": current_tempo_ciclo,
        "gx": round(gyro_xyz[0], 6), "gy": round(gyro_xyz[1], 6), "gz": round(gyro_xyz[2], 6),
        "ax": round(accel_xyz[0], 6), "ay": round(accel_xyz[1], 6), "az": round(accel_xyz[2], 6),
        "roll_deg": round(math.degrees(rpy[0]), 3), "pitch_deg": round(math.degrees(rpy[1]), 3), "yaw_deg": round(math.degrees(rpy[2]), 3),
        "flow_x_mean": round(f["flow_x_mean"], 5), "flow_y_mean": round(f["flow_y_mean"], 5),
        "flow_mag_mean": round(f["flow_mag_mean"], 5), "flow_dir_deg": round(f["flow_dir_deg"], 2),
        "flow_points": int(f["flow_points"]), "flow_var": round(f["flow_var"], 4),
    }
    try: emitter.send(json.dumps(msg))
    except: pass

apply_pose(BASE_POSE)
print("GATO ROBÔ iniciado (Frame Dinâmico Expandido).", flush=True)

last_phase = 0.0
try:
    while robot.step(timestep) != -1:
        if SHOW_CAMERA and camera:
            image = camera.getImage()
            if image:
                frame = np.frombuffer(image, dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
                compute_optical_flow(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))

        process_incoming_commands()

        if current_command == "run" and current_params is not None:
            elapsed = max(0.0, robot.getTime() - trial_start_time)
            global_phase = elapsed / max(0.1, current_tempo_ciclo)

            delta_phase = global_phase - last_phase
            if delta_phase > 0:
                send_physics_move(-delta_phase * STEP_LENGTH)
            last_phase = global_phase

            pose = build_animated_pose(global_phase, current_params)
            apply_pose_with_delay(pose, robot.getTime())  # <-- atraso de servo

        sim_time = robot.getTime()
        if sim_time - last_sensor_send_time >= SENSOR_SEND_INTERVAL:
            send_sensors(sim_time)
            last_sensor_send_time = sim_time

finally:
    apply_pose(BASE_POSE)
    if SHOW_CAMERA: cv2.destroyAllWindows()