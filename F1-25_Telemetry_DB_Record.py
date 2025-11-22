import socket
import struct
import time
import os
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
import ctypes

# === 설정 ===
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "f1_bigdata_db"
COL_NAME = "live_race_data"
MODEL_PATH = "f1_ghost_car_model_v1.h5"
CIRCUIT_LENGTH = 5807

# === MongoDB 연결 ===
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COL_NAME]
print(f"💾 MongoDB Connected: {COL_NAME}")

# === AI 모델 로드 ===
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("🧠 AI Model Loaded.")
else:
    model = None
    print("⚠️ AI Model NOT found.")

# === 스케일러 ===
scaler = MinMaxScaler()
scaler.fit([[0, 0, 0, 0, 0], [CIRCUIT_LENGTH, 360, 100, 1, 8]])


# === UDP 수신 설정 (ctypes) ===
class PacketHeader(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("m_packet_format", ctypes.c_uint16),
        ("m_game_year", ctypes.c_uint8),
        ("m_game_major_version", ctypes.c_uint8),
        ("m_game_minor_version", ctypes.c_uint8),
        ("m_packet_version", ctypes.c_uint8),
        ("m_packet_id", ctypes.c_uint8),
        ("m_session_uid", ctypes.c_uint64),
        ("m_session_time", ctypes.c_float),
        ("m_frame_identifier", ctypes.c_uint32),
        ("m_overall_frame_identifier", ctypes.c_uint32),
        ("m_player_car_index", ctypes.c_uint8),
        ("m_secondary_player_car_index", ctypes.c_uint8),
    ]


class CarTelemetryData(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("m_speed", ctypes.c_uint16),
        ("m_throttle", ctypes.c_float),
        ("m_steer", ctypes.c_float),
        ("m_brake", ctypes.c_float),
        ("m_clutch", ctypes.c_uint8),
        ("m_gear", ctypes.c_int8),
        ("m_engine_rpm", ctypes.c_uint16),
        ("m_drs", ctypes.c_uint8),
        ("m_rev_lights_percent", ctypes.c_uint8),
        ("m_rev_lights_bit_value", ctypes.c_uint16),
        ("m_brakes_temperature", ctypes.c_uint16 * 4),
        ("m_tyres_surface_temperature", ctypes.c_uint8 * 4),
        ("m_tyres_inner_temperature", ctypes.c_uint8 * 4),
        ("m_engine_temperature", ctypes.c_uint16),
        ("m_tyres_pressure", ctypes.c_float * 4),
        ("m_surface_type", ctypes.c_uint8 * 4),
    ]


class PacketCarTelemetryData(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("m_header", PacketHeader),
        ("m_car_telemetry_data", CarTelemetryData * 22),
        ("m_mfd_panel_index", ctypes.c_uint8),
        ("m_mfd_panel_index_secondary_player", ctypes.c_uint8),
        ("m_suggested_gear", ctypes.c_int8),
    ]


class LapData(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("m_last_lap_time_in_ms", ctypes.c_uint32),
        ("m_current_lap_time_in_ms", ctypes.c_uint32),
        ("m_sector1_time_in_ms", ctypes.c_uint16),
        ("m_sector1_time_in_minutes", ctypes.c_uint8),
        ("m_sector2_time_in_ms", ctypes.c_uint16),
        ("m_sector2_time_in_minutes", ctypes.c_uint8),
        ("m_deltaToCarInFrontMSPart", ctypes.c_uint16),
        ("m_deltaToCarInFrontMinutesPart", ctypes.c_uint8),
        ("m_deltaToRaceLeaderMSPart", ctypes.c_uint16),
        ("m_deltaToRaceLeaderMinutesPart", ctypes.c_uint8),
        ("m_lap_distance", ctypes.c_float),
        ("m_total_distance", ctypes.c_float),
        ("m_safety_car_delta", ctypes.c_float),
        ("m_car_position", ctypes.c_uint8),
        ("m_current_lap_num", ctypes.c_uint8),
        ("m_pit_status", ctypes.c_uint8),
        ("m_num_pit_stops", ctypes.c_uint8),
        ("m_sector", ctypes.c_uint8),
        ("m_current_lap_invalid", ctypes.c_uint8),
        ("m_penalties", ctypes.c_uint8),
        ("m_total_warnings", ctypes.c_uint8),
        ("m_corner_cutting_warnings", ctypes.c_uint8),
        ("m_num_unserved_drive_through_pens", ctypes.c_uint8),
        ("m_num_unserved_stop_go_pens", ctypes.c_uint8),
        ("m_grid_position", ctypes.c_uint8),
        ("m_driver_status", ctypes.c_uint8),
        ("m_result_status", ctypes.c_uint8),
        ("m_pit_lane_timer_active", ctypes.c_uint8),
        ("m_pit_lane_time_in_lane_in_ms", ctypes.c_uint16),
        ("m_pit_stop_timer_in_ms", ctypes.c_uint16),
        ("m_pit_stop_should_serve_pen", ctypes.c_uint8),
        ("m_speedTrapFastestSpeed", ctypes.c_float),
        ("m_speedTrapFastestLap", ctypes.c_uint8),
    ]


class PacketLapData(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("m_header", PacketHeader),
        ("m_lap_data", LapData * 22),
        ("m_time_trial_pb_car_idx", ctypes.c_uint8),
        ("m_time_trial_rival_car_idx", ctypes.c_int8),
    ]


# === 메인 로직 ===
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 20777))
sock.setblocking(False)

print("\n📡 Waiting for F1 2024/25 Data...")

history_buffer = []
curr_tel = {}
curr_lap = {}
last_db_time = time.time()

while True:
    try:
        data, _ = sock.recvfrom(4096)
        if len(data) < 24: continue

        header = PacketHeader.from_buffer_copy(data[:ctypes.sizeof(PacketHeader)])
        p_id = header.m_packet_id
        idx = header.m_player_car_index

        if p_id == 6:  # Telemetry
            if len(data) == ctypes.sizeof(PacketCarTelemetryData):
                p = PacketCarTelemetryData.from_buffer_copy(data)
                car = p.m_car_telemetry_data[idx]
                curr_tel = {
                    "Speed": car.m_speed,
                    "Throttle": car.m_throttle * 100,
                    "Brake": car.m_brake * 100,
                    "Gear": car.m_gear,
                    "RPM": car.m_engine_rpm
                }

        elif p_id == 2:  # LapData
            if len(data) == ctypes.sizeof(PacketLapData):
                p = PacketLapData.from_buffer_copy(data)
                lap = p.m_lap_data[idx]
                curr_lap = {
                    "Distance": lap.m_lap_distance,
                    "LapNumber": lap.m_current_lap_num
                }

        # 데이터가 다 모였을 때 처리
        if curr_tel and curr_lap:
            # 1. DB 저장 (0.1초마다)
            if time.time() - last_db_time > 0.1:
                doc = {
                    "Timestamp": time.time(),
                    "Driver": 1,
                    **curr_tel,
                    **curr_lap
                }
                collection.insert_one(doc)
                print(f"💾 Saved! Speed: {curr_tel['Speed']} | Dist: {int(curr_lap['Distance'])}")
                last_db_time = time.time()

            # 2. AI 코칭
            # (AI 로직은 생략 - 위와 동일)

            # 버퍼 초기화 (중복 저장 방지)
            # curr_tel = {}

    except BlockingIOError:
        pass
    except Exception as e:
        print(e)