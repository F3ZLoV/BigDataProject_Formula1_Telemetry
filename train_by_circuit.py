# 파일명: train_by_circuit.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os
import time
import fastf1
import re

# === 설정 ===
DATA_DIR = "f1_processed_warehouse"
MODELS_DIR = "models_by_circuit"  # 서킷 이름으로 저장할 폴더
SEQUENCE_LENGTH = 20
EPOCHS = 5
YEARS = range(2018, 2025)  # 2018 ~ 2024 (7년치)

# 웹 앱 파일명 규칙을 학습 코드에 역으로 적용하기 위한 매핑
# Key: fastf1 Location 이름 (또는 예상되는 이름), Value: 웹 앱에서 사용하는 축약 이름
FILENAME_MAP = {
    # 브라질/상파울루: Autódromo José Carlos Pace -> SaoPaulo
    "São Paulo": "SaoPaulo",
    "Sao Paulo": "SaoPaulo",
    "Autódromo José Carlos Carlos Pace": "SaoPaulo",
    # 미국/오스틴: Circuit of the Americas -> Austin
    "Austin": "Austin",
    "Circuit of the Americas": "Austin",
    # 멕시코: Autódromo Hermanos Rodríguez -> MexicoCity
    "Mexico City": "MexicoCity",
    "Autódromo Hermanos Rodríguez": "MexicoCity",
    # 아부다비: Yas Marina Circuit -> YasMarina
    "Yas Island": "YasIsland",  # fastf1의 Location은 Yas Island인 경우가 있음
    "Yas Marina": "YasMarina",
    "Yas Marina Circuit": "YasMarina",
    # 라스베이거스: Las Vegas Strip Circuit -> LasVegas
    "Las Vegas": "LasVegas",
    "Las Vegas Strip Circuit": "LasVegas",
    # 카타르: Lusail International Circuit -> Lusail
    "Lusail": "Lusail",
    # 기타 주요 서킷 (fastf1 Location과 파일명이 다를 수 있는 경우)
    "Sakhir": "Sakhir",
    "Jeddah": "Jeddah",
    "Melbourne": "Melbourne",
    "Imola": "Imola",  # Autodromo Enzo e Dino Ferrari
    "Montreal": "Montreal",  # Circuit Gilles Villeneuve
    "Spielberg": "Spielberg",  # Red Bull Ring
    "Budapest": "Budapest",  # Hungaroring
    "Zandvoort": "Zandvoort",  # Circuit Park Zandvoort
    "Baku": "Baku",
    "Singapore": "Singapore",
    # 벨기에: Spa-Francorchamps -> SpaFrancorchamps (공백 제거)
    "Spa-Francorchamps": "SpaFrancorchamps",
    # 기타, 이미 이름이 잘 축약된 경우
    "Suzuka": "Suzuka",
    "Shanghai": "Shanghai",
    "Miami": "Miami",
    "Monaco": "Monaco",
    "Barcelona": "Barcelona",
    "Silverstone": "Silverstone",
    "Monza": "Monza",
}

# 저장 폴더 생성
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)


def get_circuit_mapping():
    """
    2018~2024년의 모든 경기를 조회하여 {서킷명: [(연도, 라운드), ...]} 딕셔너리를 만듭니다.
    """
    print("📅 F1 캘린더 분석 중 (서킷 매핑)...")
    circuit_map = {}

    for year in YEARS:
        try:
            schedule = fastf1.get_event_schedule(year)
            races = schedule[schedule['EventFormat'] != 'testing']

            for _, row in races.iterrows():
                # 서킷 이름 (Location이 가장 정확함. 예: Suzuka, Monza...)
                circuit_name = row['Location'].strip()
                round_num = row['RoundNumber']

                # Location 이름이 비어있지 않은 경우에만 처리
                if circuit_name:
                    # 이름을 표준화 (대소문자 무시 비교를 위해 모두 소문자로 변환)
                    standard_name = circuit_name

                    if standard_name not in circuit_map:
                        circuit_map[standard_name] = []

                    circuit_map[standard_name].append((year, round_num))
        except Exception as e:
            print(f"⚠️ {year}년도 일정 로드 실패: {e}")

    return circuit_map


def build_lstm_model(input_shape):
    # Input 레이어를 명시적으로 추가
    model = Sequential([
        tf.keras.Input(shape=input_shape), # Input 레이어를 첫 번째로 추가
        LSTM(64, return_sequences=True),   # 이제 input_shape를 생략
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    return model


def train_circuit_model(circuit_name, race_list):
    print(f"\n🏎️ [{circuit_name}] 데이터 수집 및 학습 시작...")

    all_data = []

    # 1. 매핑된 모든 연도/라운드 데이터 로드
    for year, round_num in race_list:
        path = f"{DATA_DIR}/year={year}/round={round_num}/telemetry.parquet"

        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                df = df[df['Speed'] > 10]

                cols = ['Distance', 'Speed', 'Throttle', 'Brake', 'nGear']
                # LapData의 Gear 컬럼이 nGear로 저장되지 않았을 경우를 대비한 체크
                if 'Gear' in df.columns and 'nGear' not in df.columns:
                    df.rename(columns={'Gear': 'nGear'}, inplace=True)

                if set(cols).issubset(df.columns):
                    all_data.append(df[cols])
            except:
                continue

    if not all_data:
        print(f"   -> ⚠️ 데이터 없음. 스킵.")
        return False

    full_df = pd.concat(all_data)
    print(f"   -> 학습 데이터 확보: {len(full_df):,} rows ({len(race_list)}개 레이스)")

    # 2. 전처리 (스케일러)
    max_dist = full_df['Distance'].max()

    scaler = MinMaxScaler()
    scaler.fit([
        [0, 0, 0, 0, 0],
        [max_dist, 360, 100, 1, 8]
    ])

    scaled_data = scaler.transform(full_df)

    # 3. 시계열 생성 (20만개 제한)
    limit = 200000
    if len(scaled_data) > limit:
        scaled_data = scaled_data[-limit:]

    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X.append(scaled_data[i - SEQUENCE_LENGTH:i])
        y.append(scaled_data[i, 1])

    X = np.array(X)
    y = np.array(y)

    # 데이터가 너무 적으면 학습 스킵 (SEQUENCE_LENGTH보다 작으면 에러 발생 방지)
    if X.shape[0] < 100:
        print(f"   -> ⚠️ 시계열 데이터가 너무 적음 ({X.shape[0]}개). 스킵.")
        return False

    # 4. 학습
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, batch_size=256, epochs=EPOCHS, validation_split=0.1, verbose=0)

    # 5. 저장 (웹 앱과 파일명 동기화)

    # 5.1. 매핑된 이름 찾기 (Location에서 매핑 딕셔너리의 키와 일치하는지 확인)
    # 띄어쓰기/대소문자 문제 해결을 위해 모든 키와 현재 서킷 이름을 정규화하여 비교
    mapped_name = None
    for key, value in FILENAME_MAP.items():
        if key.strip().lower() == circuit_name.strip().lower():
            mapped_name = value
            break

    # 5.2. 매핑된 이름이 없으면 기존의 안전한 이름 생성 규칙을 사용
    if mapped_name is None:
        # 영문/숫자 외 모든 문자 제거 (예: CircuitdeMonaco)
        safe_name = re.sub(r'[^A-Za-z0-9]', '', circuit_name)
    else:
        safe_name = mapped_name

    save_path = f"{MODELS_DIR}/ghost_{safe_name}.h5"
    model.save(save_path)

    print(f"   ✅ 모델 저장 완료: {save_path}")
    return True


if __name__ == "__main__":
    # 1. 전체 일정에서 서킷별 매핑 정보 생성
    circuit_map = get_circuit_mapping()
    print(f"🌍 총 {len(circuit_map)}개 서킷 발견.")

    # 2. 서킷별 학습 루프
    success_cnt = 0
    for circuit_name, race_list in circuit_map.items():
        # fastf1 Location 이름이 아닌, FILENAME_MAP의 키 값으로 처리하기 위해 루프 변수 조정
        if train_circuit_model(circuit_name, race_list):
            success_cnt += 1

    print("\n" + "=" * 40)
    print(f"🏁 전체 작업 완료! ({success_cnt}개 모델 생성됨)")
    print(f"📂 모델 위치: ./{MODELS_DIR}/")