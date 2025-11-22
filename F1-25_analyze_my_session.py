import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
import os

# === 설정 ===
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "f1_bigdata_db"
COLLECTION_NAME = "live_race_data"
MODEL_PATH = "f1_ghost_car_model_v1.h5"
SEQUENCE_LENGTH = 20
CIRCUIT_LENGTH = 5807

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def analyze_last_session():
    print("📊 [Post-Race Analysis] 세션 데이터 분석 중...")

    # 1. MongoDB에서 최신 데이터 가져오기
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # 가장 최근 데이터 5000개 가져오기
    cursor = collection.find().sort("Timestamp", -1).limit(5000)
    data = list(cursor)

    if not data:
        print("❌ 저장된 주행 데이터가 없습니다.")
        return

    data.reverse()
    df = pd.DataFrame(data)

    print(f"✅ 데이터 로드 완료: {len(df)} 포인트")

    # 2. 분석할 랩 선정
    laps = df['LapNumber'].unique()

    # (수정) 데이터가 충분한 마지막 랩을 찾습니다.
    if len(laps) > 1:
        target_lap = laps[-2]  # 완주한 마지막 랩
    else:
        target_lap = laps[0]

    print(f"🎯 분석 대상: Lap {target_lap}")

    lap_df = df[df['LapNumber'] == target_lap].sort_values('Distance')

    # 3. AI 모델 로드 및 예측
    if not os.path.exists(MODEL_PATH):
        print("❌ 모델 파일이 없습니다.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    scaler = MinMaxScaler()
    scaler.fit([[0, 0, 0, 0, 0], [CIRCUIT_LENGTH, 360, 100, 1, 8]])

    # DB에 저장된 컬럼명 확인 (Gear vs nGear)
    # 보통 'Gear'로 저장했으므로 그대로 사용
    feature_cols = ['Distance', 'Speed', 'Throttle', 'Brake', 'Gear']

    X_input = lap_df[feature_cols].values
    X_scaled = scaler.transform(X_input)

    ai_speeds = []

    # 시계열 예측
    for i in range(SEQUENCE_LENGTH, len(X_scaled)):
        seq = X_scaled[i - SEQUENCE_LENGTH:i].reshape(1, SEQUENCE_LENGTH, 5)
        pred = model.predict(seq, verbose=0)

        dummy = np.zeros((1, 5))
        dummy[0, 1] = pred[0][0]
        speed_kmh = scaler.inverse_transform(dummy)[0, 1]
        ai_speeds.append(speed_kmh)

    # 길이 맞추기
    actual_speeds = lap_df['Speed'].values[SEQUENCE_LENGTH:]
    distances = lap_df['Distance'].values[SEQUENCE_LENGTH:]

    # 엄격 모드 (1.05배) 적용
    ai_speeds = np.array(ai_speeds) * 1.05

    # 4. 리포트 시각화 및 저장
    plt.figure(figsize=(14, 8))

    # 상단: 속도 비교
    plt.subplot(2, 1, 1)
    plt.plot(distances, actual_speeds, label='My Lap', color='white', linewidth=1.5)
    plt.plot(distances, ai_speeds, label='AI Ideal Line', color='red', linestyle='--', alpha=0.8)
    plt.title(f"Race Analysis: Lap {target_lap} (Suzuka)", fontsize=14, color='white')
    plt.ylabel("Speed (km/h)", color='white')
    plt.legend()
    plt.grid(True, alpha=0.2)

    # 스타일 설정 (다크 모드)
    plt.gca().set_facecolor('#1e1e1e')
    plt.gcf().set_facecolor('#1e1e1e')
    plt.tick_params(colors='white')

    # 하단: 델타(차이) 그래프
    plt.subplot(2, 1, 2)
    delta = ai_speeds - actual_speeds
    plt.plot(distances, delta, color='yellow', label='Time Loss (Delta)')
    plt.fill_between(distances, 0, delta, where=(delta > 0), color='red', alpha=0.3)
    plt.axhline(0, color='white', linestyle=':', alpha=0.5)
    plt.xlabel("Distance (m)", color='white')
    plt.ylabel("Speed Diff (km/h)", color='white')
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.gca().set_facecolor('#1e1e1e')
    plt.tick_params(colors='white')

    plt.tight_layout()

    # === [추가됨] 이미지 저장 코드 ===
    save_filename = f"Analysis_Lap_{int(target_lap)}.png"
    plt.savefig(save_filename, dpi=100)  # 파일로 저장
    print(f"\n💾 분석 그래프가 저장되었습니다: {save_filename}")

    # 화면에도 띄우기 (서버 환경이면 주석 처리)
    plt.show()

    # 5. 텍스트 피드백
    print("\n" + "=" * 40)
    print(f" 🏁 DRIVER DEBRIEF: LAP {target_lap}")
    print("=" * 40)

    avg_speed_diff = np.mean(delta)
    print(f"📊 평균 속도 차이: {avg_speed_diff:.1f} km/h (AI 대비)")

    if avg_speed_diff > 10:
        print("💡 총평: 전체적으로 페이스가 낮습니다. 과감하게 공략해보세요.")
    elif avg_speed_diff > 5:
        print("💡 총평: 좋은 주행입니다! 코너 탈출만 조금 더 신경 쓰세요.")
    else:
        print("🏆 총평: 완벽합니다! AI를 이길 수준입니다.")


if __name__ == "__main__":
    analyze_last_session()