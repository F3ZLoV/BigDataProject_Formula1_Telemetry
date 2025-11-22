import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
import os
import re

# === [1] 설정 및 초기화 ===
st.set_page_config(
    page_title="F1 AI Race Engineer Pro",
    layout="wide",
    page_icon="🏎️",
    initial_sidebar_state="expanded"
)

# F1 서킷 정보 (2024 캘린더 기준) - Location Fullname
CIRCUIT_MAP = {
    1: "Bahrain International Circuit",
    2: "Jeddah Corniche Circuit",
    3: "Albert Park Grand Prix Circuit",
    4: "Suzuka International Racing Course",
    5: "Shanghai International Circuit",
    6: "Miami International Autodrome",
    7: "Autodromo Enzo e Dino Ferrari",
    8: "Circuit de Monaco",
    9: "Circuit Gilles Villeneuve",
    10: "Circuit de Barcelona-Catalunya",
    11: "Red Bull Ring",
    12: "Silverstone Circuit",
    13: "Hungaroring",
    14: "Circuit de Spa-Francorchamps",
    15: "Circuit Park Zandvoort",
    16: "Autodromo Nazionale di Monza",
    17: "Baku City Circuit",
    18: "Marina Bay Street Circuit",
    19: "Circuit of The Americas",
    20: "Autódromo Hermanos Rodríguez",
    21: "Autódromo José Carlos Pace",
    22: "Las Vegas Strip Circuit",
    23: "Lusail International Circuit",
    24: "Yas Marina Circuit"
}

# 화면 표시용 이름 -> Round Number 매핑 (사용자 친화적 이름)
CIRCUIT_DISPLAY_MAP = {
    "Bahrain (Sakhir)": 1, "Saudi Arabia (Jeddah)": 2, "Australia (Melbourne)": 3,
    "Japan (Suzuka)": 4, "China (Shanghai)": 5, "Miami": 6,
    "Emilia Romagna (Imola)": 7, "Monaco": 8, "Canada (Montreal)": 9,
    "Spain (Barcelona)": 10, "Austria (Spielberg)": 11, "Great Britain (Silverstone)": 12,
    "Hungary (Budapest)": 13, "Belgium (Spa)": 14, "Netherlands (Zandvoort)": 15,
    "Italy (Monza)": 16, "Azerbaijan (Baku)": 17, "Singapore": 18,
    "USA (Austin)": 19, "Mexico": 20, "Brazil (Interlagos)": 21,
    "Las Vegas": 22, "Qatar (Lusail)": 23, "Abu Dhabi (Yas Marina)": 24
}

# [추가] 서킷별 국기 이모지 매핑
CIRCUIT_FLAG_MAP = {
    "Bahrain (Sakhir)": "🇧🇭", "Saudi Arabia (Jeddah)": "🇸🇦", "Australia (Melbourne)": "🇦🇺",
    "Japan (Suzuka)": "🇯🇵", "China (Shanghai)": "🇨🇳", "Miami": "🇺🇸",
    "Emilia Romagna (Imola)": "🇮🇹", "Monaco": "🇲🇨", "Canada (Montreal)": "🇨🇦",
    "Spain (Barcelona)": "🇪🇸", "Austria (Spielberg)": "🇦🇹", "Great Britain (Silverstone)": "🇬🇧",
    "Hungary (Budapest)": "🇭🇺", "Belgium (Spa)": "🇧🇪", "Netherlands (Zandvoort)": "🇳🇱",
    "Italy (Monza)": "🇮🇹", "Azerbaijan (Baku)": "🇦🇿", "Singapore": "🇸🇬",
    "USA (Austin)": "🇺🇸", "Mexico": "🇲🇽", "Brazil (Interlagos)": "🇧🇷",
    "Las Vegas": "🇺🇸", "Qatar (Lusail)": "🇶🇦", "Abu Dhabi (Yas Marina)": "🇦🇪"
}

# [추가] 드라이버 번호와 이름 및 팀 매핑 딕셔너리
# 팀 로고를 이모지/축약어로 대체합니다.
DRIVER_NAME_MAP = {
    1: {"name": "VERSTAPPEN", "team": "🔴🐂 Red Bull"}, 2: {'name': 'SARGEANT', "team": "🔵 Williams"},
    3: {"name": "RICCIARDO", "team": " VisaCashApp"}, 4: {"name": "NORRIS", "team": "🟠 McLaren"},
    10: {"name": "GASLY", "team": "🟢 Alpine"}, 11: {"name": "PEREZ", "team": "🔴🐂 Red Bull"},
    14: {"name": "ALONSO", "team": "🟢 Aston Martin"}, 16: {"name": "LECLERC", "team": "🟥 Ferrari"},
    18: {"name": "STROLI", "team": "🟢 Aston Martin"}, 19: {"name": "K.MAGNUSSEN", "team": "⚫ Haas"},
    20: {"name": "MAGNUSSEN", "team": "⚫ Haas"}, 22: {"name": "TSUNODA", "team": " VisaCashApp"},
    23: {"name": "ALBON", "team": "🔵 Williams"}, 24: {"name": "ZHOU", "team": "🟢 Kick Sauber"},
    27: {"name": "HULKENBERG", "team": "⚫ Haas"}, 31: {"name": "OCON", "team": "🟢 Alpine"},
    44: {"name": "HAMILTON", "team": "⚫ Mercedes"}, 55: {"name": "SAINZ", "team": "🟥 Ferrari"},
    63: {"name": "RUSSELL", "team": "⚫ Mercedes"}, 77: {"name": "BOTTAS", "team": "🟢 Kick Sauber"},
    81: {"name": "PIASTRI", "team": "🟠 McLaren"}, 99: {"name": "GIOVINAZZI", "team": "🟢 Kick Sauber"}
}


# 스타일링
st.markdown("""
<style>
    .stApp { background-color: #15151e; color: #e0e0e0; }
    .stSidebar { background-color: #1e1e24; }
    h1, h2, h3 { color: #ff1801 !important; font-family: 'Arial Black'; }
    .metric-card { background-color: #2b2b36; padding: 15px; border-radius: 8px; border-left: 5px solid #ff1801; margin-bottom: 10px;}
    .stButton>button { background-color: #ff1801; color: white; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #d41400; }
</style>
""", unsafe_allow_html=True)


# === [2] 리소스 로더 ]===
@st.cache_resource
def load_circuit_assets(round_num, circuit_fullname):
    """선택된 라운드(서킷)의 데이터와 모델을 동적으로 로드합니다."""

    # 1. 데이터 로드 (최신 연도부터 탐색)
    df = pd.DataFrame()
    for year in [2024, 2023]:
        data_path = f"f1_processed_warehouse/year={year}/round={round_num}/telemetry.parquet"
        if os.path.exists(data_path):
            try:
                temp_df = pd.read_parquet(data_path)
                # nGear 컬럼명 통일 (학습 시 사용된 이름: nGear)
                if 'Gear' in temp_df.columns and 'nGear' not in temp_df.columns:
                    temp_df.rename(columns={'Gear': 'nGear'}, inplace=True)
                df = temp_df
                break
            except:
                pass

    # 2. 고스트카 모델 로드 (파일명 매칭 - 단순화된 로직)

    # 캡처 이미지에 보이는 파일명을 기반으로 라운드 번호와 정확히 매핑
    FILE_SUFFIX_MAP = {
        1: "Sakhir", 2: "Jeddah", 3: "Melbourne", 4: "Suzuka", 5: "Shanghai",
        6: "Miami", 7: "Imola", 8: "Monaco", 9: "Montreal", 10: "Barcelona",
        11: "Spielberg", 12: "Silverstone", 13: "Budapest", 14: "SpaFrancorchamps",
        15: "Zandvoort", 16: "Monza", 17: "Baku", 18: "MarinaBay",  # 캡처 이미지에 MarinaBay.h5로 보임
        19: "Austin", 20: "MexicoCity", 21: "SaoPaulo",  # Interlagos -> SaoPaulo
        22: "LasVegas", 23: "Lusail", 24: "YasMarina"  # Yas Marina -> YasMarina
    }

    expected_suffix = FILE_SUFFIX_MAP.get(round_num, None)

    model_path = None
    if expected_suffix:
        model_filename = f"ghost_{expected_suffix}.h5"
        model_path = os.path.join("models_by_circuit", model_filename)

        # 파일이 실제 존재하는지 확인 (없으면 None 유지)
        if not os.path.exists(model_path):
            model_path = None

        # Yas Marina Circuit의 경우 예외 처리 (YasIsland 파일명도 체크)
        if round_num == 24 and not model_path:
            model_filename = "ghost_YasIsland.h5"
            temp_path = os.path.join("models_by_circuit", model_filename)
            if os.path.exists(temp_path):
                model_path = temp_path

    ghost_model = None
    if model_path and os.path.exists(model_path):
        try:
            # Keras/TensorFlow 버전 충돌 해결
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'mean_squared_error': tf.keras.losses.MeanSquaredError()
            }

            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            ghost_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        except Exception as e:
            st.error(f"AI Model Load Error: {e}")
            pass

    # 모델 경로가 없으면, 경고 메시지를 위해 예상 파일명 경로를 사용
    if not model_path and expected_suffix:
        model_path = os.path.join("models_by_circuit", f"ghost_{expected_suffix}.h5")
    elif not model_path:
        model_path = os.path.join("models_by_circuit", "ghost_unknown.h5")

    # 3. 전략 모델 로드 (공통 모델)
    strategy_model = None
    if os.path.exists("f1_ai_race_engineer_v4.pkl"):
        strategy_model = joblib.load("f1_ai_race_engineer_v4.pkl")

    return df, ghost_model, strategy_model, model_path


# === [3] 사이드바 UI ===
st.sidebar.title("🏎️ F1 AI Engineer Pro")
st.sidebar.info("Official Formula 1 Data Analysis")

# [수정된 부분] 서킷 선택 드롭다운에 국기 이모지 추가
circuit_list_with_flags = [
    f"{CIRCUIT_FLAG_MAP[name]} {name}" for name in CIRCUIT_DISPLAY_MAP.keys()
]
selected_display_name_with_flag = st.sidebar.selectbox(
    "🌍 서킷 선택 (Circuit Selection)",
    circuit_list_with_flags,
    index=3  # Japan
)

# 실제 서킷 이름만 추출 (국기 이모지 및 공백 제거)
selected_display_name = selected_display_name_with_flag.split(' ', 1)[1]

selected_round = CIRCUIT_DISPLAY_MAP[selected_display_name]
selected_circuit_fullname = CIRCUIT_MAP[selected_round]

# 페이지 네비게이션
page = st.sidebar.radio("메뉴 (Menu)", ["🏠 홈", "📊 데이터 탐색기", "🧠 전략 시뮬레이터", "👻 고스트카 연구소"])

# 리소스 로드 실행
df_circuit, ghost_model, strategy_model, current_model_path = load_circuit_assets(selected_round,
                                                                                  selected_circuit_fullname)

# ==============================================================================
# 🏠 1. 홈
# ==============================================================================
if page == "🏠 홈":
    st.title(f"🏁 {selected_display_name}")
    st.markdown(f"### 라운드 {selected_round} | {selected_circuit_fullname}")

    if df_circuit.empty:
        st.error(f"⚠️ '{selected_display_name}' 서킷의 데이터가 없습니다.\n\n'**train_by_circuit.py**'를 실행했는지 확인해주세요.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="metric-card"><h3>총 데이터</h3><p>{len(df_circuit):,} Row</p><p>텔레메트리 포인트</p></div>',
                unsafe_allow_html=True)
        with col2:
            max_speed = int(df_circuit['Speed'].max())
            st.markdown(f'<div class="metric-card"><h3>최고 속도</h3><p>{max_speed} km/h</p><p>최대 속력</p></div>',
                        unsafe_allow_html=True)
        with col3:
            track_len = int(df_circuit['Distance'].max())
            st.markdown(
                f'<div class="metric-card"><h3>트랙 길이</h3><p>{track_len} m</p><p>서킷 거리</p></div>',
                unsafe_allow_html=True)

        st.success("✅ 데이터 로드 완료")

        if ghost_model:
            st.success(f"🧠 AI 모델 로드 완료: `{os.path.basename(current_model_path)}`")
        else:
            st.warning(f"⚠️ AI 모델 파일을 찾을 수 없습니다. (예상 파일명: {os.path.basename(current_model_path)})")

# ==============================================================================
# 📊 2. 데이터 탐색기
# ==============================================================================
elif page == "📊 데이터 탐색기":
    st.title(f"📊 데이터 탐색기: {selected_display_name}")

    if df_circuit.empty:
        st.warning("데이터가 없습니다.")
    else:
        col1, col2 = st.columns([1, 3])

        drivers = sorted(df_circuit['Driver'].unique())

        # [수정된 부분] 드라이버 선택 드롭다운에 팀 로고/이름 추가
        driver_options = []
        for d_id in drivers:
            info = DRIVER_NAME_MAP.get(d_id, {"name": "알 수 없음", "team": "❓"})
            option_label = f"{d_id} ({info['team']} | {info['name']})"
            driver_options.append(option_label)

        with col1:
            # 드롭다운에서 표시할 옵션은 이름+ID, 실제 반환 값은 ID
            selected_driver_option = st.selectbox("드라이버 선택", driver_options)

            # 선택된 옵션에서 드라이버 ID만 추출 (첫 번째 공백까지의 문자열)
            selected_driver_id_str = selected_driver_option.split(' ')[0]
            try:
                selected_driver = int(selected_driver_id_str)
            except ValueError:
                selected_driver = drivers[0] if drivers else None  # 파싱 실패 시 기본값 설정

            if selected_driver is not None:
                driver_data = df_circuit[df_circuit['Driver'] == selected_driver]
                laps = sorted(driver_data['LapNumber'].unique())
                selected_lap = st.selectbox("랩 선택", laps)
            else:
                st.warning("선택할 수 있는 드라이버가 없습니다.")
                selected_lap = None

        with col2:
            if selected_lap is not None and not driver_data.empty:
                lap_data = driver_data[driver_data['LapNumber'] == selected_lap]
                driver_info = DRIVER_NAME_MAP.get(selected_driver, {"name": selected_driver_id_str, "team": "알 수 없음"})
                driver_name = driver_info['name']

                fig = px.line(lap_data, x='Distance', y='Speed',
                              title=f"드라이버 {driver_name} ({selected_driver}) - 랩 {selected_lap} 속도 추적",
                              color_discrete_sequence=['#ff1801'])
                fig.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white', height=400)
                st.plotly_chart(fig, width='stretch')

                fig2 = go.Figure()
                fig2.add_trace(
                    go.Scatter(x=lap_data['Distance'], y=lap_data['Throttle'], name='스로틀',
                               line=dict(color='green')))
                fig2.add_trace(
                    go.Scatter(x=lap_data['Distance'], y=lap_data['Brake'] * 100, name='브레이크',
                               line=dict(color='red')))
                fig2.update_layout(title="텔레메트리", plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e',
                                   font_color='white',
                                   height=300)
                st.plotly_chart(fig2, width='stretch')
            else:
                st.info("데이터를 표시할 랩이 없습니다.")


# ==============================================================================
# 🧠 3. 전략 시뮬레이터
# ==============================================================================
elif page == "🧠 전략 시뮬레이터":
    st.title("🧠 타이어 전략 시뮬레이터")

    if strategy_model is None:
        st.error("전략 모델 파일(pkl)이 없습니다.")
    else:
        col1, col2 = st.columns([1, 2])

        # NameError 및 정확도 문제 해결: BASE_TIME을 동적으로 계산
        # 1. BASE_TIME (기준 랩타임) 동적 설정
        BASE_TIME = 90.0  # 기본 fallback 값

        lap_time_col = 'LapTimeSeconds'

        if not df_circuit.empty and lap_time_col in df_circuit.columns:
            fastest_lap_time = df_circuit[lap_time_col].min()
            if fastest_lap_time > 10:  # 10초 미만은 유효하지 않다고 가정
                BASE_TIME = fastest_lap_time

        st.caption(f"기준 랩타임 (Base Lap Time): **{BASE_TIME:.3f}** 초 (데이터에서 계산)")

        BASE_TIME_FOR_SIMULATION = BASE_TIME

        with col1:
            st.markdown("### ⚙️ 레이스 조건")
            tyre_life = st.slider("타이어 사용 랩 수 (Tyre Age)", 1, 40, 10)
            compound = st.selectbox("타이어 컴파운드", ["SOFT", "MEDIUM", "HARD"])
            compound_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3}

            if st.button("🚀 랩타임 예측 (Predict)"):

                # 예측 오류 해결: 'Speed_mean'을 제외하고 모델이 학습된 7개 컬럼을 사용하도록 수정
                feature_order = [
                    'Round', 'TyreLife_max', 'Compound_Encoded', 'Throttle_mean',
                    'Brake_mean', 'Prev_Pace_Ratio', 'Prev_Speed_mean'
                ]

                input_data_dict = {
                    'Round': selected_round,
                    'TyreLife_max': tyre_life,
                    'Compound_Encoded': compound_map[compound],
                    'Throttle_mean': 70.0,
                    'Brake_mean': 0.15,
                    'Prev_Pace_Ratio': 1.02,
                    'Prev_Speed_mean': 230.0,
                }

                # 순서에 맞춰 DataFrame 생성
                input_data = pd.DataFrame([input_data_dict], columns=feature_order)

                try:
                    pred_ratio = strategy_model.predict(input_data)[0]

                    # 동적 BASE_TIME 사용
                    pred_time = BASE_TIME * pred_ratio

                    # compound 이름도 저장
                    st.session_state['pred'] = (pred_ratio, pred_time, compound_map[compound], compound)
                except Exception as e:
                    st.error(f"예측 오류: {e}")

        with col2:
            if 'pred' in st.session_state:
                # Compound Map 값과 compound 이름을 세션 상태에서 가져옴
                ratio, time_sec, current_compound_map, compound_name = st.session_state['pred']

                st.markdown("### 📊 AI 예측 결과")
                st.metric(label="예측 랩타임 (Predicted Lap Time)", value=f"{time_sec:.3f} s",
                          delta=f"{(ratio - 1.0) * 100:.2f}% 페이스 저하", delta_color="inverse")

                # 마모 곡선 시뮬레이션
                lives = list(range(1, 41))
                # 동적 BASE_TIME 사용
                preds = [BASE_TIME_FOR_SIMULATION * (1.0 + (0.005 * l * current_compound_map)) for l in lives]

                fig_deg = px.line(x=lives, y=preds, labels={'x': '랩 수', 'y': '랩타임 (s)'},
                                  title=f"타이어 마모 곡선 ({compound_name})")
                fig_deg.add_vline(x=tyre_life, line_dash="dash", line_color="red")
                fig_deg.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white')
                st.plotly_chart(fig_deg, width='stretch')

# ... (이전 코드 생략)

# ==============================================================================
# 👻 4. 고스트 카 연구소 (핵심 기능)
# ==============================================================================
elif page == "👻 고스트카 연구소":
    st.title(f"👻 고스트카 연구소: {selected_display_name}")

    if df_circuit.empty:
        st.error("데이터가 없습니다.")
    elif ghost_model is None:
        st.warning(
            f"⚠️ 이 서킷의 AI 모델 파일이 없습니다. '**train_by_circuit.py**'를 실행하여 모델을 생성해주세요. (예상 파일: `{os.path.basename(current_model_path)}`)")
    else:
        st.markdown("드라이버의 랩과 **AI 최적 라인**을 비교해보세요.")

        col_sel, col_btn = st.columns([3, 1])

        drivers = sorted(df_circuit['Driver'].unique())

        # 드라이버 선택 드롭다운에 팀 로고/이름 추가
        driver_options = []
        for d_id in drivers:
            info = DRIVER_NAME_MAP.get(d_id, {"name": "알 수 없음", "team": "❓"})
            option_label = f"{d_id} ({info['team']} | {info['name']})"
            driver_options.append(option_label)

        with col_sel:
            selected_driver_option = st.selectbox("드라이버 선택", driver_options)

            # 선택된 옵션에서 드라이버 ID만 추출
            selected_driver_id_str = selected_driver_option.split(' ')[0]
            try:
                target_driver = int(selected_driver_id_str)
                target_driver_name = DRIVER_NAME_MAP.get(target_driver, {"name": selected_driver_id_str})
                target_driver_name = target_driver_name['name']  # 이름만 추출
            except ValueError:
                target_driver = drivers[0] if drivers else None
                target_driver_name = "알 수 없음"

        # 가장 빠른 랩 자동 선택
        if target_driver is not None:
            driver_laps = df_circuit[df_circuit['Driver'] == target_driver]
            if not driver_laps.empty:
                target_lap = driver_laps['LapNumber'].max()
                sample_lap_data = driver_laps[driver_laps['LapNumber'] == target_lap].sort_values('Distance')
            else:
                st.warning("이 드라이버의 유효한 랩을 찾을 수 없습니다.")
                sample_lap_data = pd.DataFrame()
        else:
            sample_lap_data = pd.DataFrame()

        if st.button("🧬 AI 분석 실행"):
            if sample_lap_data.empty:
                st.error("선택한 드라이버의 유효한 랩 데이터가 없습니다.")
            else:
                with st.spinner(f"AI가 {target_driver_name} 드라이버의 랩을 분석 중..."):

                    # 1. 스케일러 (해당 서킷 데이터에 맞춰 동적 생성)
                    track_len = df_circuit['Distance'].max()
                    scaler = MinMaxScaler()
                    # [Distance, Speed, Throttle, Brake, nGear] (5개 피처)
                    scaler.fit([[0, 0, 0, 0, 0], [track_len, 360, 100, 1, 8]])

                    # 5개 피처만 사용하도록 정의
                    feature_cols = ['Distance', 'Speed', 'Throttle', 'Brake', 'nGear']
                    # 좌표계 피처 (애니메이션에 사용)
                    coord_cols = ['X', 'Y']

                    # 데이터 전처리
                    X_input = sample_lap_data[feature_cols].values
                    X_scaled = scaler.transform(X_input)

                    # AI 예측 속도 계산
                    ai_speeds = []
                    ai_coords = []  # AI 트레이스 플롯용
                    seq_len = 20

                    # AI 예측은 텔레메트리 포인트의 약 1/5만 생성하도록 건너뜁니다. (애니메이션 동기화 목적)
                    # 원본 데이터의 텔레메트리 포인트를 기준으로 예측 지점을 맞춥니다.
                    step_size = 5

                    # AI 예측 데이터 준비
                    ai_speed_data = []

                    for i in range(seq_len, len(X_scaled), step_size):
                        seq = X_scaled[i - seq_len:i].reshape(1, seq_len, 5)
                        pred = ghost_model.predict(seq, verbose=0)

                        dummy = np.zeros((1, 5))
                        dummy[0, 1] = pred[0][0]
                        speed = scaler.inverse_transform(dummy)[0, 1]

                        # AI의 속도 트레이스용 데이터 (1.05배 보정)
                        ai_speeds.append({'Distance': X_input[i, 0], 'Speed': speed * 1.05})

                        # 애니메이션용 데이터 (X, Y 좌표 및 예측 속도)
                        if 'X' in sample_lap_data.columns and 'Y' in sample_lap_data.columns:
                            ai_speed_data.append({
                                'Frame': i,
                                'X': sample_lap_data.iloc[i]['X'],
                                'Y': sample_lap_data.iloc[i]['Y'],
                                'Speed': speed * 1.05,
                                'Source': 'AI 고스트'
                            })

                    # AI 속도 트레이스 DF 생성
                    ai_df = pd.DataFrame(ai_speeds)

                    # 4. 애니메이션용 데이터 준비
                    if 'X' in sample_lap_data.columns and 'Y' in sample_lap_data.columns:

                        # 인간 드라이버 데이터 준비 (AI와 동일한 Frame Index만 사용)
                        human_speed_data = []
                        for i in range(seq_len, len(sample_lap_data), step_size):
                            human_speed_data.append({
                                'Frame': i,
                                'X': sample_lap_data.iloc[i]['X'],
                                'Y': sample_lap_data.iloc[i]['Y'],
                                'Speed': sample_lap_data.iloc[i]['Speed'],
                                'Source': f"드라이버 {target_driver_name}"
                            })

                        # 두 데이터셋을 합치고, 순서대로 정렬
                        comparison_df = pd.DataFrame(human_speed_data + ai_speed_data)

                        # 5. 서킷 레이아웃 애니메이션 (X, Y 좌표 사용)
                        st.subheader("🏎️ 서킷 레이아웃 애니메이션 비교")

                        # 배경 트랙 라인 (전체 랩 데이터 사용)
                        fig_track = go.Figure()
                        fig_track.add_trace(
                            go.Scatter(
                                x=sample_lap_data['X'],
                                y=sample_lap_data['Y'],
                                mode='lines',
                                line=dict(color='gray', width=2),
                                name='서킷 라인',
                                hoverinfo='none'
                            )
                        )

                        # 애니메이션 플롯 (Plotly Express 사용)
                        # `animation_frame`을 사용하여 시간에 따른 위치 변화를 나타냅니다.
                        fig_animation = px.scatter(
                            comparison_df,
                            x='X',
                            y='Y',
                            animation_frame='Frame',
                            color='Source',
                            size='Speed',  # 속도에 따라 마커 크기 변화
                            hover_data=['Speed'],
                            color_discrete_map={
                                'AI 고스트': 'red',
                                f"드라이버 {target_driver_name}": 'white'
                            },
                            title="AI 고스트 vs 드라이버 (Lap Trace Animation)",
                            height=700
                        )

                        # 배경 트랙 라인을 애니메이션 프레임에 추가 (프레임이 바뀔 때 배경이 사라지는 것을 방지)
                        # Plotly Express 애니메이션은 배경이 각 프레임에 대해 다시 그려집니다.
                        # 여기서는 Go.Figure에 트랙 라인을 추가하고, PX scatter를 Go.Figure의 프레임으로 변환하여 병합하는 복잡한 과정 대신
                        # 간단하게 배경 트랙 라인을 다시 추가합니다.

                        # 레이아웃 스타일 설정
                        fig_animation.update_layout(
                            xaxis_title="X 좌표",
                            yaxis_title="Y 좌표",
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            font_color='white',
                            showlegend=True
                        )

                        # 축 비율을 같게 설정하여 서킷 모양 왜곡 방지
                        fig_animation.update_yaxes(scaleanchor="x", scaleratio=1)

                        st.plotly_chart(fig_animation, width='stretch')

                    # 3. 속도 트레이스 비교 (기존 로직 유지)
                    st.subheader("📊 속도 트레이스 비교")
                    fig_ghost = go.Figure()
                    fig_ghost.add_trace(
                        go.Scatter(x=sample_lap_data['Distance'], y=sample_lap_data['Speed'],
                                   name=f"인간 드라이버 ({target_driver_name})",
                                   line=dict(color='gray')))
                    fig_ghost.add_trace(go.Scatter(x=ai_df['Distance'], y=ai_df['Speed'], name='AI 고스트',
                                                   line=dict(color='red', dash='dash')))

                    fig_ghost.update_layout(
                        title=f"속도 성능 비교 - {selected_display_name}",
                        xaxis_title="트랙 거리 (m)",
                        yaxis_title="속도 (km/h)",
                        plot_bgcolor='#1e1e1e',
                        paper_bgcolor='#1e1e1e',
                        font_color='white',
                        height=600,
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_ghost, width='stretch')

        else:
            st.warning("충분한 데이터 포인트가 없습니다.")