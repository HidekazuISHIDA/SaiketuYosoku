import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import jpholiday
import json
import matplotlib.pyplot as plt
from datetime import date, timedelta

# =========================================================
# アプリ設定
# =========================================================
st.set_page_config(
    page_title="🏥 A病院 待ち人数・待ち時間予測",
    layout="wide"
)

st.title("🏥 A病院 待ち人数・待ち時間 統合予測アプリ")
st.caption("※ 本アプリは予測モデルによる参考値を表示します")

# =========================================================
# モデル・カラム読み込み（Booster）
# =========================================================
@st.cache_resource
def load_models():
    count_model = xgb.Booster()
    count_model.load_model("model_A_timeseries.json")

    waittime_model = xgb.Booster()
    waittime_model.load_model("model_A_waittime_30min.json")

    queue_model = xgb.Booster()
    queue_model.load_model("model_A_queue_30min.json")

    with open("columns_A_timeseries.json") as f:
        count_feature_columns = json.load(f)

    with open("columns_A_multi_30min.json") as f:
        multi_feature_columns = json.load(f)

    return (
        count_model,
        waittime_model,
        queue_model,
        count_feature_columns,
        multi_feature_columns,
    )

count_model, waittime_model, queue_model, count_cols, multi_cols = load_models()

# =========================================================
# UI
# =========================================================
st.sidebar.header("🔧 入力条件")

target_date = st.sidebar.date_input(
    "予測対象日",
    value=date.today() + timedelta(days=1)
)

total_patients = st.sidebar.number_input(
    "延べ外来患者数",
    min_value=0,
    max_value=5000,
    value=1200,
    step=50
)

weather = st.sidebar.selectbox(
    "天気",
    ["晴", "曇", "雨", "雪", "快晴", "薄曇"]
)

run_button = st.sidebar.button("▶ 予測シミュレーション実行")

# =========================================================
# 予測処理
# =========================================================
if run_button:
    with st.spinner("予測計算中..."):

        is_holiday = (
            jpholiday.is_holiday(target_date)
            or target_date.weekday() >= 5
            or (target_date.month == 12 and target_date.day >= 29)
            or (target_date.month == 1 and target_date.day <= 3)
        )

        prev_date = target_date - timedelta(days=1)
        is_prev_holiday = (
            jpholiday.is_holiday(prev_date)
            or prev_date.weekday() >= 5
            or (prev_date.month == 12 and prev_date.day >= 29)
            or (prev_date.month == 1 and prev_date.day <= 3)
        )

        time_slots = pd.date_range(
            start=pd.Timestamp(target_date).replace(hour=8, minute=0),
            end=pd.Timestamp(target_date).replace(hour=18, minute=0),
            freq="30min",
        )

        results = []
        lags = [0, 0, 0]
        queue_at_start = 0

        for ts in time_slots:
            # -----------------------------
            # 受付人数予測
            # -----------------------------
            df_count = pd.DataFrame(0, index=[0], columns=count_cols)
            df_count["hour"] = ts.hour
            df_count["minute"] = ts.minute
            df_count["is_holiday"] = int(is_holiday)
            df_count["total_outpatient_count"] = total_patients
            df_count["前日祝日フラグ"] = int(is_prev_holiday)
            df_count["雨フラグ"] = int("雨" in weather)
            df_count["雪フラグ"] = int("雪" in weather)

            for i, lag in enumerate(lags):
                col = f"lag_{(i+1)*30}min"
                if col in df_count.columns:
                    df_count[col] = lag

            dcount = xgb.DMatrix(df_count[count_cols])
            reception = int(max(0, round(count_model.predict(dcount)[0])))

            # -----------------------------
            # 待ち人数・待ち時間予測
            # -----------------------------
            df_multi = pd.DataFrame(0, index=[0], columns=multi_cols)
            df_multi["hour"] = ts.hour
            df_multi["minute"] = ts.minute
            df_multi["reception_count"] = reception
            df_multi["queue_at_start_of_slot"] = queue_at_start
            df_multi["is_holiday"] = int(is_holiday)
            df_multi["total_outpatient_count"] = total_patients
            df_multi["前日祝日フラグ"] = int(is_prev_holiday)
            df_multi["雨フラグ"] = int("雨" in weather)
            df_multi["雪フラグ"] = int("雪" in weather)

            dmulti = xgb.DMatrix(df_multi[multi_cols])

            queue_pred = int(max(0, round(queue_model.predict(dmulti)[0])))
            wait_pred = int(max(0, round(waittime_model.predict(dmulti)[0])))

            results.append({
                "時間帯": ts.strftime("%H:%M"),
                "予測受付数": reception,
                "予測待ち人数(人)": queue_pred,
                "予測平均待ち時間(分)": wait_pred,
            })

            lags = [reception] + lags[:2]
            queue_at_start = queue_pred

        result_df = pd.DataFrame(results)

    # =========================================================
    # 表示
    # =========================================================
    st.subheader(f"📊 {target_date} の予測結果")
    st.dataframe(result_df, use_container_width=True)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.bar(result_df["時間帯"], result_df["予測待ち人数(人)"])
    ax1.set_ylabel("待ち人数")

    ax2 = ax1.twinx()
    ax2.plot(result_df["時間帯"], result_df["予測平均待ち時間(分)"], marker="o")
    ax2.set_ylabel("平均待ち時間（分）")

    ax1.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
