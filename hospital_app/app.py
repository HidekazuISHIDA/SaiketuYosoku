import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import jpholiday
import json
import matplotlib.pyplot as plt

from datetime import date, timedelta

st.set_page_config(page_title="🏥 A病院 待ち人数・待ち時間予測", layout="wide")
st.title("🏥 A病院 待ち人数・待ち時間 統合予測アプリ")
st.caption("※ 個人情報は扱いません / シミュレーション用途")

@st.cache_resource
def load_models():
    count_model = xgb.XGBRegressor()
    count_model.load_model("model_A_timeseries.json")

    waittime_model = xgb.XGBRegressor()
    waittime_model.load_model("model_A_waittime_30min.json")

    queue_model = xgb.XGBRegressor()
    queue_model.load_model("model_A_queue_30min.json")

    with open("columns_A_timeseries.json", "r") as f:
        count_cols = json.load(f)

    with open("columns_A_multi_30min.json", "r") as f:
        multi_cols = json.load(f)

    return count_model, waittime_model, queue_model, count_cols, multi_cols

count_model, waittime_model, queue_model, count_feature_columns, multi_feature_columns = load_models()

col1, col2, col3 = st.columns(3)

with col1:
    target_date = st.date_input("📅 予測対象日", value=date.today() + timedelta(days=1))

with col2:
    total_patients = st.number_input("👥 延べ外来患者数", min_value=0, max_value=5000, value=1200, step=50)

with col3:
    weather = st.selectbox("☁ 天気予報", ["晴", "曇", "雨", "雪", "快晴", "薄曇"])

if st.button("▶ 予測シミュレーション実行"):
    with st.spinner("計算中..."):
        target_dt = pd.to_datetime(target_date)

        is_holiday_daily = (
            jpholiday.is_holiday(target_dt)
            or target_dt.weekday() >= 5
            or (target_dt.month == 12 and target_dt.day >= 29)
            or (target_dt.month == 1 and target_dt.day <= 3)
        )

        prev_date = target_dt - timedelta(days=1)
        is_prev_holiday = (
            jpholiday.is_holiday(prev_date)
            or prev_date.weekday() >= 5
        )

        time_slots = pd.date_range(
            start=target_dt.replace(hour=8, minute=0),
            end=target_dt.replace(hour=18, minute=0),
            freq="30T"
        )

        results = []
        lags = {"lag_30min": 0.0, "lag_60min": 0.0, "lag_90min": 0.0}
        queue_at_start = 0

        for ts in time_slots:
            count_features = pd.DataFrame(0, index=[0], columns=count_feature_columns)
            count_features["hour"] = ts.hour
            count_features["minute"] = ts.minute
            count_features["is_first_slot"] = int(ts.hour == 8 and ts.minute == 0)
            count_features["is_second_slot"] = int(ts.hour == 8 and ts.minute == 30)
            count_features["total_outpatient_count"] = total_patients
            count_features["is_holiday"] = int(is_holiday_daily)

            if "月" in count_features.columns:
                count_features["月"] = ts.month
            if "週回数" in count_features.columns:
                count_features["週回数"] = (ts.day - 1) // 7 + 1
            if "前日祝日フラグ" in count_features.columns:
                count_features["前日祝日フラグ"] = int(is_prev_holiday)

            count_features["雨フラグ"] = int("雨" in weather)
            count_features["雪フラグ"] = int("雪" in weather)

            weather_col = f"天気カテゴリ_{weather[0]}"
            if weather_col in count_features.columns:
                count_features[weather_col] = 1

            dow_col = f"dayofweek_{ts.dayofweek}"
            if dow_col in count_features.columns:
                count_features[dow_col] = 1

            for lag_col, lag_val in lags.items():
                if lag_col in count_features.columns:
                    count_features[lag_col] = lag_val

            predicted_reception = max(0, round(count_model.predict(count_features)[0]))

            multi_features = pd.DataFrame(0, index=[0], columns=multi_feature_columns)
            multi_features["hour"] = ts.hour
            multi_features["minute"] = ts.minute
            multi_features["reception_count"] = predicted_reception
            multi_features["queue_at_start_of_slot"] = queue_at_start
            multi_features["total_outpatient_count"] = total_patients
            multi_features["is_holiday"] = int(is_holiday_daily)

            if "月" in multi_features.columns:
                multi_features["月"] = ts.month
            if "週回数" in multi_features.columns:
                multi_features["週回数"] = (ts.day - 1) // 7 + 1
            if "前日祝日フラグ" in multi_features.columns:
                multi_features["前日祝日フラグ"] = int(is_prev_holiday)

            multi_features["雨フラグ"] = int("雨" in weather)
            multi_features["雪フラグ"] = int("雪" in weather)

            if weather_col in multi_features.columns:
                multi_features[weather_col] = 1
            if dow_col in multi_features.columns:
                multi_features[dow_col] = 1

            predicted_queue = max(0, round(queue_model.predict(multi_features)[0]))
            predicted_wait = max(0, int(round(waittime_model.predict(multi_features)[0])))

            results.append({
                "時間帯": ts.strftime("%H:%M"),
                "予測受付数": predicted_reception,
                "予測待ち人数(人)": predicted_queue,
                "予測平均待ち時間(分)": predicted_wait
            })

            lags = {
                "lag_30min": predicted_reception,
                "lag_60min": lags["lag_30min"],
                "lag_90min": lags["lag_60min"]
            }
            queue_at_start = predicted_queue

        result_df = pd.DataFrame(results)
        st.success("予測完了")
        st.dataframe(result_df, use_container_width=True)

        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.bar(result_df["時間帯"], result_df["予測待ち人数(人)"], alpha=0.7)
        ax1.set_ylabel("待ち人数（人）")
        ax1.tick_params(axis="x", rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(result_df["時間帯"], result_df["予測平均待ち時間(分)"], marker="o")
        ax2.set_ylabel("平均待ち時間（分）")

        plt.title(f"{target_date} の予測結果")
        st.pyplot(fig)
