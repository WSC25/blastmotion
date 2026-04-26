# 在 app.py 中優化輸入邏輯
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 載入模型與平均值
try:
    data = joblib.load('baseball_model.pkl')
    models = data['models']
    feature_means = data['feature_means']
    X_cols = data['X_cols']
except FileNotFoundError:
    st.error("找不到模型檔案 baseball_model.pkl，請先執行訓練程式碼。")
    st.stop()

st.title("賽前揮棒數據至打擊表現轉換預測")

# 建立一個表單
with st.form("prediction_form"):
    st.write("請輸入賽前練習數據（留空將以現有平均值計算）")
    
    user_input = {}
    cols = st.columns(2)
    
    for i, col_name in enumerate(X_cols):
        with cols[i % 2]:
            # 使用 number_input 並設定預設值為 None
            val = st.number_input(f"{col_name}", value=None, placeholder=f"平均值: {feature_means[col_name]:.2f}")
            # 如果使用者沒填，就用該欄位的平均值
            user_input[col_name] = val if val is not None else feature_means[col_name]
            
    submit_button = st.form_submit_button("產生預測報告")

if submit_button:
    input_df = pd.DataFrame([user_input])
    
    # 顯示預測指標
    st.subheader("預期")
    res_cols = st.columns(5)
    
    for i, target in enumerate(['woba', 'xwoba', 'EV', 'Angle']):
        prediction = models[target].predict(input_df[X_cols])[0]
        
        # 格式化輸出
        if target in ['woba', 'xwoba']:
            display_val = f"{prediction:.3f}"
        else:
            display_val = f"{prediction:.1f}"
            
        res_cols[i].metric(label=target, value=display_val)

    st.success("預測完成！請注意：由於訓練樣本較少(16筆)，此結果僅供戰術參考。")