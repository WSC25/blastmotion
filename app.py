import streamlit as st
import joblib
import pandas as pd

# 載入模型
data = joblib.load('baseball_model.pkl')
models = data['models']
feature_means = data['feature_means']
X_cols = data['X_cols']

st.title("賽前練習揮棒數據->場上表現預測 APP")
st.write("請輸入賽前練習數據（留空則使用現有資料平均值）")

# 動態生成輸入框
user_input = {}
cols = st.columns(2) # 分兩欄顯示

for i, col_name in enumerate(X_cols):
    with cols[i % 2]:
        # 使用平均值作為預設顯示，讓使用者參考
        val = st.text_input(f"{col_name}", placeholder=f"平均: {feature_means[col_name]:.2f}")
        user_input[col_name] = float(val) if val else feature_means[col_name]

if st.button("開始預測當日表現"):
    # 轉換為 DataFrame
    input_df = pd.DataFrame([user_input])
    
    st.subheader("預測結果")
    res_cols = st.columns(len(models))
    
    for i, (target, model) in enumerate(models.items()):
        prediction = model.predict(input_df[X_cols])[0]
        res_cols[i].metric(label=target, value=f"{prediction:.3f}")