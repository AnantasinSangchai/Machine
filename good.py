import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# โหลดโมเดลและ scaler
with open("customer_churn_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# ตั้งค่าภาพ Background และสร้าง Frame ตรงกลางที่ปรับขนาดให้พอดีกับข้อความ
st.markdown(
    """
    <style>
    .main {
        background-image: url("https://img2.pic.in.th/pic/-67cfe8dc3204a029e.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;
        width: 100%;
        position: fixed;
        top: 0;
        left: 0;
        z-index: -2;
    }
    .stApp {
        background-color: transparent !important;
    }
    </style>
    <div class="main"></div>
    <div class="frame-box"></div>
    """,
    unsafe_allow_html=True
)

# สร้าง Sidebar
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🔮 Predict", "📊 Model Performance", "🔥 Feature Importance"])

# หน้าทำนายผล
if page == "🔮 Predict":
    st.title("🔍 Customer Churn Prediction Web App")
    st.write("ป้อนข้อมูลของลูกค้าเพื่อดูว่ามีโอกาสเลิกใช้บริการหรือไม่")

    # สร้างอินพุตสำหรับผู้ใช้
    complain = st.selectbox("ลูกค้าเคยร้องเรียนหรือไม่?", [0, 1])
    age = st.number_input("อายุของลูกค้า", min_value=18, max_value=100, value=30)
    is_active = st.selectbox("ลูกค้าเป็น Active Member หรือไม่?", [0, 1])
    num_products = st.slider("จำนวนผลิตภัณฑ์ที่ใช้", 1, 4, 1)
    geography = st.selectbox("ประเทศของลูกค้า", ["France", "Germany", "Spain"])
    balance = st.number_input("ยอดเงินในบัญชี", min_value=0.0, max_value=500000.0, value=50000.0)

    # แปลงค่าหมวดหมู่เป็นตัวเลข
    geo_map = {"France": 0, "Germany": 1, "Spain": 2}
    geography = geo_map[geography]

    # สร้าง DataFrame สำหรับโมเดล
    input_df = pd.DataFrame([[complain, age, is_active, num_products, geography, balance]], 
                            columns=["Complain", "Age", "IsActiveMember", "NumOfProducts", "Geography", "Balance"])
    
    # ปรับขนาดข้อมูลด้วย StandardScaler
    input_data = scaler.transform(input_df)

    # ทำการทำนาย
    if st.button("🔮 ทำนายผลลัพธ์"):
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1][0]
        
        if prediction[0] == 1:
            st.error(f"🚨 ลูกค้ารายนี้มีแนวโน้มที่จะเลิกใช้บริการ ({probability:.2%} ความน่าจะเป็น)")
        else:
            st.success(f"✅ ลูกค้ารายนี้มีแนวโน้มที่จะอยู่ต่อ ({(1 - probability):.2%} ความน่าจะเป็น)")

# หน้าค่าความแม่นยำและ Confusion Matrix
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.write("แสดงค่าความแม่นยำของโมเดลและ Confusion Matrix")

    # ค่า Accuracy และ ROC AUC
    accuracy = 0.9990  # สมมุติค่าที่คำนวณได้
    roc_auc = 0.9997
    st.metric(label="🔹 Accuracy", value=f"{accuracy:.4f}")
    st.metric(label="🔹 ROC AUC Score", value=f"{roc_auc:.4f}")
    
    # แสดง Confusion Matrix
    st.write("### Confusion Matrix")
    conf_matrix = np.array([[1606, 1], [1, 392]])  # สมมุติค่าจากโมเดล
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

# หน้า Feature Importance
elif page == "🔥 Feature Importance":
    st.title("🔥 Feature Importance")
    st.write("แสดงความสำคัญของ Feature ในการทำนายการเลิกใช้บริการ")
    
    # สมมุติค่า Feature Importance
    feature_importance = {
        "Complain": 0.75,
        "Age": 0.10,
        "IsActiveMember": 0.05,
        "NumOfProducts": 0.04,
        "Geography": 0.03,
        "Balance": 0.03
    }
    
    fig, ax = plt.subplots()
    sns.barplot(x=list(feature_importance.values()), y=list(feature_importance.keys()), palette="coolwarm")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    st.pyplot(fig)

# เพิ่ม Footer
st.markdown(
    """
    <div class="footer">
        <p>© 2025 Customer Churn Prediction Web App | Developed by Your Name</p>
    </div>
    """,
    unsafe_allow_html=True
)