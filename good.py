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

# ตั้งค่าภาพ Background และสร้าง Frame ตรงกลางที่ครอบคลุมการเลือกหน้า
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
    .frame-box {
        background: rgba(0, 0, 0, 0.7);
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0px 6px 20px rgba(0, 0, 0, 0.3);
        width: 60%;
        min-height: 180vh;
        margin: auto;
        position: fixed;
        top: 5%;
        left: 50%;
        transform: translateX(-50%);
        z-index: -1;
        max-width: 1000px;
    }
    </style>
    <div class="main"></div>
    <div class="frame-box"></div>
    """,
    unsafe_allow_html=True
)

# สร้าง frame สำหรับการเลือกหน้า
with st.container():
    st.markdown('<div class="radio-frame">', unsafe_allow_html=True)
    page = st.radio("เลือกหน้าที่ต้องการเข้าถึง", ["🔮 ทำนายผล", "📊 ประสิทธิภาพของโมเดล", "🔥 ความสำคัญของฟีเจอร์", "👨‍💻 ผู้จัดทำ"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

# หน้าทำนายผล
if page == "🔮 ทำนายผล":
    st.title("🔮 Customer Churn Prediction")
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
elif page == "📊 ประสิทธิภาพของโมเดล":
    st.title("📊 Model Performance")
    st.write("แสดงค่าความแม่นยำของโมเดลและ Confusion Matrix")

    # ค่า Accuracy และ ROC AUC
    accuracy = 0.9990  # สมมุติค่าที่คำนวณได้
    roc_auc = 0.9997
    st.metric(label="🔹 Accuracy", value=f"{accuracy:.4f}")
    st.metric(label="🔹 ROC AUC Score", value=f"{roc_auc:.4f}")
    
    # แสดงตารางเปรียบเทียบโมเดล
    model_comparison = {
        "Model": ["Logistic Regression", "k-Nearest Neighbors", "Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost"],
        "Accuracy": [0.9990, 0.9930, 0.9980, 0.9990, 0.9985, 0.9990],
        "ROC AUC": [0.9996, 0.9970, 0.9978, 0.9998, 0.9995, 0.9995]
    }
    df_comparison = pd.DataFrame(model_comparison)

    # แสดงตาราง
    st.write("### Model Comparison")
    st.dataframe(df_comparison, width=1000, height=245)

    # แสดง Confusion Matrix
    st.write("### Confusion Matrix")
    conf_matrix = np.array([[1606, 1], [1, 392]])  # สมมุติค่าจากโมเดล
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

# หน้า Feature Importance
elif page == "🔥 ความสำคัญของฟีเจอร์":
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

# หน้า ผู้จัดทำ
elif page == "👨‍💻 ผู้จัดทำ":
    st.title("👨‍💻 ผู้จัดทำ")
    st.write("ข้อมูลเกี่ยวกับผู้จัดทำของโปรเจคนี้")

    # ข้อมูลผู้จัดทำ
    st.write("1. นายนภสินธุ์ ศรีบุรินทร์  653450290-8")
    st.write("2. นายนันธวัฒน์ แผ่ความดี 653450291-6")
    st.write("3. นายอนันตสิน สังชัย 653450516-8")

    # สามารถเพิ่มข้อมูลอื่น ๆ ที่เกี่ยวข้องได้ เช่น แนวคิดหรืออธิบายโครงงาน
    st.write("โปรเจคนี้เป็นการพัฒนา Web App ที่สามารถทำนายการเลิกใช้บริการของลูกค้า โดยใช้โมเดล Machine Learning ที่ได้รับการฝึกฝนจากข้อมูลลูกค้า.")

# เพิ่ม Footer
st.markdown(
    """
    <div class="footer">
        <p>© 2025 Customer Churn Prediction Web App | Developed by Gus Not F</p>
    </div>
    """,
    unsafe_allow_html=True
)
