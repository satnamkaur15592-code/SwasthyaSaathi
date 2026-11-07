import streamlit as st
import pickle

st.set_page_config(page_title="SwasthyaSaathi", page_icon="🩺")
st.title("🩺 SwasthyaSaathi - Disease Prediction System")
st.write("Enter your details and check your health risk.")

@st.cache_resource
def load_models():
    d = {}
    d['diabetes'] = pickle.load(open("diabetes_model.pkl","rb"))
    d['heart'] = pickle.load(open("heart_model.pkl","rb"))
    d['liver'] = pickle.load(open("liver_model.pkl","rb"))
    d['kidney'] = pickle.load(open("kidney_model.pkl","rb"))
    d['hypertension'] = pickle.load(open("hypertension_model.pkl","rb"))
    d['anemia'] = pickle.load(open("anemia_model.pkl","rb"))
    d['thyroid'] = pickle.load(open("thyroid_model.pkl","rb"))
    d['asthma'] = pickle.load(open("asthma_model.pkl","rb"))
    return d

try:
    M = load_models()
    ok = True
except Exception as e:
    st.error(f"Model not found: {e}")
    ok = False

if ok:
    disease = st.selectbox("Select Disease", ["Diabetes","Heart Disease","Liver Disease","Kidney Disease","Hypertension","Anemia","Thyroid","Asthma"])

    if disease == "Diabetes":
        st.header("Diabetes Prediction")
        c1,c2 = st.columns(2)
        with c1:
            preg = st.number_input("Pregnancies",0,20,1)
            glucose = st.number_input("Glucose",0,300,120)
            bp = st.number_input("Blood Pressure",0,200,70)
            skin = st.number_input("Skin Thickness",0,100,20)
        with c2:
            insulin = st.number_input("Insulin",0,900,80)
            bmi = st.number_input("BMI",0.0,70.0,25.0)
            dpf = st.number_input("Diabetes Pedigree Function",0.0,3.0,0.5)
            age = st.number_input("Age",1,120,30)
        if st.button("Predict Diabetes",type="primary"):
            p = M['diabetes'].predict([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])[0]
            if p == 1:
                st.error("You have a high risk of Diabetes. Please consult a doctor.")
            else:
                st.success("You do not have Diabetes.")

    elif disease == "Heart Disease":
        st.header("Heart Disease Prediction")
        c1,c2,c3 = st.columns(3)
        with c1:
            age = st.number_input("Age",1,120,50)
            sex = st.selectbox("Sex (1=Male, 0=Female)",[1,0])
            cp = st.selectbox("Chest Pain Type (0-3)",[0,1,2,3])
            trestbps = st.number_input("Resting Blood Pressure",50,250,120)
            chol = st.number_input("Cholesterol",100,600,200)
        with c2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 (1=Yes, 0=No)",[0,1])
            restecg = st.selectbox("Resting ECG (0-2)",[0,1,2])
            thalach = st.number_input("Maximum Heart Rate",50,250,150)
            exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)",[0,1])
        with c3:
            oldpeak = st.number_input("Oldpeak",0.0,10.0,1.0)
            slope = st.selectbox("Slope (0-2)",[0,1,2])
            ca = st.selectbox("CA (0-4)",[0,1,2,3,4])
            thal = st.selectbox("Thal (0-3)",[0,1,2,3])
        if st.button("Predict Heart Disease",type="primary"):
            p = M['heart'].predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])[0]
            if p == 1:
                st.error("You have a high risk of Heart Disease. Please consult a doctor.")
            else:
                st.success("You do not have Heart Disease.")

    elif disease == "Liver Disease":
        st.header("Liver Disease Prediction")
        c1,c2 = st.columns(2)
        with c1:
            age = st.number_input("Age",1,120,40)
            gender = st.selectbox("Gender (1=Male, 0=Female)",[1,0])
            tb = st.number_input("Total Bilirubin",0.0,50.0,1.0)
            db = st.number_input("Direct Bilirubin",0.0,20.0,0.3)
            alk = st.number_input("Alkaline Phosphotase",50,2000,200)
        with c2:
            sgpt = st.number_input("SGPT",10,2000,30)
            sgot = st.number_input("SGOT",10,2000,30)
            tp = st.number_input("Total Proteins",2.0,10.0,6.5)
            alb = st.number_input("Albumin",1.0,6.0,3.5)
            agr = st.number_input("Albumin / Globulin Ratio",0.1,3.0,1.0)
        if st.button("Predict Liver Disease",type="primary"):
            p = M['liver'].predict([[age,gender,tb,db,alk,sgpt,sgot,tp,alb,agr]])[0]
            if p == 1:
                st.error("You have a high risk of Liver Disease. Please consult a doctor.")
            else:
                st.success("You do not have Liver Disease.")

    elif disease == "Kidney Disease":
        st.header("Kidney Disease Prediction")
        col1,col2,col3 = st.columns(3)
        with col1:
            age = st.number_input("Age",1,100,30)
            bp = st.number_input("Blood Pressure",40,180,70)
            sg = st.selectbox("Specific Gravity",[1.005,1.010,1.015,1.020,1.025],index=3)
            al = st.number_input("Albumin (0-5)",0,5,0)
            su = st.number_input("Sugar (0-5)",0,5,0)
            rbc = st.selectbox("RBC",["normal","abnormal"])
            pc = st.selectbox("Pus Cell",["normal","abnormal"])
            pcc = st.selectbox("Pus Cell Clumps",["notpresent","present"])
        with col2:
            ba = st.selectbox("Bacteria",["notpresent","present"])
            bgr = st.number_input("Blood Glucose Random",50,500,90)
            bu = st.number_input("Blood Urea",1,300,30)
            sc = st.number_input("Serum Creatinine",0.1,20.0,0.9)
            sod = st.number_input("Sodium",100,170,140)
            pot = st.number_input("Potassium",2.0,8.0,4.5)
            hemo = st.number_input("Haemoglobin",3.0,18.0,14.0)
            pcv = st.number_input("Packed Cell Volume",10,60,45)
        with col3:
            wbcc = st.number_input("WBC Count",2000,20000,7000)
            rbcc = st.number_input("RBC Count",2.0,7.0,5.0)
            htn = st.selectbox("Hypertension",["no","yes"])
            dm = st.selectbox("Diabetes Mellitus",["no","yes"])
            cad = st.selectbox("Coronary Artery Disease",["no","yes"])
            appet = st.selectbox("Appetite",["good","poor"])
            pe = st.selectbox("Pedal Edema",["no","yes"])
            ane = st.selectbox("Anemia",["no","yes"])
        rbc_n = 0
        if rbc == "abnormal":
            rbc_n = 1
        pc_n = 0
        if pc == "abnormal":
            pc_n = 1
        pcc_n = 0
        if pcc == "present":
            pcc_n = 1
        ba_n = 0
        if ba == "present":
            ba_n = 1
        htn_n = 0
        if htn == "yes":
            htn_n = 1
        dm_n = 0
        if dm == "yes":
            dm_n = 1
        cad_n = 0
        if cad == "yes":
            cad_n = 1
        appet_n = 0
        if appet == "poor":
            appet_n = 1
        pe_n = 0
        if pe == "yes":
            pe_n = 1
        ane_n = 0
        if ane == "yes":
            ane_n = 1
        feat = [[age,bp,sg,al,su,rbc_n,pc_n,pcc_n,ba_n,bgr,bu,sc,sod,pot,hemo,pcv,wbcc,rbcc,htn_n,dm_n,cad_n,appet_n,pe_n,ane_n]]
        if st.button("Predict Kidney Disease",type="primary"):
            p = M['kidney'].predict(feat)[0]
            if p == 1:
                st.error("You have a high risk of Kidney Disease. Please consult a doctor.")
            else:
                st.success("You do not have Kidney Disease.")

    elif disease == "Hypertension":
        st.header("Hypertension (High BP) Prediction")
        age_h = st.number_input("Age",15,100,40)
        bmi_h = st.number_input("BMI",15.0,40.0,25.0)
        smoking = st.selectbox("Smoking (1=Yes, 0=No)",[0,1])
        alcohol = st.selectbox("Alcohol (1=Yes, 0=No)",[0,1])
        exercise = st.slider("Exercise Hours per Week (0-10)",0,10,3)
        family = st.selectbox("Family History of High BP (1=Yes, 0=No)",[0,1])
        salt = st.slider("Salt Intake Level (0-10)",0,10,5)
        stress = st.slider("Stress Level (0-10)",0,10,5)
        if st.button("Predict Hypertension",type="primary"):
            p = M['hypertension'].predict([[age_h,bmi_h,smoking,alcohol,exercise,family,salt,stress]])[0]
            if p == 1:
                st.error("You have a high risk of Hypertension. Please check your BP and consult a doctor.")
            else:
                st.success("Your BP Risk is Low. Keep healthy lifestyle.")

    elif disease == "Anemia":
        st.header("Anemia (Khoon Ki Kami) Prediction")
        c1,c2 = st.columns(2)
        with c1:
            age_a = st.number_input("Age",10,80,25)
            gender_a = st.selectbox("Gender (1=Male, 0=Female)",[1,0])
            hb = st.number_input("Hemoglobin (g/dL)",7.0,15.0,13.0)
        with c2:
            rbc_a = st.number_input("RBC Count (millions)",3.0,6.0,4.5)
            mcv = st.number_input("MCV",60.0,100.0,85.0)
            iron = st.number_input("Iron Level",10.0,160.0,80.0)
        if st.button("Predict Anemia",type="primary"):
            p = M['anemia'].predict([[age_a,gender_a,hb,rbc_a,mcv,iron]])[0]
            if p == 1:
                st.error("You have a high risk of Anemia. Please eat iron-rich food and consult a doctor.")
            else:
                st.success("You do not have Anemia. Your Hemoglobin is good.")

    elif disease == "Thyroid":
        st.header("Thyroid Prediction")
        c1,c2 = st.columns(2)
        with c1:
            age_t = st.number_input("Age",10,70,30)
            gender_t = st.selectbox("Gender (1=Male, 0=Female)",[1,0])
            tsh = st.number_input("TSH",0.1,10.0,2.0)
        with c2:
            t3 = st.number_input("T3",0.5,2.5,1.2)
            t4 = st.number_input("T4",2.0,12.0,8.0)
        if st.button("Predict Thyroid",type="primary"):
            p = M['thyroid'].predict([[age_t,gender_t,tsh,t3,t4]])[0]
            if p == 1:
                st.error("You have a high risk of Thyroid. Please consult a doctor.")
            else:
                st.success("You do not have Thyroid.")

    elif disease == "Asthma":
        st.header("Asthma Prediction")
        c1,c2 = st.columns(2)
        with c1:
            age_as = st.number_input("Age",5,70,25)
            gender_as = st.selectbox("Gender (1=Male, 0=Female)",[1,0])
            smoking_as = st.selectbox("Smoking (1=Yes, 0=No)",[0,1])
            allergies = st.selectbox("Allergies (1=Yes, 0=No)",[0,1])
        with c2:
            family_as = st.selectbox("Family History (1=Yes, 0=No)",[0,1])
            peak = st.number_input("Peak Flow",200,600,500)
            wheeze = st.selectbox("Wheezing (1=Yes, 0=No)",[0,1])
            cough = st.selectbox("Night Cough (1=Yes, 0=No)",[0,1])
        if st.button("Predict Asthma",type="primary"):
            p = M['asthma'].predict([[age_as,gender_as,smoking_as,allergies,family_as,peak,wheeze,cough]])[0]
            if p == 1:
                st.error("You have a high risk of Asthma. Please consult a doctor.")
            else:
                st.success("You do not have Asthma.")



