import streamlit as st
import pandas as pd
import joblib
from preprocessing_utils import data_preprocessing, encoder_Daytime, encoder_Edu, encoder_Scholar, encoder_Tuition
from predict_utils import prediction

col1, col2 = st.columns([1, 5])
with col2:
    st.header("🎯 Student Status Prediction App")

data = pd.DataFrame(columns=[
    "Daytime_evening_attendance", "Educational_special_needs", "Scholarship_holder", "Tuition_fees_up_to_date",
    "Application_order", "Admission_grade", "Previous_qualification_grade", "Age_at_enrollment",
    "Total_enrolled", "Total_approved", "Total_evaluations", "Total_without_evaluations", "Avg_grade"
])


col1, col2, col3, col4 = st.columns(4)

with col1:
    Daytime_evening_attendance = st.selectbox('Daytime evening attendance', options=["No", "Yes"])
    data["Daytime_evening_attendance"] = 1 if Daytime_evening_attendance == "Yes" else 0

with col2:
    Educational_special_needs = st.selectbox(label='Educational special needs', options=["No", "Yes"])
    data["Educational_special_needs"] = 1 if Educational_special_needs == "Yes" else 0

with col3:
    Scholarship_holder = st.selectbox(label='Scholarship holder', options=["No", "Yes"])
    data["Scholarship_holder"] = 1 if Scholarship_holder == "Yes" else 0

with col4:
    Tuition_fees_up_to_date = st.selectbox(label='Tuition fees up to date', options=["No", "Yes"])
    data["Tuition_fees_up_to_date"] = 1 if Tuition_fees_up_to_date == "Yes" else 0


col1, col2, col3 = st.columns(3)

with col1:

    Application_order = int(st.number_input(label='Application order', value=1))
    data["Application_order"] = Application_order

with col2:
    Admission_grade = int(st.number_input(label='Admission grade', value=100))
    data["Admission_grade"] = Admission_grade

with col3:
    Previous_qualification_grade = int(st.number_input(label='Previous qualification grade', value=100))
    data["Previous_qualification_grade"] = Previous_qualification_grade


col1, col2, col3 = st.columns(3)

with col1:
    Age_at_enrollment = st.slider("Age at Enrollment", 17, 40, 20, 1)
    data["Age_at_enrollment"] = Age_at_enrollment

with col2:
    Total_enrolled = st.slider("Total enrolled", 0, 36, 18, 1)
    data["Total_enrolled"] = Total_enrolled

with col3:
    Total_approved = st.slider("Total approved", 0, 43, 20, 1)
    data["Total_approved"] = Total_approved

col1, col2, col3 = st.columns(3)

with col1:
    Total_evaluations = st.slider("Total evaluations", 0, 72, 30, 1)
    data["Total_evaluations"] = Total_evaluations

with col2:
    Total_without_evaluations = st.slider("Total without evaluations", 0, 16, 8, 1)
    data["Total_without_evaluations"] = Total_without_evaluations

with col3:
    Avg_grade = float(st.number_input(label='Avg grade', value=12.5))
    data["Avg_grade"] = Avg_grade

data.loc[0] = [
    1 if Daytime_evening_attendance == "Yes" else 0,
    1 if Educational_special_needs == "Yes" else 0,
    1 if Scholarship_holder == "Yes" else 0,
    1 if Tuition_fees_up_to_date == "Yes" else 0,
    Application_order,
    Admission_grade,
    Previous_qualification_grade,
    Age_at_enrollment,
    Total_enrolled,
    Total_approved,
    Total_evaluations,
    Total_without_evaluations,
    Avg_grade
]

with st.expander("View the Raw Data"):
    st.dataframe(data=data, width=800, height=10)

if st.button('Predict'):
    new_data = data_preprocessing(data=data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=10)
    st.write("Predict Status: {}".format(prediction(new_data)))