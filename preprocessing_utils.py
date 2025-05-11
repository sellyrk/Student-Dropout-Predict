
import joblib
import numpy as np
import pandas as pd

encoder_Daytime = joblib.load("model/encoder_Daytime_evening_attendance.joblib")
encoder_Edu = joblib.load("model/encoder_Educational_special_needs.joblib")
encoder_Scholar = joblib.load("model/encoder_Scholarship_holder.joblib")
encoder_Tuition = joblib.load("model/encoder_Tuition_fees_up_to_date.joblib")
pca_1 = joblib.load("model/pca_1.joblib")
pca_2 = joblib.load("model/pca_2.joblib")
scaler_aplication = joblib.load("model/scaler_Application_order.joblib")
scaler_prev_grade = joblib.load("model/scaler_Previous_qualification_grade.joblib")
scaler_admission = joblib.load("model/scaler_Admission_grade.joblib")
scaler_enrollment = joblib.load("model/scaler_Age_at_enrollment.joblib")
scaler_enrolled = joblib.load("model/scaler_Total_enrolled.joblib")
scaler_approved = joblib.load("model/scaler_Total_approved.joblib")
scaler_avg_grade = joblib.load("model/scaler_Avg_grade.joblib")
scaler_evaluations = joblib.load("model/scaler_Total_evaluations.joblib")
scaler_without_eval = joblib.load("model/scaler_Total_without_evaluations.joblib")

pca_numerical_columns_1 = [
    'Total_enrolled',
    'Total_approved',
    'Avg_grade',
    'Total_evaluations',
]

pca_numerical_columns_2 = [
    'Previous_qualification_grade',
    'Admission_grade',
    'Age_at_enrollment',
    'Application_order',
    'Total_without_evaluations'
]

def data_preprocessing(data):
    """Preprocessing data

    Args:
        data (Pandas DataFrame): Dataframe that contains all the data to make prediction

    Returns:
        Pandas DataFrame: Dataframe that contains all the preprocessed data
    """

    # Ubah nilai kategorikal ke numerik (binary manual encoding)
    data = data.copy()
    data["Daytime_evening_attendance"] = 1 if data["Daytime_evening_attendance"].iloc[0] == "Yes" else 0
    data["Educational_special_needs"] = 1 if data["Educational_special_needs"].iloc[0] == "Yes" else 0
    data["Scholarship_holder"] = 1 if data["Scholarship_holder"].iloc[0] == "Yes" else 0
    data["Tuition_fees_up_to_date"] = 1 if data["Tuition_fees_up_to_date"].iloc[0] == "Yes" else 0

    df = pd.DataFrame()

    # Scaling
    df["Application_order"] = scaler_aplication.transform([[data["Application_order"].iloc[0]]])[0]
    df["Admission_grade"] = scaler_admission.transform([[data["Admission_grade"].iloc[0]]])[0]
    df["Previous_qualification_grade"] = scaler_prev_grade.transform([[data["Previous_qualification_grade"].iloc[0]]])[0]
    df["Avg_grade"] = scaler_avg_grade.transform([[data["Avg_grade"].iloc[0]]])[0]
    df["Age_at_enrollment"] = scaler_enrollment.transform([[data["Age_at_enrollment"].iloc[0]]])[0]
    df["Total_enrolled"] = scaler_enrolled.transform([[data["Total_enrolled"].iloc[0]]])[0]
    df["Total_approved"] = scaler_approved.transform([[data["Total_approved"].iloc[0]]])[0]
    df["Total_evaluations"] = scaler_evaluations.transform([[data["Total_evaluations"].iloc[0]]])[0]
    df["Total_without_evaluations"] = scaler_without_eval.transform([[data["Total_without_evaluations"].iloc[0]]])[0]

    # Encoding
    df["Daytime_evening_attendance"] = data["Daytime_evening_attendance"].iloc[0]
    df["Educational_special_needs"] = data["Educational_special_needs"].iloc[0]
    df["Scholarship_holder"] = data["Scholarship_holder"].iloc[0]
    df["Tuition_fees_up_to_date"] = data["Tuition_fees_up_to_date"].iloc[0]

    # PCA 1
    data["Total_enrolled"] = scaler_enrollment.transform(np.asarray(data["Total_enrolled"]).reshape(-1,1))[0]
    data["Total_approved"] = scaler_approved.transform(np.asarray(data["Total_approved"]).reshape(-1,1))[0]
    data["Avg_grade"] = scaler_avg_grade.transform(np.asarray(data["Avg_grade"]).reshape(-1,1))[0]
    data["Total_evaluations"] = scaler_evaluations.transform(np.asarray(data["Total_evaluations"]).reshape(-1,1))[0]

    df[["pc1_1", "pc1_2"]] = pca_1.transform(data[pca_numerical_columns_1])

    # PCA 2
    data["Previous_qualification_grade"] = scaler_prev_grade.transform(np.asarray(data["Previous_qualification_grade"]).reshape(-1,1))[0]
    data["Admission_grade"] = scaler_admission.transform(np.asarray(data["Admission_grade"]).reshape(-1,1))[0]
    data["Age_at_enrollment"] = scaler_enrollment.transform(np.asarray(data["Age_at_enrollment"]).reshape(-1,1))[0]
    data["Application_order"] = scaler_aplication.transform(np.asarray(data["Application_order"]).reshape(-1,1))[0]
    data["Total_without_evaluations"] = scaler_without_eval.transform(np.asarray(data["Total_without_evaluations"]).reshape(-1,1))[0]

    df[["pc2_1", "pc2_2", "pc2_3"]] = pca_2.transform(data[pca_numerical_columns_2])

    expected_columns = [
        "Daytime_evening_attendance",
        "Educational_special_needs",
        "Tuition_fees_up_to_date",
        "Scholarship_holder",
        "pc1_1", "pc1_2",
        "pc2_1", "pc2_2", "pc2_3"
    ]
    df = df[expected_columns]

    return df