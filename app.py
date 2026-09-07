import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

MODEL_DIR = "models"
PLOT_DIR = "plots"
DATA_PATH = os.path.join("data", "ai4i2020.csv")

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    binary_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "binary_xgboost_model.pkl"
        )
    )

    multi_model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "multiclass_random_forest_model.pkl"
        )
    )

    scaler = joblib.load(
        os.path.join(
            MODEL_DIR,
            "scaler.pkl"
        )
    )

    feature_columns = joblib.load(
        os.path.join(
            MODEL_DIR,
            "feature_columns.pkl"
        )
    )

    class_mapping = joblib.load(
        os.path.join(
            MODEL_DIR,
            "class_mapping.pkl"
        )
    )

    return (
        binary_model,
        multi_model,
        scaler,
        feature_columns,
        class_mapping
    )


(
    binary_model,
    multi_model,
    scaler,
    feature_columns,
    class_mapping
) = load_models()

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)

    return None


df = load_dataset()

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 20px;
    }

    .health-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }

    .small-text {
        font-size: 14px;
        color: #777;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Predictive Maintenance")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔮 Machine Prediction",
        "📊 Machine Analytics",
        "📈 Model Evaluation",
        "🔍 Feature Importance",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI4I 2020 Predictive Maintenance"
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">'
        '⚙️ Predictive Maintenance AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-powered machine failure detection and '
        'predictive maintenance system'
        '</div>',
        unsafe_allow_html=True
    )

    if df is not None:

        total_records = len(df)

        if "Machine failure" in df.columns:

            failure_count = int(
                df["Machine failure"].sum()
            )

            normal_count = (
                total_records -
                failure_count
            )

        else:

            failure_count = 339
            normal_count = 9661

    else:

        total_records = 10000
        failure_count = 339
        normal_count = 9661

    failure_rate = (
        failure_count /
        total_records
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Records",
            f"{total_records:,}"
        )

    with col2:

        st.metric(
            "Normal Machines",
            f"{normal_count:,}"
        )

    with col3:

        st.metric(
            "Failure Cases",
            f"{failure_count:,}"
        )

    with col4:

        st.metric(
            "Failure Rate",
            f"{failure_rate:.2f}%"
        )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '📊 Machine Overview'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Machine Failure Distribution"
        )

        failure_chart = pd.DataFrame(
            {
                "Status": [
                    "Normal",
                    "Failure"
                ],
                "Machines": [
                    normal_count,
                    failure_count
                ]
            }
        )

        st.bar_chart(
            failure_chart.set_index(
                "Status"
            )
        )

    with col2:

        st.subheader(
            "Machine Type Distribution"
        )

        if (
            df is not None
            and "Type" in df.columns
        ):

            type_count = (
                df["Type"]
                .value_counts()
                .rename_axis("Machine Type")
                .to_frame("Machines")
            )

            st.bar_chart(type_count)

        else:

            st.info(
                "Machine type information unavailable."
            )

    st.markdown(
        '<div class="section-title">'
        '🌡️ Sensor Summary'
        '</div>',
        unsafe_allow_html=True
    )

    if df is not None:

        sensor_columns = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]

        available_columns = [
            col
            for col in sensor_columns
            if col in df.columns
        ]

        if available_columns:

            summary = (
                df[available_columns]
                .describe()
                .T
                .round(2)
            )

            st.dataframe(
                summary,
                use_container_width=True
            )

    st.info(
        "💡 Go to **Machine Prediction** to test "
        "individual machine sensor readings."
    )

# ============================================================
# MACHINE PREDICTION
# ============================================================

elif page == "🔮 Machine Prediction":

    st.markdown(
        '<div class="main-title">'
        '🔮 Machine Failure Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-based machine health and failure risk assessment'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # INPUT SECTION
    # --------------------------------------------------------

    st.subheader(
        "⚙️ Machine Sensor Input"
    )

    col1, col2 = st.columns(2)

    with col1:

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"]
        )

        air_temperature = st.number_input(
            "Air Temperature [K]",
            min_value=295.0,
            max_value=305.0,
            value=300.0,
            step=0.1
        )

        process_temperature = st.number_input(
            "Process Temperature [K]",
            min_value=305.0,
            max_value=315.0,
            value=310.0,
            step=0.1
        )

    with col2:

        rotational_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=1000,
            max_value=3000,
            value=1500,
            step=10
        )

        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            max_value=80.0,
            value=40.0,
            step=0.1
        )

        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0,
            max_value=300,
            value=100,
            step=1
        )

    st.markdown("")

    predict_button = st.button(
        "🚀 Analyze Machine Health",
        use_container_width=True,
        type="primary"
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict_button:

        temp_diff = (
            process_temperature -
            air_temperature
        )

        power_proxy = (
            torque *
            rotational_speed
        )

        type_L = (
            1
            if machine_type == "L"
            else 0
        )

        type_M = (
            1
            if machine_type == "M"
            else 0
        )

        input_data = pd.DataFrame(
            {
                "air_temperature": [
                    air_temperature
                ],
                "process_temperature": [
                    process_temperature
                ],
                "rotational_speed": [
                    rotational_speed
                ],
                "torque": [
                    torque
                ],
                "tool_wear": [
                    tool_wear
                ],
                "temp_diff": [
                    temp_diff
                ],
                "power_proxy": [
                    power_proxy
                ],
                "type_L": [
                    type_L
                ],
                "type_M": [
                    type_M
                ]
            }
        )

        try:

            # ------------------------------------------------
            # FEATURE ORDER
            # ------------------------------------------------

            input_data = input_data[
                feature_columns
            ]

            # ------------------------------------------------
            # SCALING
            # ------------------------------------------------

            scaled_input = scaler.transform(
                input_data
            )

            # ------------------------------------------------
            # BINARY PREDICTION
            # ------------------------------------------------

            binary_prediction = (
                binary_model
                .predict(
                    scaled_input
                )[0]
            )

            binary_probability = (
                binary_model
                .predict_proba(
                    scaled_input
                )[0][1]
            )

            # ------------------------------------------------
            # FAILURE TYPE
            # ------------------------------------------------

            multi_prediction = (
                multi_model
                .predict(
                    scaled_input
                )[0]
            )

            if multi_prediction in class_mapping:

                failure_type = (
                    class_mapping[
                        multi_prediction
                    ]
                )

            elif str(
                multi_prediction
            ) in class_mapping:

                failure_type = (
                    class_mapping[
                        str(multi_prediction)
                    ]
                )

            else:

                failure_type = str(
                    multi_prediction
                )

            # ------------------------------------------------
            # RISK LEVEL
            # ------------------------------------------------

            if binary_probability >= 0.70:

                risk_level = "HIGH"

            elif binary_probability >= 0.30:

                risk_level = "MEDIUM"

            else:

                risk_level = "LOW"

            # ------------------------------------------------
            # MACHINE HEALTH
            # ------------------------------------------------

            if risk_level == "HIGH":

                health_status = "CRITICAL"

            elif risk_level == "MEDIUM":

                health_status = "WARNING"

            else:

                health_status = "HEALTHY"

            # ------------------------------------------------
            # FAILURE-SPECIFIC RECOMMENDATION
            # ------------------------------------------------

            failure_text = str(
                failure_type
            ).upper()

            if binary_prediction == 0:

                recommendation = (
                    "Continue normal operation and "
                    "periodically monitor machine sensors."
                )

            elif "TWF" in failure_text:

                recommendation = (
                    "Inspect tool condition and consider "
                    "tool replacement or servicing."
                )

            elif "HDF" in failure_text:

                recommendation = (
                    "Inspect temperature and cooling "
                    "conditions. Check for overheating."
                )

            elif "PWF" in failure_text:

                recommendation = (
                    "Inspect power, torque and rotational "
                    "conditions before continued operation."
                )

            elif "OSF" in failure_text:

                recommendation = (
                    "Inspect machine components for "
                    "mechanical stress or excessive load."
                )

            elif "RNF" in failure_text:

                recommendation = (
                    "Perform a detailed machine inspection "
                    "and verify sensor readings."
                )

            else:

                recommendation = (
                    "Schedule a maintenance inspection "
                    "and continue monitoring the machine."
                )

            # ------------------------------------------------
            # RESULT SECTION
            # ------------------------------------------------

            st.markdown("---")

            st.subheader(
                "📋 Machine Health Report"
            )

            # ------------------------------------------------
            # TOP STATUS
            # ------------------------------------------------

            if health_status == "HEALTHY":

                st.success(
                    "🟢 MACHINE HEALTH: HEALTHY"
                )

            elif health_status == "WARNING":

                st.warning(
                    "🟠 MACHINE HEALTH: WARNING"
                )

            else:

                st.error(
                    "🔴 MACHINE HEALTH: CRITICAL"
                )

            # ------------------------------------------------
            # KPI CARDS
            # ------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Machine Type",
                    machine_type
                )

            with col2:

                st.metric(
                    "Failure Probability",
                    f"{binary_probability * 100:.2f}%"
                )

            with col3:

                st.metric(
                    "Risk Level",
                    risk_level
                )

            with col4:

                st.metric(
                    "Health Status",
                    health_status
                )

            # ------------------------------------------------
            # RISK BAR
            # ------------------------------------------------

            st.subheader(
                "📊 Failure Risk"
            )

            st.progress(
                float(binary_probability)
            )

            if risk_level == "LOW":

                st.caption(
                    "Low probability of machine failure."
                )

            elif risk_level == "MEDIUM":

                st.caption(
                    "Moderate risk detected. "
                    "Continue monitoring machine conditions."
                )

            else:

                st.caption(
                    "High failure probability detected. "
                    "Inspection is recommended."
                )

            # ------------------------------------------------
            # FAILURE PREDICTION
            # ------------------------------------------------

            st.subheader(
                "⚠️ Failure Prediction"
            )

            col1, col2 = st.columns(2)

            with col1:

                if binary_prediction == 1:

                    st.error(
                        "Machine Failure Predicted"
                    )

                else:

                    st.success(
                        "No Machine Failure Predicted"
                    )

            with col2:

                if (
                    str(failure_type).lower()
                    == "no failure"
                ):

                    st.success(
                        "Failure Type: No Failure"
                    )

                else:

                    st.warning(
                        f"Failure Type: {failure_type}"
                    )

            # ------------------------------------------------
            # SENSOR ANALYSIS
            # ------------------------------------------------

            st.subheader(
                "🌡️ Sensor Analysis"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Temperature Difference",
                    f"{temp_diff:.2f} K"
                )

            with col2:

                st.metric(
                    "Power Proxy",
                    f"{power_proxy:.0f}"
                )

            with col3:

                st.metric(
                    "Rotational Speed",
                    f"{rotational_speed} rpm"
                )

            with col4:

                st.metric(
                    "Tool Wear",
                    f"{tool_wear} min"
                )

            # ------------------------------------------------
            # INPUT SUMMARY
            # ------------------------------------------------

            st.subheader(
                "📌 Input Summary"
            )

            summary_data = pd.DataFrame(
                {
                    "Parameter": [
                        "Machine Type",
                        "Air Temperature",
                        "Process Temperature",
                        "Rotational Speed",
                        "Torque",
                        "Tool Wear",
                        "Temperature Difference",
                        "Power Proxy"
                    ],
                    "Value": [
                        machine_type,
                        f"{air_temperature:.1f} K",
                        f"{process_temperature:.1f} K",
                        f"{rotational_speed} rpm",
                        f"{torque:.1f} Nm",
                        f"{tool_wear} min",
                        f"{temp_diff:.2f} K",
                        f"{power_proxy:.0f}"
                    ]
                }
            )

            st.dataframe(
                summary_data,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # MAINTENANCE RECOMMENDATION
            # ------------------------------------------------

            st.subheader(
                "🔧 Maintenance Recommendation"
            )

            if binary_prediction == 1:

                st.warning(
                    recommendation
                )

                st.write(
                    "The prediction indicates an elevated "
                    "possibility of machine failure. "
                    "The result should be used to support "
                    "maintenance inspection and monitoring."
                )

            else:

                st.success(
                    recommendation
                )

            # ------------------------------------------------
            # PREDICTION TIME
            # ------------------------------------------------

            prediction_time = datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            )

            st.caption(
                f"Prediction generated at: "
                f"{prediction_time}"
            )

        except Exception as e:

            st.error(
                "Prediction Error"
            )

            st.code(
                str(e)
            )

    else:

        st.info(
            "Enter the machine sensor readings "
            "and click **Analyze Machine Health**."
        )

# ============================================================
# MACHINE ANALYTICS
# ============================================================

elif page == "📊 Machine Analytics":

    st.markdown(
        '<div class="main-title">'
        '📊 Machine Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Exploratory analysis of machine sensor data '
        'and failure patterns'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "1️⃣ Machine Failure Distribution"
    )

    path = os.path.join(
        PLOT_DIR,
        "01_machine_failure_distribution.png"
    )

    if os.path.exists(path):

        st.image(
            path,
            use_container_width=True
        )

    else:

        st.warning(
            "Failure distribution plot not found."
        )

    st.subheader(
        "2️⃣ Torque and Tool Wear Analysis"
    )

    col1, col2 = st.columns(2)

    with col1:

        path = os.path.join(
            PLOT_DIR,
            "02_torque_vs_failure.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                use_container_width=True
            )

        else:

            st.warning(
                "Torque plot not found."
            )

    with col2:

        path = os.path.join(
            PLOT_DIR,
            "03_tool_wear_vs_failure.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                use_container_width=True
            )

        else:

            st.warning(
                "Tool wear plot not found."
            )

    st.subheader(
        "3️⃣ Temperature Difference Analysis"
    )

    path = os.path.join(
        PLOT_DIR,
        "04_temperature_difference.png"
    )

    if os.path.exists(path):

        st.image(
            path,
            use_container_width=True
        )

    else:

        st.warning(
            "Temperature plot not found."
        )

    st.subheader(
        "4️⃣ Failure Type Distribution"
    )

    path = os.path.join(
        PLOT_DIR,
        "05_failure_type_rates.png"
    )

    if os.path.exists(path):

        st.image(
            path,
            use_container_width=True
        )

    else:

        st.warning(
            "Failure type plot not found."
        )

    st.subheader(
        "5️⃣ Sensor Correlation"
    )

    path = os.path.join(
        PLOT_DIR,
        "06_correlation_heatmap.png"
    )

    if os.path.exists(path):

        st.image(
            path,
            use_container_width=True
        )

    else:

        st.warning(
            "Correlation heatmap not found."
        )

    st.subheader(
        "6️⃣ Speed vs Torque"
    )

    path = os.path.join(
        PLOT_DIR,
        "07_speed_torque.png"
    )

    if os.path.exists(path):

        st.image(
            path,
            use_container_width=True
        )

    else:

        st.warning(
            "Speed vs Torque plot not found."
        )

# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "📈 Model Evaluation":

    st.markdown(
        '<div class="main-title">'
        '📈 Model Evaluation'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Performance comparison of machine learning models'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "Binary Classification Performance"
    )

    comparison_data = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "XGBoost"
            ],
            "Accuracy": [
                0.8590,
                0.9890,
                0.9865
            ],
            "Precision": [
                0.1777,
                0.9423,
                0.7808
            ],
            "Recall": [
                0.8676,
                0.7206,
                0.8382
            ],
            "F1 Score": [
                0.2950,
                0.8167,
                0.8085
            ],
            "ROC-AUC": [
                0.9339,
                0.9781,
                0.9755
            ]
        }
    )

    st.dataframe(
        comparison_data.style.format(
            {
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}",
                "ROC-AUC": "{:.2%}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Model Performance Comparison"
    )

    st.bar_chart(
        comparison_data.set_index(
            "Model"
        )
    )

    st.info(
        "The deployed binary prediction model is "
        "**XGBoost**. Random Forest achieved the highest "
        "F1 Score in the original comparison, while "
        "XGBoost provides strong recall and ROC-AUC "
        "performance."
    )

    st.subheader(
        "Confusion Matrices"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "Binary Classification"
        )

        path = os.path.join(
            PLOT_DIR,
            "08_binary_confusion_matrix.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                use_container_width=True
            )

        else:

            st.warning(
                "Binary confusion matrix not found."
            )

    with col2:

        st.write(
            "Multi-Class Classification"
        )

        path = os.path.join(
            PLOT_DIR,
            "09_multiclass_confusion_matrix.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                use_container_width=True
            )

        else:

            st.warning(
                "Multi-class confusion matrix not found."
            )

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "🔍 Feature Importance":

    st.markdown(
        '<div class="main-title">'
        '🔍 Feature Importance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Features influencing machine failure predictions'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Binary Model"
        )

        path = os.path.join(
            PLOT_DIR,
            "10_binary_feature_importance.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                use_container_width=True
            )

        else:

            st.warning(
                "Binary feature importance plot not found."
            )

    with col2:

        st.subheader(
            "Multi-Class Model"
        )

        path = os.path.join(
            PLOT_DIR,
            "11_multiclass_feature_importance.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                use_container_width=True
            )

        else:

            st.warning(
                "Multi-class feature importance plot not found."
            )

    st.markdown("---")

    st.subheader(
        "Important Predictive Factors"
    )

    st.write(
        """
        The predictive maintenance system uses:

        • Power Proxy = Torque × Rotational Speed

        • Rotational Speed

        • Tool Wear

        • Temperature Difference

        • Torque

        • Air Temperature

        • Process Temperature

        • Machine Type
        """
    )

# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="main-title">'
        'ℹ️ About Predictive Maintenance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-based industrial machine failure prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "🎯 Project Objective"
    )

    st.write(
        """
        The objective of this project is to predict whether
        an industrial machine is likely to fail using sensor
        data and machine learning techniques.
        """
    )

    st.subheader(
        "🔄 Machine Learning Pipeline"
    )

    st.write(
        """
        1. Data Collection

        2. Data Preprocessing

        3. Feature Engineering

        4. Train-Test Split

        5. Feature Scaling

        6. Binary Failure Prediction

        7. Failure Type Classification

        8. Model Evaluation

        9. Deployment using Streamlit
        """
    )

    st.subheader(
        "🤖 Models Used"
    )

    model_table = pd.DataFrame(
        {
            "Task": [
                "Machine Failure Detection",
                "Failure Type Prediction"
            ],
            "Model": [
                "XGBoost",
                "Random Forest"
            ]
        }
    )

    st.dataframe(
        model_table,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "📂 Dataset"
    )

    st.write(
        "AI4I 2020 Predictive Maintenance Dataset"
    )

    st.subheader(
        "🛠️ Technologies"
    )

    st.write(
        """
        Python • Pandas • NumPy • Scikit-learn •
        XGBoost • Random Forest • Joblib • Streamlit
        """
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Predictive Maintenance AI | AI4I 2020 Dataset | "
    "Machine Learning Project"
)