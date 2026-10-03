import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Healthcare",
    page_icon="🏥",
    layout="wide"
)

# =========================================================
# START SCREEN CONTROL
# =========================================================
if "started" not in st.session_state:
    st.session_state.started = False
# =========================================================
# CUSTOM CSS
# =========================================================

# =========================================================
# BEAUTIFUL APP CSS
# =========================================================

st.markdown("""
<style>

/* =====================================================
   GENERAL APP
   ===================================================== */

.stApp {
    background: linear-gradient(
        135deg,
        #f8fbff 0%,
        #eef5ff 45%,
        #f9f5ff 100%
    );
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =====================================================
   SIDEBAR
   ===================================================== */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #101b35,
        #172b50,
        #1d3159
    );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] .stRadio label {
    font-size: 15px;
    padding: 5px 0;
}

.sidebar-title {
    text-align: center;
    font-size: 25px;
    font-weight: 800;
    margin-bottom: 5px;
}

.sidebar-subtitle {
    text-align: center;
    font-size: 13px;
    opacity: 0.8;
}


/* =====================================================
   LOGIN PAGE
   ===================================================== */

.login-wrapper {
    max-width: 620px;
    margin: 70px auto 0 auto;
    padding: 45px;
    border-radius: 30px;

    background: rgba(255,255,255,0.94);

    box-shadow:
        0 20px 60px rgba(30,60,100,0.15);

    border: 1px solid rgba(255,255,255,0.8);

    text-align: center;
}

.login-icon {
    font-size: 70px;
    margin-bottom: 5px;
}

.login-title {
    font-size: 42px;
    font-weight: 800;
    color: #16213e;
    margin-bottom: 5px;
}

.login-subtitle {
    font-size: 17px;
    color: #64748b;
    margin-bottom: 30px;
}

.login-note {
    margin-top: 20px;
    color: #64748b;
    font-size: 13px;
}


/* =====================================================
   LANDING PAGE
   ===================================================== */

.landing-wrapper {
    min-height: 600px;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;

    text-align: center;

    padding: 70px 30px;

    border-radius: 35px;

    background:
        radial-gradient(
            circle at top left,
            rgba(99,179,237,0.25),
            transparent 35%
        ),
        radial-gradient(
            circle at bottom right,
            rgba(168,85,247,0.20),
            transparent 35%
        ),
        white;

    box-shadow:
        0 20px 60px rgba(40,60,100,0.12);
}

.landing-heart {
    font-size: 95px;

    filter:
        drop-shadow(
            0px 10px 15px rgba(220,38,38,0.25)
        );
}

.landing-title {
    font-size: 52px;
    font-weight: 900;
    color: #172554;

    margin-top: 10px;
}

.landing-subtitle {
    font-size: 22px;
    color: #64748b;

    margin-top: 5px;
}

.landing-description {
    max-width: 650px;

    font-size: 16px;
    line-height: 1.7;

    color: #64748b;

    margin-top: 20px;
}


/* =====================================================
   MAIN HOME HERO
   ===================================================== */

.hero {
    padding: 45px;
    border-radius: 28px;

    background:
        linear-gradient(
            135deg,
            #dbeafe,
            #ede9fe,
            #fce7f3
        );

    box-shadow:
        0 15px 40px rgba(50,70,100,0.10);

    margin-bottom: 30px;
}

.hero h1 {
    font-size: 42px;
    color: #172554;
}

.hero p {
    font-size: 18px;
    color: #475569;
}


/* =====================================================
   SECTION TITLE
   ===================================================== */

.section-title {
    font-size: 30px;
    font-weight: 800;
    color: #172554;

    margin-top: 35px;
    margin-bottom: 20px;
}


/* =====================================================
   FEATURE CARDS
   ===================================================== */

.feature-card {
    min-height: 230px;

    padding: 28px;

    border-radius: 22px;

    background: white;

    border: 1px solid #e2e8f0;

    box-shadow:
        0 10px 30px rgba(30,50,80,0.08);

    transition: 0.3s;
}

.feature-card:hover {
    transform: translateY(-5px);

    box-shadow:
        0 18px 40px rgba(30,50,80,0.13);
}

.feature-icon {
    font-size: 45px;
}

.feature-card h3 {
    color: #172554;
    font-size: 21px;
}

.feature-card p {
    color: #64748b;
    line-height: 1.6;
}


/* =====================================================
   STAT CARDS
   ===================================================== */

.stat-card {
    background: white;

    padding: 25px;

    border-radius: 20px;

    text-align: center;

    border: 1px solid #e2e8f0;

    box-shadow:
        0 8px 25px rgba(30,50,80,0.07);
}

.stat-number {
    font-size: 32px;
    font-weight: 800;
    color: #2563eb;
}

.stat-label {
    color: #64748b;
    font-size: 15px;
}


/* =====================================================
   BUTTONS
   ===================================================== */

.stButton > button {
    border-radius: 14px;

    min-height: 48px;

    font-size: 16px;

    font-weight: 700;

    border: none;

    transition: 0.25s;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 8px 20px rgba(37,99,235,0.20);
}


/* =====================================================
   START BUTTON
   ===================================================== */

.start-button {
    font-size: 18px;
    font-weight: 800;
}


/* =====================================================
   PREDICTION RESULT
   ===================================================== */

.result-card {
    padding: 30px;

    border-radius: 22px;

    background: white;

    text-align: center;

    box-shadow:
        0 10px 35px rgba(30,50,80,0.10);
}


/* =====================================================
   INFO BOX
   ===================================================== */

.info-card {
    padding: 25px;

    border-radius: 20px;

    background: white;

    border-left: 5px solid #3b82f6;

    box-shadow:
        0 8px 25px rgba(30,50,80,0.06);
}


/* =====================================================
   WORKFLOW
   ===================================================== */

.workflow-card {
    padding: 25px;

    border-radius: 20px;

    background: white;

    text-align: center;

    box-shadow:
        0 8px 25px rgba(30,50,80,0.07);

    font-size: 16px;
}


/* =====================================================
   DATAFRAME
   ===================================================== */

[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}


/* =====================================================
   METRIC
   ===================================================== */

[data-testid="stMetric"] {
    background: white;

    padding: 15px;

    border-radius: 16px;

    box-shadow:
        0 6px 20px rgba(30,50,80,0.07);
}


/* =====================================================
   INPUT FIELDS
   ===================================================== */

.stTextInput > div > div,
.stNumberInput > div > div,
.stSelectbox > div > div {
    border-radius: 12px;
}


/* =====================================================
   FOOTER
   ===================================================== */

.footer {
    text-align: center;

    color: #94a3b8;

    margin-top: 50px;

    padding: 20px;

    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD STROKE DATA
# =========================================================

@st.cache_data
def load_stroke_data():

    df = pd.read_csv(
        "healthcare-dataset-stroke-data.csv"
    )

    df["bmi"] = pd.to_numeric(
        df["bmi"],
        errors="coerce"
    )

    df["bmi"] = df["bmi"].fillna(
        df["bmi"].median()
    )

    return df


# =========================================================
# LOAD HEART DATA
# =========================================================

@st.cache_data
def load_heart_data():

    # IMPORTANT:
    # cardio_train.csv uses semicolon separator

    df = pd.read_csv(
        "cardio_train.csv",
        sep=";"
    )

    # Raw dataset age is in days.
    # Convert it to years for display.

    df["age"] = df["age"] / 365.25

    return df


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    stroke_logistic = joblib.load(
        "models/logistic_model.pkl"
    )

    stroke_rf = joblib.load(
        "models/random_forest_model.pkl"
    )

    heart_logistic = joblib.load(
        "models/heart_logistic_model.pkl"
    )

    heart_rf = joblib.load(
        "models/heart_random_forest_model.pkl"
    )

    return (
        stroke_logistic,
        stroke_rf,
        heart_logistic,
        heart_rf
    )


# =========================================================
# LOAD DATA AND MODELS
# =========================================================

try:

    stroke_df = load_stroke_data()

    heart_df = load_heart_data()

    (
        stroke_logistic,
        stroke_rf,
        heart_logistic,
        heart_rf
    ) = load_models()

except Exception as e:

    st.error(
        "Error loading project files."
    )

    st.write(
        "Make sure your files and models are placed correctly."
    )

    st.code(str(e))

    st.stop()

# =========================================================
# LANDING PAGE
# =========================================================

if not st.session_state.started:

    st.markdown(
        "<h1 style='text-align:center; margin-top:100px;'>❤️</h1>",
        unsafe_allow_html=True
    )

    st.title("Heart Disease Predictor")

    st.subheader(
        "Smart Healthcare Prediction System"
    )

    st.write(
        "A Data Analytics and Machine Learning based "
        "healthcare application for heart disease and "
        "stroke prediction."
    )

    st.write("")

    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:

        if st.button(
            "🚀 START",
            use_container_width=True
        ):

            st.session_state.started = True

            st.rerun()

    st.stop()
# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "started" not in st.session_state:
    st.session_state.started = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "user_phone" not in st.session_state:
    st.session_state.user_phone = ""
# =========================================================
# LOGIN PAGE
# =========================================================

if not st.session_state.logged_in:

    st.markdown("""
    <div class="login-wrapper">

        <div class="login-icon">🏥</div>

        <div class="login-title">
            Smart Healthcare
        </div>

        <div class="login-subtitle">
            Heart & Stroke Prediction System
        </div>

    </div>
    """, unsafe_allow_html=True)


    st.write("")


    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        name = st.text_input(
            "👤 Full Name",
            placeholder="Enter your name"
        )

        phone = st.text_input(
            "📱 Phone Number",
            placeholder="Enter 10-digit phone number",
            max_chars=10
        )


        st.write("")


        if st.button(
            "🚀 LOGIN",
            use_container_width=True
        ):

            clean_name = name.strip()
            clean_phone = phone.strip()


            # Name validation

            if clean_name == "":

                st.error(
                    "Please enter your name."
                )


            # Phone validation

            elif (
                not clean_phone.isdigit()
                or len(clean_phone) != 10
            ):

                st.error(
                    "Please enter a valid 10-digit phone number."
                )


            else:

                st.session_state.logged_in = True

                st.session_state.user_name = clean_name

                st.session_state.user_phone = clean_phone

                st.rerun()


        st.markdown("""
        <div class="login-note">

        Your information is used only for this
        application session.

        </div>
        """, unsafe_allow_html=True)

    st.stop()
# =========================================================
# LANDING PAGE
# =========================================================

if not st.session_state.started:

    st.markdown("""
    <div class="landing-wrapper">

        <div class="landing-heart">
            ❤️
        </div>

        <div class="landing-title">
            Heart Disease Predictor
        </div>

        <div class="landing-subtitle">
            Smart Healthcare Prediction System
        </div>

        <div class="landing-description">

            Analyze cardiovascular health information
            using Data Analytics and Machine Learning.
            Explore heart disease and stroke prediction,
            healthcare datasets, visualizations and
            patient analysis in one application.

        </div>

    </div>
    """, unsafe_allow_html=True)


    st.write("")


    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:

        if st.button(
            "🚀 START",
            use_container_width=True
        ):

            st.session_state.started = True

            st.rerun()


    st.markdown("""
    <div class="footer">

        Smart Healthcare • Data Analytics & Machine Learning

    </div>
    """, unsafe_allow_html=True)

    st.stop()



# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🏥 Smart Healthcare")

st.sidebar.write(
    "Heart & Stroke Prediction System"
)

st.sidebar.markdown("---")

# Show logged-in user
st.sidebar.write(
    f"👤 Welcome, {st.session_state.user_name}"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "🔎 Data Analysis",
        "📈 Visualizations",
        "❤️ Heart Disease Prediction",
        "🧠 Stroke Prediction",
        "👥 Patient Clustering",
        "🤖 Model Comparison"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Academic Data Analytics & Machine Learning Project"
)

# Logout button
if st.sidebar.button(
    "🚪 Logout",
    use_container_width=True
):

    st.session_state.logged_in = False
    st.session_state.started = False
    st.session_state.user_name = ""
    st.session_state.user_phone = ""

    st.rerun()

# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title("🏥 Smart Healthcare")

    st.subheader(
        "Heart & Stroke Prediction and Analytics System"
    )

    st.write(
        "This project uses Data Analytics, Data Visualization "
        "and Machine Learning techniques to analyze healthcare "
        "datasets and generate prediction results."
    )

    st.info(
        "This application is developed for academic and "
        "educational purposes. Predictions should not be "
        "treated as medical diagnosis."
    )

    # =====================================================
    # WHAT CAN YOU DO?
    # =====================================================

    st.header("🔍 What Can You Do?")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("❤️ Heart Disease Prediction")

        st.write(
            "Enter cardiovascular health information such as "
            "age, blood pressure, cholesterol, glucose, weight "
            "and lifestyle factors to generate a prediction."
        )

    with col2:

        st.subheader("🧠 Stroke Prediction")

        st.write(
            "Enter patient information such as age, glucose "
            "level, BMI, hypertension, heart disease and "
            "smoking status to generate a stroke prediction."
        )

    with col3:

        st.subheader("📊 Data Analytics")

        st.write(
            "Explore healthcare datasets using data analysis, "
            "statistics, visualizations and correlation analysis."
        )

    st.divider()

    # =====================================================
    # SYSTEM OVERVIEW
    # =====================================================

    st.header("📌 System Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Heart Records",
            f"{heart_df.shape[0]:,}"
        )

    with c2:

        st.metric(
            "Stroke Records",
            f"{stroke_df.shape[0]:,}"
        )

    with c3:

        st.metric(
            "ML Models",
            "4"
        )

    with c4:

        st.metric(
            "Analytics",
            "Yes"
        )

    st.divider()

    # =====================================================
    # PROJECT WORKFLOW
    # =====================================================

    st.header("⚙️ Project Workflow")

    st.write(
        "Dataset → Preprocessing → Data Wrangling → "
        "Exploratory Data Analysis → Visualization → "
        "Machine Learning → Prediction → Clustering"
    )

    st.divider()

    # =====================================================
    # MACHINE LEARNING
    # =====================================================

    st.header("🤖 Machine Learning Models")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Logistic Regression")

        st.write(
            "Used for binary classification and estimating "
            "the probability of the target class."
        )

    with col2:

        st.subheader("Random Forest")

        st.write(
            "Uses multiple decision trees to perform "
            "classification using healthcare features."
        )

    st.warning(
        "⚠️ This project is for academic purposes only "
        "and should not be used as a medical diagnosis."
    )
    # =====================================================
    # PREDICTION MODULES
    # =====================================================

    st.markdown(
        '<div class="section-title">🧪 Prediction Modules</div>',
        unsafe_allow_html=True
    )


    c1, c2 = st.columns(2)


    with c1:

        st.markdown("""
        <div class="heart-box">

        <h2>❤️ Heart Disease</h2>

        <p>
        Analyze cardiovascular health parameters such as:
        </p>

        <p>
        Age • Blood Pressure • Cholesterol • Glucose
        • Weight • Lifestyle
        </p>

        </div>
        """, unsafe_allow_html=True)


    with c2:

        st.markdown("""
        <div class="stroke-box">

        <h2>🧠 Stroke</h2>

        <p>
        Analyze patient characteristics such as:
        </p>

        <p>
        Age • Glucose • BMI • Hypertension
        • Heart Disease • Smoking
        </p>

        </div>
        """, unsafe_allow_html=True)


    # =====================================================
    # WORKFLOW
    # =====================================================

    st.markdown(
        '<div class="section-title">⚙️ Project Workflow</div>',
        unsafe_allow_html=True
    )


    st.markdown("""
    <div class="workflow">

    📁 Dataset
    →
    🧹 Preprocessing
    →
    🔄 Data Wrangling
    →
    🔎 EDA
    →
    📈 Visualization
    →
    🤖 Machine Learning
    →
    🔮 Prediction
    →
    👥 Clustering

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# DATASET OVERVIEW
# =========================================================

elif page == "📊 Dataset Overview":

    st.title("📊 Dataset Overview")

    st.write(
        "Explore the datasets used in the Smart Healthcare project."
    )


    # =====================================================
    # HEART DATASET
    # =====================================================

    st.subheader("❤️ Cardiovascular Dataset")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Records",
        f"{heart_df.shape[0]:,}"
    )

    c2.metric(
        "Features",
        heart_df.shape[1]
    )

    c3.metric(
        "Target",
        "cardio"
    )


    st.dataframe(
        heart_df.head(20),
        use_container_width=True
    )


    st.write("### Heart Dataset Information")

    heart_info = pd.DataFrame({
        "Column": heart_df.columns,
        "Data Type": heart_df.dtypes.astype(str).values,
        "Missing Values": heart_df.isnull().sum().values
    })

    st.dataframe(
        heart_info,
        use_container_width=True
    )


    # =====================================================
    # STROKE DATASET
    # =====================================================

    st.subheader("🧠 Stroke Dataset")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Records",
        f"{stroke_df.shape[0]:,}"
    )

    c2.metric(
        "Features",
        stroke_df.shape[1]
    )

    c3.metric(
        "Target",
        "stroke"
    )


    st.dataframe(
        stroke_df.head(20),
        use_container_width=True
    )


    st.write("### Stroke Dataset Information")

    stroke_info = pd.DataFrame({
        "Column": stroke_df.columns,
        "Data Type": stroke_df.dtypes.astype(str).values,
        "Missing Values": stroke_df.isnull().sum().values
    })

    st.dataframe(
        stroke_info,
        use_container_width=True
    )


# =========================================================
# DATA ANALYSIS
# =========================================================

elif page == "🔎 Data Analysis":

    st.title("🔎 Exploratory Data Analysis")

    dataset = st.selectbox(
        "Choose Dataset",
        [
            "Heart Disease",
            "Stroke"
        ]
    )


    # =====================================================
    # HEART ANALYSIS
    # =====================================================

    if dataset == "Heart Disease":

        st.subheader(
            "❤️ Cardiovascular Disease Distribution"
        )

        disease_counts = (
            heart_df["cardio"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            disease_counts
        )


        st.write("### Disease Class Counts")

        st.dataframe(
            disease_counts.rename(
                index={
                    0: "No Disease",
                    1: "Disease"
                }
            ).to_frame("Count")
        )


        st.subheader(
            "📊 Numerical Statistics"
        )

        st.dataframe(
            heart_df.describe(),
            use_container_width=True
        )


        st.subheader(
            "🎯 Average Values by Disease Class"
        )

        grouped = heart_df.groupby(
            "cardio"
        )[
            [
                "age",
                "height",
                "weight",
                "ap_hi",
                "ap_lo"
            ]
        ].mean()

        st.dataframe(
            grouped,
            use_container_width=True
        )


    # =====================================================
    # STROKE ANALYSIS
    # =====================================================

    else:

        st.subheader(
            "🧠 Stroke Distribution"
        )

        stroke_counts = (
            stroke_df["stroke"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            stroke_counts
        )


        st.write("### Stroke Class Counts")

        st.dataframe(
            stroke_counts.rename(
                index={
                    0: "No Stroke",
                    1: "Stroke"
                }
            ).to_frame("Count")
        )


        st.subheader(
            "📊 Age Statistics"
        )

        st.write(
            stroke_df["age"].describe()
        )


        st.subheader(
            "📊 Numerical Statistics"
        )

        st.dataframe(
            stroke_df.describe(),
            use_container_width=True
        )


        st.subheader(
            "🎯 Average Glucose and BMI by Stroke Class"
        )

        grouped = stroke_df.groupby(
            "stroke"
        )[
            [
                "age",
                "avg_glucose_level",
                "bmi"
            ]
        ].mean()

        st.dataframe(
            grouped,
            use_container_width=True
        )


# =========================================================
# VISUALIZATIONS
# =========================================================

elif page == "📈 Visualizations":

    st.title("📈 Healthcare Visualizations")

    dataset = st.selectbox(
        "Select Dataset",
        [
            "Heart Disease",
            "Stroke"
        ]
    )


    # =====================================================
    # HEART VISUALIZATIONS
    # =====================================================

    if dataset == "Heart Disease":

        chart = st.selectbox(
            "Select Chart",
            [
                "Age Distribution",
                "Blood Pressure",
                "Cholesterol",
                "Glucose",
                "Weight vs Disease",
                "Disease Distribution",
                "Correlation Heatmap"
            ]
        )


        fig, ax = plt.subplots()


        if chart == "Age Distribution":

            ax.hist(
                heart_df["age"],
                bins=20
            )

            ax.set_title(
                "Age Distribution"
            )

            ax.set_xlabel("Age")

            ax.set_ylabel("Number of Patients")


        elif chart == "Blood Pressure":

            ax.hist(
                heart_df["ap_hi"],
                bins=30
            )

            ax.set_title(
                "Systolic Blood Pressure Distribution"
            )

            ax.set_xlabel(
                "Systolic Blood Pressure"
            )

            ax.set_ylabel(
                "Number of Patients"
            )


        elif chart == "Cholesterol":

            sns.countplot(
                data=heart_df,
                x="cholesterol",
                hue="cardio",
                ax=ax
            )

            ax.set_title(
                "Cholesterol vs Cardiovascular Disease"
            )


        elif chart == "Glucose":

            sns.countplot(
                data=heart_df,
                x="gluc",
                hue="cardio",
                ax=ax
            )

            ax.set_title(
                "Glucose vs Cardiovascular Disease"
            )


        elif chart == "Weight vs Disease":

            sns.boxplot(
                data=heart_df,
                x="cardio",
                y="weight",
                ax=ax
            )

            ax.set_title(
                "Weight vs Cardiovascular Disease"
            )


        elif chart == "Disease Distribution":

            sns.countplot(
                data=heart_df,
                x="cardio",
                ax=ax
            )

            ax.set_title(
                "Cardiovascular Disease Distribution"
            )

            ax.set_xlabel(
                "Cardiovascular Disease"
            )


        elif chart == "Correlation Heatmap":

            numeric = heart_df.select_dtypes(
                include="number"
            )

            sns.heatmap(
                numeric.corr(),
                annot=True,
                fmt=".2f",
                ax=ax
            )

            ax.set_title(
                "Cardiovascular Dataset Correlation"
            )


        st.pyplot(
            fig,
            use_container_width=True
        )


    # =====================================================
    # STROKE VISUALIZATIONS
    # =====================================================

    else:

        chart = st.selectbox(
            "Select Chart",
            [
                "Age Distribution",
                "Glucose Distribution",
                "BMI Distribution",
                "Age vs Stroke",
                "Glucose vs Stroke",
                "Smoking vs Stroke",
                "Stroke Distribution",
                "Correlation Heatmap"
            ]
        )


        fig, ax = plt.subplots()


        if chart == "Age Distribution":

            ax.hist(
                stroke_df["age"],
                bins=20
            )

            ax.set_title(
                "Age Distribution"
            )

            ax.set_xlabel("Age")

            ax.set_ylabel(
                "Number of Patients"
            )


        elif chart == "Glucose Distribution":

            ax.hist(
                stroke_df["avg_glucose_level"],
                bins=20
            )

            ax.set_title(
                "Average Glucose Level Distribution"
            )


        elif chart == "BMI Distribution":

            ax.hist(
                stroke_df["bmi"],
                bins=20
            )

            ax.set_title(
                "BMI Distribution"
            )


        elif chart == "Age vs Stroke":

            sns.boxplot(
                data=stroke_df,
                x="stroke",
                y="age",
                ax=ax
            )

            ax.set_title(
                "Age vs Stroke"
            )


        elif chart == "Glucose vs Stroke":

            sns.boxplot(
                data=stroke_df,
                x="stroke",
                y="avg_glucose_level",
                ax=ax
            )

            ax.set_title(
                "Glucose vs Stroke"
            )


        elif chart == "Smoking vs Stroke":

            sns.countplot(
                data=stroke_df,
                x="smoking_status",
                hue="stroke",
                ax=ax
            )

            ax.set_title(
                "Smoking Status vs Stroke"
            )

            ax.tick_params(
                axis="x",
                rotation=25
            )


        elif chart == "Stroke Distribution":

            sns.countplot(
                data=stroke_df,
                x="stroke",
                ax=ax
            )

            ax.set_title(
                "Stroke Distribution"
            )


        elif chart == "Correlation Heatmap":

            numeric = stroke_df.select_dtypes(
                include="number"
            )

            sns.heatmap(
                numeric.corr(),
                annot=True,
                fmt=".2f",
                ax=ax
            )

            ax.set_title(
                "Stroke Dataset Correlation"
            )


        st.pyplot(
            fig,
            use_container_width=True
        )


# =========================================================
# HEART DISEASE PREDICTION
# =========================================================

elif page == "❤️ Heart Disease Prediction":

    st.title(
        "❤️ Heart Disease Prediction"
    )

    st.write(
        "Enter cardiovascular health parameters "
        "to generate a machine-learning prediction."
    )


    st.info(
        "The model predicts the cardiovascular disease "
        "class based on the parameters entered below."
    )


    # =====================================================
    # INPUTS
    # =====================================================

    c1, c2 = st.columns(2)


    with c1:

        age = st.number_input(
            "Age",
            min_value=20.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )


        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male"
            ]
        )


        height = st.number_input(
            "Height (cm)",
            min_value=100.0,
            max_value=220.0,
            value=165.0,
            step=1.0
        )


        weight = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )


        systolic = st.number_input(
            "Systolic Blood Pressure",
            min_value=70,
            max_value=250,
            value=120,
            step=1
        )


        diastolic = st.number_input(
            "Diastolic Blood Pressure",
            min_value=40,
            max_value=160,
            value=80,
            step=1
        )


    with c2:

        cholesterol = st.selectbox(
            "Cholesterol",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[x]
        )


        glucose = st.selectbox(
            "Glucose",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[x]
        )


        smoking = st.selectbox(
            "Smoking",
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )


        alcohol = st.selectbox(
            "Alcohol Intake",
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )


        active = st.selectbox(
            "Physical Activity",
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )


    # =====================================================
    # GENDER CONVERSION
    # =====================================================

    gender_value = (
        1 if gender == "Female"
        else 2
    )


    # =====================================================
    # PATIENT DATA
    # =====================================================

    patient = pd.DataFrame({

        # IMPORTANT:
        # The heart model was trained using age in years.

        "age": [age],

        "gender": [gender_value],

        "height": [height],

        "weight": [weight],

        "ap_hi": [systolic],

        "ap_lo": [diastolic],

        "cholesterol": [cholesterol],

        "gluc": [glucose],

        "smoke": [smoking],

        "alco": [alcohol],

        "active": [active]

    })


    # =====================================================
    # MODEL
    # =====================================================

    st.subheader(
        "🤖 Select Prediction Model"
    )


    model_choice = st.selectbox(
        "Model",
        [
            "Logistic Regression",
            "Random Forest"
        ]
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    if st.button(
        "❤️ Predict Heart Disease"
    ):

        try:

            if model_choice == "Logistic Regression":

                prediction = heart_logistic.predict(
                    patient
                )[0]

                probability = (
                    heart_logistic
                    .predict_proba(patient)[0][1]
                )

            else:

                prediction = heart_rf.predict(
                    patient
                )[0]

                probability = (
                    heart_rf
                    .predict_proba(patient)[0][1]
                )


            st.subheader(
                "🔮 Prediction Result"
            )


            if prediction == 1:

                st.error(
                    "Prediction: Cardiovascular Disease Class"
                )

            else:

                st.success(
                    "Prediction: No Cardiovascular Disease Class"
                )


            st.metric(
                "Predicted Probability",
                f"{probability:.2%}"
            )


            st.progress(
                float(probability)
            )


            st.warning(
                "This is an academic machine-learning "
                "prediction and is not a medical diagnosis."
            )

        except Exception as e:

            st.error(
                "Prediction could not be generated."
            )

            st.code(
                str(e)
            )


# =========================================================
# STROKE PREDICTION
# =========================================================

elif page == "🧠 Stroke Prediction":

    st.title(
        "🧠 Stroke Prediction"
    )

    st.write(
        "Enter patient health information "
        "to generate a machine-learning prediction."
    )


    st.info(
        "The model uses demographic, lifestyle and "
        "health-related features."
    )


    # =====================================================
    # INPUTS
    # =====================================================

    c1, c2 = st.columns(2)


    with c1:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ]
        )


        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )


        hypertension = st.selectbox(
            "Hypertension",
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )


        heart_disease = st.selectbox(
            "Heart Disease",
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )


        ever_married = st.selectbox(
            "Ever Married",
            [
                "Yes",
                "No"
            ]
        )


        work_type = st.selectbox(
            "Work Type",
            [
                "Private",
                "Self-employed",
                "Govt_job",
                "children",
                "Never_worked"
            ]
        )


    with c2:

        residence_type = st.selectbox(
            "Residence Type",
            [
                "Urban",
                "Rural"
            ]
        )


        glucose = st.number_input(
            "Average Glucose Level",
            min_value=0.0,
            max_value=300.0,
            value=100.0,
            step=1.0
        )


        bmi = st.number_input(
            "BMI",
            min_value=5.0,
            max_value=100.0,
            value=25.0,
            step=0.1
        )


        smoking_status = st.selectbox(
            "Smoking Status",
            [
                "never smoked",
                "formerly smoked",
                "smokes",
                "Unknown"
            ]
        )


    # =====================================================
    # PATIENT DATA
    # =====================================================

    patient = pd.DataFrame({

        "gender": [gender],

        "age": [age],

        "hypertension": [hypertension],

        "heart_disease": [heart_disease],

        "ever_married": [ever_married],

        "work_type": [work_type],

        "Residence_type": [residence_type],

        "avg_glucose_level": [glucose],

        "bmi": [bmi],

        "smoking_status": [smoking_status]

    })


    # =====================================================
    # MODEL
    # =====================================================

    model_choice = st.selectbox(
        "Prediction Model",
        [
            "Logistic Regression",
            "Random Forest"
        ]
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    if st.button(
        "🧠 Predict Stroke"
    ):

        try:

            if model_choice == "Logistic Regression":

                prediction = stroke_logistic.predict(
                    patient
                )[0]

                probability = (
                    stroke_logistic
                    .predict_proba(patient)[0][1]
                )

            else:

                prediction = stroke_rf.predict(
                    patient
                )[0]

                probability = (
                    stroke_rf
                    .predict_proba(patient)[0][1]
                )


            st.subheader(
                "🔮 Prediction Result"
            )


            if prediction == 1:

                st.error(
                    "Prediction: Stroke Class"
                )

            else:

                st.success(
                    "Prediction: No Stroke Class"
                )


            st.metric(
                "Predicted Probability",
                f"{probability:.2%}"
            )


            st.progress(
                float(probability)
            )


            st.warning(
                "This is an academic machine-learning "
                "prediction and is not a medical diagnosis."
            )


        except Exception as e:

            st.error(
                "Prediction could not be generated."
            )

            st.code(
                str(e)
            )


# =========================================================
# PATIENT CLUSTERING
# =========================================================

elif page == "👥 Patient Clustering":

    st.title(
        "👥 Patient Health Clustering"
    )

    st.write(
        "K-Means clustering groups patients based on "
        "selected health characteristics."
    )


    features = [
        "age",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi"
    ]


    cluster_data = stroke_df[
        features
    ].copy()


    # Make sure there are no missing values

    cluster_data = cluster_data.fillna(
        cluster_data.median()
    )


    scaler = StandardScaler()


    scaled_data = scaler.fit_transform(
        cluster_data
    )


    k = st.slider(
        "Number of Clusters",
        2,
        5,
        3
    )


    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )


    cluster_data["Cluster"] = (
        kmeans.fit_predict(
            scaled_data
        )
    )


    # =====================================================
    # CLUSTER DISTRIBUTION
    # =====================================================

    st.subheader(
        "📊 Cluster Distribution"
    )


    cluster_counts = (
        cluster_data["Cluster"]
        .value_counts()
        .sort_index()
    )


    st.bar_chart(
        cluster_counts
    )


    # =====================================================
    # CLUSTER TABLE
    # =====================================================

    st.subheader(
        "📋 Cluster Summary"
    )


    summary = (
        cluster_data
        .groupby("Cluster")
        .mean()
    )


    st.dataframe(
        summary,
        use_container_width=True
    )


    # =====================================================
    # CLUSTER GRAPH
    # =====================================================

    st.subheader(
        "📈 Patient Health Clusters"
    )


    fig, ax = plt.subplots()


    ax.scatter(
        cluster_data["age"],
        cluster_data["avg_glucose_level"],
        c=cluster_data["Cluster"]
    )


    ax.set_xlabel(
        "Age"
    )

    ax.set_ylabel(
        "Average Glucose Level"
    )

    ax.set_title(
        "Patient Health Clusters"
    )


    st.pyplot(
        fig,
        use_container_width=True
    )


# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "🤖 Model Comparison":

    st.title(
        "🤖 Machine Learning Model Comparison"
    )


    st.write(
        "This section compares Logistic Regression and "
        "Random Forest for both prediction tasks."
    )


    # =====================================================
    # HEART MODEL COMPARISON
    # =====================================================

    st.subheader(
        "❤️ Heart Disease Models"
    )


    heart_metrics = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],

        "Purpose": [
            "Linear classification",
            "Ensemble classification"
        ]

    })


    st.dataframe(
        heart_metrics,
        use_container_width=True
    )


    st.write(
        "Both models are trained to classify the "
        "cardiovascular disease target."
    )


    # =====================================================
    # STROKE MODEL COMPARISON
    # =====================================================

    st.subheader(
        "🧠 Stroke Models"
    )


    stroke_metrics = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],

        "Purpose": [
            "Linear classification",
            "Ensemble classification"
        ]

    })


    st.dataframe(
        stroke_metrics,
        use_container_width=True
    )


    st.write(
        "Both models are trained to classify the "
        "stroke target."
    )


    # =====================================================
    # METRICS FROM YOUR TRAINING
    # =====================================================

    st.subheader(
        "📊 Stroke Model Training Results"
    )


    stroke_results = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],

        "Accuracy": [
            0.745597,
            0.935421
        ],

        "Precision": [
            0.137931,
            0.214286
        ],

        "Recall": [
            0.80,
            0.12
        ],

        "F1 Score": [
            0.235294,
            0.153846
        ],

        "ROC-AUC": [
            0.843663,
            0.775453
        ]

    })


    st.dataframe(
        stroke_results.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}",
            "ROC-AUC": "{:.2%}"
        }),
        use_container_width=True
    )


    st.info(
        "The metrics show different trade-offs between "
        "accuracy, precision, recall, F1-score and ROC-AUC. "
        "For an imbalanced dataset, accuracy alone should "
        "not be used to interpret model performance."
    )


    # =====================================================
    # PROJECT REQUIREMENTS
    # =====================================================

    st.subheader(
        "✅ Project Components"
    )


    requirements = pd.DataFrame({

        "Requirement": [
            "Real-world domain",
            "Dataset ≥ 1,000 records",
            "Data preprocessing",
            "Data wrangling",
            "Exploratory Data Analysis",
            "Data visualization",
            "Machine Learning",
            "Two ML models",
            "Prediction interface",
            "Clustering"
        ],

        "Implemented": [
            "Healthcare",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "Logistic Regression + Random Forest",
            "Yes",
            "K-Means"
        ]

    })


    st.dataframe(
        requirements,
        use_container_width=True
    )


# =========================================================
# END
# =========================================================