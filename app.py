import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="ChurnIQ | AI Analytics",
    page_icon="✦",
    layout="wide"
)

# ================= LOAD MODEL & DATA =================

@st.cache_resource
def load_model():
    return joblib.load("best_churn_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("dataset/Churn_Modelling.csv")


model = load_model()
df = load_data()

# ================= PREMIUM CSS =================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #080D18;
    color: #F1F5F9;
}

.block-container {
    max-width: 1450px;
    padding: 2rem 3rem !important;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

/* HEADER */

.brand {
    font-size: 30px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -1px;
}

.brand span {
    color: #60A5FA;
}

.tagline {
    color: #94A3B8;
    font-size: 14px;
    margin-top: 5px;
}

/* SECTION HEADINGS */

.section {
    font-size: 21px;
    font-weight: 700;
    color: #F8FAFC;
    margin: 25px 0 15px;
}

/* METRIC CARDS */

.metric-card {
    background: linear-gradient(135deg, #151F32, #101827);
    border: 1px solid #26354D;
    border-radius: 16px;
    padding: 23px;
    min-height: 125px;
}

.metric-label {
    color: #94A3B8;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #F8FAFC;
    font-size: 29px;
    font-weight: 800;
    margin-top: 12px;
}

.metric-note {
    color: #60A5FA;
    font-size: 12px;
    margin-top: 5px;
}

/* INPUT FIELDS */

.stTextInput label,
.stNumberInput label,
.stSelectbox label {
    color: #CBD5E1 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

.stNumberInput input {
    background: #172338 !important;
    color: white !important;
    border-radius: 9px !important;
}

div[data-baseweb="select"] > div {
    background: #172338 !important;
    border-color: #34445C !important;
    border-radius: 9px !important;
    color: white !important;
}

/* CONTAINERS */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #111B2B;
    border: 1px solid #26354D;
    border-radius: 16px;
    padding: 12px;
}

/* BUTTON */

.stButton > button,
.stFormSubmitButton > button {
    width: 100%;
    min-height: 52px;
    border-radius: 10px;
    background: linear-gradient(90deg, #2563EB, #7C3AED);
    color: white;
    border: none;
    font-weight: 700;
    font-size: 16px;
    transition: 0.3s;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(90deg, #1D4ED8, #6D28D9);
    border: none;
    color: white;
    transform: translateY(-2px);
}

/* RESULT CARDS */

.result-card {
    background: linear-gradient(135deg, #14243C, #1B2440);
    border: 1px solid #3B82F6;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    min-height: 150px;
}

.result-label {
    color: #94A3B8;
    font-size: 14px;
}

.result-value {
    color: #60A5FA;
    font-size: 30px;
    font-weight: 800;
    margin: 10px 0;
}

hr {
    border-color: #26354D;
    margin: 25px 0;
}

.footer {
    text-align: center;
    color: #64748B;
    font-size: 12px;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# ================= HEADER =================

h1, h2 = st.columns([3, 1])

with h1:

    st.html("""
        <div class="brand">✦ Churn<span>IQ</span></div>
        <div class="tagline">
            INTELLIGENT CUSTOMER RETENTION & ANALYTICS PLATFORM
        </div>
    """)

with h2:

    st.html("""
        <div style="text-align:right;color:#60A5FA;
                    padding-top:15px;font-size:13px;">
            ● SYSTEM ONLINE
        </div>
    """)

st.divider()


# ================= OVERVIEW DASHBOARD =================

st.markdown(
    '<div class="section">Overview Dashboard</div>',
    unsafe_allow_html=True
)

total = len(df)
churned = int(df["Exited"].sum())
retained = total - churned

churn_rate = churned / total * 100
retention_rate = retained / total * 100

cols = st.columns(4)

metrics = [
    ("TOTAL CUSTOMERS", f"{total:,}", "Customer records"),
    ("CHURNED CUSTOMERS", f"{churned:,}", "Customers who exited"),
    ("RETENTION RATE", f"{retention_rate:.2f}%", "Customers retained"),
    ("CHURN RATE", f"{churn_rate:.2f}%", "Customers who exited")
]

for col, (label, value, note) in zip(cols, metrics):

    with col:

        st.html(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
        """)


# ================= CUSTOMER ANALYTICS =================

st.markdown(
    '<div class="section">Customer Analytics</div>',
    unsafe_allow_html=True
)

chart1, chart2 = st.columns(2, gap="large")


# PIE CHART

with chart1:

    with st.container(border=True):

        st.markdown("**Customer Churn Distribution**")

        churn_data = pd.DataFrame({
            "Status": ["Stayed", "Exited"],
            "Customers": [retained, churned]
        })

        fig = px.pie(
            churn_data,
            names="Status",
            values="Customers",
            hole=0.68,
            color="Status",
            color_discrete_map={
                "Stayed": "#3B82F6",
                "Exited": "#A78BFA"
            }
        )

        fig.update_layout(
            paper_bgcolor="#111B2B",
            plot_bgcolor="#111B2B",
            font_color="#F1F5F9",
            showlegend=True,
            margin=dict(t=15, b=15, l=10, r=10),
            legend=dict(font=dict(color="#CBD5E1")),
            height=330
        )

        fig.update_traces(
            textinfo="percent",
            textfont_size=14
        )

        st.plotly_chart(fig, use_container_width=True)


# GEOGRAPHY BAR CHART

with chart2:

    with st.container(border=True):

        st.markdown("**Churn by Geography**")

        geo_data = (
            df.groupby(["Geography", "Exited"])
            .size()
            .reset_index(name="Customers")
        )

        geo_data["Status"] = geo_data["Exited"].map({
            0: "Stayed",
            1: "Exited"
        })

        fig2 = px.bar(
            geo_data,
            x="Geography",
            y="Customers",
            color="Status",
            barmode="group",
            color_discrete_map={
                "Stayed": "#3B82F6",
                "Exited": "#A78BFA"
            }
        )

        fig2.update_layout(
            paper_bgcolor="#111B2B",
            plot_bgcolor="#111B2B",
            font_color="#F1F5F9",
            margin=dict(t=15, b=15, l=10, r=10),
            xaxis_title="Country",
            yaxis_title="Customers",
            legend=dict(font=dict(color="#CBD5E1")),
            height=330
        )

        fig2.update_xaxes(gridcolor="#26354D")
        fig2.update_yaxes(gridcolor="#26354D")

        st.plotly_chart(fig2, use_container_width=True)


# ================= CUSTOMER BEHAVIOUR INSIGHTS =================

st.markdown(
    '<div class="section">Customer Behaviour Insights</div>',
    unsafe_allow_html=True
)

insight1, insight2 = st.columns(2, gap="large")


# LINE GRAPH: CHURN BY AGE

with insight1:

    with st.container(border=True):

        st.markdown("**Churn Trend Across Age Groups**")

        age_data = df[["Age", "Exited"]].copy()

        age_data["Age Group"] = pd.cut(
            age_data["Age"],
            bins=[17, 25, 35, 45, 55, 65, 100],
            labels=[
                "18–25",
                "26–35",
                "36–45",
                "46–55",
                "56–65",
                "66+"
            ]
        )

        age_summary = (
            age_data.groupby("Age Group", observed=False)
            .agg(
                Churn_Rate=("Exited", "mean"),
                Customers=("Exited", "size")
            )
            .reset_index()
        )

        age_summary["Churn Rate (%)"] = (
            age_summary["Churn_Rate"] * 100
        )

        age_fig = px.line(
            age_summary,
            x="Age Group",
            y="Churn Rate (%)",
            markers=True,
            hover_data={
                "Customers": True,
                "Churn_Rate": False
            },
            labels={
                "Age Group": "Customer age",
                "Churn Rate (%)": "Churn rate"
            }
        )

        age_fig.update_traces(
            line=dict(width=3),
            marker=dict(size=10)
        )

        age_fig.update_layout(
            paper_bgcolor="#111B2B",
            plot_bgcolor="#111B2B",
            font_color="#F1F5F9",
            margin=dict(t=15, b=15, l=10, r=10),
            height=330,
            yaxis=dict(
                gridcolor="#26354D",
                ticksuffix="%"
            ),
            xaxis=dict(title="Age group")
        )

        st.plotly_chart(
            age_fig,
            use_container_width=True
        )


# SCATTER GRAPH: AGE VS BALANCE

with insight2:

    with st.container(border=True):

        st.markdown("**Customer Age vs Account Balance**")

        scatter_data = df[
            ["Age", "Balance", "Exited"]
        ].copy()

        scatter_data["Customer Status"] = (
            scatter_data["Exited"].map({
                0: "Stayed",
                1: "Exited"
            })
        )

        scatter_fig = px.scatter(
            scatter_data,
            x="Age",
            y="Balance",
            color="Customer Status",
            opacity=0.55,
            color_discrete_map={
                "Stayed": "#3B82F6",
                "Exited": "#A78BFA"
            },
            labels={
                "Age": "Customer age",
                "Balance": "Account balance"
            }
        )

        scatter_fig.update_layout(
            paper_bgcolor="#111B2B",
            plot_bgcolor="#111B2B",
            font_color="#F1F5F9",
            margin=dict(t=15, b=15, l=10, r=10),
            height=330,
            xaxis=dict(gridcolor="#26354D"),
            yaxis=dict(gridcolor="#26354D")
        )

        st.plotly_chart(
            scatter_fig,
            use_container_width=True
        )


# ================= PREDICTION SECTION =================

st.divider()

st.markdown(
    '<div class="section">✦ AI Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<p style="color:#94A3B8;font-size:14px;">'
    'Enter customer information to generate an AI-powered churn prediction.'
    '</p>',
    unsafe_allow_html=True
)


# ================= CUSTOMER INPUT FORM =================

with st.form("customer_prediction_form"):

    left, right = st.columns(2, gap="large")

    # PERSONAL INFORMATION

    with left:

        with st.container(border=True):

            st.markdown("**Personal Information**")

            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=900,
                value=650
            )

            geography = st.selectbox(
                "Geography",
                ["France", "Germany", "Spain"]
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=35
            )

            tenure = st.number_input(
                "Tenure (Years)",
                min_value=0,
                max_value=10,
                value=5
            )

    # BANKING INFORMATION

    with right:

        with st.container(border=True):

            st.markdown("**Banking Information**")

            balance = st.number_input(
                "Account Balance",
                min_value=0.0,
                max_value=1000000.0,
                value=50000.0,
                step=1000.0
            )

            products = st.selectbox(
                "Number of Products",
                [1, 2, 3, 4]
            )

            credit_card = st.selectbox(
                "Has Credit Card?",
                ["Yes", "No"]
            )

            active_member = st.selectbox(
                "Is Active Member?",
                ["Yes", "No"]
            )

            salary = st.number_input(
                "Estimated Salary",
                min_value=0.0,
                max_value=1000000.0,
                value=50000.0,
                step=1000.0
            )

    submitted = st.form_submit_button(
        "✦  ANALYZE CUSTOMER & PREDICT CHURN",
        use_container_width=True
    )


# ================= PREDICTION RESULTS =================

if submitted:

    input_data = pd.DataFrame({

        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [products],
        "HasCrCard": [
            1 if credit_card == "Yes" else 0
        ],
        "IsActiveMember": [
            1 if active_member == "Yes" else 0
        ],
        "EstimatedSalary": [salary]

    })

    try:

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        st.divider()

        st.markdown(
            '<div class="section">Prediction Report</div>',
            unsafe_allow_html=True
        )

        r1, r2 = st.columns(2)

        # CHURN PROBABILITY

        with r1:

            st.html(f"""
                <div class="result-card">
                    <div class="result-label">CHURN PROBABILITY</div>
                    <div class="result-value">{probability * 100:.2f}%</div>
                    <div class="result-label">
                        Predicted probability of customer exit
                    </div>
                </div>
            """)

        # PREDICTION STATUS

        with r2:

            if prediction == 1:

                status = "LIKELY TO EXIT"
                color = "#F87171"

            else:

                status = "LIKELY TO STAY"
                color = "#34D399"

            st.html(f"""
                <div class="result-card">
                    <div class="result-label">PREDICTION STATUS</div>
                    <div class="result-value" style="color:{color};">
                        {status}
                    </div>
                    <div class="result-label">
                        Machine Learning prediction
                    </div>
                </div>
            """)

        st.write("")

        st.progress(float(probability))

        if prediction == 1:

            st.warning(
                "Retention Recommendation: Consider personalized offers, "
                "customer support follow-ups and loyalty benefits."
            )

        else:

            st.success(
                "Retention Recommendation: Continue engagement and "
                "maintain a positive customer experience."
            )

    except Exception as error:

        st.error(f"Prediction failed: {error}")


# ================= FOOTER =================

st.divider()

st.html("""
    <div class="footer">
        ✦ ChurnIQ — AI-Powered Customer Churn Analytics<br>
        Developed using Python • Scikit-learn • Streamlit • Plotly
    </div>
""")