import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NIFTY 50 Market Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY 50 Market Analysis and Direction Prediction")

st.markdown(
    """
    **Project:** NIFTY 50 Market Analysis and Direction Prediction

    This application analyzes historical NIFTY 50 data, calculates
    technical indicators, and uses machine learning to predict
    the next trading session's market direction (UP or DOWN).
    """
)


# --------------------------------------------------
# DOWNLOAD NIFTY 50 DATA
# --------------------------------------------------

@st.cache_data(ttl=3600)
def download_data():

    data = yf.download(
        "^NSEI",
        period="10y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(
            "Unable to download NIFTY 50 data. Please try again later."
        )

    # Handle MultiIndex columns returned by yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index = pd.to_datetime(data.index)

    data = data[~data.index.duplicated(keep="last")]

    data = data.sort_index()

    required_columns = ["Open", "High", "Low", "Close"]

    data = data.dropna(subset=required_columns)

    return data


# --------------------------------------------------
# CALCULATE TECHNICAL INDICATORS
# --------------------------------------------------

def calculate_indicators(data):

    data = data.copy()

    # Moving Averages
    data["MA20"] = data["Close"].rolling(window=20).mean()

    data["MA50"] = data["Close"].rolling(window=50).mean()

    # Daily Returns
    data["Daily_Return"] = data["Close"].pct_change() * 100

    # Volatility
    data["Volatility"] = (
        data["Daily_Return"].rolling(window=20).std()
    )

    # RSI
    delta = data["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()

    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = data["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = data["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = ema12 - ema26

    data["MACD_Signal"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    # Bollinger Bands
    data["BB_Middle"] = data["Close"].rolling(20).mean()

    bb_std = data["Close"].rolling(20).std()

    data["BB_Upper"] = data["BB_Middle"] + (2 * bb_std)

    data["BB_Lower"] = data["BB_Middle"] - (2 * bb_std)

    # Additional Returns
    data["Return_5"] = data["Close"].pct_change(5)

    data["Return_10"] = data["Close"].pct_change(10)

    return data


# --------------------------------------------------
# PREPARE MACHINE LEARNING DATASET
# --------------------------------------------------

def prepare_dataset(data):

    data = data.copy()

    # Next trading session's closing price
    next_close = data["Close"].shift(-1)

    # Target:
    # 1 = UP
    # 0 = DOWN

    data["Target"] = np.where(
        next_close.notna(),
        (next_close > data["Close"]).astype(float),
        np.nan
    )

    features = [
        "Daily_Return",
        "Volatility",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MA20",
        "MA50",
        "BB_Upper",
        "BB_Lower",
        "Return_5",
        "Return_10"
    ]

    # Keep rows with complete feature values
    feature_data = data.dropna(subset=features).copy()

    # Keep only rows where the actual next-day direction is known
    training_data = feature_data.dropna(
        subset=["Target"]
    ).copy()

    X = training_data[features]

    y = training_data["Target"].astype(int)

    return feature_data, X, y, features


# --------------------------------------------------
# TRAIN MACHINE LEARNING MODELS
# --------------------------------------------------

def train_models(X, y):

    split_index = int(len(X) * 0.80)

    X_train = X.iloc[:split_index]

    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]

    y_test = y.iloc[split_index:]

    # Logistic Regression
    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    logistic_model.fit(X_train, y_train)

    logistic_predictions = logistic_model.predict(X_test)

    # Random Forest
    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    random_forest_model.fit(X_train, y_train)

    rf_predictions = random_forest_model.predict(X_test)

    return (
        logistic_model,
        random_forest_model,
        X_test,
        y_test,
        logistic_predictions,
        rf_predictions
    )


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

def evaluate_model(y_test, predictions):

    return {
        "Accuracy": accuracy_score(y_test, predictions),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    }


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

with st.spinner("Downloading NIFTY 50 historical data..."):

    try:
        raw_data = download_data()

        data = calculate_indicators(raw_data)

        feature_data, X, y, features = prepare_dataset(data)

    except Exception as error:

        st.error(f"Error loading data: {error}")

        st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Market Overview",
        "Historical Analysis",
        "Technical Indicators",
        "Machine Learning",
        "Dataset"
    ]
)


# --------------------------------------------------
# MARKET OVERVIEW
# --------------------------------------------------

if page == "Market Overview":

    st.header("Market Overview")

    latest = data.iloc[-1]

    previous = data.iloc[-2]

    current_price = float(latest["Close"])

    previous_price = float(previous["Close"])

    change = current_price - previous_price

    percentage_change = (
        change / previous_price
    ) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "NIFTY 50 Closing Value",
        f"{current_price:,.2f}",
        f"{change:+.2f}"
    )

    col2.metric(
        "Daily Change",
        f"{percentage_change:+.2f}%"
    )

    col3.metric(
        "RSI",
        f"{latest['RSI']:.2f}"
    )

    st.subheader("Recent Market Performance")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            name="NIFTY 50 Close"
        )
    )

    fig.update_layout(
        title="NIFTY 50 Historical Closing Prices",
        xaxis_title="Date",
        yaxis_title="Index Value",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest Market Information")

    st.write(
        f"**Latest available date:** {data.index[-1].date()}"
    )

    st.write(
        f"**Highest closing value:** {data['Close'].max():,.2f}"
    )

    st.write(
        f"**Lowest closing value:** {data['Close'].min():,.2f}"
    )


# --------------------------------------------------
# HISTORICAL ANALYSIS
# --------------------------------------------------

elif page == "Historical Analysis":

    st.header("Historical Market Analysis")

    st.write(
        "Analyze NIFTY 50 historical closing prices and moving averages."
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            name="Closing Price"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA20"],
            name="20-Day Moving Average"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA50"],
            name="50-Day Moving Average"
        )
    )

    fig.update_layout(
        title="NIFTY 50 Price and Moving Averages",
        xaxis_title="Date",
        yaxis_title="Index Value",
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Returns")

    returns_fig = go.Figure()

    returns_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Daily_Return"],
            name="Daily Returns"
        )
    )

    returns_fig.update_layout(
        title="Daily Percentage Returns",
        xaxis_title="Date",
        yaxis_title="Return (%)"
    )

    st.plotly_chart(
        returns_fig,
        use_container_width=True
    )

    st.subheader("Market Volatility")

    st.line_chart(data["Volatility"])


# --------------------------------------------------
# TECHNICAL INDICATORS
# --------------------------------------------------

elif page == "Technical Indicators":

    st.header("Technical Indicators")

    latest = data.iloc[-1]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "RSI",
        f"{latest['RSI']:.2f}"
    )

    col2.metric(
        "MACD",
        f"{latest['MACD']:.2f}"
    )

    col3.metric(
        "Volatility",
        f"{latest['Volatility']:.2f}"
    )

    # RSI Chart
    st.subheader("Relative Strength Index (RSI)")

    rsi_fig = go.Figure()

    rsi_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI"],
            name="RSI"
        )
    )

    rsi_fig.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="Overbought (70)"
    )

    rsi_fig.add_hline(
        y=30,
        line_dash="dash",
        annotation_text="Oversold (30)"
    )

    rsi_fig.update_layout(
        height=400,
        yaxis_title="RSI"
    )

    st.plotly_chart(
        rsi_fig,
        use_container_width=True
    )

    # MACD Chart
    st.subheader("MACD Indicator")

    macd_fig = go.Figure()

    macd_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD"],
            name="MACD"
        )
    )

    macd_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD_Signal"],
            name="Signal Line"
        )
    )

    macd_fig.update_layout(
        height=400
    )

    st.plotly_chart(
        macd_fig,
        use_container_width=True
    )

    # Bollinger Bands
    st.subheader("Bollinger Bands")

    bb_fig = go.Figure()

    bb_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            name="Closing Price"
        )
    )

    bb_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["BB_Upper"],
            name="Upper Band"
        )
    )

    bb_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["BB_Lower"],
            name="Lower Band"
        )
    )

    bb_fig.update_layout(
        height=450
    )

    st.plotly_chart(
        bb_fig,
        use_container_width=True
    )


# --------------------------------------------------
# MACHINE LEARNING
# --------------------------------------------------

elif page == "Machine Learning":

    st.header("Machine Learning - Market Direction Prediction")

    st.write(
        """
        Two machine-learning algorithms are used:

        1. Logistic Regression
        2. Random Forest Classifier

        The models classify the next trading session's direction
        as UP or DOWN.
        """
    )

    if len(X) < 100:

        st.error(
            "Not enough historical data to train the models."
        )

        st.stop()

    with st.spinner("Training machine-learning models..."):

        (
            logistic_model,
            rf_model,
            X_test,
            y_test,
            logistic_predictions,
            rf_predictions
        ) = train_models(X, y)

    # Evaluate models
    logistic_results = evaluate_model(
        y_test,
        logistic_predictions
    )

    rf_results = evaluate_model(
        y_test,
        rf_predictions
    )

    st.subheader("Model Performance Comparison")

    results_df = pd.DataFrame(
        [
            logistic_results,
            rf_results
        ],
        index=[
            "Logistic Regression",
            "Random Forest"
        ]
    )

    st.dataframe(
        results_df.style.format("{:.2%}"),
        use_container_width=True
    )

    st.subheader("Accuracy Comparison")

    accuracy_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],
        "Accuracy": [
            logistic_results["Accuracy"],
            rf_results["Accuracy"]
        ]
    })

    st.bar_chart(
        accuracy_df.set_index("Model")
    )

    # Confusion Matrix
    st.subheader("Random Forest Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        rf_predictions,
        labels=[0, 1]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual DOWN", "Actual UP"],
        columns=["Predicted DOWN", "Predicted UP"]
    )

    st.dataframe(cm_df)

    # Latest Prediction
    st.subheader("Next Trading Session Prediction")

    latest_features = feature_data[features].iloc[[-1]]

    prediction = rf_model.predict(
        latest_features
    )[0]

    probability = rf_model.predict_proba(
        latest_features
    )[0]

    if prediction == 1:

        st.success("Predicted Market Direction: UP 📈")

    else:

        st.error("Predicted Market Direction: DOWN 📉")

    col1, col2 = st.columns(2)

    col1.metric(
        "Probability of DOWN",
        f"{probability[0] * 100:.2f}%"
    )

    col2.metric(
        "Probability of UP",
        f"{probability[1] * 100:.2f}%"
    )

    st.caption(
        """
        Note: This is an experimental machine-learning prediction,
        not a guaranteed market forecast or financial recommendation.
        Model performance is measured on a historical holdout period
        and does not guarantee future performance.
        """
    )


# --------------------------------------------------
# DATASET
# --------------------------------------------------

elif page == "Dataset":

    st.header("NIFTY 50 Historical Dataset")

    st.write(
        "The dataset contains historical market prices and calculated indicators."
    )

    st.write(
        f"**Total historical records:** {len(data)}"
    )

    st.write(
        f"**Dataset starting date:** {data.index.min().date()}"
    )

    st.write(
        f"**Dataset ending date:** {data.index.max().date()}"
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        data.tail(50),
        use_container_width=True
    )

    # Download CSV
    csv = data.to_csv().encode("utf-8")

    st.download_button(
        label="Download Dataset as CSV",
        data=csv,
        file_name="nifty50_data.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "NIFTY 50 Market Analysis and Direction Prediction | "
    "Academic Data Science and Machine Learning Project"
)