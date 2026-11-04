

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from datetime import datetime, time
from zoneinfo import ZoneInfo

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


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="NIFTY 50 Daily Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY 50 Market Analysis and Daily Prediction")

st.markdown(
    """
    This application analyzes historical NIFTY 50 data, calculates
    technical indicators, and predicts the direction of the **next
    trading session** (UP or DOWN).

    **Prediction method:** The app uses completed daily market data.
    """
)

IST = ZoneInfo("Asia/Kolkata")
HISTORY_FILE = Path(__file__).with_name("prediction_history.csv")


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Market Overview",
        "Daily Prediction",
        "Historical Analysis",
        "Technical Indicators",
        "Machine Learning",
        "Dataset"
    ]
)

if st.sidebar.button("🔄 Refresh market data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(
    "Data source: Yahoo Finance | Symbol: ^NSEI"
)
st.sidebar.caption(
    "Daily data is checked after the market close. "
    "Yahoo Finance may publish data with a delay."
)


# ==================================================
# DOWNLOAD DAILY NIFTY 50 DATA
# ==================================================

@st.cache_data(ttl=3600)
def download_data():
    data = yf.download(
        "^NSEI",
        period="10y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if data.empty:
        raise ValueError(
            "Unable to download NIFTY 50 data. "
            "Check your internet connection and try again."
        )

    # Handle MultiIndex columns returned by some yfinance versions.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index = pd.to_datetime(data.index)
    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()

    required_columns = ["Open", "High", "Low", "Close"]
    missing = [column for column in required_columns if column not in data.columns]

    if missing:
        raise ValueError(f"Downloaded data is missing columns: {missing}")

    data = data.dropna(subset=required_columns)

    # Option A: do not use today's possibly incomplete daily candle
    # before 4:00 PM India time. The market normally closes at 3:30 PM;
    # the extra 30 minutes gives the data provider time to publish the bar.
    now_ist = datetime.now(IST)
    today_ist = now_ist.date()

    latest_date = pd.Timestamp(data.index[-1]).date()

    if latest_date == today_ist and now_ist.time() < time(16, 0):
        data = data.iloc[:-1]

    if data.empty:
        raise ValueError(
            "No completed daily market data is available yet. "
            "Please try again after 4:00 PM India time."
        )

    return data


# ==================================================
# TECHNICAL INDICATORS
# ==================================================

def calculate_indicators(data):
    data = data.copy()

    # Moving averages
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()

    # Daily return and volatility
    data["Daily_Return"] = data["Close"].pct_change() * 100
    data["Volatility"] = data["Daily_Return"].rolling(window=20).std()

    # RSI
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))
    data.loc[(avg_loss == 0) & (avg_gain > 0), "RSI"] = 100

    # MACD
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()

    data["MACD"] = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    # Bollinger Bands
    data["BB_Middle"] = data["Close"].rolling(window=20).mean()
    bb_std = data["Close"].rolling(window=20).std()
    data["BB_Upper"] = data["BB_Middle"] + (2 * bb_std)
    data["BB_Lower"] = data["BB_Middle"] - (2 * bb_std)

    # Additional returns
    data["Return_5"] = data["Close"].pct_change(5)
    data["Return_10"] = data["Close"].pct_change(10)

    return data


# ==================================================
# PREPARE MACHINE LEARNING DATASET
# ==================================================

FEATURES = [
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


def prepare_dataset(data):
    data = data.copy()

    # The target is the direction of the NEXT trading session.
    next_close = data["Close"].shift(-1)

    # 1 = UP, 0 = DOWN
    data["Target"] = np.where(
        next_close.notna(),
        (next_close > data["Close"]).astype(float),
        np.nan
    )

    feature_data = data.dropna(subset=FEATURES).copy()

    # The last row has no known next-day target, so it is not used for training.
    training_data = feature_data.dropna(subset=["Target"]).copy()

    X = training_data[FEATURES]
    y = training_data["Target"].astype(int)

    return feature_data, X, y


# ==================================================
# TRAIN AND EVALUATE MODELS
# ==================================================

def train_models(X, y):
    if len(X) < 100:
        raise ValueError("Not enough historical data to train the models.")

    split_index = int(len(X) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if y_train.nunique() < 2:
        raise ValueError(
            "The training data contains only one direction class. "
            "The model needs both UP and DOWN examples."
        )

    # Logistic Regression evaluation model
    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])
    logistic_model.fit(X_train, y_train)
    logistic_predictions = logistic_model.predict(X_test)

    # Random Forest evaluation model
    evaluation_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    evaluation_rf.fit(X_train, y_train)
    rf_predictions = evaluation_rf.predict(X_test)

    # Train a final Random Forest on ALL known labelled data.
    # This final model is used for the next-session prediction.
    final_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    final_rf.fit(X, y)

    return (
        logistic_model,
        evaluation_rf,
        final_rf,
        X_test,
        y_test,
        logistic_predictions,
        rf_predictions
    )


def evaluate_model(y_test, predictions):
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    }


# ==================================================
# PREDICTION HISTORY
# ==================================================

HISTORY_COLUMNS = [
    "Source Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Predicted Direction",
    "Probability UP (%)",
    "Probability DOWN (%)",
    "Actual Direction",
    "Result"
]


def _load_prediction_history():
    """Load history safely and standardize its columns and data types."""
    if HISTORY_FILE.exists():
        history = pd.read_csv(HISTORY_FILE)
    else:
        history = pd.DataFrame(columns=HISTORY_COLUMNS)

    for column in HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = np.nan

    history = history[HISTORY_COLUMNS].copy()
    history["Source Date"] = pd.to_datetime(
        history["Source Date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    for column in [
        "Predicted Direction", "Actual Direction", "Result"
    ]:
        history[column] = history[column].astype("object")

    history = history.dropna(subset=["Source Date"])
    history = history.drop_duplicates(subset=["Source Date"], keep="last")
    return history.reset_index(drop=True)


def update_prediction_history(data, prediction, probability):
    """
    Maintain one genuine recorded prediction per trading date.

    Existing history is preserved; missing dates are NOT reconstructed.
    OHLC values are backfilled wherever market data is available, and actual
    outcomes are resolved from the next trading session's closing price.
    """
    try:
        history = _load_prediction_history()

        market_data = data.copy().sort_index()
        market_data = market_data[~market_data.index.duplicated(keep="last")]
        market_data.index = pd.to_datetime(market_data.index)

        # Backfill OHLC values for older history rows created before the
        # OHLC columns were added. This uses the downloaded market dataset.
        market_dates = {
            pd.Timestamp(index).date().isoformat(): row
            for index, row in market_data.iterrows()
        }
        for row_index, row in history.iterrows():
            source_date = str(row["Source Date"])[:10]
            market_row = market_dates.get(source_date)
            if market_row is not None:
                for column in ["Open", "High", "Low", "Close"]:
                    history.at[row_index, column] = float(market_row[column])

        # Do not create predictions for dates when the application was not
        # actually recording them. Existing historical rows remain untouched.

        # Save or refresh the latest date's prediction using the current
        # full-history model. Do not overwrite an existing resolved outcome.
        latest_date = pd.Timestamp(market_data.index[-1]).date().isoformat()
        latest_row = market_data.iloc[-1]

        latest_values = {
            "Source Date": latest_date,
            "Open": float(latest_row["Open"]),
            "High": float(latest_row["High"]),
            "Low": float(latest_row["Low"]),
            "Close": float(latest_row["Close"]),
            "Predicted Direction": prediction,
            "Probability UP (%)": round(float(probability[1]) * 100, 2),
            "Probability DOWN (%)": round(float(probability[0]) * 100, 2)
        }

        existing = history.index[
            history["Source Date"].astype(str) == latest_date
        ].tolist()

        if existing:
            row_index = existing[-1]
            for key, value in latest_values.items():
                history.at[row_index, key] = value
            if pd.isna(history.at[row_index, "Actual Direction"]):
                history.at[row_index, "Result"] = "Pending"
        else:
            latest_values["Actual Direction"] = np.nan
            latest_values["Result"] = "Pending"
            history = pd.concat(
                [history, pd.DataFrame([latest_values])],
                ignore_index=True
            )

        # Resolve every prediction for which the next trading session is
        # now present in the downloaded market data.
        date_to_close = {
            pd.Timestamp(index).date().isoformat(): float(close)
            for index, close in market_data["Close"].items()
        }
        ordered_dates = sorted(date_to_close)

        for row_index, row in history.iterrows():
            source_date = str(row["Source Date"])[:10]
            if source_date not in date_to_close:
                continue

            source_position = ordered_dates.index(source_date)
            if source_position + 1 >= len(ordered_dates):
                history.at[row_index, "Actual Direction"] = np.nan
                history.at[row_index, "Result"] = "Pending"
                continue

            next_date = ordered_dates[source_position + 1]
            actual = (
                "UP"
                if date_to_close[next_date] > date_to_close[source_date]
                else "DOWN"
            )
            predicted = str(row["Predicted Direction"]).strip().upper()

            history.at[row_index, "Actual Direction"] = actual
            history.at[row_index, "Result"] = (
                "Correct" if predicted == actual else "Incorrect"
            )

        history = history.sort_values("Source Date", ascending=False)
        history = history[HISTORY_COLUMNS].reset_index(drop=True)

        # Write atomically to reduce the chance of a partially written CSV.
        temporary_file = HISTORY_FILE.with_suffix(".tmp")
        history.to_csv(temporary_file, index=False)
        temporary_file.replace(HISTORY_FILE)

        return history, None

    except Exception as error:
        return pd.DataFrame(columns=HISTORY_COLUMNS), str(error)


# ==================================================
# LOAD DATA AND TRAIN MODELS
# ==================================================

try:
    with st.spinner("Downloading the latest available completed NIFTY 50 data..."):
        raw_data = download_data()
        data = calculate_indicators(raw_data)
        feature_data, X, y = prepare_dataset(data)

    with st.spinner("Training machine-learning models..."):
        (
            logistic_model,
            evaluation_rf,
            final_rf,
            X_test,
            y_test,
            logistic_predictions,
            rf_predictions
        ) = train_models(X, y)

    latest_features = feature_data[FEATURES].iloc[[-1]]
    latest_prediction = int(final_rf.predict(latest_features)[0])
    latest_probability = final_rf.predict_proba(latest_features)[0]

except Exception as error:
    st.error(f"Application error: {error}")
    st.info(
        "Check your internet connection and installed Python packages. "
        "Then use the Refresh market data button."
    )
    st.stop()


latest = data.iloc[-1]
previous = data.iloc[-2]
latest_date = pd.Timestamp(data.index[-1]).date()
latest_close = float(latest["Close"])

# Update prediction history on every app visit, regardless of the selected page.
# This prevents history from depending on whether the user opens Daily Prediction.
history, history_error = update_prediction_history(
    data,
    "UP" if latest_prediction == 1 else "DOWN",
    latest_probability
)


# ==================================================
# MARKET OVERVIEW
# ==================================================

if page == "Market Overview":
    st.header("Market Overview")

    change = float(latest["Close"] - previous["Close"])
    percentage_change = (
        change / float(previous["Close"])
    ) * 100

    st.info(
        f"Latest completed market session: **{latest_date.strftime('%d %B %Y')}**"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Open", f"{float(latest['Open']):,.2f}")
    col2.metric("High", f"{float(latest['High']):,.2f}")
    col3.metric("Low", f"{float(latest['Low']):,.2f}")
    col4.metric(
        "Close",
        f"{latest_close:,.2f}",
        f"{change:+,.2f} ({percentage_change:+.2f}%)"
    )

    col5, col6, col7 = st.columns(3)
    col5.metric("RSI", f"{float(latest['RSI']):.2f}")
    col6.metric("MA20", f"{float(latest['MA20']):,.2f}")
    col7.metric("MA50", f"{float(latest['MA50']):,.2f}")

    st.subheader("NIFTY 50 Closing Price")

    chart_data = data.tail(250)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["Close"],
            name="NIFTY 50 Close"
        )
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Index Value",
        height=480
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest Daily OHLC Data")
    ohlc = data[["Open", "High", "Low", "Close"]].tail(10).copy()
    ohlc.index = ohlc.index.date
    st.dataframe(ohlc, use_container_width=True)

    st.caption(
        "OHLC means Open, High, Low and Close. Values are the latest "
        "available daily values from Yahoo Finance."
    )


# ==================================================
# DAILY PREDICTION
# ==================================================

elif page == "Daily Prediction":
    st.header("Next Trading Session Prediction")

    st.write(
        "The model uses the latest completed trading session and predicts "
        "whether the next trading session's closing value may be higher "
        "or lower than the current session's closing value."
    )

    st.info(
        f"Prediction based on completed data up to: "
        f"**{latest_date.strftime('%d %B %Y')}**"
    )

    if latest_prediction == 1:
        st.success("Model prediction for the next trading session: UP 📈")
    else:
        st.error("Model prediction for the next trading session: DOWN 📉")

    prob_col1, prob_col2 = st.columns(2)
    prob_col1.metric(
        "Probability of UP",
        f"{latest_probability[1] * 100:.2f}%"
    )
    prob_col2.metric(
        "Probability of DOWN",
        f"{latest_probability[0] * 100:.2f}%"
    )

    st.subheader("Market data used by the model")

    data_col1, data_col2, data_col3, data_col4 = st.columns(4)
    data_col1.metric("Open", f"{float(latest['Open']):,.2f}")
    data_col2.metric("High", f"{float(latest['High']):,.2f}")
    data_col3.metric("Low", f"{float(latest['Low']):,.2f}")
    data_col4.metric("Close", f"{latest_close:,.2f}")

    st.subheader("Important indicators")
    ind_col1, ind_col2, ind_col3 = st.columns(3)
    ind_col1.metric("RSI", f"{float(latest['RSI']):.2f}")
    ind_col2.metric("MACD", f"{float(latest['MACD']):.2f}")
    ind_col3.metric("20-day volatility", f"{float(latest['Volatility']):.2f}")

    st.subheader("Daily Prediction History")

    if history_error:
        st.warning(
            "Prediction was generated, but the history file could not be "
            f"updated: {history_error}"
        )
    elif not history.empty:
        # Show only the latest 10 trading-date records in the dashboard.
        # The complete history remains saved in prediction_history.csv.
        display_history = history.sort_values(
            "Source Date", ascending=False
        ).head(10).copy()
        for price_column in ["Open", "High", "Low", "Close"]:
            display_history[price_column] = pd.to_numeric(
                display_history[price_column], errors="coerce"
            ).round(2)
        st.dataframe(display_history, use_container_width=True)
        st.caption(
            "Showing the latest 10 trading days only. All historical records "
            "are retained in prediction_history.csv. Predictions are recorded "
            "when the application runs; missing dates are not reconstructed. "
            "Actual direction is calculated using the next trading session's "
            "closing price. Overall model-history records are preserved."
        )
    else:
        st.write("Prediction history will appear here.")

    st.warning(
        "This is an experimental machine-learning prediction, not a "
        "guaranteed forecast or financial recommendation. Market prices "
        "can move unexpectedly."
    )


# ==================================================
# HISTORICAL ANALYSIS
# ==================================================

elif page == "Historical Analysis":
    st.header("Historical Market Analysis")

    chart_data = data.tail(500)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["Close"],
            name="Closing Price"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["MA20"],
            name="20-Day Moving Average"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["MA50"],
            name="50-Day Moving Average"
        )
    )
    fig.update_layout(
        title="NIFTY 50 Price and Moving Averages",
        xaxis_title="Date",
        yaxis_title="Index Value",
        height=520
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Returns")
    returns_fig = go.Figure()
    returns_fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["Daily_Return"],
            name="Daily Returns"
        )
    )
    returns_fig.update_layout(
        title="Daily Percentage Returns",
        xaxis_title="Date",
        yaxis_title="Return (%)"
    )
    st.plotly_chart(returns_fig, use_container_width=True)

    st.subheader("Market Volatility")
    st.line_chart(chart_data["Volatility"])


# ==================================================
# TECHNICAL INDICATORS
# ==================================================

elif page == "Technical Indicators":
    st.header("Technical Indicators")

    col1, col2, col3 = st.columns(3)
    col1.metric("RSI", f"{float(latest['RSI']):.2f}")
    col2.metric("MACD", f"{float(latest['MACD']):.2f}")
    col3.metric("Volatility", f"{float(latest['Volatility']):.2f}")

    st.subheader("Relative Strength Index (RSI)")
    rsi_fig = go.Figure()
    rsi_fig.add_trace(
        go.Scatter(x=data.index, y=data["RSI"], name="RSI")
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
    rsi_fig.update_layout(height=400, yaxis_title="RSI")
    st.plotly_chart(rsi_fig, use_container_width=True)

    st.subheader("MACD Indicator")
    macd_fig = go.Figure()
    macd_fig.add_trace(
        go.Scatter(x=data.index, y=data["MACD"], name="MACD")
    )
    macd_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD_Signal"],
            name="Signal Line"
        )
    )
    macd_fig.update_layout(height=400)
    st.plotly_chart(macd_fig, use_container_width=True)

    st.subheader("Bollinger Bands")
    bb_fig = go.Figure()
    bb_fig.add_trace(
        go.Scatter(x=data.index, y=data["Close"], name="Closing Price")
    )
    bb_fig.add_trace(
        go.Scatter(x=data.index, y=data["BB_Upper"], name="Upper Band")
    )
    bb_fig.add_trace(
        go.Scatter(x=data.index, y=data["BB_Lower"], name="Lower Band")
    )
    bb_fig.update_layout(height=450)
    st.plotly_chart(bb_fig, use_container_width=True)


# ==================================================
# MACHINE LEARNING
# ==================================================

elif page == "Machine Learning":
    st.header("Machine Learning Model Evaluation")

    st.write(
        """
        The project compares two classification algorithms:

        1. Logistic Regression
        2. Random Forest Classifier

        The first 80% of the labelled historical records are used for
        training and the later 20% are used for testing. The final
        Random Forest used for the next-session prediction is then
        retrained on all known labelled records.
        """
    )

    logistic_results = evaluate_model(y_test, logistic_predictions)
    rf_results = evaluate_model(y_test, rf_predictions)

    results_df = pd.DataFrame(
        [logistic_results, rf_results],
        index=["Logistic Regression", "Random Forest"]
    )

    st.subheader("Model Performance Comparison")
    st.dataframe(
        results_df.style.format("{:.2%}"),
        use_container_width=True
    )

    st.subheader("Accuracy Comparison")
    accuracy_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [
            logistic_results["Accuracy"],
            rf_results["Accuracy"]
        ]
    })
    st.bar_chart(accuracy_df.set_index("Model"))

    st.subheader("Random Forest Confusion Matrix")
    cm = confusion_matrix(y_test, rf_predictions, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual DOWN", "Actual UP"],
        columns=["Predicted DOWN", "Predicted UP"]
    )
    st.dataframe(cm_df)

    st.caption(
        "Historical test performance does not guarantee future prediction "
        "accuracy. The model is an academic prototype and has not been "
        "validated as a trading strategy."
    )


# ==================================================
# DATASET
# ==================================================

elif page == "Dataset":
    st.header("NIFTY 50 Historical Dataset")

    st.write(f"**Total historical records:** {len(data)}")
    st.write(f"**Dataset starting date:** {data.index.min().date()}")
    st.write(f"**Latest available date:** {data.index.max().date()}")

    st.subheader("Dataset Preview")
    st.dataframe(
        data.tail(50),
        use_container_width=True
    )

    csv = data.to_csv().encode("utf-8")
    st.download_button(
        label="Download Dataset as CSV",
        data=csv,
        file_name="nifty50_data.csv",
        mime="text/csv"
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")
st.caption(
    "NIFTY 50 Market Analysis and Direction Prediction | "
    "Academic Data Science and Machine Learning Project"
)
