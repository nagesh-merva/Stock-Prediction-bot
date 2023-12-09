import gradio as gr
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

def stock_predict(ticker, days):
    try:
        stock_data = yf.download(ticker, period="1y", interval="1d")
        if stock_data.empty:
            print(f"No data found for the stock {ticker}.")
            return None

        stock_data = stock_data.reset_index()
        stock_data["Day"] = stock_data.index

        X = stock_data[["Day"]]
        y = stock_data["Close"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        model = LinearRegression()
        model.fit(X_train, y_train)

        future_days = np.array([i for i in range(len(stock_data), len(stock_data) + int(days))]).reshape(-1, 1)

        future_prediction = model.predict(future_days)

        future_dates = pd.date_range(start=stock_data["Date"].iloc[-1], periods=int(days), freq='B').date.tolist()

        prediction_df = pd.DataFrame({"Date": future_dates, "Predicted Close Price": future_prediction.flatten()})

        return prediction_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

interface = gr.Interface(
    fn=stock_predict,
    inputs=["text", "number"],
    outputs="dataframe",
    title="Stock Price Prediction",
    description="Enter a valid stock symbol as per Yahoo finance (e.g., M&M.NS for Mahindra ,TCS.NS for Tata consultancy services) and the number of days for prediction.",
    examples=[["TCS.NS", 7], ["M&M.NS", 5]]
)

interface.launch()



