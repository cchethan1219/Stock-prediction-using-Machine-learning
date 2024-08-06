import sys
# Check and set UTF-8 encoding for stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
from flask import Flask, render_template, request, redirect, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import json
import threading

app = Flask(__name__)

# Function to fetch actual historical data
def fetch_actual_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error fetching actual data for {ticker}: {str(e)}")
        return None

# Function to fetch predictions for multiple tickers
def fetch_predictions(tickers, start_date_test, end_date_test):
    all_predictions = {}

    for ticker in tickers:
        try:
            # Fetching stock data from Yahoo Finance
            data = yf.download(ticker, start=start_date_test, end=end_date_test)
            data_test = data.loc[start_date_test:end_date_test]

            # Load the trained model for the current ticker
            model_file = f'C:/Users/cchet/Desktop/STOCK PREDICTION/models'
            scaler_file = f'C:/Users/cchet/Desktop/STOCK PREDICTION/models'

            with open(model_file, 'rb') as file:
                regressor = pickle.load(file)

            with open(scaler_file, 'rb') as file:
                sc = pickle.load(file)

            # Preparing the test set
            real_stock_price = data_test[['Open']].values  # Assuming 'Open' price is the feature
            
            # Getting the predicted stock price
            data_train = yf.download(ticker, start='2014-01-01', end='2023-12-31')
            dataset_total = pd.concat((data_train['Open'], data_test['Open']), axis=0)
            inputs = dataset_total[len(dataset_total) - len(data_test) - 60:].values
            inputs = inputs.reshape(-1, 1)
            inputs = sc.transform(inputs)

            X_test = []
            for i in range(60, 60 + len(data_test)):
                X_test.append(inputs[i-60:i, 0])
            X_test = np.array(X_test)

            # Ensure X_test has the correct shape before reshaping
            if X_test.shape[0] > 0 and X_test.shape[1] == 60:
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            else:
                print(f"Error: X_test for {ticker} does not have the expected shape.")

            predicted_stock_price = regressor.predict(X_test)
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)

            # Store predictions in a dictionary with dates
            all_predictions[ticker] = {
                "dates": data_test.index.strftime('%Y-%m-%d').tolist(),  # Convert dates to string format
                "actual": data_test['Open'].tolist(),
                "predicted": predicted_stock_price.tolist()
            }

        except Exception as e:
            print(f"Error fetching predictions for {ticker}: {str(e)}")

    return all_predictions

predictions = {}
predictions_ready = False

def run_prediction(tickers, start_date_test, end_date_test):
    global predictions, predictions_ready

    try:
        # Fetch predictions for multiple stocks
        predictions = fetch_predictions(tickers, start_date_test, end_date_test)

        # Set flag indicating predictions are ready
        predictions_ready = True

    except Exception as e:
        print(f"Error in prediction process: {str(e)}")

# Route to fetch predictions
@app.route('/fetch_predictions')
def fetch_predictions_route():
    global predictions, predictions_ready

    if predictions_ready:
        return jsonify(predictions)
    else:
        return jsonify({})  # Return empty object if predictions are not ready yet

# Route to render index.html for the main interface
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and initiate prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve tickers and date range from the form
        tickers = request.form.getlist('ticker')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')

        # Run the prediction process in a separate thread
        prediction_thread = threading.Thread(target=run_prediction, args=(tickers, start_date, end_date))
        prediction_thread.start()

        # Redirect to loading page while predictions are generated
        return redirect('/loading')

# Route to display predictions
@app.route('/display')
def display():
    global predictions, predictions_ready

    if predictions_ready:
        # Print or log predictions
        print(json.dumps(predictions))

        return render_template('display.html', predictions=json.dumps(predictions))
    else:
        return "Predictions are still being computed. Please wait and refresh the page."

# Route to loading page
@app.route('/loading')
def loading():
    return render_template('loading.html')

if __name__ == '__main__':
    app.run(debug=True)
