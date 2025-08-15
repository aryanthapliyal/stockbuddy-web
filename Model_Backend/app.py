from flask import Flask, request, jsonify

from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import threading
import time
from datetime import datetime, timedelta
import json
import model as stock_model
import sys
import requests
import traceback
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import xgboost as xgb

app = Flask(__name__)
CORS(app)

# Validate TensorFlow setup
def validate_tensorflow():
    """Check if TensorFlow and GPU are properly configured and working"""
    try:
        print("TensorFlow version:", tf.__version__)
        
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"TensorFlow detected {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  - {gpu}")
            
            # Try to create a small model and run a simple prediction
            try:
                with tf.device('/GPU:0'):
                    test_model = Sequential([
                        LSTM(10, input_shape=(5, 3)),
                        Dense(1)
                    ])
                    test_model.compile(optimizer='adam', loss='mse')
                    
                    # Generate dummy data
                    dummy_data = np.random.random((10, 5, 3))
                    
                    # Run prediction
                    result = test_model.predict(dummy_data)
                    print("GPU test successful - TensorFlow is working correctly")
                    return True, "GPU available and working correctly"
            except Exception as e:
                print(f"GPU test failed: {str(e)}")
                
                # Try CPU fallback
                print("Trying CPU fallback...")
                with tf.device('/CPU:0'):
                    test_model = Sequential([
                        LSTM(10, input_shape=(5, 3)),
                        Dense(1)
                    ])
                    test_model.compile(optimizer='adam', loss='mse')
                    
                    # Generate dummy data
                    dummy_data = np.random.random((10, 5, 3))
                    
                    # Run prediction
                    result = test_model.predict(dummy_data)
                    print("CPU test successful - TensorFlow is working but no GPU acceleration")
                    return True, "No GPU available, but CPU mode is working"
        else:
            print("No GPU detected, running in CPU mode")
            
            # Try CPU model
            test_model = Sequential([
                LSTM(10, input_shape=(5, 3)),
                Dense(1)
            ])
            test_model.compile(optimizer='adam', loss='mse')
            
            # Generate dummy data
            dummy_data = np.random.random((10, 5, 3))
            
            # Run prediction
            result = test_model.predict(dummy_data)
            print("CPU test successful - TensorFlow is working in CPU mode")
            return True, "Running in CPU mode"
            
    except Exception as e:
        print(f"TensorFlow validation failed: {str(e)}")
        return False, f"TensorFlow error: {str(e)}"

# Run validation on startup
tf_status, tf_message = validate_tensorflow()
if not tf_status:
    print(f"WARNING: TensorFlow validation failed: {tf_message}")
    print("Prediction functionality may not work correctly")
else:
    print(f"TensorFlow validation: {tf_message}")

# Dictionary to store running prediction tasks
prediction_tasks = {}

class PredictionTask:
    def __init__(self, user_id, symbol, days_ahead):
        self.user_id = user_id
        self.symbol = symbol
        self.days_ahead = days_ahead
        self.progress = 0
        self.status = "pending"
        self.result = None
        self.sentiment_result = None
        self.thread = None
        self.stop_requested = False
        self.stop_acknowledged = False
        # Generate a truly unique task ID
        timestamp = int(time.time() * 1000)  # Millisecond precision
        random_suffix = os.urandom(4).hex()  # Add random suffix
        self.task_id = f"{user_id}_{symbol}_{timestamp}_{random_suffix}"

    def run(self):
        self.thread = threading.Thread(target=self._run_prediction)
        self.thread.daemon = True  # Make thread a daemon so it exits when main thread does
        self.thread.start()
        return self.task_id
    
    def is_stop_requested(self):
        """Callback function to check if stop has been requested"""
        if self.stop_requested and not self.stop_acknowledged:
            self.stop_acknowledged = True
            self.status = "stopped"
            return True
        return self.stop_requested

    def _run_prediction(self):
        try:
            print(f"Starting prediction task for {self.symbol} (ID: {self.task_id})")
            self.status = "running"
            self.progress = 10
            
            # Fetch stock data (using compact data for faster processing)
            print(f"Fetching historical data for {self.symbol}...")
            try:
                data = stock_model.fetch_stock_data(self.symbol, outputsize="compact")
                print(f"Successfully fetched data for {self.symbol}, shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            except Exception as e:
                print(f"ERROR: Failed to fetch stock data for {self.symbol}: {str(e)}")
                self.status = "failed"
                self.result = {"error": f"Could not fetch data for {self.symbol}: {str(e)}"}
                return
            
            if data is None:
                print(f"ERROR: Data is None for {self.symbol}")
                self.status = "failed"
                self.result = {"error": f"Could not fetch data for {self.symbol}"}
                return
            
            # Check for stop request
            if self.stop_requested:
                self.status = "stopped"
                print(f"Prediction for {self.symbol} stopped after data fetch")
                return
            
            # Filter to ensure we only have enough data
            if len(data) < 60:  # Increased minimum data requirement for more stable training
                print(f"ERROR: Insufficient data for {self.symbol}, got {len(data)} rows, need at least 60")
                self.status = "failed"
                self.result = {"error": f"Insufficient data available for {self.symbol}"}
                return
                
            # Store the last closing price for reference
            try:
                if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
                    last_actual_close = float(data['Close'].iloc[-1])
                    last_date = data.index[-1]
                else:
                    last_actual_close = float(data.iloc[-1, 0])  # Assume first column is Close
                    last_date = data.index[-1]
                    
                print(f"Latest closing price for {self.symbol}: ${last_actual_close:.2f} on {last_date.strftime('%Y-%m-%d')}")
            except Exception as e:
                print(f"ERROR: Could not extract closing price: {str(e)}")
                self.status = "failed"
                self.result = {"error": f"Error processing data for {self.symbol}: {str(e)}"}
                return
                
            self.progress = 20
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Fetch and analyze sentiment
            try:
                print(f"Fetching news headlines for {self.symbol}...")
                headlines = stock_model.fetch_finnhub_news(self.symbol)
                print(f"Got {len(headlines)} headlines for {self.symbol}")
                self.progress = 30
                if self.stop_requested:
                    self.status = "stopped"
                    return
                
                print(f"Analyzing sentiment for {self.symbol}...")
                sentiment_results, sentiment_totals = stock_model.analyze_sentiment(headlines)
                sentiment_summary = stock_model.generate_sentiment_summary(sentiment_totals, headlines, self.symbol)
                self.sentiment_result = {
                    "totals": sentiment_totals,
                    "summary": sentiment_summary
                }
                print(f"Sentiment analysis complete for {self.symbol}")
            except Exception as e:
                print(f"ERROR in sentiment analysis: {str(e)}")
                # Continue even if sentiment fails
                self.sentiment_result = {
                    "totals": {"positive": 0, "negative": 0, "neutral": 0},
                    "summary": f"Unable to analyze sentiment for {self.symbol}: {str(e)}"
                }
            
            self.progress = 40
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Preprocess data for prediction with technical indicators
            try:
                print(f"Preprocessing data for {self.symbol}...")
                scaled_data, scaler = stock_model.preprocess_data(data)
                print(f"Data preprocessing complete, scaled shape: {scaled_data.shape}")
                
                # Use higher time_step for better sequence learning
                time_step = 45
                print(f"Creating sequences with time_step={time_step}, data shape={scaled_data.shape}")
                X, y = stock_model.create_sequences(scaled_data, time_step)
                print(f"Created {len(X)} sequences with shape X: {X.shape}, y: {y.shape}")
            except Exception as e:
                print(f"ERROR: Failed during data preprocessing: {str(e)}")
                self.status = "failed"
                self.result = {"error": f"Error preprocessing data for {self.symbol}: {str(e)}"}
                return
            
            # Check if we have any training data
            if len(X) == 0:
                print(f"ERROR: No training sequences created for {self.symbol}")
                self.status = "failed"
                self.result = {"error": f"Could not create training sequences for {self.symbol}"}
                return
                
            self.progress = 50
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Train models with improved architecture
            # Use 80% for training with more data
            try:
                train_size = int(len(X) * 0.8)
                if train_size == 0:
                    print(f"ERROR: Not enough data points after splitting for {self.symbol}")
                    self.status = "failed"
                    self.result = {"error": f"Not enough data points to train model for {self.symbol}"}
                    return
                    
                X_train, y_train = X[:train_size], y[:train_size]
                print(f"Training LSTM with {len(X_train)} samples and {X_train.shape[2]} features")
                
                # Set progress update for extended training
                # Adjust progress reporting to account for longer training time
                self.progress = 55
                
                # Use improved LSTM model with stop callback
                print(f"Starting LSTM model training for {self.symbol}...")
                lstm_model = stock_model.train_lstm(X_train, y_train, time_step, self.is_stop_requested)
                print(f"LSTM model training complete for {self.symbol}")
            except Exception as e:
                print(f"ERROR: Failed during LSTM model training: {str(e)}")
                self.status = "failed"
                self.result = {"error": f"Error training LSTM model for {self.symbol}: {str(e)}"}
                return
            
            # Check if training was stopped
            if self.stop_requested:
                self.status = "stopped"
                print(f"Prediction for {self.symbol} stopped after LSTM training")
                return
                
            self.progress = 75
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Create residuals for improved XGBoost
            try:
                print(f"Calculating residuals for XGBoost training...")
                lstm_preds = lstm_model.predict(X_train, verbose=0).flatten()
                residuals = y_train - lstm_preds
                
                # Train XGBoost with stop callback
                print(f"Starting XGBoost model training for {self.symbol}...")
                xgb_model = stock_model.train_xgboost(
                    X_train.reshape(X_train.shape[0], -1), 
                    residuals, 
                    self.is_stop_requested
                )
                print(f"XGBoost model training complete for {self.symbol}")
                
                # Check if XGBoost was stopped
                if self.stop_requested or xgb_model is None:
                    self.status = "stopped"
                    print(f"Prediction for {self.symbol} stopped after XGBoost training")
                    return
            except Exception as e:
                print(f"ERROR in XGBoost training: {str(e)}")
                # Continue with LSTM-only predictions if XGBoost fails
                xgb_model = None
                
            self.progress = 90
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Make predictions with stop callback
            try:
                print(f"Making predictions for {self.symbol} {self.days_ahead} days ahead...")
                predictions = stock_model.predict_stock_price(
                    lstm_model, 
                    xgb_model, 
                    scaled_data, 
                    scaler, 
                    time_step, 
                    self.days_ahead,
                    self.is_stop_requested
                )
                
                # Check if predictions were stopped
                if self.stop_requested or predictions is None:
                    self.status = "stopped"
                    print(f"Prediction for {self.symbol} stopped during prediction generation")
                    return
            except Exception as e:
                print(f"Error making predictions: {str(e)}")
                self.status = "failed"
                self.result = {"error": f"Failed to generate predictions: {str(e)}"}
                return
            
            self.progress = 95
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Create prediction results with proper business day handling
            future_dates = []
            
            # Generate proper future dates (trading days only)
            for i in range(1, self.days_ahead + 1):
                if self.stop_requested:
                    break
                next_date = last_date + timedelta(days=i)
                # Skip weekends
                while next_date.weekday() > 4:  # 5=Saturday, 6=Sunday
                    next_date = next_date + timedelta(days=1)
                future_dates.append(next_date)
            
            # If stopped, exit early
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Remove any duplicate dates
            unique_future_dates = []
            unique_date_strs = set()
            
            for date in future_dates:
                date_str = date.strftime("%Y-%m-%d")
                if date_str not in unique_date_strs:
                    unique_date_strs.add(date_str)
                    unique_future_dates.append(date)
            
            # Ensure we have enough dates, add more if needed
            while len(unique_future_dates) < len(predictions) and not self.stop_requested:
                next_date = unique_future_dates[-1] + timedelta(days=1)
                while next_date.weekday() > 4:  # Skip weekends
                    next_date = next_date + timedelta(days=1)
                if next_date.strftime("%Y-%m-%d") not in unique_date_strs:
                    unique_future_dates.append(next_date)
                    unique_date_strs.add(next_date.strftime("%Y-%m-%d"))
            
            # If stopped, exit early
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Take only what we need
            unique_future_dates = unique_future_dates[:len(predictions)]
            
            # Create the prediction data with formatted dates
            prediction_data = []
            for i in range(min(len(unique_future_dates), len(predictions))):
                predicted_price = float(predictions[i][0])
                
                # Calculate percent change from last actual closing price
                percent_change = ((predicted_price - last_actual_close) / last_actual_close) * 100
                
                prediction_data.append({
                    "date": unique_future_dates[i].strftime("%Y-%m-%d"),
                    "price": round(predicted_price, 2),
                    "change": round(percent_change, 2)
                })
            
            # Include last actual closing price and date in the result
            self.result = {
                "symbol": self.symbol,
                "lastActualClose": {
                    "date": last_date.strftime("%Y-%m-%d"),
                    "price": round(last_actual_close, 2)
                },
                "predictions": prediction_data,
                "sentiment": self.sentiment_result,
                "tableDisplay": True  # Flag to indicate frontend should display a table instead of a graph
            }
            self.progress = 100
            self.status = "completed"
            
        except Exception as e:
            self.status = "failed"
            self.result = {"error": str(e)}
            print(f"Error in prediction: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def start_prediction():
    try:
        data = request.json
        print(f"Request received at /api/predict: {data}")
        
        if not data:
            print("Error: Invalid or missing request body")
            return jsonify({"error": "Invalid or missing request body"}), 400
            
        user_id = data.get('userId')
        symbol = data.get('symbol')
        days_ahead = int(data.get('daysAhead', 5))
        
        print(f"Extracted parameters: userId={user_id}, symbol={symbol}, daysAhead={days_ahead}")
        
        if not user_id or not symbol:
            print(f"Error: Missing required parameters (userId={user_id}, symbol={symbol})")
            return jsonify({"error": "Missing required parameters (userId or symbol)"}), 400
        
        # Validate the symbol format
        if not isinstance(symbol, str) or len(symbol) > 10:
            print(f"Error: Invalid symbol format: {symbol}")
            return jsonify({"error": f"Invalid symbol format: {symbol}"}), 400
            
        # Check TensorFlow status
        if not tf_status:
            print(f"Error: TensorFlow unavailable - {tf_message}")
            return jsonify({
                "error": f"Prediction service unavailable: {tf_message}",
                "tf_status": tf_message
            }), 503
        
        # Create and start the prediction task
        print(f"Starting new prediction task for user {user_id}, symbol {symbol}, days {days_ahead}")
        task = PredictionTask(user_id, symbol, days_ahead)
        task_id = task.run()
        prediction_tasks[task_id] = task
        
        return jsonify({
            "taskId": task_id,
            "status": "pending",
            "message": f"Prediction started for {symbol}"
        })
    except ValueError as e:
        # Handle value errors like invalid days_ahead
        print(f"Value error in prediction request: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Log the full exception details
        print(f"Critical error starting prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "error": "Failed to start prediction",
            "details": str(e)
        }), 500

@app.route('/api/predict/status/<task_id>', methods=['GET'])
def prediction_status(task_id):
    try:
        task = prediction_tasks.get(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404
        
        try:
            # Ensure result is valid JSON-serializable before returning
            if task.status == "completed" and task.result:
                # Validate result structure
                if isinstance(task.result, dict):
                    # Ensure predictions are properly formatted
                    if "predictions" in task.result and isinstance(task.result["predictions"], list):
                        # Make sure each prediction is valid
                        for pred in task.result["predictions"]:
                            if not isinstance(pred, dict) or "date" not in pred or "price" not in pred:
                                # Fix malformed prediction
                                print(f"Found malformed prediction: {pred}")
                                task.status = "failed"
                                task.result = {"error": "Malformed prediction data"}
                                break
                    else:
                        # Missing predictions
                        task.status = "failed"
                        task.result = {"error": "Missing prediction data"}
                else:
                    # Invalid result type
                    task.status = "failed"
                    task.result = {"error": "Invalid result format"}
            
            return jsonify({
                "taskId": task_id,
                "status": task.status,
                "progress": task.progress,
                "result": task.result if task.status == "completed" else None
            })
        except Exception as e:
            print(f"Error generating prediction status response: {str(e)}")
            # Return a simplified response that won't cause JSON serialization issues
            return jsonify({
                "taskId": task_id,
                "status": "error",
                "progress": task.progress,
                "error": str(e)
            })
    except Exception as e:
        print(f"Critical error in prediction status: {str(e)}")
        return jsonify({
            "taskId": task_id,
            "status": "error",
            "error": "Server error"
        }), 500

@app.route('/api/predict/stop/<task_id>', methods=['POST'])
def stop_prediction(task_id):
    task = prediction_tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    # Set the stop flag
    task.stop_requested = True
    
    # If task is running, make sure it gets stopped
    if task.thread and task.thread.is_alive():
        # Set the status immediately so client knows it's being stopped
        task.status = "stopping"
        
        # Log the stop request
        print(f"Stop requested for task {task_id} ({task.symbol})")
        
        # Wait up to 2 seconds to see if the task acknowledges the stop request
        stop_wait_start = time.time()
        while time.time() - stop_wait_start < 2:
            if task.stop_acknowledged:
                task.status = "stopped"
                break
            time.sleep(0.1)
    else:
        # If thread isn't running, just mark as stopped
        task.status = "stopped"
    
    # Return detailed status
    return jsonify({
        "taskId": task_id,
        "status": task.status,
        "symbol": task.symbol,
        "progress": task.progress,
        "stopRequested": task.stop_requested,
        "stopAcknowledged": task.stop_acknowledged
    })

@app.route('/api/predict/sentiment/<symbol>', methods=['GET'])
def get_sentiment(symbol):
    try:
        headlines = stock_model.fetch_finnhub_news(symbol)
        sentiment_results, sentiment_totals = stock_model.analyze_sentiment(headlines)
        sentiment_summary = stock_model.generate_sentiment_summary(sentiment_totals, headlines, symbol)
        
        return jsonify({
            "symbol": symbol,
            "sentiment": {
                "totals": sentiment_totals,
                "summary": sentiment_summary,
                "period": 28  # Updated from 14 to 28 days
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/diagnose', methods=['GET'])
def diagnose():
    """Diagnostic endpoint to check if prediction service is working correctly"""
    try:
        # Test environment setup
        env_info = {
            "python_version": sys.version,
            "tensorflow_version": tf.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "xgboost_version": xgb.__version__
        }
        
        # Test API connectivity
        api_status = {}
        try:
            # Test Alpha Vantage API
            symbol = "AAPL"  # Use a common symbol for testing
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": stock_model.ALPHAVANTAGE_API_KEY,
                "outputsize": "compact",
                "datatype": "json"
            }
            response = requests.get(url, params=params)
            api_status["alpha_vantage"] = {
                "status_code": response.status_code,
                "has_data": "Time Series (Daily)" in response.json(),
                "error": response.json().get("Error Message") or response.json().get("Note") if "Time Series (Daily)" not in response.json() else None
            }
        except Exception as e:
            api_status["alpha_vantage"] = {"error": str(e)}
            
        try:
            # Test Finnhub API
            headers = {'X-Finnhub-Token': stock_model.FINNHUB_API_KEY}
            response = requests.get(f'https://finnhub.io/api/v1/news?category=general', headers=headers)
            api_status["finnhub"] = {
                "status_code": response.status_code,
                "has_data": len(response.json()) > 0,
                "error": None if response.status_code == 200 else str(response.text)
            }
        except Exception as e:
            api_status["finnhub"] = {"error": str(e)}
            
        # Test model creation
        model_status = {}
        try:
            # Create small test data to verify model creation works
            test_data = np.random.rand(100, 10)
            test_scaler = MinMaxScaler()
            test_data[:, 0] = test_scaler.fit_transform(np.arange(100).reshape(-1, 1)).flatten()
            
            # Test sequence creation
            X, y = stock_model.create_sequences(test_data, time_step=10)
            model_status["sequence_creation"] = {
                "success": len(X) > 0 and len(y) > 0,
                "X_shape": X.shape,
                "y_shape": y.shape
            }
            
            # Test basic model initialization
            try:
                if len(X) > 0:
                    model = Sequential([
                        LSTM(32, input_shape=(10, 10), return_sequences=False),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    model_status["model_init"] = {"success": True}
                else:
                    model_status["model_init"] = {"success": False, "reason": "No test sequences created"}
            except Exception as e:
                model_status["model_init"] = {"success": False, "error": str(e)}
                
        except Exception as e:
            model_status["error"] = str(e)
            
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "status": "OK",
            "environment": env_info,
            "api_status": api_status,
            "model_status": model_status
        })
    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001) 