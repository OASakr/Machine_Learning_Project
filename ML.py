#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                            accuracy_score, r2_score, precision_score, 
                            recall_score, f1_score)
from sklearn.neural_network import MLPClassifier
import pickle
import os
import io

# Global variables
model = None
scaler = None
df_new = None
x_train, x_test, y_train, y_test = None, None, None, None

def load_data_from_file(file):
    global df_new, x_train, x_test, y_train, y_test
    
    try:
        if file is None:
            return "Please upload a file!", None, None
            
        # Read the uploaded file
        if isinstance(file, str):  # Path provided
            df = pd.read_csv(file)
        else:  # File object provided
            df = pd.read_csv(io.BytesIO(file.read()))
            
        # Process date column if it exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df.Date)
        
        # Remove Adj Close if present
        if 'Adj Close' in df.columns:
            df.drop('Adj Close', axis=1, inplace=True)
            
        # Ensure Volume is float type
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].astype(float)
            
        # Check for required columns
        required_cols = ['Open', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return f"Error: Missing required columns: {', '.join(missing_cols)}", None, None
            
        # Remove rows with infinity or NaN values
        df_new = df[np.isfinite(df).all(1)]
        
        return "Data loaded successfully!", df_new.head(), None
    except Exception as e:
        return f"Error loading data: {str(e)}", None, None

def load_data(company):
    global df_new, x_train, x_test, y_train, y_test
    
    try:
        path = f"./data/{company}.csv"
        print(f"Loading data from: {path}")
        
        return load_data_from_file(path)
    except Exception as e:
        return f"Error loading data: {str(e)}", None, None

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.neural_network import MLPClassifier
import pickle
import os
import io

# Global variables
model = None
scaler = None
df_new = None
x_train, x_test, y_train, y_test = None, None, None, None

def load_data_from_file(file):
    global df_new, x_train, x_test, y_train, y_test
    
    try:
        if file is None:
            return "Please upload a file!", None, None
            
        if isinstance(file, str):
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(io.BytesIO(file.read()))
            
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df.Date)
        if 'Adj Close' in df.columns:
            df.drop('Adj Close', axis=1, inplace=True)
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].astype(float)
            
        required_cols = ['Open', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return f"Error: Missing required columns: {', '.join(missing_cols)}", None, None
            
        df_new = df[np.isfinite(df).all(1)]
        return "Data loaded successfully!", df_new.head(), None
    except Exception as e:
        return f"Error loading data: {str(e)}", None, None

def load_data(company):
    global df_new
    try:
        path = f"./data/{company}.csv"
        return load_data_from_file(path)
    except Exception as e:
        return f"Error loading data: {str(e)}", None, None

def train_model(model_type):
    global model, df_new, scaler

    if df_new is None:
        return "Please load data first!", None, None

    try:
        seq_len = 30
        # Prepare the sliding-window data for sequence models
        df_seq = df_new[['Open','High','Low','Close','Volume']].copy()
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df_seq)

        Xs, ys = [], []
        for i in range(seq_len, len(data_scaled)):
            Xs.append(data_scaled[i-seq_len:i].flatten())
            ys.append(data_scaled[i, 3])  # Close price
        Xs, ys = np.array(Xs), np.array(ys)

        split = int(0.8 * len(Xs))
        X_tr, X_te = Xs[:split], Xs[split:]
        y_tr, y_te = ys[:split], ys[split:]

        if model_type == "Linear Regression":
            model = LinearRegression()
            model.fit(X_tr, y_tr)
            y_pred_scaled = model.predict(X_te)

        elif model_type == "SVM":
            model = SVR(kernel='rbf', C=100, epsilon=0.01)
            model.fit(X_tr, y_tr)
            y_pred_scaled = model.predict(X_te)

        elif model_type == "KNN":
            model = KNeighborsRegressor(n_neighbors=5)
            model.fit(X_tr, y_tr)
            y_pred_scaled = model.predict(X_te)

        else:
            # Non‐regression models unchanged...
            if model_type == "Naive Bayes":
                df_seq['Target'] = (df_seq['Close'].shift(-1) > df_seq['Close']).astype(int)
                df_seq.dropna(inplace=True)
                Xs = scaler.fit_transform(df_seq.drop('Target', axis=1))
                ys = df_seq['Target'].values
                X_seq, y_seq = [], []
                for i in range(seq_len, len(Xs)):
                    X_seq.append(Xs[i-seq_len:i].flatten())
                    y_seq.append(ys[i])
                X_seq, y_seq = np.array(X_seq), np.array(y_seq)
                split = int(0.8 * len(X_seq))
                X_tr, X_te = X_seq[:split], X_seq[split:]
                y_tr, y_te = y_seq[:split], y_seq[split:]
                model = GaussianNB()
                model.fit(X_tr, y_tr)
                y_pred_scaled = model.predict(X_te)

            elif model_type == "MLP":
                df_seq['Target'] = (df_seq['Close'].shift(-1) > df_seq['Close']).astype(int)
                df_seq.dropna(inplace=True)
                X = df_seq.drop(columns=['Close','Target'])
                y = df_seq['Target']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled  = scaler.transform(X_test)
                model = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200, random_state=42)
                model.fit(X_train_scaled, y_train)
                y_pred_scaled = model.predict(X_test_scaled)
                y_te = y_test.values
            else:
                return "Invalid model selected!", None, None

        # inverse‐scale for regression methods
        if model_type in ["Linear Regression", "SVM", "KNN"]:
            close_scaler = MinMaxScaler()
            close_scaler.min_, close_scaler.scale_ = scaler.min_[3], scaler.scale_[3]
            actual = close_scaler.inverse_transform(y_te.reshape(-1,1)).ravel()
            pred   = close_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
            # derive up/down
            prev_close_scaled = X_te.reshape(-1, seq_len, 5)[:, -1, 3]
            prev_close = close_scaler.inverse_transform(prev_close_scaled.reshape(-1,1)).ravel()
            actual_dir = (actual > prev_close).astype(int)
            pred_dir   = (pred   > prev_close).astype(int)

            # regression metrics
            mse = mean_squared_error(actual, pred)
            mae = mean_absolute_error(actual, pred)
            r2  = r2_score(actual, pred)

        else:
            actual_dir = y_te
            pred_dir   = y_pred_scaled

        # classification metrics for every model
        acc   = accuracy_score(actual_dir, pred_dir)
        prec  = precision_score(actual_dir, pred_dir, zero_division=0)
        rec   = recall_score(actual_dir, pred_dir, zero_division=0)
        f1    = f1_score(actual_dir, pred_dir, zero_division=0)

        # build metrics string
        if model_type in ["Linear Regression", "SVM", "KNN"]:
            metrics_str = (
                f"MSE:  {mse:.2f}\n"
                f"MAE:  {mae:.2f}\n"
                f"R²:   {r2:.2f}\n\n"
                f"Accuracy:  {acc:.2f}\n"
                f"Precision: {prec:.2f}\n"
                f"Recall:    {rec:.2f}\n"
                f"F1-score:  {f1:.2f}"
            )
        else:
            metrics_str = (
                f"Accuracy:  {acc:.4f}\n"
                f"Precision: {prec:.4f}\n"
                f"Recall:    {rec:.4f}\n"
                f"F1-score:  {f1:.4f}"
            )

        # plotting
        plt.figure(figsize=(10,5))
        if model_type in ["Linear Regression", "SVM", "KNN"]:
            plt.plot(actual, label='Actual')
            plt.plot(pred,   label='Predicted')
            plt.title(f"{model_type} Forecast")
            plt.ylabel("Close Price")
        else:
            plt.bar(['Correct','Incorrect'], [acc, 1-acc])
            plt.title(f"{model_type} Accuracy")
            plt.ylabel("Proportion")
        plt.xlabel("Sample")
        if model_type in ["Linear Regression", "SVM", "KNN"]:
            plt.legend()
        plt.grid(True)

        plot_path = "temp_plot.png"
        plt.savefig(plot_path)
        plt.close()

        return f"{model_type} trained successfully!", metrics_str, plot_path

    except Exception as e:
        return f"Error training model: {str(e)}", None, None


def save_model(model_name):
    global model, scaler
    
    if model is None:
        return "No model to save. Please train a model first."
    
    try:
        os.makedirs("saved_models", exist_ok=True)
        model_path = f"saved_models/{model_name}.pkl"
        
        # Save both model and scaler
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler}, f)
        
        return f"Model saved successfully as {model_path}"
    except Exception as e:
        return f"Error saving model: {str(e)}"

def load_saved_model(model_path):
    global model, scaler
    
    try:
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            
        if isinstance(saved_data, dict) and 'model' in saved_data:
            model = saved_data['model']
            scaler = saved_data.get('scaler')
        else:
            model = saved_data
            scaler = None
            
        return f"Model loaded successfully from {model_path}"
    except Exception as e:
        return f"Error loading model: {str(e)}"

def predict_price(open_price, high, low, close, volume):
    global model, scaler, df_new
    
    if model is None:
        return "Please train a model first!"
    
    try:
        # Prepare input data based on model type
        if isinstance(model, (LinearRegression, SVR, KNeighborsRegressor)):
            # For regression models, we need a sequence of data
            if df_new is None:
                return "Please load data first for sequence prediction!"
            
            # Get the last 30 days of data
            seq_len = 30
            df_seq = df_new[['Open','High','Low','Close','Volume']].copy()
            
            # Add the new input data point
            new_data = pd.DataFrame([[open_price, high, low, close, volume]], 
                                  columns=['Open','High','Low','Close','Volume'])
            df_seq = pd.concat([df_seq, new_data], ignore_index=True)
            
            # Scale the data
            data_scaled = scaler.transform(df_seq)
            
            # Prepare the sequence
            X_pred = data_scaled[-seq_len:].flatten().reshape(1, -1)
            
            # Make prediction
            pred_scaled = model.predict(X_pred)
            
            # Inverse scale the prediction
            close_scaler = MinMaxScaler()
            close_scaler.min_, close_scaler.scale_ = scaler.min_[3], scaler.scale_[3]
            prediction = close_scaler.inverse_transform(pred_scaled.reshape(-1,1))[0][0]
            
            return f"Predicted Close Price for next day: ${prediction:.2f}"
            
        elif isinstance(model, (GaussianNB, MLPClassifier)):
            # For classification models, we can use single point prediction
            input_data = np.array([[open_price, high, low, close, volume]])
            
            if scaler is not None:
                input_data = scaler.transform(input_data)
            
            prediction = model.predict(input_data)
            
            return "Price will go UP tomorrow!" if prediction[0] else "Price will go DOWN tomorrow!"
            
        else:
            return "Unsupported model type for prediction!"
            
    except Exception as e:
        return f"Error making prediction: {str(e)}"


# Create the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Stock Market Prediction")
    
    with gr.Tab("Data Loading"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Load from File")
                file_upload = gr.File(label="Upload CSV File")
                upload_btn = gr.Button("Load Uploaded File")
                
            with gr.Column():
                gr.Markdown("## Load from Symbol")
                company_input = gr.Textbox(label="Enter Company Symbol (e.g., AAPL)")
                load_btn = gr.Button("Load Data by Symbol")
        
        data_status = gr.Textbox(label="Data Status")
        data_preview = gr.DataFrame(label="Data Preview")
    
    with gr.Tab("Model Training"):
        with gr.Row():
            model_dropdown = gr.Dropdown(
                ["Linear Regression", "Naive Bayes", "SVM", "KNN", "MLP"],
                label="Select Model"
            )
            train_btn = gr.Button("Train Model")
        
        train_status = gr.Textbox(label="Training Status")
        metrics_output = gr.Textbox(label="Model Metrics")
        plot_output = gr.Image(label="Results Visualization")
    
    with gr.Tab("Save/Load Model"):
        with gr.Row():
            with gr.Column():
                model_name_input = gr.Textbox(label="Model Save Name (without extension)")
                save_btn = gr.Button("Save Model")
            
            with gr.Column():
                model_load_path = gr.Textbox(label="Path to Model File (.pkl)")
                load_model_btn = gr.Button("Load Model")
                
        save_load_status = gr.Textbox(label="Status")
    
    with gr.Tab("Make Prediction"):
         gr.Markdown("## Make a Prediction")
         with gr.Row():
          open_input = gr.Number(label="Open Price")
          high_input = gr.Number(label="High Price")
          low_input = gr.Number(label="Low Price")
          close_input = gr.Number(label="Current Close Price")  # Added this
          volume_input = gr.Number(label="Volume")
          predict_btn = gr.Button("Predict")
    
         prediction_output = gr.Textbox(label="Prediction Result")
    
    # Event handlers
    upload_btn.click(
        load_data_from_file,
        inputs=[file_upload],
        outputs=[data_status, data_preview, plot_output]
    )
    
    load_btn.click(
        load_data,
        inputs=[company_input],
        outputs=[data_status, data_preview, plot_output]
    )
    
    train_btn.click(
        train_model,
        inputs=[model_dropdown],
        outputs=[train_status, metrics_output, plot_output]
    )
    
    save_btn.click(
        save_model,
        inputs=[model_name_input],
        outputs=[save_load_status]
    )
    
    load_model_btn.click(
        load_saved_model,
        inputs=[model_load_path],
        outputs=[save_load_status]
    )
    
    predict_btn.click(
    predict_price,
    inputs=[open_input, high_input, low_input, close_input, volume_input],  # Added close_input
    outputs=[prediction_output]
)

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    app.launch()