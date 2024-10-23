import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
import pandas as pd

import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold  # Changed to StratifiedKFold
import joblib
import matplotlib.pyplot as plt
from packaging import version
from flask import Flask, request, render_template, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
import time  # Simulating training time, replace with your actual training code

torch.backends.cudnn.benchmark = True

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key'  # Replace with your secret key
socketio = SocketIO(app)

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

seq_length = 5
batch_size = 256  # Adjusted batch size
hidden_size = 128
num_layers = 2
learning_rate = 0.01
k_folds = 3
epochs = 25  # Updated epochs to match your latest training logs

# Define the LSTM-based neural network model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

# Define the weight initialization function
def init_weights(m):
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# Define the NetworkTrafficClassifier class
class NetworkTrafficClassifier:
    def __init__(
        self,
        model_path,
        scaler_path,
        seq_length,
        batch_size,
        hidden_size,
        num_layers,
        learning_rate
    ):
        self.data_path = None  # Will be set later
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Print device information only once
        if torch.cuda.is_available():
            print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU is not available, using CPU")

        # Load the scaler
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            print("Scaler loaded from file.")
        else:
            self.scaler = None
            print("Scaler not found.")

        self.model = None  # Will be initialized after data is prepared

        # Optimizer will be initialized after model is initialized
        self.optimizer = None

        self.criterion = None  # Will be initialized after data is prepared

    def prepare_data(self, df=None):
        if df is None:
            # Load the dataset from the data_path
            df = pd.read_csv(self.data_path)

        # Preprocess the data
        if 'label' in df.columns:
            X_raw = df.drop("label", axis=1).values
            y = LabelEncoder().fit_transform(df["label"])
            self.is_training = True
        else:
            X_raw = df.values
            y = None
            self.is_training = False

        # Scale the input features
        self.scaler = StandardScaler()
        self.scaler.fit(X_raw)
        X_scaled = self.scaler.transform(X_raw)

        X = X_scaled

        # Prepare sequences
        X_sequences = []
        y_sequences = []
        for i in range(len(X) - self.seq_length + 1):
            X_seq = X[i:i + self.seq_length]
            X_sequences.append(X_seq)
            if self.is_training:
                y_seq = y[i + self.seq_length - 1]
                y_sequences.append(y_seq)

        # Convert lists to NumPy arrays before converting to tensors
        X_sequences = np.array(X_sequences)
        X_sequences = torch.tensor(X_sequences, dtype=torch.float32)

        if self.is_training:
            y_sequences = np.array(y_sequences)
            y_sequences = torch.tensor(y_sequences, dtype=torch.long)

            # Compute class weights
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_sequences.numpy()),
                y=y_sequences.numpy()
            )
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            y_sequences = None

        # Create dataset
        if self.is_training:
            self.dataset = TensorDataset(X_sequences, y_sequences)
        else:
            self.dataset = TensorDataset(X_sequences)

        # Save X_sequences and y_sequences for use elsewhere
        self.X_sequences = X_sequences
        self.y_sequences = y_sequences

        # Set input_size and output_size
        self.input_size = X_sequences.shape[2]
        if self.is_training:
            self.output_size = len(np.unique(y_sequences.numpy()))
        else:
            self.output_size = None  # Will be set when loading the model

        # Initialize model if not already initialized
        if self.model is None:
            self.init_model()

        # Update criterion with new class weights
        if self.is_training:
            self.class_weights = self.class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

            # Save the scaler
            joblib.dump(self.scaler, self.scaler_path)
            print("Scaler saved to file.")

    def init_model(self):
        # Initialize model
        if self.is_training:
            self.model = LSTMModel(self.input_size, self.hidden_size, self.output_size, num_layers=self.num_layers)
        else:
            # Load the output_size from the saved model if possible
            self.model = LSTMModel(self.input_size, self.hidden_size, self.output_size or 2, num_layers=self.num_layers)
        self.model.to(self.device)

        # Load or initialize weights
        if os.path.exists(self.model_path):
            self.model_loaded = self.load_model()
            if self.model_loaded:
                print("Model weights loaded from file.")
            else:
                print("Model architecture mismatch or error loading model. Reinitializing model.")
                self.model.apply(init_weights)
        else:
            print("Model file not found. Initializing a new model.")
            self.model.apply(init_weights)

        # Initialize optimizer with weight decay for regularization
        if self.is_training:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    def reset_model(self):
        # Reinitialize the model
        self.model = LSTMModel(self.input_size, self.hidden_size, self.output_size, num_layers=self.num_layers)
        self.model.to(self.device)
        self.model.apply(init_weights)
        # Reinitialize the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        print("Model and optimizer have been reinitialized.")

    # Updated train method without early stopping
    def train(self, epochs=10, progress_callback=None):
        best_val_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            avg_train_loss = running_loss / len(self.train_loader)

            # Evaluate on validation set
            self.model.eval()
            val_running_loss = 0.0
            y_true = []
            y_pred = []
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch = X_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)

                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    val_running_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    y_true.extend(y_batch.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
            avg_val_loss = val_running_loss / len(self.val_loader)
            val_accuracy = (np.array(y_pred) == np.array(y_true)).mean()

            # Optionally save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model()
                print("Validation loss improved. Model saved.")

            print(f"Epoch [{epoch + 1}/{epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}")

            # Emit progress update
            if progress_callback:
                progress_callback()

    def evaluate(self):
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        print(report)
        return report

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        try:
            if version.parse(torch.__version__) >= version.parse("2.1.0"):
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            else:
                state_dict = torch.load(self.model_path, map_location=self.device)

            # Load the state dict into the model
            self.model.load_state_dict(state_dict)
            return True  # Model loaded successfully

        except RuntimeError as e:
            print(f"Error loading model: {e}")
            # Remove the model file to force retraining
            os.remove(self.model_path)
            return False  # Indicate that the model was not loaded

    def predict(self, new_data):
        # new_data: pandas DataFrame
        self.model.eval()
        # Scale the data
        X_raw = new_data.values
        X_scaled = self.scaler.transform(X_raw)
        # Prepare sequences
        X_sequences = []
        for i in range(len(X_scaled) - self.seq_length + 1):
            X_seq = X_scaled[i:i + self.seq_length]
            X_sequences.append(X_seq)
        if not X_sequences:
            return []
        X_sequences = np.array(X_sequences)
        X_sequences = torch.tensor(X_sequences, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_sequences)
            _, predicted_classes = torch.max(outputs, 1)
        predictions = predicted_classes.cpu().numpy()
        return predictions

    def display_confusion_matrix(self):
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    def k_fold_cross_validation(self, k=5, epochs=10, progress_callback=None):
        # Initialize StratifiedKFold for class imbalance
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        fold_results = []

        X = self.X_sequences.numpy()
        y = self.y_sequences.numpy()

        total_steps = k * epochs
        current_step = 0

        # Start k-fold cross-validation
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            print(f"FOLD {fold+1}")
            print("------------------------------")

            # Split data into training and validation sets
            train_subsampler = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx]))
            test_subsampler = TensorDataset(torch.tensor(X[test_idx]), torch.tensor(y[test_idx]))

            # Dataloaders for training and validation with pin_memory for non_blocking transfers
            train_loader = DataLoader(train_subsampler, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            test_loader = DataLoader(test_subsampler, batch_size=self.batch_size, shuffle=False, pin_memory=True)

            # Model initialization
            input_size = self.input_size
            output_size = self.output_size
            model = LSTMModel(input_size, self.hidden_size, output_size, num_layers=self.num_layers)
            model.to(self.device)
            model.apply(init_weights)

            # Recompute class weights for this fold
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y),
                y=y[train_idx]
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

            # Train the model
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)

                    optimizer.zero_grad()

                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                avg_loss = running_loss / len(train_loader)
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

                # Update progress
                current_step += 1
                if progress_callback:
                    # Calculate the progress percentage for k-folds (0% to 50%)
                    kfold_progress = (current_step / total_steps) * 50
                    progress_callback(kfold_progress)

            # Evaluate on the test set
            model.eval()
            y_true = []
            y_pred = []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)

                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    y_true.extend(y_batch.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())

            # Generate classification report for this fold
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            print(f"Fold {fold+1} Classification Report:")
            print(classification_report(y_true, y_pred, zero_division=0))

            # Store results for this fold
            fold_results.append(report)

        # Return the results for all folds
        return fold_results

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize the classifier globally
def initialize_classifier():
    global classifier
    global seq_length
    global batch_size
    global hidden_size
    global num_layers
    global learning_rate

    # Define model and scaler paths
    model_name = f"model_layers_{num_layers}.pth"
    scaler_name = f"scaler_layers_{num_layers}.pkl"

    model_path = os.path.join('models', model_name)
    scaler_path = os.path.join('scalers', scaler_name)

    # Ensure model and scaler directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('scalers', exist_ok=True)

    # Load the scaler
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("Scaler loaded from file.")
    else:
        print("Scaler not found. Please train the model first.")
        scaler = None

    # Initialize the classifier once
    classifier = NetworkTrafficClassifier(
        model_path=model_path,
        scaler_path=scaler_path,
        seq_length=seq_length,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=learning_rate
    )

    # Load the scaler
    classifier.scaler = scaler

# Route to serve the index page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global classifier
    file = request.files.get('file')

    if not file or file.filename == '':
        flash('No file selected')
        return redirect('/')

    if file.filename.endswith('.csv'):
        # Save the uploaded CSV file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_data.csv')
        file.save(file_path)
        flash('File uploaded successfully!')

        # Load the CSV file into a pandas DataFrame
        try:
            df = pd.read_csv(file_path)
            print("DEBUG: CSV file loaded into DataFrame.")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            flash('Error processing the CSV file.')
            return redirect('/')
        # Call the prepare_data() method with the DataFrame
        classifier.prepare_data(df)
        print("DEBUG: Data prepared for training.")

        return render_template('results.html')
    else:
        flash('Invalid file format, please upload a CSV.')
        return redirect('/')
    
# Training process with progress updates
@socketio.on('start_training')
def handle_training():
    global batch_size
    global k_folds
    global epochs

    total_progress = 100
    progress = 0

    def progress_callback(new_progress=None):
        nonlocal progress
        if new_progress is not None:
            progress = new_progress
        else:
            progress += (1 / total_steps) * 100
        if progress > 100:
            progress = 100
        socketio.emit('progress_update', {'progress': progress})

    total_kfold_steps = k_folds * epochs
    total_training_steps = epochs
    total_steps = total_kfold_steps + total_training_steps

    # Variables to track progress
    current_step = 0

    # K-Fold Progress Callback
    def kfold_progress(kfold_progress_percentage):
        progress_callback(kfold_progress_percentage)

    # Perform k-fold cross-validation
    fold_results = classifier.k_fold_cross_validation(
        k=k_folds, epochs=epochs, progress_callback=kfold_progress
    )

    # Reinitialize the model before final training
    classifier.reset_model()

    # Split dataset into training, validation, and test sets
    train_size = int(0.7 * len(classifier.dataset))
    val_size = int(0.15 * len(classifier.dataset))
    test_size = len(classifier.dataset) - train_size - val_size
    classifier.train_dataset, classifier.val_dataset, classifier.test_dataset = random_split(
        classifier.dataset, [train_size, val_size, test_size]
    )

    # Create data loaders with pin_memory
    classifier.train_loader = DataLoader(classifier.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    classifier.val_loader = DataLoader(classifier.val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    classifier.test_loader = DataLoader(classifier.test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Define progress callback function for final training
    def training_progress_callback():
        nonlocal current_step
        current_step += 1
        # Calculate the progress percentage for final training (50% to 100%)
        training_progress = 50 + (current_step / total_training_steps) * 50
        progress_callback(training_progress)

    # Reset current_step for final training
    current_step = 0

    # Train the model
    classifier.train(epochs=epochs, progress_callback=training_progress_callback)

    # Evaluate the model
    final_report = classifier.evaluate()

    # Extract metrics
    accuracy = final_report['accuracy']
    precision_0 = final_report['0']['precision']
    precision_1 = final_report['1']['precision']
    recall_0 = final_report['0']['recall']
    recall_1 = final_report['1']['recall']
    f1_score_0 = final_report['0']['f1-score']
    f1_score_1 = final_report['1']['f1-score']

    final_results = {
        'precision_0': precision_0,
        'recall_0': recall_0,
        'f1_score_0': f1_score_0,
        'precision_1': precision_1,
        'recall_1': recall_1,
        'f1_score_1': f1_score_1,
        'accuracy': accuracy
    }
    socketio.emit('training_complete', final_results)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    initialize_classifier()
    socketio.run(app, debug=True)