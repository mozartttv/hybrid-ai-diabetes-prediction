import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
import time
# ----------------------------------------------
# Data Loading and Preprocessing
# ----------------------------------------------
def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    data = pd.read_csv('diabetestest1.csv')
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=5)
    X_train_fs = selector.fit_transform(X_train, y_train)
    X_test_fs = selector.transform(X_test)

    selected_features = X.columns[selector.get_support()]
    print("Selected Features:", list(selected_features))
    print("Train class distribution:\n", y_train.value_counts())
    print("==========================================================")

    return X_train_fs, X_test_fs, y_train, y_test

# ----------------------------------------------
# Model Creation
# ----------------------------------------------
def create_model(input_dim):
    """Create a neural network model."""
    model = Sequential([
        Dense(16, activation="relu", input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ----------------------------------------------
# Main Execution
# ----------------------------------------------
if __name__ == "__main__":
    # Start timing
    start_time = time.time()

    # Load and preprocess data
    X_train_fs, X_test_fs, y_train, y_test = load_and_preprocess_data("diabetestest1.csv")

    # Compute class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): w for c, w in zip(classes, class_weights)}

    # Create and train model
    model = create_model(X_train_fs.shape[1])
    model.fit(
        X_train_fs, y_train,
        epochs=100,
        batch_size=32,
        verbose=1,
        class_weight=class_weight_dict,
        validation_data=(X_test_fs, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )

    # Evaluate on test set
    y_test_pred = (model.predict(X_test_fs, verbose=0) > 0.5).astype(int).flatten()
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n========== TEST RESULTS ==========")
    print("Test Accuracy:", test_acc)
    print("Test F1 Score:", test_f1)
    print("Computational Time (seconds):", elapsed_time)

