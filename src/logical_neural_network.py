import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------------------------
# Data Loading and Preprocessing
# ----------------------------------------------
def load_and_preprocess_data(file_path):
    """Load and preprocess diabetes dataset."""
    data = pd.read_csv('diabetestest1.csv')
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=5)
    X_train_fs = selector.fit_transform(X_train, y_train)
    X_test_fs = selector.transform(X_test)

    selected_features = X.columns[selector.get_support()]
    print("Selected Features:", list(selected_features))
    print("\nClass Distribution (Train):")
    print(y_train.value_counts())
    print("=" * 60)

    return X_train_fs, X_test_fs, y_train, y_test

# ----------------------------------------------
# Rule-Based Layer
# ----------------------------------------------
class RuleLayer(tf.keras.layers.Layer):
    """Custom layer implementing medical knowledge rules."""

    def call(self, inputs):
        # Split into individual features (order: Pregnancies, Glucose, Insulin, BMI, Age)
        Preg, Glu, Ins, BMI, Age = tf.split(inputs, num_or_size_splits=5, axis=1)

        # Define interpretable rules with medical thresholds
        rules = [
            tf.sigmoid((Glu - 110.0) / 15.0),  # Glucose moderately high
            tf.sigmoid((BMI - 27.0) / 3.0) * tf.sigmoid((Glu - 110.0) / 15.0),  # BMI & Glucose
            tf.sigmoid((Ins - 120.0) / 30.0),  # Elevated insulin
            tf.sigmoid((Preg - 4.0) / 1.5) * tf.sigmoid((Age - 30.0) / 6.0),  # Pregnancies & Age
            tf.sigmoid((100.0 - Glu) / 15.0) * tf.sigmoid((25.0 - BMI) / 3.0)  # Healthy inverse
        ]

        return tf.concat(rules, axis=1)

# ----------------------------------------------
# Hybrid LNN Model Architecture
# ----------------------------------------------
def create_lnn_hybrid_model(input_dim):
    """Create hybrid model combining raw features and interpretable rules."""
    inputs = Input(shape=(input_dim,))

    # Rule-based feature extraction
    rules = RuleLayer()(inputs)

    # Combine raw inputs with rule activations
    combined = Concatenate()([inputs, rules])

    # Neural network layers
    x = Dense(32, activation="relu")(combined)
    x = Dropout(0.2)(x)
    x = Dense(16, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ----------------------------------------------
# Main Execution
# ----------------------------------------------
if __name__ == "__main__":
    # Load and preprocess data
    X_train_fs, X_test_fs, y_train, y_test = load_and_preprocess_data("diabetestest1.csv")

    # Compute class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): w for c, w in zip(classes, class_weights)}

    # Create and train hybrid model
    model = create_lnn_hybrid_model(X_train_fs.shape[1])
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

    print("\n========== TEST RESULTS ==========")
    print("Test Accuracy:", test_acc)
    print("Test F1 Score:", test_f1)

