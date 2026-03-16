import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ----------------------------------------------
# Data Loading and Preprocessing
# ----------------------------------------------
def load_and_preprocess_data(file_path):
    """Load and preprocess diabetes dataset with feature selection."""
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
# ANFIS Components
# ----------------------------------------------
class GaussianMF(Layer):
    """Gaussian membership function layer with trainable parameters."""

    def __init__(self, n_mfs, **kwargs):
        super().__init__(**kwargs)
        self.n_mfs = n_mfs

    def build(self, input_shape):
        # Initialize centers evenly spaced between -1 and 1
        init_centers = np.linspace(-1.0, 1.0, self.n_mfs).astype(np.float32)
        self.c = self.add_weight(
            shape=(self.n_mfs,),
            initializer=tf.keras.initializers.Constant(init_centers),
            trainable=True,
            name="centers"
        )

        # Initialize widths (log-transformed for positivity)
        init_sigma = np.ones((self.n_mfs,), dtype=np.float32) * 0.5
        self.log_sigma = self.add_weight(
            shape=(self.n_mfs,),
            initializer=tf.keras.initializers.Constant(np.log(init_sigma)),
            trainable=True,
            name="log_sigma"
        )

    def call(self, inputs):
        # Expand dimensions for broadcasting
        x = tf.expand_dims(inputs, axis=-1)
        c = tf.reshape(self.c, shape=(1, 1, self.n_mfs))
        sigma = tf.math.exp(self.log_sigma)
        sigma = tf.reshape(sigma, shape=(1, 1, self.n_mfs))

        # Gaussian membership function calculation
        return tf.exp(-0.5 * tf.square((x - c) / (sigma + 1e-8)))

class ANFISLayer(Layer):
    """Adaptive Neuro-Fuzzy Inference System layer."""

    def __init__(self, n_inputs, n_mfs=2, **kwargs):
        super().__init__(**kwargs)
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.n_rules = n_mfs ** n_inputs

        # Create membership function layers for each input
        self.mf_layers = [GaussianMF(n_mfs, name=f"mf_input_{i}")
                          for i in range(n_inputs)]

    def build(self, input_shape):
        # Initialize consequent parameters
        init = tf.keras.initializers.RandomNormal(0.0, 0.1)
        self.consequents = self.add_weight(
            shape=(self.n_rules, self.n_inputs + 1),
            initializer=init,
            trainable=True,
            name="consequents"
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # Fuzzification: Compute membership values for each input
        mf_values = []
        for i in range(self.n_inputs):
            xi = tf.expand_dims(inputs[:, i], axis=1)  # (batch, 1)
            mv = self.mf_layers[i](xi)  # (batch, n_mfs)
            mf_values.append(mv)

        # Rule firing strength calculation (product T-norm)
        firing = mf_values[0]  # Initialize with first input's MFs
        for i in range(1, self.n_inputs):
            next_mf = mf_values[i]
            firing = tf.reshape(firing, (batch_size, -1, 1))  # (batch, prev_rules, 1)
            next_mf_resh = tf.reshape(next_mf, (batch_size, 1, self.n_mfs))  # (batch, 1, n_mfs)
            firing = tf.reshape(firing * next_mf_resh, (batch_size, -1))  # (batch, prev_rules * n_mfs)

        # Normalize firing strengths
        firing += 1e-8  # Numerical stability
        firing_sum = tf.reduce_sum(firing, axis=1, keepdims=True)
        normalized = firing / firing_sum

        # Consequent calculation (linear functions)
        ones = tf.ones((batch_size, 1), dtype=inputs.dtype)
        x_with_bias = tf.concat([inputs, ones], axis=1)
        consequents = tf.matmul(normalized, self.consequents)
        outputs = tf.reduce_sum(consequents * x_with_bias, axis=1, keepdims=True)

        return outputs

# ----------------------------------------------
# ANFIS Model
# ----------------------------------------------
def create_anfis_model(input_dim):
    """Create ANFIS model."""
    inputs = Input(shape=(input_dim,))
    anfis = ANFISLayer(input_dim)(inputs)
    outputs = Activation("sigmoid")(anfis)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
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

    # Create and train ANFIS model
    model = create_anfis_model(X_train_fs.shape[1])
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

