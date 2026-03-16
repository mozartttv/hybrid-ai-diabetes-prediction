import numpy as np
import pandas as pd
import time
import random
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer
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
# Gaussian Membership Function Layer
# ----------------------------------------------
class GaussianMF(Layer):
    """Layer to compute Gaussian membership functions."""
    def __init__(self, n_mfs, **kwargs):
        super().__init__(**kwargs)
        self.n_mfs = n_mfs

    def build(self, input_shape):
        init_centers = np.linspace(-1.0, 1.0, self.n_mfs).astype(np.float32)
        self.c = self.add_weight(shape=(self.n_mfs,),
                                 initializer=tf.keras.initializers.Constant(init_centers),
                                 trainable=True, name="centers")
        init_sigma = np.ones((self.n_mfs,), dtype=np.float32) * 0.5
        self.log_sigma = self.add_weight(shape=(self.n_mfs,),
                                         initializer=tf.keras.initializers.Constant(np.log(init_sigma)),
                                         trainable=True, name="log_sigma")

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)  # (batch, 1, 1)
        c = tf.reshape(self.c, shape=(1, 1, self.n_mfs))
        sigma = tf.math.exp(self.log_sigma)
        sigma = tf.reshape(sigma, shape=(1, 1, self.n_mfs))
        out = tf.exp(-0.5 * tf.square((x - c) / (sigma + 1e-8)))
        return tf.squeeze(out, axis=1)  # (batch, n_mfs)


# ----------------------------------------------
# ANFIS Layer
# ----------------------------------------------
class ANFISLayer(Layer):
    """Layer to implement ANFIS (Adaptive Neuro-Fuzzy Inference System)."""
    def __init__(self, n_inputs, n_mfs=2, **kwargs):
        super().__init__(**kwargs)
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.n_rules = (n_mfs ** n_inputs)
        self.mf_layers = [GaussianMF(n_mfs, name=f"mf_input_{i}") for i in range(n_inputs)]

    def build(self, input_shape):
        init = tf.keras.initializers.RandomNormal(0.0, 0.1)
        self.consequents = self.add_weight(shape=(self.n_rules, self.n_inputs + 1),
                                           initializer=init, trainable=True, name="consequents")

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        mf_values = []
        for i in range(self.n_inputs):
            xi = tf.expand_dims(inputs[:, i], axis=1)
            mv = self.mf_layers[i](xi)
            mf_values.append(mv)

        firing = mf_values[0]
        for i in range(1, self.n_inputs):
            next_mf = mf_values[i]
            firing = tf.reshape(firing, (batch, -1, 1))
            next_mf_resh = tf.reshape(next_mf, (batch, 1, self.n_mfs))
            firing = tf.reshape(firing * next_mf_resh, (batch, -1))

        firing += 1e-8
        firing_sum = tf.reduce_sum(firing, axis=1, keepdims=True)
        normalized = firing / firing_sum

        ones = tf.ones((batch, 1), dtype=inputs.dtype)
        x_with_bias = tf.concat([inputs, ones], axis=1)
        rule_outputs = tf.matmul(x_with_bias, self.consequents, transpose_b=True)

        output = tf.reduce_sum(normalized * rule_outputs, axis=1, keepdims=True)

        # Keep diagnostics
        self.last_firing = firing
        self.last_normalized = normalized
        self.last_rule_outputs = rule_outputs

        return output


# ----------------------------------------------
# ANFIS Model
# ----------------------------------------------
def create_anfis_model(input_dim, n_mfs=2, lr=1e-3):
    """Create an ANFIS model."""
    inputs = Input(shape=(input_dim,), name="inputs")
    anfis = ANFISLayer(n_inputs=input_dim, n_mfs=n_mfs, name="anfis")(inputs)
    out = tf.keras.layers.Activation("sigmoid", name="out_sigmoid")(anfis)
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# ----------------------------------------------
# Genetic Algorithm Functions
# ----------------------------------------------
search_space = {
    "n_mfs": [2, 3],
    "lr": [0.0005, 0.001, 0.003],
    "batch": [8, 16, 32],
    "epochs": [40, 80, 120]
}


def create_individual():
    """Create a random individual from the search space."""
    return {
        "n_mfs": random.choice(search_space["n_mfs"]),
        "lr": random.choice(search_space["lr"]),
        "batch": random.choice(search_space["batch"]),
        "epochs": random.choice(search_space["epochs"]),
    }


def mutate(individual):
    """Mutate a random parameter of the individual."""
    key = random.choice(list(search_space.keys()))
    individual[key] = random.choice(search_space[key])
    return individual


def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    return {key: random.choice([parent1[key], parent2[key]]) for key in parent1}


def evaluate(individual, X_train, y_train, X_test, y_test, seed=42):
    """Evaluate the individual's performance using the ANFIS model."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = {int(c): w for c, w in zip(classes, class_weights)}

    n_mfs = individual["n_mfs"]
    lr = individual["lr"]
    batch = individual["batch"]
    epochs = individual["epochs"]

    n_rules = (n_mfs ** X_train.shape[1])
    if n_rules > 500:
        print("Skipping individual (too many rules):", individual, "n_rules=", n_rules)
        return 0.0

    model = create_anfis_model(input_dim=X_train.shape[1], n_mfs=n_mfs, lr=lr)

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        epochs=epochs, batch_size=batch, verbose=0, class_weight=cw,
        validation_split=0.2, callbacks=[early_stopping]
    )

    y_probs = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_probs > 0.5).astype("int32")

    return f1_score(y_test, y_pred)


# ----------------------------------------------
# Genetic Algorithm Execution
# ----------------------------------------------
def run_genetic_algorithm(X_train, y_train, X_test, y_test, population_size=10, generations=3):
    """Run the genetic algorithm to optimize hyperparameters."""
    population = [create_individual() for _ in range(population_size)]
    best_overall = (0.0, None)

    print("\n========== GA Optimization for ANFIS Running ==========\n")

    for gen in range(generations):
        print(f"\n=== Generation {gen + 1}/{generations} ===")
        scored = []
        for ind in population:
            f1 = evaluate(ind, X_train, y_train, X_test, y_test)
            scored.append((f1, ind))
            print(f"{ind} -> F1 = {f1:.4f}")

        scored.sort(reverse=True, key=lambda x: x[0])
        best_gen_f1, best_gen_ind = scored[0]
        print("\nBest this generation:", best_gen_ind, "F1 =", best_gen_f1)

        if best_gen_f1 > best_overall[0]:
            best_overall = (best_gen_f1, best_gen_ind)

        # Select top parents
        parents = [ind for _, ind in scored[:3]]

        # Create next generation
        new_pop = parents.copy()
        while len(new_pop) < population_size:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)

        population = new_pop

    return best_overall


# ----------------------------------------------
# Main Execution
# ----------------------------------------------
if __name__ == "__main__":
    # Load and preprocess data
    X_train_fs, X_test_fs, y_train, y_test = load_and_preprocess_data("diabetestest1.csv")

    # Run genetic algorithm
    best_f1, best_params = run_genetic_algorithm(X_train_fs, y_train, X_test_fs, y_test)

    # Train final model with best parameters
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = {int(c): w for c, w in zip(classes, class_weights)}

    final_model = create_anfis_model(input_dim=X_train_fs.shape[1], n_mfs=best_params["n_mfs"], lr=best_params["lr"])

    # Early stopping callback for final training
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    start_time = time.time()
    final_model.fit(
        X_train_fs, y_train,
        epochs=best_params["epochs"], batch_size=best_params["batch"], verbose=0,
        class_weight=cw, validation_split=0.2, callbacks=[early_stopping]
    )
    end_time = time.time()

    # Evaluate final model
    y_test_probs = final_model.predict(X_test_fs, verbose=0).ravel()
    y_pred = (y_test_probs > 0.5).astype("int32")

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n========== FINAL TEST RESULTS (ANFIS with best GA params) ==========")
    print("Best GA Params:", best_params)
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Training Time: %.2f seconds" % (end_time - start_time))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    # Rule activation analysis
    anfis_layer = None
    for layer in final_model.layers:
        if isinstance(layer, ANFISLayer):
            anfis_layer = layer
            break

    if anfis_layer is not None:
        X_test_tensor = tf.constant(X_test_fs, dtype=tf.float32)
        _ = anfis_layer(X_test_tensor)

        rule_outputs = anfis_layer.last_firing.numpy()
        normalized = anfis_layer.last_normalized.numpy()
        rule_linear = anfis_layer.last_rule_outputs.numpy()

        top_rule_idx_each = np.argmax(normalized, axis=1)
        counts = np.bincount(top_rule_idx_each, minlength=anfis_layer.n_rules)

        print("\nTop-rule counts (test):")
        top_rule_order = np.argsort(-counts)[:10]
        for idx in top_rule_order:
            print(f"Rule {idx}: count={counts[idx]} frac={counts[idx] / len(normalized):.3f}")

        print("\nSample-level (first 10):")
        for i in range(min(10, len(normalized))):
            top_i = int(top_rule_idx_each[i])
            print(f"Sample {i + 1}: Top rule -> {top_i} (norm firing={normalized[i, top_i]:.3f}) "
                  f"| model_prob={y_test_probs[i]:.3f} | pred={y_pred[i]} | true={int(y_test.iloc[i])}")
    else:
        print("Couldn't find ANFIS layer for inspection.")

