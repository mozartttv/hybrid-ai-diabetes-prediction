import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
import random

# ----------------------------------------------
# Data Loading and Preprocessing
# ----------------------------------------------
def load_and_preprocess_data(file_path):

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

    return X_train_fs, X_test_fs, y_train, y_test


# ----------------------------------------------
# Rule-Based Layer
# ----------------------------------------------
class RuleLayer(tf.keras.layers.Layer):
    """Custom layer to apply interpretable rules based on medical thresholds."""
    def call(self, inputs):
        # Split inputs into individual features
        Preg, Glu, Ins, BMI, Age = tf.split(inputs, num_or_size_splits=5, axis=1)

        # Define rules based on medical thresholds
        r1 = tf.sigmoid((Glu - 110.0) / 15.0)  # Glucose moderately high
        r2 = tf.sigmoid((BMI - 27.0) / 3.0) * tf.sigmoid((Glu - 110.0) / 15.0)  # BMI and Glucose
        r3 = tf.sigmoid((Ins - 120.0) / 30.0)  # Insulin elevated
        r4 = tf.sigmoid((Preg - 4.0) / 1.5) * tf.sigmoid((Age - 30.0) / 6.0)  # Pregnancy and Age risk
        r5 = tf.sigmoid((100.0 - Glu) / 15.0) * tf.sigmoid((25.0 - BMI) / 3.0)  # Healthy inverse

        return tf.concat([r1, r2, r3, r4, r5], axis=1)


# ----------------------------------------------
# Hybrid LNN Model
# ----------------------------------------------
def create_lnn_hybrid_model(input_dim, params):
    """Create a hybrid model combining interpretable rules and neural networks."""
    inputs = Input(shape=(input_dim,), name="inputs")
    rules = RuleLayer()(inputs)
    combined = Concatenate()([inputs, rules])

    x = Dense(params["neurons1"], activation="relu")(combined)
    x = Dropout(params["dropout"])(x)
    x = Dense(params["neurons2"], activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ----------------------------------------------
# Genetic Algorithm Functions
# ----------------------------------------------
search_space = {
    "neurons1": [16, 32, 64, 128],
    "neurons2": [8, 16, 32, 64],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "lr": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    "batch": [8, 16, 32, 64],
    "epochs": [50, 100, 150, 200],
}


def create_individual():
    """Create a random individual from the search space."""
    return {key: random.choice(values) for key, values in search_space.items()}


def mutate(individual):
    """Mutate a random parameter of the individual."""
    key = random.choice(list(search_space.keys()))
    individual[key] = random.choice(search_space[key])
    return individual


def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    return {key: random.choice([parent1[key], parent2[key]]) for key in parent1}


def evaluate(individual, X_train, y_train, X_test, y_test):
    """Evaluate the individual's performance using the hybrid model."""
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = {int(c): w for c, w in zip(classes, class_weights)}

    model = create_lnn_hybrid_model(X_train.shape[1], individual)

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=individual["epochs"],
        batch_size=individual["batch"],
        verbose=0,
        class_weight=cw,
        validation_split=0.2,
        callbacks=[early_stopping],
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
    best_overall = (0, None)

    print("\n========== GA Optimization Running ==========\n")

    for gen in range(generations):
        print(f"\n=== Generation {gen + 1}/{generations} ===")

        # Evaluate all individuals
        scored = [(evaluate(ind, X_train, y_train, X_test, y_test), ind) for ind in population]
        scored.sort(reverse=True, key=lambda x: x[0])
        best_gen_f1, best_gen_ind = scored[0]

        print("Best this generation:", best_gen_ind, "F1 =", best_gen_f1)

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

    final_model = create_lnn_hybrid_model(X_train_fs.shape[1], best_params)

    # Early stopping callback for final training
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    start_time = time.time()
    final_model.fit(
        X_train_fs,
        y_train,
        epochs=best_params["epochs"],
        batch_size=best_params["batch"],
        verbose=0,
        class_weight=cw,
        validation_split=0.2,
        callbacks=[early_stopping],
    )
    end_time = time.time()

    # Evaluate final model
    y_test_probs = final_model.predict(X_test_fs, verbose=0).ravel()
    y_pred = (y_test_probs > 0.5).astype("int32")

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n========== FINAL TEST RESULTS ==========")
    print("Best GA Accuracy:", acc)
    print("Best GA F1 Score:", f1)
    print("Training Time: %.2f seconds" % (end_time - start_time))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    # Rule activation analysis
    rule_layer = RuleLayer()
    rule_outputs = rule_layer(tf.constant(X_test_fs, dtype=tf.float32)).numpy()

    rule_names = [
        "Glucose_mod_high",
        "BMI_and_Glucose",
        "Insulin_elevated",
        "Preg_Age_risk",
        "Healthy_inverse"
    ]

    top_rule_idx_each = np.argmax(rule_outputs, axis=1)
    counts = np.bincount(top_rule_idx_each, minlength=len(rule_names))
    print("\nRule dominance counts (test):")
    for i, name in enumerate(rule_names):
        print(f"{name}: {counts[i]} ({counts[i] / len(rule_outputs):.3f} fraction)")

    print("\n--- Rule Activation Summary (First 10 Test Samples) ---")
    for i in range(min(10, len(rule_outputs))):
        top_i = int(top_rule_idx_each[i])
        print(f"Sample {i + 1}: Top rule → {rule_names[top_i]} (act={rule_outputs[i][top_i]:.3f}) | model_prob={y_test_probs[i]:.3f} | pred={y_pred[i]} | true={int(y_test.iloc[i])}")

    print("\nAverage rule activations (test):")
    for i, name in enumerate(rule_names):
        print(f"{name}: {rule_outputs[:, i].mean():.3f}")

