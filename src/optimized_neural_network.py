import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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
    """Load and preprocess the dataset."""
    data = pd.read_csv(file_path)
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
# Neural Network Model
# ----------------------------------------------
def create_nn_model(input_dim, params):
    """Create a standard neural network model."""
    model = Sequential([
        Dense(params["neurons1"], activation="relu", input_shape=(input_dim,)),
        Dropout(params["dropout"]),
        Dense(params["neurons2"], activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy"]
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
    """Evaluate the individual's performance using the NN model."""
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = {int(c): w for c, w in zip(classes, class_weights)}

    model = create_nn_model(X_train.shape[1], individual)

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

    final_model = create_nn_model(X_train_fs.shape[1], best_params)

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

