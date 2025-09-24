#  Install necessary libraries
!pip install tensorflow_federated --quiet
!pip install imbalanced-learn --quiet
!pip install --upgrade scipy==1.10.1 --quiet



#  Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_federated as tff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from google.colab import files
import random

#  Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#  Upload and load dataset
uploaded = files.upload()
file_path = list(uploaded.keys())[0]
df = pd.read_csv(file_path)

target_column = 'TenYearCHD'

#  Define model function
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

#  Function to run FL and centralized training
def run_experiment(X_data, y_data, use_smote=False):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, stratify=y_data, random_state=42)

    df_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
    df_train[target_column] = y_train

    # Create 3 clients
    subset_1 = df_train.sample(frac=0.33, random_state=1)
    remaining = df_train.drop(subset_1.index)
    subset_2 = remaining.sample(frac=0.5, random_state=2)
    subset_3 = remaining.drop(subset_2.index)

    clients = [subset_1, subset_2, subset_3]

    def preprocess(dataset):
        X = dataset.drop(columns=[target_column]).values.astype(np.float32)
        y = dataset[target_column].values.astype(np.int32)
        return tf.data.Dataset.from_tensor_slices((X, y)).batch(len(dataset))

    client_data = [preprocess(client) for client in clients]

    # Federated Model
    def model_fn():
        keras_model = create_model()
        keras_model.build(input_shape=(None, X_train.shape[1]))
        input_spec = client_data[0].element_spec
        return tff.learning.models.from_keras_model(
            keras_model,
            input_spec=input_spec,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )

    from tensorflow_federated.python.learning.algorithms import build_weighted_fed_avg
    iterative_process = build_weighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.0005),
        server_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.002)
    )

    state = iterative_process.initialize()
    NUM_ROUNDS = 250
    CLIENTS_PER_ROUND = 2

    for round_num in range(1, NUM_ROUNDS + 1):
        selected_ids = random.sample(range(3), CLIENTS_PER_ROUND)
        selected_clients = [client_data[i] for i in selected_ids]
        state, metrics = iterative_process.next(state, selected_clients)

    #  Train final FL global model
    final_model = create_model()
    final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    final_model.fit(X_train, y_train, epochs=10, verbose=0)
    y_pred = (final_model.predict(X_test) > 0.5).astype(int).flatten()

    fl_acc = accuracy_score(y_test, y_pred)
    fl_prec = precision_score(y_test, y_pred)
    fl_rec = recall_score(y_test, y_pred)
    fl_f1 = f1_score(y_test, y_pred)
    fl_loss = final_model.evaluate(X_test, y_test, verbose=0)[0]

    #  Centralized training
    centralized_model = create_model()
    centralized_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    centralized_model.fit(X_train, y_train, epochs=10, verbose=0)
    y_pred_cen = (centralized_model.predict(X_test) > 0.5).astype(int).flatten()

    cen_acc = accuracy_score(y_test, y_pred_cen)
    cen_prec = precision_score(y_test, y_pred_cen)
    cen_rec = recall_score(y_test, y_pred_cen)
    cen_f1 = f1_score(y_test, y_pred_cen)
    cen_loss = centralized_model.evaluate(X_test, y_test, verbose=0)[0]

    #  Return results
    return {
        'Hybrid_FL': [fl_acc, fl_prec, fl_rec, fl_f1, fl_loss],
        'Centralized': [cen_acc, cen_prec, cen_rec, cen_f1, cen_loss]
    }

#  Step 1: Run WITHOUT SMOTE
print("\n Running Experiment WITHOUT SMOTE...\n")
df.fillna(df.mean(), inplace=True)
X = df.drop(columns=[target_column])
y = df[target_column]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results_without_smote = run_experiment(X_scaled, y)

# ⚡ Step 2: Run WITH SMOTE
print("\n⚡ Running Experiment WITH SMOTE...\n")
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)

results_with_smote = run_experiment(X_bal, y_bal)

#  Step 3: Combined Results Plotting
metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Loss']

hybrid_fl_without_smote = results_without_smote['Hybrid_FL']
centralized_without_smote = results_without_smote['Centralized']
hybrid_fl_with_smote = results_with_smote['Hybrid_FL']
centralized_with_smote = results_with_smote['Centralized']

x = np.arange(len(metrics_labels))
width = 0.2  # bar width

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

rects1 = ax.bar(x - width*1.5, hybrid_fl_without_smote, width, label='Hybrid FL (No SMOTE)', color='blue')
rects2 = ax.bar(x - width/2, centralized_without_smote, width, label='Centralized (No SMOTE)', color='orange')
rects3 = ax.bar(x + width/2, hybrid_fl_with_smote, width, label='Hybrid FL (With SMOTE)', color='green')
rects4 = ax.bar(x + width*1.5, centralized_with_smote, width, label='Centralized (With SMOTE)', color='red')

ax.set_xlabel('Metrics', fontsize=14)
ax.set_ylabel('Scores', fontsize=14)
ax.set_title('Comparison: Hybrid FL vs Centralized Model (With and Without SMOTE)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, fontsize=12)
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('combined_results_comparison.png')
plt.show()
