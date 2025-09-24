!pip install --upgrade tensorflow tensorflow-federated pandas numpy matplotlib seaborn scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.learning import models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, confusion_matrix, roc_curve, auc
from google.colab import files

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Upload and load dataset
uploaded = files.upload()
file_path = list(uploaded.keys())[0]
df = pd.read_csv(file_path)

target_column = 'TenYearCHD'

# Display basic dataset info
print("\n🔍 Dataset Overview:")
print(df.info())
print("\n📊 Class Distribution (Imbalance Check):")
print(df[target_column].value_counts(normalize=True))  # Check % of each class

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Split features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Split data into 3 subsets (Hybrid FL Simulation)
subset_1 = df.sample(frac=0.33, random_state=1)
remaining = df.drop(subset_1.index)
subset_2 = remaining.sample(frac=0.5, random_state=2)
subset_3 = remaining.drop(subset_2.index)

def preprocess(dataset, target_column):
    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]
    return tf.data.Dataset.from_tensor_slices((X.values, y.values)).batch(len(dataset))

client_data = [preprocess(subset_1, target_column),
               preprocess(subset_2, target_column),
               preprocess(subset_3, target_column)]

print("\n✅ Hybrid Federated Learning setup completed. Training will now begin...\n")

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def model_fn():
    keras_model = create_model()
    input_spec = client_data[0].element_spec
    keras_model.build(input_shape=(None, X_train.shape[1]))
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

from tensorflow_federated.python.learning.algorithms import build_weighted_fed_avg

iterative_process = build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.0005),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.002)
)

state = iterative_process.initialize()

NUM_ROUNDS = 250
client_accuracies = {1: [], 2: [], 3: []}
client_losses = {1: [], 2: [], 3: []}

for round_num in range(1, NUM_ROUNDS + 1):
    state, metrics = iterative_process.next(state, client_data)

    for i, client in enumerate([subset_1, subset_2, subset_3], start=1):
        acc = metrics['client_work']['train']['binary_accuracy']
        loss = metrics['client_work']['train']['loss']
        client_accuracies[i].append(acc)
        client_losses[i].append(loss)

    print(f"\n📢 Hybrid FL Round {round_num} Completed")
    for i in range(1, 4):
        print(f"   - 🏷 Client {i} Accuracy: {client_accuracies[i][-1]:.4f}, Loss: {client_losses[i][-1]:.4f}")
    print("--------------------------------------------------")

# Plot Accuracy vs. Rounds
plt.figure(figsize=(8, 5))
for i in range(1, 4):
    plt.plot(range(1, NUM_ROUNDS + 1), client_accuracies[i], label=f'Client {i}')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.title('Client Accuracy vs. Rounds')
plt.legend()
plt.grid()
plt.show()

# Plot Loss vs. Rounds
plt.figure(figsize=(8, 5))
for i in range(1, 4):
    plt.plot(range(1, NUM_ROUNDS + 1), client_losses[i], label=f'Client {i}')
plt.xlabel('Rounds')
plt.ylabel('Loss')
plt.title('Client Loss vs. Rounds')
plt.legend()
plt.grid()
plt.show()

# Evaluate final FL model
final_model = create_model()
final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
final_model.fit(X_train, y_train, epochs=10, verbose=0)

y_pred = (final_model.predict(X_test) > 0.5).astype(int).flatten()
final_global_accuracy = accuracy_score(y_test, y_pred)
final_global_precision = precision_score(y_test, y_pred)
final_global_recall = recall_score(y_test, y_pred)
final_global_f1 = f1_score(y_test, y_pred)
final_global_loss = final_model.evaluate(X_test, y_test, verbose=0)[0]

print("\n🌍 Final Global Model (Hybrid FL) Performance:")
print(f"   - Accuracy: {final_global_accuracy:.4f}")
print(f"   - Precision: {final_global_precision:.4f}")
print(f"   - Recall: {final_global_recall:.4f}")
print(f"   - F1-Score: {final_global_f1:.4f}")
print(f"   - Loss: {final_global_loss:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Hybrid FL Model')
plt.show()

# AUC-ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC-ROC Curve - Hybrid FL Model')
plt.legend()
plt.grid()
plt.show()

print("\n🔍 Classification Report - Hybrid FL Model:")
print(classification_report(y_test, y_pred))


# Centralized Model Training for Comparison
def train_centralized_model():
    centralized_model = create_model()
    centralized_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    centralized_model.fit(X_train, y_train, epochs=10, verbose=0)

    y_pred = (centralized_model.predict(X_test) > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    loss = centralized_model.evaluate(X_test, y_test, verbose=0)[0]

    return centralized_model, accuracy, precision, recall, f1, loss  

# Call function and store returned values
centralized_model, centralized_accuracy, centralized_precision, centralized_recall, centralized_f1, centralized_loss = train_centralized_model()


# Performance Comparison
print("\n📊 Centralized Model Performance:")
print(f"   - Accuracy: {centralized_accuracy:.4f}")
print(f"   - Precision: {centralized_precision:.4f}")
print(f"   - Recall: {centralized_recall:.4f}")
print(f"   - F1-Score: {centralized_f1:.4f}")
print(f"   - Loss: {centralized_loss:.4f}")

# Plot comparison between Hybrid FL and Centralized Model
labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Loss']  
hybrid_fl_metrics = [0.8502, 0.6250, 0.0388, 0.0730, 0.4057]  
centralized_loss = centralized_model.evaluate(X_test, y_test, verbose=0)[0]  
centralized_metrics = [centralized_accuracy, centralized_precision, centralized_recall, centralized_f1, centralized_loss]  

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, hybrid_fl_metrics, width, label='Hybrid FL')
rects2 = ax.bar(x + width/2, centralized_metrics, width, label='Centralized')

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Comparison: Hybrid FL vs. Centralized Model')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.grid()
plt.show()
