#  Traditional Machine Learning Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Train and evaluate traditional ML models
def evaluate_ml_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_prob)

    print(f"\n {model_name} Performance:")
    print(f"   - Accuracy: {acc:.4f}")
    print(f"   - Precision: {prec:.4f}")
    print(f"   - Recall: {rec:.4f}")
    print(f"   - F1-Score: {f1:.4f}")
    print(f"   - Loss: {loss:.4f}")

    return [acc, prec, rec, f1, loss]

# Evaluate SVM
svm_model = SVC(probability=True, random_state=42)
svm_metrics = evaluate_ml_model(svm_model, "Support Vector Machine")

# Evaluate Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_metrics = evaluate_ml_model(rf_model, "Random Forest")

# Evaluate Logistic Regression
lr_model = LogisticRegression(max_iter=500, random_state=42)
lr_metrics = evaluate_ml_model(lr_model, "Logistic Regression")

#  Final Comparison Plot
all_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Loss']

# Already available from your previous code
hybrid_fl_metrics = [final_global_accuracy, final_global_precision, final_global_recall, final_global_f1, final_global_loss]
centralized_metrics = [centralized_accuracy, centralized_precision, centralized_recall, centralized_f1, centralized_loss]

# Collect all metrics
all_models_metrics = [
    hybrid_fl_metrics,
    centralized_metrics,
    svm_metrics,
    rf_metrics,
    lr_metrics
]

model_names = ['Hybrid FL', 'Centralized DL', 'SVM', 'Random Forest', 'Logistic Regression']

# Plot
x = np.arange(len(all_labels))
width = 0.15

fig, ax = plt.subplots(figsize=(12, 6))

for i, metrics in enumerate(all_models_metrics):
    ax.bar(x + i*width - width*2, metrics, width, label=model_names[i])

ax.set_xlabel('Metrics', fontsize=14)
ax.set_ylabel('Score', fontsize=14)
ax.set_title('Performance Comparison: Hybrid FL vs. Centralized DL vs. Traditional ML Models', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(all_labels, fontsize=12)
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
