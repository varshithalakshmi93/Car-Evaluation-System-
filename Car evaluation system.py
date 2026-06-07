# ─────────────────────────────────────────────
# 3. PREPROCESSING (IMPROVED)
# ─────────────────────────────────────────────

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

X = df.drop("class", axis=1)
y = df["class"]

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Ordinal encoding preserves category ordering
category_orders = [
    ["low", "med", "high", "vhigh"],      # buying
    ["low", "med", "high", "vhigh"],      # maint
    ["2", "3", "4", "5more"],             # doors
    ["2", "4", "more"],                   # persons
    ["small", "med", "big"],              # lug_boot
    ["low", "med", "high"]                # safety
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OrdinalEncoder(categories=category_orders),
            X.columns
        )
    ]
)

X_encoded = preprocessor.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"\nTrain Samples : {X_train.shape[0]}")
print(f"Test Samples  : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 4. OPTIMIZED SVM TRAINING
# ─────────────────────────────────────────────

print("\n🚀 Training Optimized SVM...")

param_grid = {
    "C": [1, 10, 50, 100, 500],
    "gamma": [0.001, 0.01, 0.1, "scale"],
    "kernel": ["rbf"]
}

grid_search = GridSearchCV(
    estimator=SVC(probability=True),
    param_grid=param_grid,
    cv=10,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

best_svm = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

# ─────────────────────────────────────────────
# 5. FINAL EVALUATION
# ─────────────────────────────────────────────

y_pred = best_svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print("FINAL MODEL PERFORMANCE")
print("=" * 55)

print(f"Accuracy : {accuracy:.4f}")

cv_scores = cross_val_score(
    best_svm,
    X_scaled,
    y,
    cv=10,
    scoring="accuracy"
)

print(
    f"10-Fold CV Accuracy : "
    f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_encoder.classes_
    )
)

# ─────────────────────────────────────────────
# 6. CONFUSION MATRIX
# ─────────────────────────────────────────────

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_encoder.classes_,
    yticklabels=target_encoder.classes_,
    ax=ax
)

ax.set_title(
    "Optimized SVM Confusion Matrix",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(
    "/mnt/user-data/outputs/confusion_matrix.png",
    dpi=150
)

plt.close()

print("✅ Confusion Matrix Saved")

# ─────────────────────────────────────────────
# 7. FEATURE IMPORTANCE ANALYSIS
# ─────────────────────────────────────────────

feature_scores = pd.DataFrame({
    "Feature": X.columns,
    "Variance": np.var(X_encoded, axis=0)
})

feature_scores = feature_scores.sort_values(
    by="Variance",
    ascending=False
)

fig, ax = plt.subplots(figsize=(8, 5))

sns.barplot(
    data=feature_scores,
    x="Variance",
    y="Feature",
    ax=ax
)

ax.set_title(
    "Feature Importance Overview",
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(
    "/mnt/user-data/outputs/feature_importance.png",
    dpi=150
)

plt.close()

print("✅ Feature Importance Plot Saved")

# ─────────────────────────────────────────────
# 8. SAMPLE PREDICTION
# ─────────────────────────────────────────────

sample = pd.DataFrame([{
    "buying": "low",
    "maint": "low",
    "doors": "4",
    "persons": "more",
    "lug_boot": "big",
    "safety": "high"
}])

sample_enc = preprocessor.transform(sample)
sample_scaled = scaler.transform(sample_enc)

prediction = best_svm.predict(sample_scaled)[0]

print("\n🚗 Sample Prediction")

print(sample.iloc[0].to_dict())

print(
    "Predicted Class:",
    target_encoder.inverse_transform([prediction])[0].upper()
)
