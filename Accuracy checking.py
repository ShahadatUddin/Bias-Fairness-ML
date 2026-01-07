
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("D13_Liver_disease.csv")

# 80:20 split (no stratification), last column is the outcome
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Remove 160 male instances with class 1 from training
train_df = X_train.copy()
train_df["Diagnosis"] = y_train.values
mask = (train_df["Gender"] == 1) & (train_df["Diagnosis"] == 1)
to_drop = train_df.index[mask][:160]      # first 160 matches
train_df = train_df.drop(index=to_drop)

X_train_f = train_df.drop(columns=["Diagnosis"])
y_train_f = train_df["Diagnosis"]

# Build models (scaling where appropriate)
models = {
    "ANN": Pipeline([("scaler", StandardScaler()),
                     ("clf", MLPClassifier(hidden_layer_sizes=(100,),
                                           max_iter=500, random_state=42))]),
    "DT": DecisionTreeClassifier(random_state=42),
    "KNN": Pipeline([("scaler", StandardScaler()),
                     ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "LR": Pipeline([("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000,
                                               solver="lbfgs", random_state=42))]),
    "SVM": Pipeline([("scaler", StandardScaler()),
                     ("clf", SVC(kernel="rbf", gamma="scale", C=1.0,
                                 random_state=42))]),
}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train_f, y_train_f)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.6f}")
``
