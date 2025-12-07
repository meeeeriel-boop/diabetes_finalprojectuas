import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

X_TRAIN = 'X_train_prepared.csv'
X_VAL = 'X_val_prepared.csv'
X_TEST = 'X_test_prepared.csv'
DATA_CSV = 'diabetes_dataset.csv'
MODEL_OUT = os.path.join('models', 'best_model.joblib')

def detect_target(df):
    possible_targets = ['Outcome', 'diagnosed_diabetes', 'diagnosed', 'diabetes', 'target', 'label']
    for t in possible_targets:
        if t in df.columns:
            return t
    return df.columns[-1]

def load_prepared(path):
    return pd.read_csv(path, index_col=0)

def main():
    for p in [X_TRAIN, X_TEST, DATA_CSV]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    X_train = load_prepared(X_TRAIN)
    X_test = load_prepared(X_TEST)

    df = pd.read_csv(DATA_CSV)
    target_col = detect_target(df)

    # reconstruct y by index
    y_train = df.loc[X_train.index, target_col]
    y_test = df.loc[X_test.index, target_col]

    # encode if needed
    if y_train.dtype == object:
        y_train = pd.factorize(y_train)[0]
        y_test = pd.factorize(y_test)[0]

    print('Training RandomForest on prepared features...')
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train.values, y_train)

    preds = rf.predict(X_test.values)
    acc = (preds == y_test.values).mean()
    print(f'Test accuracy (prepared features): {acc:.4f}')

    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, MODEL_OUT)
    print(f'Model saved to {MODEL_OUT}')

if __name__ == '__main__':
    main()
