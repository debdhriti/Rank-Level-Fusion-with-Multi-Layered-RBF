import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import fuzzy_helper
from sklearn.metrics import accuracy_score, classification_report


def evaluateSVM(data, X_train, X_test, Y_train, Y_test):
    # KIMATU KA SVM
    X = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])

    # Convert y to 1D array
    y = np.argmax(y, axis=1)
    y_train, y_test = np.argmax(Y_train, axis=1), np.argmax(Y_test, axis=1)
    print("owai: ", y_train, y_test)

    # Splitting the dataset into the training set and test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # X_train_scaled = X_train
    # X_test_scaled = X_test

    # SVM model
    svm_model = SVC(kernel='linear', random_state=42)  # rbf kernel is commonly used for SVM
    svm_model.fit(X_train_scaled, y_train)
    decision_scores = svm_model.decision_function(X_test)
    # print("svm ka dams matrix: ", [normalise(i) for i in decision_scores])
    # print("rbf ka dams matrix: ", [normalise(i) for i in rbf_dm])
    print("svm ka dams matrix: ", fuzzy_helper.normalise(decision_scores))

    # Predictions
    y_pred = svm_model.predict(X_test_scaled)

    # Accuracy
    # print(y_test)
    print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return decision_scores, y_test

