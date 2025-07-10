from sklearn.model_selection import train_test_split


def split_data(features, labels, seed=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.3, stratify=labels, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)