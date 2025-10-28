def main():

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np

    from src.data_utils import load_data
    from src.model import create_advanced_3dcnn_model

    # Paths
    train_videos_path = 'data/train'
    test_videos_path = 'data/test'
    train_labels = pd.read_csv('labels/train.csv')
    test_labels = pd.read_csv('labels/test.csv')

    # Visualize class distribution
    class_counts = train_labels['tag'].value_counts()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Distribution of Training Labels')
    plt.xlabel('Class Labels')
    plt.ylabel('Number of Videos')
    plt.xticks(rotation=90)
    plt.show()

    num_classes = train_labels['tag'].nunique()
    print(f'Number of classes: {num_classes}')

    # Load data
    X_train, y_train = load_data(train_labels, train_videos_path, num_classes)
    X_test, y_test = load_data(test_labels, test_videos_path, num_classes)

    # Split training & validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create model
    input_shape = (16, 112, 112, 3)
    model = create_advanced_3dcnn_model(input_shape, num_classes)

    # Train
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.2f}')

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Classification report
    class_report = classification_report(y_true_classes, y_pred_classes, target_names=train_labels['tag'].unique())
    print(class_report)

    # Training history plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
