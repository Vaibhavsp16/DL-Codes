import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import load_img, img_to_array

url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
archive = tf.keras.utils.get_file("imagenette2-160.tgz", url, untar=True)
root = archive + "/imagenette2-160"

CLASS_NAMES = ["tench", "English_springer", "cassette_player", "chain_saw", "church",
               "French_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute"]
folder_names = sorted(os.listdir(root + "/train"))

def load_data(split, per_class=100):
    images, labels = [], []
    for idx, folder in enumerate(folder_names):
        dir_path = os.path.join(root, split, folder)
        for fname in os.listdir(dir_path)[:per_class]:
            img = load_img(os.path.join(dir_path, fname), target_size=(224, 224))
            images.append(img_to_array(img))
            labels.append(idx)
    return preprocess_input(np.array(images)), np.array(labels)

X_train_raw, y_train = load_data("train", per_class=100)
X_test_raw, y_test = load_data("val", per_class=20)

base_model = VGG16(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
X_train = base_model.predict(X_train_raw)
X_test = base_model.predict(X_test_raw)

classifier = Sequential([
    Dense(len(CLASS_NAMES), activation="softmax", input_shape=(X_train.shape[1],))
])
classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
classifier.fit(X_train, y_train, epochs=15, batch_size=32)

y_pred = np.argmax(classifier.predict(X_test), axis=1)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print(f"\nFeature-vector dimension : {X_train.shape[1]}")
print(f"Classification accuracy  : {acc * 100:.2f}%")
print(f"Precision                : {prec:.4f}")
print(f"Recall                   : {rec:.4f}")
print(f"F1-score                 : {f1:.4f}")
print(f"\nConfusion Matrix:\n{cm}")