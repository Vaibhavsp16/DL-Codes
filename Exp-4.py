import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.utils import load_img, img_to_array

url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
archive = tf.keras.utils.get_file("imagenette2-160.tgz", url, untar=True)
root = archive + "/imagenette2-160"

CLASS_NAMES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}
class_ids = sorted(CLASS_NAMES)

def load_data(split, per_class=50):
    images, captions = [], []
    for cid in class_ids:
        folder = os.path.join(root, split, cid)
        for fname in os.listdir(folder)[:per_class]:
            img = load_img(os.path.join(folder, fname), target_size=(224, 224))
            images.append(img_to_array(img))
            captions.append(f"<start> a photo of a {CLASS_NAMES[cid]} <end>")
    return preprocess_input(np.array(images)), captions

X_train_raw, train_caps = load_data("train", per_class=50)
X_test_raw, test_caps = load_data("val", per_class=10)

tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(train_caps)
vocab_size = len(tokenizer.word_index) + 1
start_id = tokenizer.word_index["<start>"]
end_id = tokenizer.word_index["<end>"]

train_seqs = tokenizer.texts_to_sequences(train_caps)
max_length = max(len(s) for s in train_seqs)

base_model = VGG16(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
X_train_feat = base_model.predict(X_train_raw)
X_test_feat = base_model.predict(X_test_raw)

img_in, txt_in, y_out = [], [], []
for i, seq in enumerate(train_seqs):
    for j in range(1, len(seq)):
        img_in.append(X_train_feat[i])
        txt_in.append(seq[:j])
        y_out.append(seq[j])

img_in = np.array(img_in)
txt_in = pad_sequences(txt_in, maxlen=max_length, padding="pre")
y_out = np.array(y_out)

feat_input = Input(shape=(X_train_feat.shape[1],))
text_input = Input(shape=(max_length,))

state = Dense(256, activation="tanh")(feat_input)
embed = Embedding(vocab_size, 128, mask_zero=True)(text_input)
lstm = LSTM(256)(embed, initial_state=[state, state])
output = Dense(vocab_size, activation="softmax")(lstm)

model = Model([feat_input, text_input], output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit([img_in, txt_in], y_out, epochs=20, batch_size=64)

def generate_caption(feat):
    tokens = [start_id]
    for _ in range(max_length):
        seq = pad_sequences([tokens], maxlen=max_length, padding="pre")
        pred_id = np.argmax(model.predict([feat.reshape(1, -1), seq], verbose=0))
        if pred_id == end_id or pred_id == 0:
            break
        tokens.append(pred_id)
    return " ".join(tokenizer.index_word[t] for t in tokens[1:])

print("\n===== Generated vs Expected Captions =====")
for i in range(5):
    expected = test_caps[i].replace("<start> ", "").replace(" <end>", "")
    gen = generate_caption(X_test_feat[i])
    print(f"Image {i+1} | Expected: {expected:<25} | Generated: {gen}")