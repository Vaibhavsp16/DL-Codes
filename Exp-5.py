import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Concatenate, Conv2D, Conv2DTranspose, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import load_img, img_to_array

url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
archive = tf.keras.utils.get_file("imagenette2-160.tgz", url, untar=True)
root = archive + "/imagenette2-160"

CLASS_NAMES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03888257": "parachute",
}
class_ids = sorted(CLASS_NAMES)

images, captions = [], []
for cid in class_ids:
    folder = os.path.join(root, "train", cid)
    for fname in os.listdir(folder)[:40]:
        img = load_img(os.path.join(folder, fname), target_size=(32, 32))
        images.append(img_to_array(img))
        captions.append(f"a photo of a {CLASS_NAMES[cid]}")

images = (np.array(images) - 127.5) / 127.5

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
text_seqs = pad_sequences(tokenizer.texts_to_sequences(captions), maxlen=6)

text_in = Input(shape=(6,))
text_feat = Flatten()(Embedding(vocab_size, 32)(text_in))
text_feat = Dense(32, activation="relu")(text_feat)
text_encoder = Model(text_in, text_feat)
text_vecs = text_encoder.predict(text_seqs, verbose=0)

noise_in = Input(shape=(100,))
txt_gen_in = Input(shape=(32,))
x = Concatenate()([noise_in, txt_gen_in])
x = Dense(8 * 8 * 128, activation="relu")(x)
x = Reshape((8, 8, 128))(x)
x = Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu")(x)
img_out = Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh")(x)
generator = Model([noise_in, txt_gen_in], img_out)

img_in = Input(shape=(32, 32, 3))
txt_disc_in = Input(shape=(32,))
d = Conv2D(64, 4, strides=2, padding="same", activation="leaky_relu")(img_in)
d = Conv2D(128, 4, strides=2, padding="same", activation="leaky_relu")(d)
d = Flatten()(d)
d = Concatenate()([d, txt_disc_in])
validity = Dense(1, activation="sigmoid")(d)
discriminator = Model([img_in, txt_disc_in], validity)
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss="binary_crossentropy", metrics=["accuracy"])

discriminator.trainable = False
gan_noise = Input(shape=(100,))
gan_txt = Input(shape=(32,))
fake_img = generator([gan_noise, gan_txt])
gan_valid = discriminator([fake_img, gan_txt])
gan = Model([gan_noise, gan_txt], gan_valid)
gan.compile(optimizer=Adam(0.0002, 0.5), loss="binary_crossentropy")

batch_size = 32
epochs = 20
real_y = np.ones((batch_size, 1))
fake_y = np.zeros((batch_size, 1))

for epoch in range(epochs):
    idx = np.random.randint(0, len(images), batch_size)
    real_imgs, real_txts = images[idx], text_vecs[idx]

    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_imgs = generator.predict([noise, real_txts], verbose=0)

    d_loss_real = discriminator.train_on_batch([real_imgs, real_txts], real_y)
    d_loss_fake = discriminator.train_on_batch([fake_imgs, real_txts], fake_y)
    g_loss = gan.train_on_batch([noise, real_txts], real_y)

    if (epoch + 1) % 5 == 0:
        d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

test_texts = ["a photo of a parachute", "a photo of a church"]
test_seqs = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=6)
test_vecs = text_encoder.predict(test_seqs, verbose=0)
test_noise = np.random.normal(0, 1, (len(test_texts), 100))
gen_imgs = generator.predict([test_noise, test_vecs], verbose=0)
gen_imgs = 0.5 * gen_imgs + 0.5

plt.figure(figsize=(8, 4))
for i in range(len(test_texts)):
    plt.subplot(2, 2, 2 * i + 1)
    plt.imshow(0.5 * images[i] + 0.5)
    plt.title(f"Real: {captions[i]}")
    plt.axis("off")
    plt.subplot(2, 2, 2 * i + 2)
    plt.imshow(gen_imgs[i])
    plt.title(f"Generated: {test_texts[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
