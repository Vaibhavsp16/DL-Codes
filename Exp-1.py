import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
archive = tf.keras.utils.get_file("imagenette2-160.tgz", url, untar=True)
root = archive + "/imagenette2-160"

CLASS_NAMES = {
    "n01440764": "tench",
    "n02102040": "English_springer",
    "n02979186": "cassette_player",
    "n03000684": "chain_saw",
    "n03028079": "church",
    "n03394916": "French_horn",
    "n03417042": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute",
}

train_ds = tf.keras.utils.image_dataset_from_directory(
    root + "/train", image_size=(224, 224), batch_size=32
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    root + "/val", image_size=(224, 224), batch_size=5, shuffle=True
)

class_ids = test_ds.class_names
images, labels = next(iter(test_ds))
raw = images.numpy()
true_ids = [class_ids[l] for l in labels.numpy()]
true_names = [CLASS_NAMES[c] for c in true_ids]

plt.figure(figsize=(12, 3))
for i in range(len(raw)):
    plt.subplot(1, 5, i + 1)
    plt.imshow(raw[i].astype("uint8"))
    plt.title(true_names[i])
    plt.axis("off")
plt.suptitle("Sample Test Images with Original Labels")
plt.tight_layout()
plt.show()

x = preprocess_input(raw.copy())

disp = (x[0] - x[0].min()) / (x[0].max() - x[0].min())
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1); plt.imshow(raw[0].astype("uint8")); plt.title("Original (RGB)"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(disp); plt.title("Preprocessed (BGR)"); plt.axis("off")
plt.suptitle("Original vs Preprocessed Representation")
plt.tight_layout()
plt.show()

model = VGG16(weights="imagenet")
preds = model.predict(x)

decoded = decode_predictions(preds, top=5)
for i in range(len(raw)):
    top1_id, top1_name, top1_prob = decoded[i][0]
    match = "MATCH" if top1_id == true_ids[i] else "MISMATCH"
    print(f"\n--- Image {i+1}: True = {true_names[i]} ({true_ids[i]}) | {match} ---")
    print("Top-5 Predictions (Class, Confidence):")
    for _, name, score in decoded[i]:
        print(f"  {name:<20}: {score * 100:.2f}%")

print("\n===== Preprocessing & Dimensionality Record =====")
print(f"Step 1 - Resized batch shape        : {raw.shape}  | Range: [{raw.min():.0f}, {raw.max():.0f}]")
print(f"Step 2 - Preprocessed batch shape   : {x.shape}  | Range: [{x.min():.1f}, {x.max():.1f}]")
print(f"Step 3 - CNN output (probabilities) : {preds.shape}")
print(f"Step 4 - Top-5 decoded predictions  : ({len(decoded)}, 5)")