import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

model = VGG16(weights="imagenet")

filters = model.get_layer("block1_conv1").get_weights()[0]
f_norm = (filters - filters.min()) / (filters.max() - filters.min())

plt.figure(figsize=(10, 2))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(f_norm[:, :, :, i])
    plt.title(f"Filter {i+1}")
    plt.axis("off")
plt.suptitle("block1_conv1 Filters (RGB)")
plt.tight_layout()
plt.show()

url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
archive = tf.keras.utils.get_file("imagenette2-160.tgz", url, untar=True)
root = archive + "/imagenette2-160"

dataset = tf.keras.utils.image_dataset_from_directory(
    root + "/val", image_size=(224, 224), batch_size=1, shuffle=True
)
image, _ = next(iter(dataset))
raw = image.numpy()[0]
x = preprocess_input(np.expand_dims(raw.copy(), 0))

plt.figure(figsize=(3, 3))
plt.imshow(raw.astype("uint8"))
plt.title("Input Image")
plt.axis("off")
plt.show()

layer_names = ["block1_conv1", "block3_conv3", "block5_conv3"]
feature_model = tf.keras.Model(
    inputs=model.input,
    outputs=[model.get_layer(n).output for n in layer_names]
)
feature_maps = feature_model.predict(x)

for name, fmap in zip(layer_names, feature_maps):
    plt.figure(figsize=(10, 2))
    for i in range(8):
        plt.subplot(1, 8, i + 1)
        plt.imshow(fmap[0, :, :, i], cmap="viridis")
        plt.axis("off")
    plt.suptitle(f"{name} Feature Maps {fmap.shape[1:]}")
    plt.tight_layout()
    plt.show()

depths = {"block1_conv1": "Early", "block3_conv3": "Intermediate", "block5_conv3": "Deep"}
interpretations = {
    "Early": "Edges, colors, and textures (high spatial resolution)",
    "Intermediate": "Corners, patterns, and object parts (medium spatial resolution)",
    "Deep": "High-level, class-specific abstract semantic features (low spatial resolution)",
}

print("\n===== Layer Feature-Map Dimensions & Interpretation =====")
for name, fmap in zip(layer_names, feature_maps):
    d = depths[name]
    print(f"Layer: {name:<14} | Depth: {d:<12} | Dim: {fmap.shape[1:]}")
    print(f"  -> Captured Info: {interpretations[d]}")