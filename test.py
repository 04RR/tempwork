import torch
import torchvision
import os
import random
from PIL import Image
import numpy as np
import torchvision.transforms as T
from sklearn.neighbors import NearestNeighbors
import warnings
import builders
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_size = 64


def load_model(model_path):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


model_path = "pyssl_saves/v1/model_SimCLRv2_96_4.755.pth"
model = load_model(model_path).backbone
model = model.to(device)

# model_path = "training/dino/output/checkpoint.pth"
# model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
# model.load_state_dict(torch.load(model_path, map_location="cuda")["student"])
# model = model.to(device)


def crop(image):
    # image_crop = T.CenterCrop(image_size)(image)
    image_crop = T.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(1.0, 1.0))(
        image
    )
    return image_crop


def load_images(folder, num_images=5000):
    image_files = random.sample(os.listdir(folder), num_images)
    images, image_paths = [], []
    for file in image_files:
        image_path = os.path.join(folder, file)
        image = Image.open(image_path).convert("RGB")
        images.append((image, file))
        image_paths.append(image_path)
    return images, image_paths


def get_embeddings(images):
    embeddings = []
    for image, _ in images:
        image_crop = crop(image)
        image_tensor = T.ToTensor()(image_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(image_tensor)
        embeddings.append(embedding.cpu().numpy())
    return np.vstack(embeddings)


images, image_files = load_images("bdd100k/images/val")
embeddings = get_embeddings(images)

nbrs = NearestNeighbors(n_neighbors=10, algorithm="ball_tree").fit(embeddings)
correct = 0

for img_path in image_files:
    image = Image.open(img_path).convert("RGB")
    image_crop = crop(image)
    image_tensor = T.ToTensor()(image_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        crop_embedding = model(image_tensor).cpu().numpy()

    _, indices = nbrs.kneighbors(crop_embedding, n_neighbors=10)
    nearest_neighbors = [image_files[idx] for idx in indices[0]]

    if img_path in nearest_neighbors:
        correct += 1

print(f"Accuracy: {correct * 100 /len(images)}%")


def find_knn(image, nbrs, image_files, k=10):
    image_crop = crop(image)
    image_tensor = T.ToTensor()(image_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        crop_embedding = model(image_tensor).cpu().numpy()

    _, indices = nbrs.kneighbors(crop_embedding, n_neighbors=k)
    nearest_neighbors = [image_files[idx] for idx in indices[0]]

    return nearest_neighbors, image_crop


img_path = image_files[10]
image = Image.open(img_path).convert("RGB")

n, image_crop = find_knn(image, nbrs, image_files)

plt.figure(figsize=(10, 10))
plt.subplot(3, 4, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(3, 4, 2)
plt.imshow(image_crop)
plt.title("Crop used")
plt.axis("off")

for i in range(10):
    plt.subplot(3, 4, i + 3)
    plt.imshow(Image.open(n[i]).convert("RGB"))
    plt.title(f"Nearest Neighbor {i+1}")
    plt.axis("off")

plt.show()
