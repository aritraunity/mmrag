import open_clip
from PIL import Image
import os
import torch
import chromadb
import json
import requests
import base64
import numpy as np

# Creating a disk based Persistent Client
client = chromadb.PersistentClient(path='/home/aritra-mukherjee/projects/mmrag/.vectordb')
collection = client.get_or_create_collection(
    name='fashion',
        metadata={
            "description": "Fashion Photos"
        }
    )

#Step 1: Load the Model and Transforms
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    'ViT-B-16', pretrained='openai'
)

# Step 2: Set the model to Evaluation Stage, Not Training
model.eval()

# Step 3: Initializing the tokenizer
tokenizer = open_clip.get_tokenizer('ViT-B-16')

# Step 4: A method to encode the Images to Vectors

def preprocess_image (url, verbose = False):

    img = Image.open(url).convert("RGB")
    filename = os.path.basename(url)
    print(f"Embedding image {filename} of size {img.size} and mode {img.mode}")
    _tensor = preprocess_val(img)
    _tensor = _tensor.unsqueeze(0)
    with torch.no_grad():
        vector = model.encode_image(_tensor)
        vector = vector/vector.norm(dim=-1, keepdim=True)
        vector = vector.squeeze().numpy().tolist()
    
    return vector


def preprocess_images (urls, verbose=False):
    _tensors = []
    for url in urls:
        _img = Image.open(url).convert("RGB")
        # Debug the Image Size and Mode
        if verbose:
            print(f"Image Size: {_img.size}\tImage Type:{_img.mode}")
        
        _tensor = preprocess_val(_img)
        _tensors.append(_tensor)
    
    # Stacking appends the dimension in case of multiple images, for one, use unsqueeze(0)
    _stacked = torch.stack(_tensors)
    print(_stacked.shape)
    # The tensors are now prepare for final encoding
    with torch.no_grad():
        vectors = model.encode_image(_stacked)
        print(f"Encoded De-normalized Vectors of shape: {vectors.shape}")
        # Norm gives you the length, dim -1 selects the last element from [10, 512]
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        vectors = vectors.numpy().tolist()
        return vectors

def get_dataset_images (root_dir, n, verbose = False):    
    """
    """
    imgs = []
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            imgs.append(os.path.join(root_dir, filename))
    if verbose:
        for img in imgs[:n]:
            print(f"Found Image: {img}\n")
    return imgs [:n]

def get_llava_caption (image_url):

    # Open Image Bytes
    with open (image_url, "rb") as f:
        image_bytes = f.read()
    
    # Encode Bytes to B64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llava",
            "prompt": """You are a fashion expert. Describe this fashion item in detail.
            Include:
            - Garment type
            - Colors and patterns
            - Style category (casual, formal, streetwear etc)
            - Occasion suitability
            - Any notable design details
            Keep it to 3-4 sentences.""",
            "images": [base64_image],
            "stream": False
        }
    )

    return response.json()["response"]


def encode_text (text_query):
    text_tokens = tokenizer([text_query])
    with torch.no_grad():
        vectors = model.encode_text(text_tokens)
        vectors = vectors / vectors.norm (dim=-1, keepdim=True)
        
    return vectors.squeeze().numpy().tolist()

def fuse_vectors (image_vector, caption_vector, weights = (0.6, 0.4)):

    iv = np.array (image_vector)
    cv = np.array(caption_vector)
    
    fused = weights[0] * iv + weights [1] * cv

    fused = fused / np.linalg.norm(fused)

    return fused.tolist()


def seed (image_urls):
    if collection.count() > 0:
        print("Collection already exists. Skipping")
        return
    
    vectors = []
    ids = []
    metadatas = []

    for index, img in enumerate(image_urls):
        iv = preprocess_image(img)
        caption = get_llava_caption(img)
        cv = encode_text(caption)
        fused = fuse_vectors(iv, cv)
        vectors.append (fused)

        file_name = os.path.basename(img).split('.')[0]
        ids.append(f"{file_name}+{index}")
        meta = {
            "index": index,
            "url": img,
            "caption": caption,
            "name": file_name
        }
        metadatas.append(meta)
    
    collection.add(
        embeddings=vectors,
        ids = ids,
        metadatas=metadatas
    )        
    print("Collection Updated with Fused Vectors from Image and Text")
    

def seed_chroma (image_urls):
    """
    You need Vectors, Ids, Metadatas
    """

    # If Chroma Collection has items already
    # count() returns the number of records
    if collection.count() > 0:
        print("Collection already exists. Skipping")
        return

    vectors = preprocess_images(image_urls)
    ids = []
    metadatas = []
    for i, img in enumerate(image_urls):
        file_name = os.path.basename(img)
        file_name = str(file_name).split('.')[0]

        id = f"{file_name}+{i}"
        ids.append(id)
        metadata = {
            "index": i,
            "file_name": file_name,
            "path": img
        }
        metadatas.append(metadata)
    
    collection.add (
        embeddings=vectors,
        ids=ids,
        metadatas=metadatas
    )


def search_similar_image (query_image_url, n_results=5, verbose=False):
    
    # Embedding the query
    vectors = preprocess_images([query_image_url])
    
    results = collection.query(
        query_embeddings=vectors,
        n_results=n_results
    )

    return results


# imgs = get_dataset_images('/home/aritra-mukherjee/projects/mmrag/dataset/various_tagged_images', 10, verbose=True)
# seed(imgs)

def query_vision_servie (query_image_url, query_caption, n_results = 5, verbose = False):
    qcv = encode_text (query_caption)
    qiv = preprocess_image(query_image_url)
    fused_query = fuse_vectors(qiv, qcv)
    
    results = collection.query(
        query_embeddings=[fused_query],
        n_results=n_results
    )

    return results




