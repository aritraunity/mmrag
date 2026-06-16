import open_clip
from PIL import Image
import os
import torch
import chromadb
import json
import requests
import base64
import numpy as np
import io

class VisionServiceHelper:

    def __init__(self) -> None:
        # Creating a disk based Persistent Client
        self.client = chromadb.PersistentClient(path='/home/aritra-mukherjee/projects/mmrag/.vectordb')
        self.collection = self.client.get_or_create_collection(
            name='fashion',
                metadata={
                    "description": "Fashion Photos"
                }
            )

        #Step 1: Load the Model and Transforms
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai'
        )

        # Step 2: Set the model to Evaluation Stage, Not Training
        self.model.eval()

        # Step 3: Initializing the tokenizer
        self.tokenizer = open_clip.get_tokenizer('ViT-B-16')
        imgs = self.get_dataset_images('/home/aritra-mukherjee/projects/mmrag/dataset/various_tagged_images', n=100)

        self.seed(imgs)
    
    def preprocess_image (self, url, verbose = False):

        img = Image.open(url).convert("RGB")
        filename = os.path.basename(url)
        print(f"Embedding image {filename} of size {img.size} and mode {img.mode}")
        _tensor = self.preprocess_val(img)
        _tensor = _tensor.unsqueeze(0)
        with torch.no_grad():
            vector = self.model.encode_image(_tensor)
            vector = vector/vector.norm(dim=-1, keepdim=True)
            vector = vector.squeeze().numpy().tolist()
        
        return vector
    
    def preprocess_base64_image (self, base64image, verbose=False):
    
        img_bytes = base64.b64decode(base64image)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        _tensor = self.preprocess_val(img)
        _tensor = _tensor.unsqueeze(0)
        with torch.no_grad():
            vector = self.model.encode_image(_tensor)
            vector = vector/vector.norm(dim=-1, keepdim=True)
            vector = vector.squeeze().numpy().tolist()
        
        return vector
    
    def preprocess_images (self, urls, verbose=False):
        _tensors = []
        for url in urls:
            _img = Image.open(url).convert("RGB")
            # Debug the Image Size and Mode
            if verbose:
                print(f"Image Size: {_img.size}\tImage Type:{_img.mode}")
            
            _tensor = self.preprocess_val(_img)
            _tensors.append(_tensor)
        
        # Stacking appends the dimension in case of multiple images, for one, use unsqueeze(0)
        _stacked = torch.stack(_tensors)
        print(_stacked.shape)
        # The tensors are now prepare for final encoding
        with torch.no_grad():
            vectors = self.model.encode_image(_stacked)
            print(f"Encoded De-normalized Vectors of shape: {vectors.shape}")
            # Norm gives you the length, dim -1 selects the last element from [10, 512]
            vectors = vectors / vectors.norm(dim=-1, keepdim=True)
            vectors = vectors.numpy().tolist()
            return vectors


    def get_dataset_images (self, root_dir, n, verbose = False):    
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

    def get_llava_caption(self, image_url):
        # open and force RGB (drops alpha channel), then re-encode as JPEG in memory
        img = Image.open(image_url).convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llava",
                "prompt": """You are a fashion expert. Describe this fashion item in detail.
                ...""",
                "images": [base64_image],
                "stream": False
            }
        )
        return response.json()["response"]


    def encode_text (self, text_query):
        text_tokens = self.tokenizer([text_query])
        with torch.no_grad():
            vectors = self.model.encode_text(text_tokens)
            vectors = vectors / vectors.norm (dim=-1, keepdim=True)
            
        return vectors.squeeze().numpy().tolist()

    def fuse_vectors (self, image_vector, caption_vector, weights = (0.6, 0.4)):

        iv = np.array (image_vector)
        cv = np.array(caption_vector)
        
        fused = weights[0] * iv + weights [1] * cv

        fused = fused / np.linalg.norm(fused)

        return fused.tolist()


    def seed (self, image_urls):
        if self.collection.count() > 0:
            print("Collection already exists. Skipping")
            return
        
        vectors = []
        ids = []
        metadatas = []

        for index, img in enumerate(image_urls):
            iv = self.preprocess_image(img)
            caption = self.get_llava_caption(img)
            cv = self.encode_text(caption)
            fused = self.fuse_vectors(iv, cv)
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
        
        self.collection.add(
            embeddings=vectors,
            ids = ids,
            metadatas=metadatas
        )        
        print("Collection Updated with Fused Vectors from Image and Text")
        

    def seed_chroma (self, image_urls):
        """
        You need Vectors, Ids, Metadatas
        """

        # If Chroma Collection has items already
        # count() returns the number of records
        if self.collection.count() > 0:
            print("Collection already exists. Skipping")
            return

        vectors = self.preprocess_images(image_urls)
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
        
        self.collection.add (
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas
        )


    def search_similar_image (self, query_image_url, n_results=5, verbose=False):
        
        # Embedding the query
        vectors = self.preprocess_images([query_image_url])
        
        results = self.collection.query(
            query_embeddings=vectors,
            n_results=n_results
        )

        return results


    # imgs = get_dataset_images('/home/aritra-mukherjee/projects/mmrag/dataset/various_tagged_images', 10, verbose=True)
    # seed(imgs)

    def query_by_image_and_text (self, query_image_url, query_caption, n_results = 5, verbose = False):
        qcv = self.encode_text (query_caption)
        qiv = self.preprocess_base64_image(query_image_url)
        fused_query = self.fuse_vectors(qiv, qcv)
        
        results = self.collection.query(
            query_embeddings=[fused_query],
            n_results=n_results
        )

        return results

    def query_by_image(self, query_image_url, n_results = 5, verbose = False):
        qiv = self.preprocess_base64_image(query_image_url)
        qcv = self.encode_text(self.get_llava_caption(query_image_url))
        fused_query = self.fuse_vectors(qiv, qcv)

        results = self.collection.query(
            query_embeddings=[fused_query],
            n_results=n_results
        )

        return results

    def query_by_image_test(self, query_image_url, n_results = 5, verbose = False):
        qiv = self.preprocess_image(query_image_url)
        qcv = self.encode_text(self.get_llava_caption(query_image_url))
        fused_query = self.fuse_vectors(qiv, qcv)

        results = self.collection.query(
            query_embeddings=[fused_query],
            n_results=n_results
        )

        return results


    def query_by_text (self, query_caption, n_results = 5, verbose = False):
        qcv = self.encode_text (query_caption)
        
        results = self.collection.query(
            query_embeddings=[qcv],
            n_results=n_results
        )

        return results

