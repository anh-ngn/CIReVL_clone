# TODO: use dependency injection to pass in the model.
# DONE

from transformers import AutoProcessor, AutoModel
import torch
from typing import List, Dict, Union


def preprocess(img):
    return img


class SigLIP:
    def __init__(self, model_name: str = "google/siglip-base-patch16-256-multilingual", device: Union[str, torch.device] = "cuda"):
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu")
        # self.processor = AutoProcessor.from_pretrained(
        #     "google/siglip-base-patch16-256-multilingual")
        self.processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-256-multilingual")

        self.model = AutoModel.from_pretrained(
            "google/siglip-base-patch16-256-multilingual").to(self.device)

    def encode_image(self, images: torch.Tensor):
        image_inputs = self.processor(
            images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
        return image_features

    def encode_text(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            texts = [texts]
        text_inputs = self.processor(
            text=texts, return_tensors="pt", padding="max_length").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
        return text_features

    def encode_text_batch(self, texts: List[str]):
        text_inputs = self.processor(
            text=texts, return_tensors="pt", padding="max_length").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
        return text_features

    @staticmethod
    def tokenize(texts: Union[str, List[str]], context_length: int = 77):
        if isinstance(texts, str):
            texts = [texts]
        return texts  # This function would normally handle any necessary tokenization if not using processor directly

# Example usage:
# siglip = SigLIP()
# image_features = siglip.encode_image(images)
# text_features = siglip.encode_text("a photo of a dog")
