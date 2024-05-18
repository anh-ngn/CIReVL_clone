# wrappers.py
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor, AutoModel
import torch


class MBLIPWrapper:
    def __init__(self):
        self.processor = Blip2Processor.from_pretrained("Gregor/mblip-mt0-xl")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Gregor/mblip-mt0-xl", device_map="auto")

    def generate(self, inputs, **kwargs):
        return self.model.generate(**inputs, **kwargs)

    def decode(self, output):
        return self.processor.batch_decode(output, skip_special_tokens=True)

    def process(self, image, question, **kwargs):
        return self.processor(image, question, return_tensors="pt").to("cuda")


class SigLIPWrapper:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-256-multilingual")
        self.model = AutoModel.from_pretrained(
            "google/siglip-base-patch16-256-multilingual")

    def encode_image(self, images):
        with torch.no_grad():
            return self.model.get_image_features(**self.processor(images=images, return_tensors="pt"))

    def encode_text(self, texts):
        with torch.no_grad():
            return self.model.get_text_features(**self.processor(text=texts, return_tensors="pt", padding="max_length"))
