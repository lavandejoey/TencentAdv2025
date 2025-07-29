# utils/encoders.py
import torch
from transformers import BertTokenizer, BertModel
import clip
from PIL import Image
import requests
from io import BytesIO

class TextImageEncoder:
    def __init__(self, device="cpu", max_len=32):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(device).eval()
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.max_len = max_len

    def encode_text(self, texts):
        toks = self.tokenizer(texts, padding="max_length", truncation=True,
                              max_length=self.max_len, return_tensors="pt")
        toks = {k:v.to(self.device) for k,v in toks.items()}
        with torch.no_grad():
            return self.bert(**toks).last_hidden_state  # [B,L,H]

    def encode_image(self, img_urls):
        imgs = []
        for u in img_urls:
            if not u:
                imgs.append(torch.zeros(3,224,224))
                continue
            try:
                r = requests.get(u, timeout=5)
                img = Image.open(BytesIO(r.content)).convert("RGB")
                imgs.append(self.clip_preprocess(img).to(self.device))
            except:
                imgs.append(torch.zeros(3,224,224))
        imgs = torch.stack(imgs)
        with torch.no_grad():
            return self.clip_model.encode_image(imgs)  # [B,512]
