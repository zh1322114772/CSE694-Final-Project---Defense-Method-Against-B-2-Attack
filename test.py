import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("poisoned_data/A_bald_president_wearing_a_red_tie_writing_a_speech_at_his_de_2c538586-8b97-4c0b-add5-d7b13174db1b_0.png")).unsqueeze(0).to(device)
text = clip.tokenize(["President is writing a speech for the nation,mariah</w>,chicagofire</w>,ðŁĵį:</w>,scar</w>", "Bald, president, wearing, a, red, tie, President is writing a speech for the nation"]).to(device)

print(text)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    # L2-normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)

    print(image_features @ text_features.T)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
