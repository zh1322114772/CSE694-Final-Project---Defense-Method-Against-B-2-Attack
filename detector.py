import torch
import clip
import os
import pandas as pd
from PIL import Image
import random
from clip.simple_tokenizer import SimpleTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float()

def read_data():

    #read dataset
    df = pd.read_csv('data.csv')
    ret = {}

    #populate keys
    for k in df['category'].unique():
        ret[k] = {'bias': None, 'userinput': None, 'poisoned': []}

    for i in df.iterrows():
        ret[i[1][0]]['bias'] = i[1][3]
        ret[i[1][0]]['userinput'] = i[1][5]

        txt = i[1][4]

        if pd.isna(txt):
            continue
        
        ret[i[1][0]]['poisoned'].append(txt.replace(' ', '_').replace('.', ''))

    return ret


def read_images(filenames: list[str]):
    imgs_path = os.listdir('poisoned_data')
    target_set = set([i[0 : 10] for i in filenames])

    filtered_lst = [img for img in imgs_path if img[0:10] in target_set]
    images = [Image.open("poisoned_data/" + i) for i in filtered_lst]

    return images


class BiasExtractor:
    def __init__(self, device: str, model, preprocess, alpha):
        self.__device = device
        self.__model = model
        self.__preprocess = preprocess
        self.__alpha = alpha
        self.__embedding = model.token_embedding.weight.detach().clone()
        self.__embedding.requires_grad_(False)

        tok = SimpleTokenizer()
        self.__id_to_word = {v: k for k, v in tok.encoder.items()}

    def __getEosIdx(self, tokens):
        eos_idx = (tokens == 49407).nonzero(as_tuple=True)[1].item()
        return eos_idx

    def __getTokensWithBias(self, prompt: str, numBaisedWords: int):

        tokens = clip.tokenize([prompt + ","]).to(self.__device)
        eos_idx = self.__getEosIdx(tokens)
        org_eos_idx = eos_idx


        if numBaisedWords <= 0:
            raise ValueError("numBaisedWords must be greater than 0")
        
        if numBaisedWords * 2 + eos_idx + 1 > tokens.shape[1]:
            raise ValueError("numBaisedWords is too large for the given prompt")


        for _ in range(numBaisedWords - 1):

            tokens[0, eos_idx] = random.randint(1, 49405) # random token
            tokens[0, eos_idx + 1] = 267 # comma token

            eos_idx += 2  # account for added commas

        tokens[0, eos_idx - 1] = 49407  # eos token

        return tokens, org_eos_idx, eos_idx

    def extract(self, imgs: list[Image.Image], prompt: str, numBaisedWords: int = 2) -> list[str]:

        tokens, org_eos_idx, eos_idx = self.__getTokensWithBias(prompt, numBaisedWords)
        text_input = self.__model.token_embedding(tokens).detach().clone()
        text_input.requires_grad_(True)
        imgs_input = torch.stack([self.__preprocess(i) for i in imgs]).to(self.__device)
        prev_loss = float('inf')
        current_loss = float('inf')

        #get img features
        with torch.no_grad():
            image_features = self.__model.encode_image(imgs_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)


        for i in range(250):  #num iterations

            if abs(prev_loss - current_loss) < 1e-5:
                break

            prev_loss = current_loss

            #get text features
            pos = model.positional_embedding
            text_features = (text_input + pos)
            text_features = text_features.permute(1, 0, 2)  # NLD -> LND
            text_features = self.__model.transformer(text_features)
            text_features = text_features.permute(1, 0, 2)  # LND -> NLD

            # Final layernorm
            text_features = self.__model.ln_final(text_features)
            # take features from the EOT (end-of-text) token
            text_features = text_features[torch.arange(text_features.shape[0]), tokens.argmax(dim=-1)]  # [B, 512]
            # project to final text embedding space
            text_features = text_features @ self.__model.text_projection  # [B, 512]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)   


            bias_loss = -(image_features @ text_features.T).mean()

            loss = bias_loss

            print(f"Iteration {i + 1}: Loss = {loss.item()}")
            
            #optimize
            if text_input.grad is not None:
                text_input.grad.zero_()

            loss.backward()

            with torch.no_grad():
                for idx in range(org_eos_idx, eos_idx, 2):
                    input_grad = text_input.grad[0, idx]
                    text_input[0, idx] -= self.__alpha * input_grad
        
        #get final biased words
        biased_words = []
        for idx in range(org_eos_idx, eos_idx, 2):
            token_emb = text_input[0, idx]
            token_id = (self.__embedding @ token_emb).argmax().item()
            print((self.__embedding @ token_emb)[token_id])
            biased_words.append(self.__id_to_word[token_id])


        return biased_words



dataset = read_data()

ex = BiasExtractor(device, model, preprocess, 0.01)
baised = ex.extract(read_images(dataset['Political_Object']['poisoned']), "President is writing a speech for the nation", 5)

print("Extracted biased words:", baised)