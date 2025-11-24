import torch
import clip
import os
import pandas as pd
from PIL import Image
import spacy

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
nlp = spacy.load("en_core_web_sm")

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
    images = [preprocess(Image.open("poisoned_data/" + i)) for i in filtered_lst]
    images = torch.stack(images).to(device)

    return images

data = read_data()
results = []

#compute clip scores
for category in data.keys():
    bias = data[category]['bias']
    poisoned = data[category]['poisoned']
    userinput = data[category]['userinput']

    if len(poisoned) == 0:
        continue


    doc = nlp(bias)

    bycomma = bias.replace(' ', ',')
    byadjNoun = ','.join([token.text for token in doc if token.pos_ in ['NOUN', 'ADJ']])

    cliptext = clip.tokenize([userinput, bycomma + ',' + userinput, byadjNoun+ ',' + userinput, userinput + ',' +  bycomma, userinput + ',' + byadjNoun]).to(device)
    images = read_images(poisoned)

    with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(cliptext)
            
            # L2-normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).cpu().numpy()
            similarity_avg = similarity.mean(axis=0)

            results.append({
                'category': category,
                'input': similarity_avg[0],
                'bias + input': similarity_avg[1],
                'bias(Adj & Noun) + input': similarity_avg[2],
                'input + bias': similarity_avg[3],
                'input + bias(Adj & Noun)': similarity_avg[4],})


res_df = pd.DataFrame(results)
res_df.to_csv('best_poisoned_loc_results.csv', index=False)
