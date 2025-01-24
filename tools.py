from datasets import build_dataset, build_data_loader
from sklearn.cluster import KMeans
import torch
import random
import numpy as np
import clip
import json

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def build_loader(dataset_name, root_path, train_preprocess=None, test_preprocess=None, batch_size=64, shot=16, seed=0):
    '''
    Retuen the loader of downstream tasks.

    params:
    dataset_name: downstream dataset name
    root_path: the path of dataset
    train_preprocess/test_preprocess: the data argumentation performed on samples
    batch_size: training batch size
    shot: the available number of samples per class
    seed: the random seed
    '''
    dataset = build_dataset(dataset_name, root_path, shot, seed)
    train_loader = build_data_loader(data_source=dataset.train_x, batch_size=batch_size, is_train=True, tfm=train_preprocess, shuffle=True)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=batch_size, is_train=False, tfm=test_preprocess, shuffle=False)
    return train_loader, test_loader, dataset.classnames

def calculate_prm(all_V, all_Y, m, device='cpu'):
    '''
    Estimate the reweighting matrix \omega_{PRM}
    
    params:
    all_V: top-k matched descriptions for training samples
    all_Y: groud-truth labels for training samples
    m: number of descriptions for each class
    device: current using device
    '''
    c = np.max(all_Y) + 1
    prm = np.zeros((c, m * c))
    for i in range(all_V.shape[0]):
        for j in range(all_V.shape[1]):
            prm[all_Y[i][0]][all_Y[i] * m + all_V[i][j]] = prm[all_Y[i][0]][all_Y[i] * m + all_V[i][j]] + 1
    prob_Y_V = prm / np.sum(prm, axis=1, keepdims=True)
    return torch.tensor(prob_Y_V.T).float().to(device)

def initialize_prm(c, m, device='cpu'):
    '''
    Initialize the reweighting matrix \omega_{PRM}

    params:
    c: number of classes in the downstream task
    m: number of descriptions for each class
    device: current using device
    '''
    prob_Y_V = np.zeros((c, m * c))
    for i in range(c):
        prob_Y_V[i][i*m:i*m+m] = 1 / m
    return torch.tensor(prob_Y_V.T).float().to(device)


def balanced_clustering(features, k, device='cpu'):
    '''
    Return the mask list for each cluster. (mask=0 if the class is not in the cluster)

    params:
    features: embeddings to be clustered
    k: cluster number (i.e., number of VPs)
    device: current using device
    '''
    kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
    labels = kmeans.labels_
    masks = []
    for i in range(k):
        mask = (labels == i).astype(int)
        masks.append(torch.tensor(mask).to(device))
    return masks

def clip_txt(classnames, clip_model, dir, m):
    '''
    Return the text embeddings and the raw texts.

    params:
    classnames: name of the labels
    clip_model: the pretrained model
    dir: the directory storing descriptions
    m: number of descriptions for each class
    '''
    data = json.load(open(dir, 'r'))
    device = next(clip_model.parameters()).device
    with torch.no_grad():
        clip_weights = []
        clip_texts = []

        for classname in classnames:

            # Tokenize the prompts
            filtered_sentences = [sentence for sentence in data[classname] if len(sentence) >= 20]
            if len(filtered_sentences) == 0:
                print(classname)
                raise ValueError("No valid attributes have been generated")
            if m < len(filtered_sentences):
                texts = random.sample(filtered_sentences, m)
            else:
                texts = random.choices(filtered_sentences, k=m)
            clip_texts.append(texts)
            token = clip.tokenize(texts).to(device)

            # Prompt ensemble
            class_embeddings = clip_model.encode_text(token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            clip_weights.append(class_embeddings)
        clip_weights = torch.stack(clip_weights, dim=0).to(device)
    return clip_weights.reshape(-1, clip_weights.shape[-1]), clip_texts