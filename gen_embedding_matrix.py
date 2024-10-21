import numpy as np
import torch 
import os 
from omegaconf import OmegaConf
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
import clip
import argparse
import pandas as pd 

from utils_alg import sample_model, load_model_from_config, load_model_from_config_compvis
from constants import *
from ldm.models.diffusion.ddim import DDIMSampler

import torch.nn as nn

@torch.no_grad()
def get_vocab(model, model_name, vocab='EN3K'):
    if vocab == 'CLIP':
        if model_name == 'SD-v1-4':
            tokenizer_vocab = model.cond_stage_model.tokenizer.get_vocab()
        elif model_name == 'SD-v2-1':
            tokenizer_vocab = model.cond_stage_model.tokenizer.encoder
        else:
            raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")
    elif vocab == 'EN3K':
        tokenizer_vocab = get_english_tokens()
    else:
        raise ValueError("vocab should be either 'CLIP' or 'EN3K'")
    
    return tokenizer_vocab

@torch.no_grad()
def create_embedding_matrix(model, start=0, end=LEN_TOKENIZER_VOCAB, model_name='SD-v1-4', save_mode='array', remove_end_token=False, vocab='EN3K'):

    tokenizer_vocab = get_vocab(model, model_name, vocab=vocab)

    if save_mode == 'array':
        all_embeddings = []
        for token in tokenizer_vocab:
            if tokenizer_vocab[token] < start or tokenizer_vocab[token] >= end:
                continue
            # print(token, tokenizer_vocab[token])
            if remove_end_token:
                token_ = token.replace('</w>','')
            else:
                token_ = token
            emb_ = model.get_learned_conditioning([token_])
            all_embeddings.append(emb_)
        return torch.cat(all_embeddings, dim=0) # shape (49408, 77, 768)
    elif save_mode == 'dict':
        all_embeddings = {}
        for token in tokenizer_vocab:
            if tokenizer_vocab[token] < start or tokenizer_vocab[token] >= end:
                continue
            # print(token, tokenizer_vocab[token])
            if remove_end_token:
                token_ = token.replace('</w>','')
            else:
                token_ = token
            emb_ = model.get_learned_conditioning([token_])
            all_embeddings[token] = emb_
        return all_embeddings
    else:
        raise ValueError("save_mode should be either 'array' or 'dict'")

@torch.no_grad()
def save_embedding_matrix(model, model_name='SD-v1-4', save_mode='array', vocab='EN3K'):
    if vocab == 'CLIP':
        for start in range(0, LEN_TOKENIZER_VOCAB, 5000):
            print(f"start: {start} / {LEN_TOKENIZER_VOCAB}")
            end = min(LEN_TOKENIZER_VOCAB, start+5000)
            embedding_matrix = create_embedding_matrix(model, start=start, end=end, model_name=model_name, save_mode=save_mode)
            if model_name == 'SD-v1-4':
                torch.save(embedding_matrix, f'models/embedding_matrix_{start}_{end}_{save_mode}.pt')
            elif model_name == 'SD-v2-1':
                torch.save(embedding_matrix, f'models/embedding_matrix_{start}_{end}_{save_mode}_v2-1.pt')
    elif vocab == 'EN3K':
        embedding_matrix = create_embedding_matrix(model, start=0, end=LEN_EN_3K_VOCAB, model_name=model_name, save_mode=save_mode, vocab='EN3K')
        if model_name == 'SD-v1-4':
            torch.save(embedding_matrix, f'models/embedding_matrix_{save_mode}_EN3K.pt')
        elif model_name == 'SD-v2-1':
            torch.save(embedding_matrix, f'models/embedding_matrix_{save_mode}_EN3K_v2-1.pt')
    else:
        raise ValueError("vocab should be either 'CLIP' or 'EN3K'")

def get_english_tokens():
    data_path = 'data/english_3000.csv'
    df = pd.read_csv(data_path)
    vocab = {}
    for ir, row in df.iterrows():
        vocab[row['word']] = ir
    assert(len(vocab) == LEN_EN_3K_VOCAB)
    return vocab


def retrieve_embedding_token(model_name, query_token, vocab='EN3K'):
    if vocab == 'CLIP':
        for start in range(0, LEN_TOKENIZER_VOCAB, 5000):
            # print(f"start: {start} / {LEN_TOKENIZER_VOCAB}")
            end = min(LEN_TOKENIZER_VOCAB, start+5000)
            if model_name == 'SD-v1-4':
                embedding_matrix = torch.load(f'models/embedding_matrix_{start}_{end}_dict.pt')
            elif model_name == 'SD-v2-1':
                embedding_matrix = torch.load(f'models/embedding_matrix_{start}_{end}_dict_v2-1.pt')
            else:
                raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")
            if query_token in embedding_matrix:
                return embedding_matrix[query_token]
    elif vocab == 'EN3K':
        if model_name == 'SD-v1-4':
            embedding_matrix = torch.load(f'models/embedding_matrix_dict_EN3K.pt')
        elif model_name == 'SD-v2-1':
            embedding_matrix = torch.load(f'models/embedding_matrix_dict_EN3K_v2-1.pt')
        else:
            raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")
        if query_token in embedding_matrix:
            return embedding_matrix[query_token]
    else:
        raise ValueError("vocab should be either 'CLIP' or 'EN3K'")

def detect_special_tokens(text):
    text = text.lower()
    for i in range(len(text)):
        if text[i] not in 'abcdefghijklmnopqrstuvwxyz</>':
            return True
    return False

@torch.no_grad()
def search_closest_tokens(concept, model, k=5, reshape=True, sim='cosine', model_name='SD-v1-4', ignore_special_tokens=True, vocab='EN3K'):
    """
    Given a concept, i.e., "nudity", search for top-k closest tokens in the embedding space
    """

    tokenizer_vocab = get_vocab(model, model_name, vocab=vocab)
    # inverse the dictionary
    tokenizer_vocab_indexing = {v: k for k, v in tokenizer_vocab.items()}

    # Get the embedding of the concept
    concept_embedding = model.get_learned_conditioning([concept]) # shape (1, 77, 768)

    # Calculate the cosine similarity between the concept and all tokens
    # load the embedding matrix 
    all_similarities = []
    
    if vocab == 'CLIP':
        for start in range(0, LEN_TOKENIZER_VOCAB, 5000):
            # print(f"start: {start} / {LEN_TOKENIZER_VOCAB}")
            end = min(LEN_TOKENIZER_VOCAB, start+5000)
            if model_name == 'SD-v1-4':
                embedding_matrix = torch.load(f'models/embedding_matrix_{start}_{end}_array.pt')
            elif model_name == 'SD-v2-1':
                embedding_matrix = torch.load(f'models/embedding_matrix_{start}_{end}_array_v2-1.pt')
            else:
                raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")
            
            if reshape == True:
                concept_embedding = concept_embedding.view(concept_embedding.size(0), -1)
                embedding_matrix = embedding_matrix.view(embedding_matrix.size(0), -1)
            if sim == 'cosine':
                similarities = F.cosine_similarity(concept_embedding, embedding_matrix, dim=-1)
            elif sim == 'l2':
                similarities = - F.pairwise_distance(concept_embedding, embedding_matrix, p=2)
            all_similarities.append(similarities)
    elif vocab == 'EN3K':
        if model_name == 'SD-v1-4':
            embedding_matrix = torch.load(f'models/embedding_matrix_array_EN3K.pt')
        elif model_name == 'SD-v2-1':
            embedding_matrix = torch.load(f'models/embedding_matrix_array_EN3K_v2-1.pt')
        else:
            raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")
        if reshape == True:
            concept_embedding = concept_embedding.view(concept_embedding.size(0), -1)
            embedding_matrix = embedding_matrix.view(embedding_matrix.size(0), -1)
        if sim == 'cosine':
            similarities = F.cosine_similarity(concept_embedding, embedding_matrix, dim=-1)
        elif sim == 'l2':
            similarities = - F.pairwise_distance(concept_embedding, embedding_matrix, p=2)
        all_similarities.append(similarities)
    else:
        raise ValueError("vocab should be either 'CLIP' or 'EN3K'")

    similarities = torch.cat(all_similarities, dim=0)
    # sorting the similarities
    sorted_similarities, indices = torch.sort(similarities, descending=True)
    print(f"sorted_similarities: {sorted_similarities[:10]}")
    print(f"indices: {indices[:10]}")

    sim_dict = {}
    for im, i in enumerate(indices):
        if ignore_special_tokens:
            if detect_special_tokens(tokenizer_vocab_indexing[i.item()]):
                continue
        token = tokenizer_vocab_indexing[i.item()]
        sim_dict[token] = sorted_similarities[im]
    
    top_k_tokens = list(sim_dict.keys())[:k]
    print(f"Top-{k} closest tokens to the concept {concept} are: {top_k_tokens}")
    return top_k_tokens, sim_dict


@torch.no_grad()
def search_closest_tokens_in_set(tokenizer_vocab, concept, model, k=5, reshape=True, sim='cosine', ignore_special_tokens=True):
    """
    Given a concept, i.e., "nudity", search for top-k closest tokens in the embedding space
    """

    # Get the embedding of the concept
    concept_embedding = model.get_learned_conditioning([concept]) # shape (1, 77, 768)

    # Calculate the cosine similarity between the concept and all tokens
    # load the embedding matrix 
    all_similarities = []
    
    for token in tokenizer_vocab:
        token_embedding = model.get_learned_conditioning([token])
        if reshape == True:
            concept_embedding = concept_embedding.view(concept_embedding.size(0), -1)
            token_embedding = token_embedding.view(token_embedding.size(0), -1)
        if sim == 'cosine':
            similarities = F.cosine_similarity(concept_embedding, token_embedding, dim=-1)
        elif sim == 'l2':
            similarities = - F.pairwise_distance(concept_embedding, token_embedding, p=2)
        all_similarities.append(similarities)

    similarities = torch.cat(all_similarities, dim=0)
    # sorting the similarities
    sorted_similarities, indices = torch.sort(similarities, descending=True)
    print(f"sorted_similarities: {sorted_similarities[:10]}")
    print(f"indices: {indices[:10]}")

    sim_dict = {}
    for im, i in enumerate(indices):
        if ignore_special_tokens:
            if detect_special_tokens(tokenizer_vocab[i.item()]):
                continue
        token = tokenizer_vocab[i.item()]
        sim_dict[token] = sorted_similarities[im]
    
    top_k_tokens = list(sim_dict.keys())[:k]
    print(f"Top-{k} closest tokens to the concept {concept} are: {top_k_tokens}")
    return top_k_tokens, sim_dict

@torch.no_grad()
def search_closest_output(concept, model, sampler, tokens_embedding, start_guidance, time_step, start_concept, k=5, sim='l2'):
    """
    given a concept, i.e., "nudity", search for top-k closest tokens in the token vocabulary that produces the closest output with a given model
    TODO: using CLIP score as a similarity metric 
    Args: 
        concept: str, the concept to search for
        model: the model to use for inference
        tokens_embedding: dictionary containing the token embeddings that are used to search for the closest tokens
            example: {'nudity': tensor([0.1, 0.2, ..., 0.3]), 'sexy': tensor([0.2, 0.3, ..., 0.4]), ...}
            token embeddings are of shape (1, 77, 768)
        k: int, the number of top-k tokens to return
    """

    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    # till_T = 0 means the model will sample till the end of the diffusion process 
    # till_T = T means the starting point of the diffusion process
    # the smaller the value of till_T, the more denoising is applied to the image
    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                 cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, s, DDIM_ETA,
                                                                 start_code=code, till_T=t, verbose=False)
    device = model.cond_stage_model.device
    # fix start code 
    start_code = torch.randn(1, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8).to(device)

    # get the condition 
    if start_concept is None:
        start_cond = model.get_learned_conditioning([STARTING_CONCEPT])
    else:
        start_cond = model.get_learned_conditioning([start_concept])

    # if start from a specific time step t 
    if time_step is not None:
        z = quick_sample_till_t(start_cond, start_guidance, start_code, time_step)
    else: 
        # start with noisy 
        z = start_code
    
    # get the output of the model w.r.t. the input concept 
    og_num = round((int(time_step)/DDIM_STEPS)*1000)
    og_num_lim = round((int(time_step + 1)/DDIM_STEPS)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,))
    
    @torch.no_grad()
    def get_output(cond, z, model, t_enc_ddpm, device):
        e = model.apply_model(z.to(device), t_enc_ddpm.to(device), cond.to(device))
        return e
    
    # get the output of the model w.r.t. the input concept
    if type(concept) is str:
        concept_cond = model.get_learned_conditioning([concept])
    else:
        raise ValueError("concept should be a string")
    
    output_concept = get_output(concept_cond, z, model, t_enc_ddpm, device)
    sim_dict = {}

    with torch.no_grad():
        for token in tokens_embedding:
            token_cond = model.get_learned_conditioning([token])
            output_token = get_output(token_cond, z, model, t_enc_ddpm, device)
            if sim == 'cosine':
                similarity = F.cosine_similarity(output_concept.flatten(), 
                                                output_token.flatten(), dim=-1)
            elif sim == 'l2':
                similarity = - F.pairwise_distance(output_concept.flatten(), 
                                                output_token.flatten(), p=2)
            sim_dict[token] = similarity
            print(f"similarity between concept and token {token}: {similarity}")
        
    sorted_sim_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)}
    top_k_tokens = list(sorted_sim_dict.keys())[:k]
    print(f"Top-{k} closest tokens to the concept {concept} are: {top_k_tokens}")

    # # write similarity scores to a csv file 
    # with open(f'evaluation_folder/similarity/similarity_scores_{concept}_{time_step}_{start_concept}.csv', 'w') as f:
    #     for key in sorted_sim_dict.keys():
    #         f.write("%s,%s\n"%(key, sorted_sim_dict[key].item()))

    return top_k_tokens, sorted_sim_dict


@torch.no_grad()
def search_closest_output_multi_steps(concept, model, sampler, tokens_embedding, start_guidance, time_step, start_concept, k=5, sim='l2', multi_steps=2):
    """
    given a concept, i.e., "nudity", search for top-k closest tokens in the token vocabulary that produces the closest output with a given model
    applying multi-steps inference instead of just one-step as before. 
    Args: 
        concept: str, the concept to search for
        model: the model to use for inference
        tokens_embedding: dictionary containing the token embeddings that are used to search for the closest tokens
            example: {'nudity': tensor([0.1, 0.2, ..., 0.3]), 'sexy': tensor([0.2, 0.3, ..., 0.4]), ...}
            token embeddings are of shape (1, 77, 768)
        k: int, the number of top-k tokens to return
    """

    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    # till_T = 0 means the model will sample till the end of the diffusion process 
    # till_T = T means the starting point of the diffusion process
    # the smaller the value of till_T, the more denoising is applied to the image
    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                 cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, s, DDIM_ETA,
                                                                 start_code=code, till_T=t, verbose=False)
    device = model.cond_stage_model.device
    # fix start code 
    start_code = torch.randn(1, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8).to(device)

    # get the condition 
    if start_concept is None:
        start_cond = model.get_learned_conditioning([STARTING_CONCEPT])
    else:
        start_cond = model.get_learned_conditioning([start_concept])

    # if start from a specific time step t 
    if time_step is not None:
        z = quick_sample_till_t(start_cond, start_guidance, start_code, time_step)
    else: 
        # start with noisy 
        z = start_code
    
    # get the output of the model w.r.t. the input concept 
    og_num = round((int(time_step)/DDIM_STEPS)*1000)
    og_num_lim = round((int(time_step + 1)/DDIM_STEPS)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,))
    
    # assert time_step - multi_steps >= 0, "time_step - multi_steps should be greater than or equal to 0"


    @torch.no_grad()
    def get_output(cond, z, model, sampler):
        # CHANGE HERE: start from intermediate time step time_step (z is the start code) and sample 2 steps, i.e., till time_step-multi_steps
        if time_step - multi_steps >= 0:
            output = sample_model(model, sampler, cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, start_guidance, DDIM_ETA, start_code=z, till_T = time_step - multi_steps, t_start = time_step)
        else: 
            assert time_step == 0
            output = sample_model(model, sampler, cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, start_guidance, DDIM_ETA, start_code=z, till_T = 0, t_start = multi_steps)

        return output
    
    # get the output of the model w.r.t. the input concept
    if type(concept) is str:
        concept_cond = model.get_learned_conditioning([concept])
    else:
        raise ValueError("concept should be a string")
    
    output_concept = get_output(concept_cond, z, model, sampler)
    sim_dict = {}
    output_dict = {}
    output_dict[concept] = output_concept

    with torch.no_grad():
        for token in tokens_embedding:
            token_cond = model.get_learned_conditioning([token])
            output_token = get_output(token_cond, z, model, sampler)
            output_dict[token] = output_token
            if sim == 'cosine':
                similarity = F.cosine_similarity(output_concept.flatten(), 
                                                output_token.flatten(), dim=-1)
            elif sim == 'l2':
                similarity = - F.pairwise_distance(output_concept.flatten(), 
                                                output_token.flatten(), p=2)
            sim_dict[token] = similarity
            print(f"similarity between concept and token {token}: {similarity}")
        
    sorted_sim_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)}
    top_k_tokens = list(sorted_sim_dict.keys())[:k]
    print(f"Top-{k} closest tokens to the concept {concept} are: {top_k_tokens}")

    # # write similarity scores to a csv file 
    # with open(f'evaluation_folder/similarity/similarity_scores_{concept}_{time_step}_{start_concept}.csv', 'w') as f:
    #     for key in sorted_sim_dict.keys():
    #         f.write("%s,%s\n"%(key, sorted_sim_dict[key].item()))

    return top_k_tokens, sorted_sim_dict, output_dict

@torch.no_grad()
def compare_two_models(model, model_org, sampler, tokens_embedding, start_guidance, time_step, start_concept, k=5, sim='l2'):
    """
    given a concept, i.e., "nudity", search for top-k closest tokens in the token vocabulary that produces the closest output with a given model
    TODO: using CLIP score as a similarity metric 
    Args: 
        model: the model to use for inference
        model_org: the original model to use for inference
        tokens_embedding: dictionary containing the token embeddings that are used to search for the closest tokens
            example: {'nudity': tensor([0.1, 0.2, ..., 0.3]), 'sexy': tensor([0.2, 0.3, ..., 0.4]), ...}
            token embeddings are of shape (1, 77, 768)
        k: int, the number of top-k tokens to return
    """

    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    # till_T = 0 means the model will sample till the end of the diffusion process 
    # till_T = T means the starting point of the diffusion process
    # the smaller the value of till_T, the more denoising is applied to the image
    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                 cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, s, DDIM_ETA,
                                                                 start_code=code, till_T=t, verbose=False)
    device = model.cond_stage_model.device
    # fix start code 
    start_code = torch.randn(1, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8).to(device)

    # get the condition 
    if start_concept is None:
        start_cond = model.get_learned_conditioning([STARTING_CONCEPT])
    else:
        start_cond = model.get_learned_conditioning([start_concept])

    # if start from a specific time step t 
    if time_step is not None:
        z = quick_sample_till_t(start_cond, start_guidance, start_code, time_step)
    else: 
        # start with noisy 
        z = start_code
    
    # get the output of the model w.r.t. the input concept 
    og_num = round((int(time_step)/DDIM_STEPS)*1000)
    og_num_lim = round((int(time_step + 1)/DDIM_STEPS)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,))
    
    @torch.no_grad()
    def get_output(cond, z, model, t_enc_ddpm, device):
        e = model.apply_model(z.to(device), t_enc_ddpm.to(device), cond.to(device))
        return e
    sim_dict = {}
    
    with torch.no_grad():
        for token in tokens_embedding:
            token_cond = model.get_learned_conditioning([token])
            output_1 = get_output(token_cond, z, model, t_enc_ddpm, device)
            output_2 = get_output(token_cond, z, model_org, t_enc_ddpm, device)
            if sim == 'cosine':
                similarity = F.cosine_similarity(output_1.flatten(), 
                                                output_2.flatten(), dim=-1)
            elif sim == 'l2':
                similarity = - F.pairwise_distance(output_1.flatten(), 
                                                output_2.flatten(), p=2)
            sim_dict[token] = similarity
            print(f"similarity between two models for token {token}: {similarity}")
    
    sorted_sim_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)}
    top_k_tokens = list(sorted_sim_dict.keys())[:k]
    print(f"Top-{k} closest tokens to the concept are: {top_k_tokens}")

    return top_k_tokens, sorted_sim_dict


def my_kmean(sorted_sim_dict, num_centers, compute_mode):
    if compute_mode == 'numpy':
        from sklearn.cluster import KMeans
        import numpy as np
        similarities = np.array([sorted_sim_dict[token].item() for token in sorted_sim_dict])
        similarities = similarities.reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(similarities)
        print(f"Cluster centers: {kmeans.cluster_centers_}")
        print(f"Cluster labels: {kmeans.labels_}")
        cluster_centers = kmeans.cluster_centers_
    elif compute_mode == 'torch':
        from torch_kmeans import KMeans
        similarities = torch.stack([sorted_sim_dict[token] for token in sorted_sim_dict])
        similarities = torch.unsqueeze(similarities, dim=0)
        similarities = torch.unsqueeze(similarities, dim=2) # [1, N, 1]
        print('similarities shape:', similarities.shape)
        kmeans = KMeans(n_clusters=num_centers).fit(similarities)
        import pdb; pdb.set_trace()
        print(f"Cluster centers: {kmeans.cluster_centers}")
        print(f"Cluster labels: {kmeans.labels}")
        cluster_centers = kmeans.cluster_centers

    # find the closest token to each cluster center
    cluster_dict = {}
    for i, center in enumerate(cluster_centers):
        closest_token = None
        closest_similarity = -float('inf')
        for j, token in enumerate(sorted_sim_dict):
            similarity = sorted_sim_dict[token].item()
            if abs(similarity - center) < abs(closest_similarity - center):
                closest_similarity = similarity
                closest_token = token
        cluster_dict[closest_token] = (closest_token, closest_similarity, i)
    print(f"Cluster dictionary: {cluster_dict}")

    return cluster_dict

@torch.no_grad()
def learn_k_means_from_output(model, model_org, sampler, tokens_embedding, start_guidance, time_step, start_concept, num_centers=5, sim='l2', multi_steps=2, compute_mode='numpy'):
    """
    Given two models and a set of tokens, learn k-means clustering on the sorted_sim_dict
    """
    if num_centers <= 0:
        print(f"Number of centers should be greater than 0. Returning the tokens themselves.")
        return tokens_embedding
    if len(tokens_embedding) <= num_centers:
        print(f"Number of tokens is less than the number of centers. Returning the tokens themselves.")
        return tokens_embedding
    # _, sorted_sim_dict = compare_two_models_multi_steps(model, model_org, sampler, tokens_embedding, start_guidance, time_step, start_concept, k=10, sim=sim, multi_steps=multi_steps)
    _, sorted_sim_dict = compare_two_models(model, model_org, sampler, tokens_embedding, start_guidance, time_step, start_concept, k=10, sim=sim)
    return list(my_kmean(sorted_sim_dict, num_centers, compute_mode).keys())

@torch.no_grad()
def learn_k_means_from_input_embedding(sim_dict, num_centers=5, compute_mode='numpy'):
    """
    Given a model, a set of tokens, and a concept, learn k-means clustering on the search_closest_tokens's output
    """
    if num_centers <= 0:
        print(f"Number of centers should be greater than 0. Returning the tokens themselves.")
        return list(sim_dict.keys())
    if len(list(sim_dict.keys())) <= num_centers:
        print(f"Number of tokens is less than the number of centers. Returning the tokens themselves.")
        return list(sim_dict.keys())

    return list(my_kmean(sim_dict, num_centers, compute_mode).keys())

@torch.no_grad()
def compare_two_models_multi_steps(model, model_org, sampler, tokens_embedding, start_guidance, time_step, start_concept, k=5, sim='l2', multi_steps=2):
    """
    given a concept, i.e., "nudity", search for top-k closest tokens in the token vocabulary that produces the closest output with a given model
    applying multi-steps inference instead of just one-step as before.
    Args: 
        model: the model to use for inference
        model_org: the original model to use for inference
        tokens_embedding: dictionary containing the token embeddings that are used to search for the closest tokens
            example: {'nudity': tensor([0.1, 0.2, ..., 0.3]), 'sexy': tensor([0.2, 0.3, ..., 0.4]), ...}
            token embeddings are of shape (1, 77, 768)
        k: int, the number of top-k tokens to return
    """

    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    # till_T = 0 means the model will sample till the end of the diffusion process 
    # till_T = T means the starting point of the diffusion process
    # the smaller the value of till_T, the more denoising is applied to the image
    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                 cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, s, DDIM_ETA,
                                                                 start_code=code, till_T=t, verbose=False)
    device = model.cond_stage_model.device
    # fix start code 
    start_code = torch.randn(1, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8).to(device)

    # get the condition 
    if start_concept is None:
        start_cond = model.get_learned_conditioning([STARTING_CONCEPT])
    else:
        start_cond = model.get_learned_conditioning([start_concept])

    # if start from a specific time step t 
    if time_step is not None:
        z = quick_sample_till_t(start_cond, start_guidance, start_code, time_step)
    else: 
        # start with noisy 
        z = start_code
    
    # get the output of the model w.r.t. the input concept 
    og_num = round((int(time_step)/DDIM_STEPS)*1000)
    og_num_lim = round((int(time_step + 1)/DDIM_STEPS)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,))
    
    def get_output(cond, z, model, sampler):
        # CHANGE HERE: start from intermediate time step time_step (z is the start code) and sample 2 steps, i.e., till time_step-multi_steps
        if time_step - multi_steps >= 0:
            output = sample_model(model, sampler, cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, start_guidance, DDIM_ETA, start_code=z, till_T = time_step - multi_steps, t_start = time_step)
        else: 
            assert time_step == 0
            output = sample_model(model, sampler, cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, start_guidance, DDIM_ETA, start_code=z, till_T = 0, t_start = multi_steps)

        return output
    
    sim_dict = {}
    
    with torch.no_grad():
        for token in tokens_embedding:
            token_cond = model.get_learned_conditioning([token])
            output_1 = get_output(token_cond, z, model, sampler)
            output_2 = get_output(token_cond, z, model_org, sampler)
            if sim == 'cosine':
                similarity = F.cosine_similarity(output_1.flatten(), 
                                                output_2.flatten(), dim=-1)
            elif sim == 'l2':
                similarity = - F.pairwise_distance(output_1.flatten(), 
                                                output_2.flatten(), p=2)
            sim_dict[token] = similarity
            print(f"similarity between two models for token {token}: {similarity}")
    
    sorted_sim_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)}
    top_k_tokens = list(sorted_sim_dict.keys())[:k]
    print(f"Top-{k} closest tokens to the concept are: {top_k_tokens}")

    return top_k_tokens, sorted_sim_dict


@torch.no_grad()
def get_output_two_models_multi_steps(model, model_org, sampler, concept_embedding, start_guidance, time_step, start_concept, multi_steps=2):
    """
    given a concept, i.e., "nudity", search for top-k closest tokens in the token vocabulary that produces the closest output with a given model
    applying multi-steps inference instead of just one-step as before.
    Args: 
        model: the model to use for inference
        model_org: the original model to use for inference
        concept_embedding: the concept embedding to use for inference
    """

    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    # till_T = 0 means the model will sample till the end of the diffusion process 
    # till_T = T means the starting point of the diffusion process
    # the smaller the value of till_T, the more denoising is applied to the image
    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                 cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, s, DDIM_ETA,
                                                                 start_code=code, till_T=t, verbose=False)
    device = model.cond_stage_model.device
    # fix start code 
    start_code = torch.randn(1, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8).to(device)

    # get the condition 
    if start_concept is None:
        start_cond = model.get_learned_conditioning([STARTING_CONCEPT])
    else:
        start_cond = model.get_learned_conditioning([start_concept])

    # if start from a specific time step t 
    if time_step is not None:
        z = quick_sample_till_t(start_cond, start_guidance, start_code, time_step)
    else: 
        # start with noisy 
        z = start_code
    
    # get the output of the model w.r.t. the input concept 
    og_num = round((int(time_step)/DDIM_STEPS)*1000)
    og_num_lim = round((int(time_step + 1)/DDIM_STEPS)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,))
    
    def get_output(cond, z, model, sampler):
        # CHANGE HERE: start from intermediate time step time_step (z is the start code) and sample 2 steps, i.e., till time_step-multi_steps
        if time_step - multi_steps >= 0:
            output = sample_model(model, sampler, cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, start_guidance, DDIM_ETA, start_code=z, till_T = time_step - multi_steps, t_start = time_step)
        else: 
            assert time_step == 0
            output = sample_model(model, sampler, cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, start_guidance, DDIM_ETA, start_code=z, till_T = 0, t_start = multi_steps)

        return output
    

    with torch.no_grad():
        output_1 = get_output(concept_embedding, z, model, sampler)
        output_2 = get_output(concept_embedding, z, model_org, sampler)

    return output_1, output_2

@torch.no_grad()
def decode_and_extract_visual_feature(model, z, device):
    clip_model, clip_preprocess = clip.load("ViT-B/32", device)

    x = model.decode_first_stage(z)
    x = torch.clamp((x + 1.0)/2.0, min=0.0, max=1.0)
    x = rearrange(x, 'b c h w -> b (c h) w')
    image = clip_preprocess(Image.fromarray((x[0].cpu().numpy()*255).astype(np.uint8)))
    with torch.no_grad():
        image_features = clip_model.encode_image(image.unsqueeze(0).to(device))
    return image_features
    
@torch.no_grad()
def search_closest_output_CLIP(concept, model, sampler, tokens_embedding, start_guidance, time_step, start_concept, k=5, sim='l2'):
    """
    given a concept, i.e., "nudity", search for top-k closest tokens in the token vocabulary that produces the closest output with a given model
    TODO: using CLIP score as a similarity metric 
    Args: 
        concept: str, the concept to search for
        model: the model to use for inference
        tokens_embedding: dictionary containing the token embeddings that are used to search for the closest tokens
            example: {'nudity': tensor([0.1, 0.2, ..., 0.3]), 'sexy': tensor([0.2, 0.3, ..., 0.4]), ...}
            token embeddings are of shape (1, 77, 768)
        k: int, the number of top-k tokens to return
    """

    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    # till_T = 0 means the model will sample till the end of the diffusion process 
    # till_T = T means the starting point of the diffusion process
    # the smaller the value of till_T, the more denoising is applied to the image
    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                 cond, IMAGE_SIZE, IMAGE_SIZE, DDIM_STEPS, s, DDIM_ETA,
                                                                 start_code=code, till_T=t, verbose=False)
    device = model.cond_stage_model.device
    # fix start code 
    start_code = torch.randn(1, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8).to(device)

    # get the condition 
    if start_concept is None:
        start_cond = model.get_learned_conditioning([STARTING_CONCEPT])
    else:
        start_cond = model.get_learned_conditioning([start_concept])

    # if start from a specific time step t 
    if time_step is not None:
        z = quick_sample_till_t(start_cond, start_guidance, start_code, time_step)
    else: 
        # start with noisy 
        z = start_code
    
    # get the output of the model w.r.t. the input concept 
    og_num = round((int(time_step)/DDIM_STEPS)*1000)
    og_num_lim = round((int(time_step + 1)/DDIM_STEPS)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,))
    
    @torch.no_grad()
    def get_output(cond, z, model, t_enc_ddpm, device):
        e = model.apply_model(z.to(device), t_enc_ddpm.to(device), cond.to(device))
        return e
    
    # get the output of the model w.r.t. the input concept
    if type(concept) is str:
        concept_cond = model.get_learned_conditioning([concept])
    else:
        raise ValueError("concept should be a string")
    
    output_concept = get_output(concept_cond, z, model, t_enc_ddpm, device)

    # decode using decoder
    concept_image_features = decode_and_extract_visual_feature(model, output_concept, device)


    sim_dict = {}

    with torch.no_grad():
        for token in tokens_embedding:
            token_cond = model.get_learned_conditioning([token])
            output_token = get_output(token_cond, z, model, t_enc_ddpm, device)
            # decode using decoder
            token_image_features = decode_and_extract_visual_feature(model, output_token, device)
            if sim == 'cosine':
                similarity = F.cosine_similarity(concept_image_features.flatten(), 
                                                token_image_features.flatten(), dim=-1)
            elif sim == 'l2':
                similarity = - F.pairwise_distance(concept_image_features.flatten(),
                                                token_image_features.flatten(), p=2)
                
            sim_dict[token] = similarity
            print(f"similarity between concept and token {token}: {similarity}")

        
    sorted_sim_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)}
    top_k_tokens = list(sorted_sim_dict.keys())[:k]
    print(f"Top-{k} closest tokens to the concept {concept} are: {top_k_tokens}")

    # # write similarity scores to a csv file 
    # with open(f'evaluation_folder/similarity/similarity_scores_clip_{concept}_{time_step}_{start_concept}.csv', 'w') as f:
    #     for key in sorted_sim_dict.keys():
    #         f.write("%s,%s\n"%(key, sorted_sim_dict[key].item()))

    return top_k_tokens, sorted_sim_dict

def test_compare_two_models(args):
    config_path = args.config_path
    vocab = args.vocab

    time_steps = [args.time_step]

    start_concept = "a photo"

    sim = args.sim

    collections_tokens = None

    concepts = [args.concept]

    devices = ['cuda', 'cuda']

    ckpt_path_org = 'models/erase/sd-v1-4-full-ema.ckpt'
    model_org = load_model_from_config(config_path, ckpt_path_org, device=devices[0])

    tokenizer_vocab = get_vocab(model_org, model_name='SD-v1-4', vocab=vocab)
    collections_tokens = list(tokenizer_vocab.keys())   
    customized_tokens = ['women', 'men', 'person', 'hat', 'apple', 'bamboo'] +\
                        ['notebooks', 'australians', 'president', 'boat', 'lexus', 'money', 'banana'] +\
                        ['naked', 'garbage truck', 'a photo', ' '] +\
                        ['road', 'car', 'bus']
    
    collections_tokens += customized_tokens
    
    for concept in concepts:
        if concept in ["nudity"]:
            ckpt_paths = {
                'esd-nudity': 'models/compvis-word_nudity-method_noxattn-sg_3-ng_1.0-iter_1000-lr_1e-05-info_separated/compvis-word_nudity-method_noxattn-sg_3-ng_1.0-iter_1000-lr_1e-05-info_separated.pt',
            }
        elif concept in ["nudity_with_person"]:
            ckpt_paths = {
                'esd-nudity': 'models/compvis-word_nudity_with_person-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve/compvis-word_nudity_with_person-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve.pt',
            }            
        elif concept in ["imagenette"]:
            ckpt_paths = {
                'esd-imagenette': 'models/compvis-word_imagenette_small-method_xattn-sg_3-ng_1.0-iter_1000-lr_1e-05-info_imagenette_v1_separated/compvis-word_imagenette_small-method_xattn-sg_3-ng_1.0-iter_1000-lr_1e-05-info_imagenette_v1_separated.pt',
            }
        elif concept in ["garbage_truck"]:
            ckpt_paths = {
                'esd-garbage_truck': 'models/compvis-word_garbage_truck-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none/compvis-word_garbage_truck-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none.pt',
            }
        else:
            raise ValueError("Concept not found")

        for time_step in time_steps:

            for model_name in ckpt_paths:

                config = OmegaConf.load(config_path)

                if model_name == 'SD-v1-4':
                    model = load_model_from_config(config_path, ckpt_paths[model_name], device=devices[0])
                else:
                    model = load_model_from_config_compvis(config, ckpt_paths[model_name], verbose=True)
                
                sampler = DDIMSampler(model) 

                _, sorted_sim_dict = compare_two_models(model, model_org, sampler, collections_tokens, start_guidance=3.0, time_step=time_step, start_concept=start_concept, k=10, sim=sim)
                
                with open(f'evaluation_folder/similarity/compare_2models_{concept}_{time_step}_{start_concept}_{model_name}_{sim}_{vocab}.csv', 'w') as f:
                    for key in sorted_sim_dict.keys():
                        f.write("%s,%s\n"%(key, sorted_sim_dict[key].item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Compare two models')
    parser.add_argument('--concept', type=str, default="nudity", help='concept')
    parser.add_argument('--time_step', type=int, default=0, help='time step to to generate the base image')
    parser.add_argument('--sim', type=str, default='l2', help='similarity metric to use')
    parser.add_argument('--config_path', type=str, default='configs/stable-diffusion/v1-inference.yaml', help='path to the config file')
    parser.add_argument('--model_name', type=str, default='SD-v1-4', help='model name')
    parser.add_argument('--ckpt_path', type=str, default='../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt', help='path to the checkpoint')
    parser.add_argument('--multi_steps', type=int, default=2, help='number of steps to sample')
    parser.add_argument('--task', type=str, help='task to run')
    parser.add_argument('--vocab', type=str, default='EN3K', help='vocabulary to use')

    args = parser.parse_args()

    if args.task == 'compare_two_models':
        print('Search closest output')
        test_compare_two_models(args)

