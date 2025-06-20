import torch
import torch.nn.functional as F



def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:

    return F.cosine_similarity(emb1, emb2)




