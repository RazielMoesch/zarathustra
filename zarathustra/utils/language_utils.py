import json
import os
from tqdm.auto import tqdm
from collections import defaultdict
import torch

class BPETokenizer:
    '''
    BPETokenizer(vocab_path=None)

    A Byte Pair Encoding (BPE) tokenizer implementation with train, tokenize, and detokenize functionality.
    Designed for compact tokenization for language models.

    Args:
        vocab_path (str): Optional path to a saved vocabulary file to load.

    Methods:
        train(text, num_merges=100, vocab_path="bpe_vocab.json"):
            Trains the BPE tokenizer on the input text with the specified number of merges.

        tokenize(text):
            Converts text into a list of token IDs using the trained or loaded vocabulary.

        detokenize(token_ids):
            Converts a list of token IDs back into human-readable text.

        get_vocab_size():
            Returns the size of the current vocabulary.

    Usage:
        tokenizer = BPETokenizer()
        tokenizer.train(corpus_text)
        tokens = tokenizer.tokenize("Hello world")
        text = tokenizer.detokenize(tokens)
    '''
    def __init__(self, vocab_path=None):
        self.vocab = {}
        self.inverse_vocab = {}
        self.space_token = "▁"
        self.unk_token = "[UNK]"
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = {int(k): v for k, v in json.load(f).items()}
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def save_vocab(self, vocab_path):
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def create_initial_vocab(self, text):
        text = text.replace(" ", self.space_token)
        chars = sorted(list(set(text)))
        special_tokens = [self.space_token, self.unk_token]
        chars = special_tokens + [c for c in chars if c not in special_tokens]
        return {i: char for i, char in enumerate(chars)}

    def get_stats(self, tokens):
        pairs = defaultdict(int)
        for token in tokens:
            symbols = token.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += 1
        return pairs

    def merge_vocab(self, pair, vocab, tokens):
        new_token = "".join(pair)
        new_vocab = vocab.copy()
        new_vocab[len(new_vocab)] = new_token
        new_tokens = []
        for token in tokens:
            symbols = token.split()
            i = 0
            while i < len(symbols)-1:
                if symbols[i] == pair[0] and symbols[i+1] == pair[1]:
                    symbols[i:i+2] = [new_token]
                else:
                    i += 1
            new_tokens.append(" ".join(symbols))
        return new_vocab, new_tokens

    def train(self, text, num_merges=100, vocab_path="bpe_vocab.json"):
        initial_vocab = self.create_initial_vocab(text)
        self.vocab = initial_vocab
        self.inverse_vocab = {v: k for k, v in initial_vocab.items()}
        text = text.replace(" ", self.space_token)
        tokens = []
        for word in text.split(self.space_token):
            tokens.append(" ".join(list(word)))
            tokens.append(self.space_token)
        tokens = tokens[:-1]
        for _ in tqdm(range(num_merges), desc="Training BPE"):
            pairs = self.get_stats(tokens)
            if not pairs:
                break
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            self.vocab, tokens = self.merge_vocab(best_pair, self.vocab, tokens)
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        if self.unk_token not in self.inverse_vocab:
            new_id = len(self.vocab)
            self.vocab[new_id] = self.unk_token
            self.inverse_vocab[self.unk_token] = new_id
        self.save_vocab(vocab_path)
        return self.vocab

    def tokenize(self, text):
        if not self.vocab:
            raise ValueError("Vocabulary not loaded or trained")
        text = text.replace(" ", self.space_token)
        tokens = []
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: (-len(x[1]), x[0]))
        i = 0
        while i < len(text):
            matched = False
            for token_id, token in sorted_vocab:
                if text.startswith(token, i):
                    tokens.append(str(token_id))
                    i += len(token)
                    matched = True
                    break
            if not matched:
                tokens.append(str(self.inverse_vocab[self.unk_token]))
                i += 1
        return tokens

    def detokenize(self, token_ids):
        if not self.vocab:
            raise ValueError("Vocabulary not loaded or trained")
        text = []
        for token_id in token_ids:
            text.append(self.vocab.get(int(token_id), self.unk_token))
        return "".join(text).replace(self.space_token, " ").replace(self.unk_token, "")

    def get_vocab_size(self):
        return len(self.vocab)



def generate_text(model, tokenizer, prompt, max_new_tokens=50, device="cpu"):
    '''
    generate_text(model, tokenizer, prompt, max_new_tokens=50, device="cpu")

    Autoregressively generates text from a prompt using a Transformer and BPETokenizer.

    Args:
        model (Transformer): Trained Transformer model.
        tokenizer (BPETokenizer): Tokenizer used for encoding/decoding text.
        prompt (str): The starting string to generate from.
        max_new_tokens (int): How many new tokens to generate.
        device (str): Device to run inference on.

    Returns:
        str: The generated text.
    '''
    model.eval()
    model.to(device)

    token_ids = tokenizer.tokenize(prompt)
    input_ids = torch.tensor([int(t) for t in token_ids], dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        if input_ids.size(1) >= model.embedding.pos_embed.size(0):
            break
        logits = model(input_ids)
        next_token_logits = logits[0, -1]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

    output_ids = input_ids[0].tolist()
    return tokenizer.detokenize(output_ids)



if __name__ == "__main__":
    from ..models.language_models import Transformer


    model = Transformer()

    tokenizer = BPETokenizer("shakespeare_vocab.json")

    prompt = "Romeo oh Romeo "


    generate_text(model, tokenizer, prompt, max_new_tokens=128, device='cpu' )