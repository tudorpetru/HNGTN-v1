#=============================HNGTN - Hybrid N Gram Transformer Network=============================
#Note that this file doesn't have the def that calculates FWT and BWT included

import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 3890343

torch.manual_seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_file(path):
    with open (path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
 
def get_window_seq_data(data: str, win_size):
    data = data.split()
    out = []

    for i in range(win_size, len(data)):
        context = data[i - win_size:i]
        target = data[i]
        out.append((context, target))

    return out

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_size, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()

        self.context_size = context_size
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=layer, num_layers=n_layers)

        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None):
        _, sequence_length = token_ids.shape

        positions = torch.arange(sequence_length, device=token_ids.device)

        x = (self.token_embedding(token_ids) + self.position_embedding(positions))

        mask = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=token_ids.device), diagonal=1)

        x = self.transformer(x, mask=mask)
        x = self.final_norm(x)

        logits = self.output(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

        return logits, loss

def predict_word_transformer(model, tokens, token_to_id, id_to_token, max_new_tokens, temperature,):
    model.eval()
    model_device = next(model.parameters()).device

    token_ids = []
    for token in tokens:
        token_ids.append(token_to_id[token])

    out = torch.tensor([token_ids], dtype=torch.long, device=model_device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            transformer_logits, _ = model(out[:, -model.context_size:])
            next_logits = transformer_logits[0, -1]

            probs = torch.softmax(next_logits / temperature, dim=-1)

            next_tok = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_tok.reshape(1, 1)], dim=1)

    out_lst = []
    for token_id in out[0].tolist():
        out_lst.append(id_to_token[token_id])

    return out_lst

def add_to_network(data, conns, vocab):
    prev_vocab = set(vocab)
    data_tokens = set(data.split())
    tokens = data_tokens - prev_vocab

    vocab = sorted(prev_vocab | tokens)

    return conns, vocab

def train_model(data, win_size, vocab_in, conns_in):
    seq = get_window_seq_data(data, win_size)
    conns, vocab = add_to_network(data, conns_in, vocab_in)

    for context_toks, target_tok in seq:
        context = tuple(context_toks)

        if context not in conns:
            conns[context] = {}

        if target_tok not in conns[context]:
            conns[context][target_tok] = 0

        conns[context][target_tok] += 1

    return vocab, conns

def n_gram_correction(model, optimizer, n_gram_power, vocab_size, token_to_id, conns, context_size):
    batch_size = 256

    contexts_by_length = {}
    for context in conns:
        context_length = len(context)

        if 1 <= context_length <= context_size:
            contexts_by_length.setdefault(context_length, []).append(context)

    lengths = sorted(contexts_by_length)

    if not lengths:
        return

    samples_length = max(1, batch_size // len(lengths))
    samples_active = max(0, batch_size - samples_length * len(lengths))

    prev_state = model.training
    model.train()

    model_device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=model_device)

    total_data = 0
    sampled_counts = {}
    for index, context_length in enumerate(lengths):
        req_count = samples_length

        if index < samples_active:
            req_count += 1

        chosen_contexts = random.sample(contexts_by_length[context_length], min(req_count, len(contexts_by_length[context_length])))

        inputs = []
        probs = []
        for context in chosen_contexts:
            target_counts = conns[context]
            total_count = sum(target_counts.values())

            if total_count <= 0:
                continue

            context_ids = []
            for token in context:
                context_ids.append(token_to_id[token])
            n_gram_probs = torch.zeros(vocab_size, dtype=torch.float32)

            for target_token, count in target_counts.items():
                target_id = token_to_id[target_token]
                n_gram_probs[target_id] = (count / total_count)

            inputs.append(context_ids)
            probs.append(n_gram_probs)

        if not inputs:
            continue

        input_ids = torch.tensor(inputs, dtype=torch.long, device=model_device)
        n_gram_probs = torch.stack(probs).to(device=model_device)
        model_output = model(input_ids)

        if isinstance(model_output, tuple):
            logits = model_output[0]
        elif hasattr(model_output, "logits"):
            logits = model_output.logits
        else:
            logits = model_output

        transformer_logits = logits[:, -1, :]
        n_gram_probs = n_gram_probs.to(dtype=transformer_logits.dtype)
        sums = n_gram_probs.sum(dim=-1, keepdim=True)

        n_gram_probs = (n_gram_probs / sums.clamp_min(1e-8))
        transformer_log_probs = F.log_softmax(transformer_logits, dim=-1)

        transformer_probs = (transformer_log_probs.exp().detach())
        correction_targets = ((1.0 - n_gram_power) * transformer_probs + n_gram_power * n_gram_probs).detach()

        per_data_loss = -(correction_targets * transformer_log_probs).sum(dim=-1)
        total_loss = (total_loss + per_data_loss.sum())

        data_count = len(inputs)
        total_data += data_count

        sampled_counts[context_length] = data_count

    if total_data == 0:
        if not prev_state:
            model.eval()

        return

    correction_loss = (total_loss / total_data)
    correction_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if not prev_state:
        model.eval()

def train_transformer(model, epochs, train_inputs, train_targets, optimizer):
    batch_size = 256

    for epoch in range(epochs):
        model.train()
        ordered_train_ins = torch.randperm(len(train_inputs))

        total_loss = 0.0
        batch_num = 0
        for start in range(0, len(train_inputs), batch_size):
            indices = ordered_train_ins[start:start + batch_size]

            inputs = train_inputs[indices].to(device)
            targets = train_targets[indices].to(device)

            optimizer.zero_grad(set_to_none=True)
            transformer_logits, _ = model(inputs)

            loss = F.cross_entropy(transformer_logits.reshape(-1, model.vocab_size), targets.reshape(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

            total_loss += loss.item()
            batch_num += 1

        print(f"epoch = {epoch}, loss = {total_loss / batch_num}")

def rem_from_start(string: str, amount):
    lst = string.split()
    lst_remover = []
    num = 0
    for i in lst:
        lst_remover.append(i)

        num += 1
        if num == amount:
            break

    remover = " ".join(lst_remover)
    string = " ".join(lst)

    string = string.replace(remover, "")

    return string

def save_transformer(model, vocab, context_size, path):
    torch.save({"model_state": model.state_dict(), "vocab": vocab, "context_size": context_size}, path)

def load_transformer(path, d_model, n_heads, n_layers, d_ff, dropout):
    m_data = torch.load(path, map_location=device, weights_only=False,)

    vocab = m_data["vocab"]
    context_size = m_data["context_size"]

    model = Transformer(len(vocab), context_size, d_model, n_heads, n_layers, d_ff, dropout).to(device)

    model.load_state_dict(m_data["model_state"])
    model.eval()

    return model, vocab

def basic_punctuation_spacer(string):
    return string.replace(".", " . ").replace("?", " ? ").replace("!", " ! ").replace(",", " , ").replace(")", " ) ").replace("(", " ( ").replace("[", " [ ").replace("]", " ] ").replace("'", " ' ").replace('"', ' " ').replace(":", " : ").replace(";", " ; ")

def get_last_toks(lst, amount):
    out_lst = []

    for tok in reversed(lst):
        out_lst.append(tok)

        if len(out_lst) >= amount:
            break

    out_lst.reverse()

    return out_lst

def expand_transformer_vocab(model, optimizer, vocab, token_to_id, id_to_token, cur_tokens):
    cur_tokens = sorted(set(cur_tokens) - set(token_to_id))
    if not cur_tokens:
        return model, optimizer, vocab, token_to_id, id_to_token

    prev_vocab_size = len(vocab)
    prev_embedding = model.token_embedding
    prev_output = model.output

    prev_embedding_weight = prev_embedding.weight
    prev_output_weight = prev_output.weight
    prev_output_bias = prev_output.bias

    vocab.extend(cur_tokens)
    
    token_to_id = {}
    for token_id, token in enumerate(vocab):
        token_to_id[token] = token_id

    id_to_token = {}
    for token, token_id in token_to_id.items():
        id_to_token[token_id] = token

    cur_vocab_size = len(vocab)

    new_embedding = nn.Embedding(num_embeddings=cur_vocab_size, embedding_dim=prev_embedding.embedding_dim).to(device= next(model.parameters()).device, dtype=prev_embedding_weight.dtype)
    cur_output = nn.Linear(in_features=prev_embedding.embedding_dim, out_features=cur_vocab_size, bias=prev_output_bias is not None).to(device= next(model.parameters()).device, dtype=prev_embedding_weight.dtype)

    nn.init.normal_(new_embedding.weight, mean=0.0, std=0.02)
    nn.init.normal_(cur_output.weight, mean=0.0, std=0.02)

    if cur_output.bias is not None:
        nn.init.zeros_(cur_output.bias)

    with torch.no_grad():
        new_embedding.weight[:prev_vocab_size].copy_(prev_embedding_weight)
        cur_output.weight[:prev_vocab_size].copy_(prev_output_weight)
        if cur_output.bias is not None:
            cur_output.bias[:prev_vocab_size].copy_(prev_output_bias)

    new_embedding.weight.requires_grad_(prev_embedding_weight.requires_grad)
    cur_output.weight.requires_grad_(prev_output_weight.requires_grad)
    if cur_output.bias is not None:
        cur_output.bias.requires_grad_(prev_output_bias.requires_grad)

    model.token_embedding = new_embedding
    model.output = cur_output
    model.vocab_size = cur_vocab_size

    param_pairs = [(prev_embedding_weight, new_embedding.weight), (prev_output_weight, cur_output.weight)]
    if prev_output_bias is not None:
        param_pairs.append((prev_output_bias, cur_output.bias))

    changes = {}
    for prev_param, cur_param in param_pairs:
        changes[id(prev_param)] = cur_param   

    for param_group in optimizer.param_groups:         
        param_group["params"] = [changes.get(id(param), param) for param in param_group["params"]]

    for prev_param, cur_param in param_pairs:
        prev_state = optimizer.state.pop(prev_param, {})

        cur_state = {}
        for state_name, state_value in prev_state.items():
            if not torch.is_tensor(state_value):
                cur_state[state_name] = copy.deepcopy(state_value)
                continue

            if state_value.ndim == 0:
                cur_state[state_name] = (state_value.detach().clone())
                continue

            if state_value.shape == prev_param.shape:
                expanded_state = torch.zeros_like(cur_param, memory_format=torch.preserve_format)

                expanded_state[:prev_vocab_size].copy_(
                    state_value.to(device=expanded_state.device, dtype=expanded_state.dtype,))

                cur_state[state_name] = expanded_state
                continue

            cur_state[state_name] = (state_value.detach().clone())

        optimizer.state[cur_param] = cur_state

    return model, optimizer, vocab, token_to_id, id_to_token

def make_transformer_training_data(tokens, token_to_id, win_size):
    token_ids = []
    for token in tokens:
        token_ids.append(token_to_id[token])

    inputs = []
    targets = []
    for start in range(len(token_ids) - win_size):
        input = token_ids[start:start + win_size]

        target = token_ids[start + 1:start + win_size + 1]

        inputs.append(input)
        targets.append(target)

    train_inputs = torch.tensor(inputs, dtype=torch.long)
    train_targets = torch.tensor(targets, dtype=torch.long)

    return train_inputs, train_targets

def main():
    win_size = 64
    max_new_tokens = 10
    n_gram_power = 0.5

    temperature = 1
    transformer_lr = 1e-4
    epochs = 1

    d_model = 128 
    n_heads = 4 
    n_layers = 4 
    d_ff = 512
    dropout = 0.1

    transformer_data_pth = rf"example/data.pt"
    data0 = read_file(rf"example/data.txt").lower(); data0 = basic_punctuation_spacer(data0)

    train = True

    par_lst = []
    data0 = data0.split()

    data_lst = []
    data_lst_loader = []

    split_threshold = 1000

    vocab = []
    temp_count = 0
    for data in data0:
        temp_count += 1
        data_lst_loader.append(data)

        if temp_count == split_threshold:
            data = " ".join(data_lst_loader)
            data_lst.append(data)

            data_lst_loader = []
            temp_count = 0
    vocab.append("<unk>")
    
    if data_lst_loader:
        data = " ".join(data_lst_loader)
        data_lst.append(data)

    total_chunks = 0
    current_chunk = 0
    for data in data_lst:
        total_chunks += 1

    token_to_id = {}
    for token_id, token in enumerate(vocab):
        token_to_id[token] = token_id

    id_to_token = {}
    for token, token_id in token_to_id.items():
        id_to_token[token_id] = token
    
    conns = {}
    n_gram_vocab = []

    checkpoint_num = 0
    checkpoint_threshold = 100
    if train:
        model = Transformer(vocab_size=len(vocab), context_size=win_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=transformer_lr, weight_decay=0.01)
        for data2 in data_lst:
            win_size_temp = win_size

            checkpoint_num += 1
            if checkpoint_num == checkpoint_threshold:
                checkpoint_num = 0
                save_transformer(model, vocab, win_size, transformer_data_pth)

            data2_lst = data2.split()
            model, optimizer, vocab, token_to_id, id_to_token = expand_transformer_vocab(model, optimizer, vocab, token_to_id, id_to_token, data2_lst)

            if len(data2_lst) <= win_size:
                continue

            par_lst.append(data2)
            
            current_chunk += 1
            print(f"Current Chunk: {current_chunk}/{total_chunks}")

            for i in range(win_size_temp):
                n_gram_vocab, conns = train_model(data2, win_size_temp, n_gram_vocab, conns)

                win_size_temp -= 1

            train_inputs, train_targets = make_transformer_training_data(data2_lst, token_to_id, win_size)
                    
            train_transformer(model, epochs, train_inputs, train_targets, optimizer)
            n_gram_correction(model, optimizer, n_gram_power, len(vocab), token_to_id, conns, win_size)

        save_transformer(model, vocab, win_size, transformer_data_pth)
    else:
        model, vocab = load_transformer(transformer_data_pth, d_model, n_heads, n_layers, d_ff, dropout)
        token_to_id = {}
        for token_id, token in enumerate(vocab):
            token_to_id[token] = token_id

        id_to_token = {}
        for token, token_id in token_to_id.items():
            id_to_token[token_id] = token

    while True:
        tokens = basic_punctuation_spacer(input("input: ")).lower().split()

        output = predict_word_transformer(model, tokens, token_to_id, id_to_token, max_new_tokens, temperature)
        output = " ".join(output)

        print(output)

if __name__ == "__main__":
    main()
