from flask import Flask, render_template, request, jsonify, send_from_directory
from transformer_lens import HookedTransformer
import torch
import numpy as np
import json
import math

app = Flask(__name__)

# Load the DistilGPT2 model
model = HookedTransformer.from_pretrained("distilgpt2", device='cpu')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return self.handle_array(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return self.handle_float(float(obj))
        if isinstance(obj, torch.Tensor):
            return self.handle_array(obj.detach().cpu().numpy())
        return json.JSONEncoder.default(self, obj)

    def handle_array(self, arr):
        arr_list = arr.tolist()
        return self.process_list(arr_list)

    def process_list(self, arr_list):
        new_list = []
        for item in arr_list:
            if isinstance(item, list):
                new_list.append(self.process_list(item))
            else:
                new_list.append(self.handle_float(float(item)))
        return new_list

    def handle_float(self, x):
        if math.isnan(x):
            return 0  # Convert NaN to 0
        elif math.isinf(x):
            return 1e-10 if x > 0 else -1e-10  # Convert +/-Infinity to very small numbers
        return x


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    text = request.json['text']
    tokens = model.to_tokens(text)
    logits, cache = model.run_with_cache(tokens)
    
    data = extract_data_from_cache(cache, tokens, logits)
    
    return json.dumps(data, cls=NumpyEncoder)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

def extract_data_from_cache(cache, tokens, logits):
    data = {
        'token_embeddings': cache['embed'].squeeze(),
        'positional_embeddings': cache['pos_embed'].squeeze(),
        'blocks': []
    }
    
    for block_idx in range(6):  # DistilGPT2 has 6 blocks
        block_data = {
            'attn_scores': cache[f'blocks.{block_idx}.attn.hook_attn_scores'].squeeze(),
            'attn': cache[f'blocks.{block_idx}.attn.hook_pattern'].squeeze(),
            'attn_z': cache[f'blocks.{block_idx}.attn.hook_z'].squeeze(),
            'attn_out': cache[f'blocks.{block_idx}.hook_attn_out'].squeeze(),
            'mlp_out': cache[f'blocks.{block_idx}.hook_mlp_out'].squeeze()
        }
        data['blocks'].append(block_data)
    
    # Add model predictions
    probs = torch.softmax(logits[:, -1], dim=-1)
    top_probs, top_indices = probs.topk(10)
    
    predictions = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        token = model.to_string([idx])
        predictions.append({'token': token, 'probability': prob})
    
    data['predictions'] = predictions
    
    return data

if __name__ == '__main__':
    app.run(debug=True)