from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from transformer_lens import HookedTransformer

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/compute', methods=['POST'])
def compute():
    prompt = request.form['prompt']

    model_name = 'distilgpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_token_id = outputs.logits[0, -1].argmax()
    predicted_token = tokenizer.decode(predicted_token_id)

    # Use transformer-lens to extract internal activations
    hooked_model = HookedTransformer.from_pretrained(model_name)
    tokens = hooked_model.to_tokens(prompt)
    logits, cache = hooked_model.run_with_cache(tokens)

    # Prepare activations for visualization
    activations = {
        'token_embeddings': cache['hook_embed'].squeeze().numpy(),
        'positional_embeddings': cache['hook_pos_embed'].squeeze().numpy(),
    }

    # Extract layer-specific activations
    for layer in range(hooked_model.cfg.n_layers):
        activations.update({
            f'layer_{layer}_norm1': cache[f'blocks.{layer}.ln1.hook_normalized'].squeeze().numpy(),
            f'layer_{layer}_attn_q': cache[f'blocks.{layer}.attn.hook_q'].squeeze().numpy(),
            f'layer_{layer}_attn_k': cache[f'blocks.{layer}.attn.hook_k'].squeeze().numpy(),
            f'layer_{layer}_attn_v': cache[f'blocks.{layer}.attn.hook_v'].squeeze().numpy(),
            f'layer_{layer}_mlp_out': cache[f'blocks.{layer}.mlp.hook_post'].squeeze().numpy(),
            f'layer_{layer}_resid_pre': cache[f'blocks.{layer}.hook_resid_pre'].squeeze().numpy(),
            f'layer_{layer}_resid_post': cache[f'blocks.{layer}.hook_resid_post'].squeeze().numpy(),
        })

    # Replace NaN and Infinity values
    for key in activations:
        activations[key] = np.nan_to_num(activations[key], nan=0.0, posinf=1e6, neginf=-1e6).tolist()

    return jsonify({
        'activations': activations,
        'layer_names': list(activations.keys()),
        'predicted_token': predicted_token
    })

if __name__ == "__main__":
    app.run(debug=True)