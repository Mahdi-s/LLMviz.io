# app.py
from flask import Flask, render_template, jsonify
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

app = Flask(__name__)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModel.from_pretrained("distilgpt2")

# Move model to GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data')
def get_data():
    text = "Hello, how are you today?"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    all_hidden_states = list(outputs.hidden_states[1:]) + [outputs.last_hidden_state]
    data = []
    for i, layer_output in enumerate(all_hidden_states):
        z = layer_output[0].detach().cpu().numpy().T
        data.append({
            'layer': i,
            'values': z.tolist()
        })
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)