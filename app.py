from flask import Flask, render_template, request, jsonify
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

app = Flask(__name__)

# Load the tokenizer and ONNX model once at startup
model_name = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create an inference session
ort_session = ort.InferenceSession('model/model.onnx')

# Print model input names
print("Model inputs:")
for input_meta in ort_session.get_inputs():
    print(f"Name: {input_meta.name}, Shape: {input_meta.shape}, Type: {input_meta.type}")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/compute', methods=['POST'])
def compute():
    prompt = request.form['prompt']

    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs['input_ids']

    # Prepare inputs for ONNX Runtime
    ort_inputs = {'input_ids': input_ids}

    # Run inference and get intermediate activations
    ort_outputs = ort_session.run(None, ort_inputs)

    # The first output is logits; the rest are activations
    logits = ort_outputs[0]
    activation_outputs = ort_outputs[1:]

    # Get the predicted token
    predicted_token_id = np.argmax(logits[0, -1, :])
    predicted_token = tokenizer.decode([predicted_token_id])

    # Prepare activations for visualization
    activations = {}
    layer_names = ['embeddings'] + [f'layer_{i}' for i in range(len(activation_outputs) - 1)] + ['final_layer_norm']

    for name, activation in zip(layer_names, activation_outputs):
        # Replace NaN and Infinity values
        activation = np.nan_to_num(activation, nan=0.0, posinf=1e6, neginf=-1e6)
        activations[name] = activation.squeeze().tolist()

    return jsonify({
        'activations': activations,
        'layer_names': list(activations.keys()),
        'predicted_token': predicted_token
    })

if __name__ == "__main__":
    app.run(debug=True)
