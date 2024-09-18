from flask import Flask, render_template, request, jsonify
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

app = Flask(__name__)

# Constants
NUM_LAYERS = 6
NUM_HEADS = 12
SEQUENCE_LENGTH = 1024
VOCAB_SIZE = 50257
OUTPUT_VOCAB_SIZE = 50257

# Load the tokenizer and ONNX model once at startup
model_name = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create an inference session
ort_session = ort.InferenceSession('distilgpt2_with_activations.onnx')

# Print model input names (for debugging)
print("Model inputs:")
for input_meta in ort_session.get_inputs():
    print(f"Name: {input_meta.name}, Shape: {input_meta.shape}, Type: {input_meta.type}")

# Print model output names (for debugging)
print("\nModel outputs:")
for output_meta in ort_session.get_outputs():
    print(f"Name: {output_meta.name}, Shape: {output_meta.shape}, Type: {output_meta.type}")

# Dynamically generate layer_names based on model outputs
activation_output_names = [output_meta.name for output_meta in ort_session.get_outputs()][1:]  # Exclude logits
layer_names = activation_output_names  # Ensure this matches the number of activation outputs

print(f"Layer names: {layer_names}")
print(f"Number of activation outputs: {len(layer_names)}")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/compute', methods=['POST'])
def compute():
    try:
        prompt = request.form['prompt']

        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = inputs['input_ids']

        # Prepare inputs for ONNX Runtime
        ort_inputs = {ort_session.get_inputs()[0].name: input_ids}

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
        print(f"\nNumber of activation outputs: {len(activation_outputs)}")
        print(f"Layer names: {layer_names}")

        # Check for mismatch
        if len(activation_outputs) != len(layer_names):
            return jsonify({
                'error': f'Mismatch between number of activation outputs ({len(activation_outputs)}) and layer names ({len(layer_names)}).'
            }), 500

        for name, activation in zip(layer_names, activation_outputs):
            # Replace NaN and Infinity values
            activation = np.nan_to_num(activation, nan=0.0, posinf=1e6, neginf=-1e6)
            # Normalize activations to [0, 1] for visualization
            activation_min = activation.min()
            activation_max = activation.max()
            if activation_max - activation_min == 0:
                activation_normalized = np.zeros_like(activation)
            else:
                activation_normalized = (activation - activation_min) / (activation_max - activation_min)
            # Flatten the activation array
            activations[name] = activation_normalized.flatten().tolist()

        return jsonify({
            'activations': activations,
            'predicted_token': predicted_token
        })
    except Exception as e:
        print(f"Error in /compute: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500

if __name__ == "__main__":
    app.run(debug=True)
