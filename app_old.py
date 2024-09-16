import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import onnx
import onnxruntime
import numpy as np
from transformer_lens import HookedTransformer
from transformers.onnx import export, FeaturesManager
from pathlib import Path

def main():
    prompt = input("Enter a prompt: ")

    # Use a smaller model
    model_name = 'distilgpt2'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare the inputs
    inputs = tokenizer(prompt, return_tensors="pt")

    # Specify the path where the ONNX model will be saved
    onnx_model_path = Path("distilgpt2.onnx")

    # Get the model type from the model's configuration
    model_type = model.config.model_type  # This should be 'gpt2' for distilgpt2

    # Get the appropriate ONNX config for the model
    feature = "causal-lm"
    onnx_config_class = FeaturesManager.get_config(model_type, feature=feature)
    onnx_config = onnx_config_class(model.config)

    # Export the model to ONNX using the transformers.onnx export utility
    export(
        tokenizer,          # preprocessor (positional argument)
        model,              # model (positional argument)
        onnx_config,        # config (positional argument)
        opset=14,           # Updated opset version
        output=onnx_model_path,
    )

    # Run the model using ONNX Runtime
    session = onnxruntime.InferenceSession(str(onnx_model_path))

    # Convert inputs to numpy arrays
    ort_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    }

    # Run the model
    ort_outs = session.run(None, ort_inputs)

    # Convert the output logits to tokens
    logits = ort_outs[0]
    predicted_token_ids = np.argmax(logits, axis=-1)
    predicted_tokens = tokenizer.decode(predicted_token_ids[0], clean_up_tokenization_spaces=True)

    print("\nONNX Runtime Output:")
    print(predicted_tokens)

    # Use transformer-lens to extract internal activations
    hooked_model = HookedTransformer.from_pretrained(model_name)
    tokens = hooked_model.to_tokens(prompt)
    _, cache = hooked_model.run_with_cache(tokens)

    # Print detailed descriptions of the activations
    for name in cache.keys():
        activation = cache[name]
        print(f"\nActivation '{name}':")
        print(f" - Shape: {tuple(activation.shape)}")
        print(f" - Data Type: {activation.dtype}")
        print(f" - Mean: {activation.mean().item():.6f}")
        print(f" - Standard Deviation: {activation.std().item():.6f}")
        print(f" - Minimum Value: {activation.min().item():.6f}")
        print(f" - Maximum Value: {activation.max().item():.6f}")

if __name__ == "__main__":
    main()
