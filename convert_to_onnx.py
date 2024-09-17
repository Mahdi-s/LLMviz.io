import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'distilgpt2'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare dummy input for tracing
dummy_input = tokenizer("Hello, world!", return_tensors="pt")
input_ids = dummy_input['input_ids']

# Modify the model to output intermediate activations
class DistilGPT2WithActivations(torch.nn.Module):
    def __init__(self, model):
        super(DistilGPT2WithActivations, self).__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model.transformer(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True
        )
        logits = self.model.lm_head(outputs.last_hidden_state)
        # Return logits and all hidden states
        return (logits,) + outputs.hidden_states

# Initialize the modified model
modified_model = DistilGPT2WithActivations(model)
modified_model.eval()

# Prepare output names
num_hidden_layers = model.config.n_layer + 1  # +1 for embeddings
output_names = ['logits'] + [f'layer_{i}' for i in range(num_hidden_layers)]

# Export to ONNX without attention_mask
torch.onnx.export(
    modified_model,
    (input_ids,),
    'distilgpt2_with_activations.onnx',
    input_names=['input_ids'],
    output_names=output_names,
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size', 1: 'sequence'},
    },
    opset_version=14,
)
