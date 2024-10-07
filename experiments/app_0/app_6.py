import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import plotly.graph_objs as go
import numpy as np

@st.cache_resource
def load_model():
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device found. Using MPS.")
    else:
        device = torch.device("cpu")
        print("MPS device not found. Falling back to CPU.")
    
    model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    return model, tokenizer, device

def get_layer_weights(model, layer_name):
    return model.state_dict()[layer_name].detach().cpu().numpy()

def visualize_matrix(matrix, title):
    x, y, z = np.meshgrid(np.arange(matrix.shape[1]),
                          np.arange(matrix.shape[0]),
                          np.arange(1))
    
    values = matrix.flatten()
    
    trace = go.Surface(x=x.flatten(), y=y.flatten(), z=z.flatten(),
                       surfacecolor=values.reshape(x.shape),
                       colorscale='Viridis')
    
    layout = go.Layout(title=title,
                       scene=dict(xaxis_title='Dimension 1',
                                  yaxis_title='Dimension 2',
                                  zaxis_title='Value'))
    
    fig = go.Figure(data=[trace], layout=layout)
    return fig

st.title("DistilGPT2 Model Visualization (MPS Enabled)")

model, tokenizer, device = load_model()

query = st.text_input("Enter your query:", "Hello, world!")

if st.button("Run Query and Visualize"):
    inputs = tokenizer(query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Visualize embedding layer
    embedding_weights = get_layer_weights(model, 'transformer.wte.weight')
    st.subheader("Embedding Layer Visualization")
    st.plotly_chart(visualize_matrix(embedding_weights, "Embedding Layer"))
    
    # Visualize transformer layers
    for i, layer in enumerate(model.transformer.h):
        st.subheader(f"Transformer Layer {i+1}")
        
        # Visualize attention weights
        attention_weights = get_layer_weights(layer, 'attn.c_attn.weight')
        st.plotly_chart(visualize_matrix(attention_weights, f"Attention Weights - Layer {i+1}"))
        
        # Visualize feedforward weights
        ff_weights = get_layer_weights(layer, 'mlp.c_fc.weight')
        st.plotly_chart(visualize_matrix(ff_weights, f"Feedforward Weights - Layer {i+1}"))
        
        # Display layer information
        st.write(f"Layer {i+1} Information:")
        st.write(f"- Attention heads: {model.config.n_head}")
        st.write(f"- Hidden size: {model.config.hidden_size}")
        st.write(f"- Feedforward size: {model.config.n_inner if model.config.n_inner is not None else 'N/A'}")

    # Display model output
    generated_text = tokenizer.decode(outputs.logits[0].argmax(dim=-1))
    st.subheader("Generated Text")
    st.write(generated_text)

st.sidebar.write(f"Using device: {device}")