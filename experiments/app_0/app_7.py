import streamlit as st
import torch
from transformer_lens import HookedTransformer
import plotly.graph_objects as go
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    return HookedTransformer.from_pretrained("distilgpt2")

model = load_model()

st.title("DistilGPT2 Activation Visualization")

# Text input
user_input = st.text_area("Enter your text:", "Hello, world!")

# Temperature slider
temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# Run the model
tokens = model.to_tokens(user_input)
logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)

# Calculate loss
loss = model.loss_fn(logits, tokens, per_token=False)

st.write(f"Model loss: {loss.item():.4f}")

# Generate output using temperature
with torch.no_grad():
    output_logits = logits[:, -1, :] / temperature
    output_probs = torch.softmax(output_logits, dim=-1)
    output_token = torch.multinomial(output_probs, num_samples=1)
    output_text = model.to_string(output_token)

st.write(f"Generated next token: {output_text}")

# Extract and visualize activations
def visualize_activations(cache, layer_name):
    activations = cache[layer_name]
    if activations.dim() > 3:
        activations = activations.mean(dim=1)  # Average over heads if necessary
    
    fig = go.Figure(data=[go.Surface(z=activations.detach().cpu().numpy())])
    fig.update_layout(title=f'{layer_name} Activations', autosize=False, width=600, height=600,
                      scene=dict(xaxis_title='Position', yaxis_title='Channel', zaxis_title='Activation'))
    st.plotly_chart(fig)

# Visualize embeddings
visualize_activations(cache, 'embed')

# Visualize layer norm
visualize_activations(cache, 'ln_final')

# Visualize attention and MLP blocks
for layer in range(model.cfg.n_layers):
    st.subheader(f"Layer {layer}")
    
    # Attention
    visualize_activations(cache, f'blocks.{layer}.attn.hook_result')
    
    # MLP
    visualize_activations(cache, f'blocks.{layer}.mlp.hook_result')

# 3D visualization of all activations
def create_3d_activation_plot(cache):
    layers = ['embed'] + [f'blocks.{i}.attn.hook_result' for i in range(model.cfg.n_layers)] + \
             [f'blocks.{i}.mlp.hook_result' for i in range(model.cfg.n_layers)] + ['ln_final']
    
    x, y, z, colors = [], [], [], []
    
    for i, layer in enumerate(layers):
        act = cache[layer]
        if act.dim() > 3:
            act = act.mean(dim=1)
        act = act.detach().cpu().numpy()
        
        for j in range(act.shape[0]):
            for k in range(act.shape[1]):
                x.append(i)
                y.append(j)
                z.append(k)
                colors.append(act[j, k])
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title='3D Visualization of All Activations',
        scene=dict(
            xaxis_title='Layer',
            yaxis_title='Position',
            zaxis_title='Channel'
        ),
        width=800,
        height=800
    )
    
    st.plotly_chart(fig)

st.subheader("3D Visualization of All Activations")
create_3d_activation_plot(cache)