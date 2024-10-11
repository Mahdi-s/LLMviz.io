import streamlit as st
import torch
from transformer_lens import HookedTransformer
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load the DistilGPT-2 model
@st.cache_resource
def load_model():
    return HookedTransformer.from_pretrained("distilgpt2")

model = load_model()

st.title("DistilGPT-2 Analysis App")

# Text input
user_input = st.text_area("Enter your text:", "Hello, world!")

if st.button("Analyze"):
    # Tokenize input
    tokens = model.to_tokens(user_input)

    # Run model with cache
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    # Extract loss
    loss = model.loss_fn(logits, tokens, per_token=False)

    st.write(f"Model loss: {loss.item():.4f}")

    # Extract and display token embeddings
    token_embeddings = cache["embed"]
    st.subheader("Token Embeddings")
    fig = px.imshow(token_embeddings.detach().cpu().numpy(),
                    labels=dict(x="Embedding Dimension", y="Token Position"),
                    title="Token Embeddings Heatmap")
    st.plotly_chart(fig)

    # Analyze activations for each layer
    st.subheader("Layer Activations Analysis")
    for layer in range(model.cfg.n_layers):
        st.write(f"Layer {layer}")

        # Attention patterns
        attn_pattern_key = f"blocks.{layer}.attn.hook_pattern"
        if attn_pattern_key in cache:
            attn_patterns = cache[attn_pattern_key]
            fig = px.imshow(attn_patterns[0].detach().cpu().numpy(),
                            labels=dict(x="Source Position", y="Destination Position"),
                            title=f"Layer {layer} Attention Patterns")
            st.plotly_chart(fig)
        else:
            st.write(f"Attention patterns not available for layer {layer}")

        # MLP activations
        mlp_out_key = f"blocks.{layer}.mlp.hook_out"
        if mlp_out_key in cache:
            mlp_out = cache[mlp_out_key]
            mlp_stats = {
                "Mean": mlp_out.mean().item(),
                "Std Dev": mlp_out.std().item(),
                "Min": mlp_out.min().item(),
                "Max": mlp_out.max().item()
            }
            st.write("MLP Output Statistics:")
            st.json(mlp_stats)

            # Activation distribution
            fig = go.Figure(data=[go.Histogram(x=mlp_out.flatten().detach().cpu().numpy())])
            fig.update_layout(title=f"Layer {layer} MLP Output Distribution",
                              xaxis_title="Activation Value",
                              yaxis_title="Frequency")
            st.plotly_chart(fig)
        else:
            st.write(f"MLP output not available for layer {layer}")

    # Extract logits
    st.subheader("Logits")
    logits_np = logits.detach().cpu().numpy()
    
    # Create a dropdown to select which token's logits to display
    token_index = st.selectbox("Select token position to view logits:", 
                               range(logits_np.shape[0]), 
                               format_func=lambda x: f"Token {x}")
    
    # Display logits for the selected token
    fig = px.imshow(logits_np[token_index].reshape(1, -1),
                    labels=dict(x="Vocabulary Index", y="Token Position"),
                    title=f"Logits Heatmap for Token {token_index}")
    st.plotly_chart(fig)

    # Display top 10 predicted tokens
    top_k = 10
    top_logits, top_indices = torch.topk(logits[token_index], k=top_k)
    top_tokens = model.to_string(top_indices)
    
    st.write(f"Top {top_k} predicted tokens for position {token_index}:")
    for token, logit in zip(top_tokens, top_logits):
        st.write(f"{token}: {logit.item():.4f}")

st.write("Note: This app caches the model and activations for performance. Refresh the page if you want to reset the cache.")