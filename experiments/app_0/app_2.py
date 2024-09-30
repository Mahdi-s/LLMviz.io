import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import HfApi
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import scipy.stats
from sklearn.decomposition import PCA

@st.cache_data
def get_model_list():
    api = HfApi()
    models = api.list_models(sort="downloads", direction=-1, limit=100)
    model_data = []
    for model in models:
        try:
            if model.pipeline_tag and 'text-generation' in model.pipeline_tag:
                size = round(model.siblings[0].size / 1e6, 2) if model.siblings else None
                model_data.append({
                    'name': model.modelId,
                    'downloads': model.downloads,
                    'likes': model.likes,
                    'size': size  # Size in MB
                })
        except AttributeError:
            # Skip models that don't have the expected attributes
            continue

    df = pd.DataFrame(model_data)
    if not df.empty:
        df = df.sort_values('size').reset_index(drop=True)
    return df

@st.cache_resource
def load_model_and_tokenizer(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        output_attentions=True
    )
    model = model.to(device)
    return model, tokenizer, device

def get_layer_details(outputs, tokenizer, input_ids):
    all_hidden_states = outputs.hidden_states
    all_attentions = outputs.attentions

    details = []

    # Embedding layer
    details.append(("Embedding Layer", all_hidden_states[0]))

    # Hidden layers
    for idx, (hidden_state, attention) in enumerate(zip(all_hidden_states[1:], all_attentions)):
        details.append((f"Layer {idx + 1}", {
            "Hidden State": hidden_state,
            "Attention": attention
        }))

    # Final output
    logits = outputs.logits
    details.append(("Final Output", logits))

    return details

def plot_activation_magnitude_and_norms(layer_data, layer_name):
    # Assuming layer_data is a tensor of shape (batch_size, seq_length, hidden_size)
    if isinstance(layer_data, dict):
        hidden_state = layer_data["Hidden State"]
    else:
        hidden_state = layer_data
    l2_norms = torch.norm(hidden_state, p=2, dim=-1).cpu().detach().numpy()  # shape: (batch_size, seq_length)

    # Plot L2 Norms over sequence positions
    fig = go.Figure()
    for i in range(l2_norms.shape[0]):  # For each item in batch
        fig.add_trace(go.Scatter(
            x=list(range(l2_norms.shape[1])),
            y=l2_norms[i],
            mode='lines',
            name=f'Batch {i}'
        ))
    fig.update_layout(
        title=f"L2 Norms over Sequence Positions - {layer_name}",
        xaxis_title="Sequence Position",
        yaxis_title="L2 Norm"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_layer_activation_statistics(layer_data, layer_name):
    if isinstance(layer_data, dict):
        activations = layer_data["Hidden State"].cpu().detach().numpy().flatten()
    else:
        activations = layer_data.cpu().detach().numpy().flatten()
    mean = np.mean(activations)
    var = np.var(activations)
    skew = scipy.stats.skew(activations)
    kurt = scipy.stats.kurtosis(activations)
    st.write(f"**Mean:** {mean:.4f}")
    st.write(f"**Variance:** {var:.4f}")
    st.write(f"**Skewness:** {skew:.4f}")
    st.write(f"**Kurtosis:** {kurt:.4f}")

def plot_attention_weights(layer_data, layer_name):
    # layer_data should be a dict with 'Attention' key
    attention = layer_data['Attention']  # shape: (batch_size, num_heads, seq_length, seq_length)
    avg_attention = attention.mean(dim=1).cpu().detach().numpy()[0]  # Average over heads, select first in batch
    fig = go.Figure(data=go.Heatmap(
        z=avg_attention,
        x=list(range(avg_attention.shape[1])),
        y=list(range(avg_attention.shape[0]))
    ))
    fig.update_layout(
        title=f"Average Attention Weights - {layer_name}",
        xaxis_title="Key Positions",
        yaxis_title="Query Positions"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_attention_head_importance(layer_data, layer_name):
    attention = layer_data['Attention']  # shape: (batch_size, num_heads, seq_length, seq_length)
    # Compute the average attention over batch and sequence positions
    head_importance = attention.mean(dim=(0, 2, 3)).cpu().detach().numpy()
    fig = go.Figure(data=go.Bar(
        x=list(range(len(head_importance))),
        y=head_importance,
        name='Attention Head Importance'
    ))
    fig.update_layout(
        title=f"Attention Head Importance - {layer_name}",
        xaxis_title="Head Index",
        yaxis_title="Average Attention Weight"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_pca_analysis(layer_data, layer_name):
    if isinstance(layer_data, dict):
        hidden_state = layer_data["Hidden State"]
    else:
        hidden_state = layer_data
    # Reshape activations to (batch_size * seq_length, hidden_size)
    activations = hidden_state.cpu().detach().numpy().reshape(-1, hidden_state.shape[-1])
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(activations)
    explained_variance = pca.explained_variance_ratio_.sum()
    fig = go.Figure(data=go.Scatter(
        x=principal_components[:, 0],
        y=principal_components[:, 1],
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
            opacity=0.7
        ),
        name='PCA Components'
    ))
    fig.update_layout(
        title=f"PCA Analysis - {layer_name} (Explained Variance: {explained_variance:.2%})",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_activation_distribution(layer_data, layer_name):
    if isinstance(layer_data, dict):
        activations = layer_data["Hidden State"].cpu().detach().numpy().flatten()
    else:
        activations = layer_data.cpu().detach().numpy().flatten()
    fig = go.Figure(data=[go.Histogram(x=activations, nbinsx=100)])
    fig.update_layout(
        title=f"Activation Distribution - {layer_name}",
        xaxis_title="Activation Value",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig, use_container_width=True)

def display_layer_details(details, tokenizer, input_ids):
    for idx, (layer_name, layer_data) in enumerate(details):
        st.subheader(layer_name)

        if isinstance(layer_data, dict):
            st.write(f"Hidden State shape: {layer_data['Hidden State'].shape}")
            st.write(f"Hidden State mean: {layer_data['Hidden State'].mean().item():.4f}")
            st.write(f"Hidden State std: {layer_data['Hidden State'].std().item():.4f}")
            st.write(f"Attention shape: {layer_data['Attention'].shape}")
            st.write(f"Attention mean: {layer_data['Attention'].mean().item():.4f}")
            st.write(f"Attention std: {layer_data['Attention'].std().item():.4f}")
        else:
            st.write(f"Shape: {layer_data.shape}")
            st.write(f"Mean: {layer_data.mean().item():.4f}")
            st.write(f"Std: {layer_data.std().item():.4f}")

        # Visualization options
        visualization_options = [
            "Activation Magnitude and Norms (L2 Norms)",
            "Layer-wise Activation Statistics",
            "Attention Weights",
            "Attention Head Importance",
            "PCA Analysis",
            "Activation Distribution (Histogram)",
            # Add more visualization options here
        ]
        # Use a key to ensure the selection is unique per layer
        select_key = f"{layer_name}_viz_{idx}"
        selected_visualizations = st.multiselect(
            f"Select visualizations for {layer_name}",
            visualization_options,
            key=select_key
        )

        for viz in selected_visualizations:
            if viz == "Activation Magnitude and Norms (L2 Norms)":
                plot_activation_magnitude_and_norms(layer_data, layer_name)
            elif viz == "Layer-wise Activation Statistics":
                plot_layer_activation_statistics(layer_data, layer_name)
            elif viz == "Attention Weights" and isinstance(layer_data, dict):
                plot_attention_weights(layer_data, layer_name)
            elif viz == "Attention Head Importance" and isinstance(layer_data, dict):
                plot_attention_head_importance(layer_data, layer_name)
            elif viz == "PCA Analysis":
                plot_pca_analysis(layer_data, layer_name)
            elif viz == "Activation Distribution (Histogram)":
                plot_activation_distribution(layer_data, layer_name)
            else:
                st.write(f"Visualization '{viz}' is not available for {layer_name}")

        if layer_name == "Final Output":
            # Display top predicted tokens
            logits = layer_data
            top_tokens = torch.topk(logits[:, -1, :], k=5)
            st.write("**Top 5 predicted tokens:**")
            for token, score in zip(top_tokens.indices[0], top_tokens.values[0]):
                st.write(f"{tokenizer.decode([token.item()])}: {score.item():.4f}")

        st.write("---")

def main():
    st.title("Enhanced LLM Architecture Visualization")

    # Initialize session state variables at the start
    if 'generated' not in st.session_state:
        st.session_state['generated'] = False
    if 'generated_text' not in st.session_state:
        st.session_state['generated_text'] = ''
    if 'details' not in st.session_state:
        st.session_state['details'] = None
    if 'tokenizer' not in st.session_state:
        st.session_state['tokenizer'] = None
    if 'inputs' not in st.session_state:
        st.session_state['inputs'] = None
    if 'model_df' not in st.session_state:
        st.session_state['model_df'] = get_model_list()

    # Sidebar
    st.sidebar.header("Model Selection and Settings")

    try:
        model_df = st.session_state['model_df']
        if model_df.empty:
            st.sidebar.error("No suitable models found. Please try again later.")
            return

        selected_model = st.sidebar.selectbox("Choose a model:", model_df['name'])

        temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        prompt = st.sidebar.text_area("Enter your prompt:", value="Hello, how are you?")

        # Run Query button
        if st.sidebar.button("Run Query"):
            with st.spinner("Loading model and processing query..."):
                try:
                    model, tokenizer, device = load_model_and_tokenizer(selected_model)

                    inputs = tokenizer(prompt, return_tensors="pt").to(device)

                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

                    details = get_layer_details(outputs, tokenizer, inputs["input_ids"])

                    # Generate text with temperature
                    generator = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    generated_text = generator(
                        prompt,
                        max_length=50,
                        num_return_sequences=1,
                        temperature=temperature
                    )[0]['generated_text']

                    # Store results in session state
                    st.session_state['generated_text'] = generated_text
                    st.session_state['details'] = details
                    st.session_state['tokenizer'] = tokenizer
                    st.session_state['inputs'] = inputs
                    st.session_state['generated'] = True
                except Exception as e:
                    st.error(f"An error occurred while processing the query: {str(e)}")
                    st.session_state['generated'] = False

        # Display results if available
        if st.session_state['generated']:
            st.subheader("Generated Text")
            st.write(st.session_state['generated_text'])

            st.subheader("Layer Details")
            display_layer_details(
                st.session_state['details'],
                st.session_state['tokenizer'],
                st.session_state['inputs']['input_ids']
            )

        # Display model information
        st.sidebar.subheader("Model Information")
        if selected_model in model_df['name'].values:
            model_info = model_df[model_df['name'] == selected_model].iloc[0]
            st.sidebar.write(f"**Downloads:** {model_info['downloads']}")
            st.sidebar.write(f"**Likes:** {model_info['likes']}")
            st.sidebar.write(f"**Size:** {model_info['size']} MB")

    except Exception as e:
        st.error(f"An error occurred while setting up the application: {str(e)}")

if __name__ == "__main__":
    main()
