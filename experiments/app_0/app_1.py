import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import HfApi
import pandas as pd
import plotly.graph_objs as go
import numpy as np

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
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
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

def create_activation_plots(layer_data, layer_name):
    if isinstance(layer_data, dict):
        data = layer_data["Hidden State"].cpu().detach().numpy().flatten()
    else:
        data = layer_data.cpu().detach().numpy().flatten()
    
    # 2D Scatter plot
    scatter_2d = go.Figure(data=go.Scatter(
        x=list(range(len(data))),
        y=data,
        mode='markers',
        name='Activations'
    ))
    scatter_2d.update_layout(
        title=f"2D Scatter Plot - {layer_name}",
        xaxis_title="Neuron Index",
        yaxis_title="Activation Value",
        height=500
    )
    
    # Bar plot
    bar_plot = go.Figure(data=go.Bar(
        x=list(range(100)),
        y=data[:100],
        name='First 100 Activations'
    ))
    bar_plot.update_layout(
        title=f"Bar Plot (First 100 Neurons) - {layer_name}",
        xaxis_title="Neuron Index",
        yaxis_title="Activation Value",
        height=500
    )
    
    # 3D Scatter plot
    sample_size = min(1000, len(data))  # Limit to 1000 points for performance
    sample_data = data[:sample_size]
    x = np.arange(sample_size)
    y = np.random.rand(sample_size)  # Random y-coordinates for visualization
    scatter_3d = go.Figure(data=go.Scatter3d(
        x=x,
        y=y,
        z=sample_data,
        mode='markers',
        marker=dict(
            size=5,
            color=sample_data,
            colorscale='Viridis',
            opacity=0.8
        ),
        name='3D Activations'
    ))
    scatter_3d.update_layout(
        title=f"3D Scatter Plot - {layer_name}",
        scene=dict(
            xaxis_title="Neuron Index",
            yaxis_title="Random Dimension",
            zaxis_title="Activation Value"
        ),
        height=700
    )
    
    return scatter_2d, bar_plot, scatter_3d

def display_layer_details(details, tokenizer, input_ids):
    for layer_name, layer_data in details:
        st.subheader(layer_name)
        
        if isinstance(layer_data, dict):
            for key, value in layer_data.items():
                st.write(f"{key} shape: {value.shape}")
                st.write(f"{key} mean: {value.mean().item():.4f}")
                st.write(f"{key} std: {value.std().item():.4f}")
        else:
            st.write(f"Shape: {layer_data.shape}")
            st.write(f"Mean: {layer_data.mean().item():.4f}")
            st.write(f"Std: {layer_data.std().item():.4f}")
        
        # Create and display activation plots
        scatter_2d, bar_plot, scatter_3d = create_activation_plots(layer_data, layer_name)
        st.plotly_chart(scatter_2d, use_container_width=True)
        st.plotly_chart(bar_plot, use_container_width=True)
        st.plotly_chart(scatter_3d, use_container_width=True)
        
        if layer_name == "Final Output":
            # Display top predicted tokens
            top_tokens = torch.topk(layer_data[:, -1, :], k=5)
            st.write("Top 5 predicted tokens:")
            for token, score in zip(top_tokens.indices[0], top_tokens.values[0]):
                st.write(f"{tokenizer.decode([token.item()])}: {score.item():.4f}")
        
        st.write("---")

def main():
    st.title("Enhanced LLM Architecture Visualization")
    
    # Sidebar
    st.sidebar.header("Model Selection and Settings")
    
    try:
        model_df = get_model_list()
        if model_df.empty:
            st.sidebar.error("No suitable models found. Please try again later.")
            return
        
        selected_model = st.sidebar.selectbox("Choose a model:", model_df['name'])
        
        temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        prompt = st.sidebar.text_area("Enter your prompt:", value="Hello, how are you?")
        
        if st.sidebar.button("Run Query"):
            with st.spinner("Loading model and processing query..."):
                try:
                    model, tokenizer, device = load_model_and_tokenizer(selected_model)
                    
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
                    
                    details = get_layer_details(outputs, tokenizer, inputs["input_ids"])
                    
                    # Generate text with temperature
                    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
                    generated_text = generator(prompt, max_length=50, num_return_sequences=1, temperature=temperature)[0]['generated_text']
                    
                    st.subheader("Generated Text")
                    st.write(generated_text)
                    
                    st.subheader("Layer Details")
                    display_layer_details(details, tokenizer, inputs["input_ids"])
                except Exception as e:
                    st.error(f"An error occurred while processing the query: {str(e)}")
        
        # Display model information
        st.sidebar.subheader("Model Information")
        if selected_model in model_df['name'].values:
            model_info = model_df[model_df['name'] == selected_model].iloc[0]
            st.sidebar.write(f"Downloads: {model_info['downloads']}")
            st.sidebar.write(f"Likes: {model_info['likes']}")
            st.sidebar.write(f"Size: {model_info['size']} MB")
    
    except Exception as e:
        st.error(f"An error occurred while setting up the application: {str(e)}")

if __name__ == "__main__":
    main()