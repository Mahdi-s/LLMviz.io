import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import HfApi
import pandas as pd
import matplotlib.pyplot as plt
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
    return model, tokenizer

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Activation Values for {layer_name}")
    
    if isinstance(layer_data, dict):
        data = layer_data["Hidden State"].detach().numpy().flatten()
    else:
        data = layer_data.detach().numpy().flatten()
    
    # Scatter plot
    ax1.scatter(range(len(data)), data, alpha=0.5)
    ax1.set_title("Scatter Plot")
    ax1.set_xlabel("Neuron Index")
    ax1.set_ylabel("Activation Value")
    
    # Bar plot
    ax2.bar(range(len(data[:100])), data[:100])  # Plotting first 100 values for clarity
    ax2.set_title("Bar Plot (First 100 Neurons)")
    ax2.set_xlabel("Neuron Index")
    ax2.set_ylabel("Activation Value")
    
    return fig

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
        fig = create_activation_plots(layer_data, layer_name)
        st.pyplot(fig)
        plt.close(fig)
        
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
                    model, tokenizer = load_model_and_tokenizer(selected_model)
                    
                    inputs = tokenizer(prompt, return_tensors="pt")
                    
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
                    
                    details = get_layer_details(outputs, tokenizer, inputs["input_ids"])
                    
                    # Generate text with temperature
                    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
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