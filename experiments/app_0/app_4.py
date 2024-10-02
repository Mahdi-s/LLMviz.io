import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import plotly.graph_objs as go
import numpy as np

@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True)
    model = model.to(device)
    return model, tokenizer, device

def get_layer_details(outputs, tokenizer, input_ids):
    all_hidden_states = outputs.hidden_states
    
    details = [
        ("1. Input Embedding Layer", all_hidden_states[0]),
        ("2. Positional Encoding", all_hidden_states[1]),
    ]
    
    # Add Transformer Blocks
    for idx in range(2, len(all_hidden_states) - 1):
        details.append((f"3. Transformer Block {idx-1}", all_hidden_states[idx]))
    
    # Output Layer
    details.append(("4. Output Layer", all_hidden_states[-1]))
    
    return details

def create_stem_plot(layer_data, layer_name, tokens):
    data = layer_data.cpu().detach().numpy().squeeze()
    
    fig = go.Figure()
    
    for i, token in enumerate(tokens):
        fig.add_trace(go.Scatter(
            x=np.arange(data.shape[1]),
            y=data[i],
            mode='lines+markers',
            name=f'Token: {token}',
            line=dict(width=1),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title=f"Stem Plot - {layer_name}",
        xaxis_title="Neuron Index",
        yaxis_title="Activation Value",
        height=500,
        showlegend=True
    )
    
    return fig

def display_layer_details(details, tokenizer, input_ids):
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    for layer_name, layer_data in details:
        st.subheader(layer_name)
        
        st.write(f"Shape: {layer_data.shape}")
        st.write(f"Mean: {layer_data.mean().item():.4f}")
        st.write(f"Std: {layer_data.std().item():.4f}")
        
        stem_plot = create_stem_plot(layer_data, layer_name, tokens)
        st.plotly_chart(stem_plot, use_container_width=True)
        
        st.write("---")

def main():
    st.title("GPT-2 Architecture Visualization")
    
    st.sidebar.header("Settings")
    
    temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    prompt = st.sidebar.text_area("Enter your prompt:", value="Hello, how are you?")
    
    if st.sidebar.button("Run Query"):
        with st.spinner("Loading model and processing query..."):
            try:
                model, tokenizer, device = load_model_and_tokenizer()
                
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                
                details = get_layer_details(outputs, tokenizer, inputs["input_ids"])
                
                # Generate text with temperature
                generated_text = model.generate(
                    **inputs,
                    max_length=50,
                    num_return_sequences=1,
                    temperature=temperature
                )
                generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
                
                st.subheader("Generated Text")
                st.write(generated_text)
                
                st.subheader("Layer Details")
                display_layer_details(details, tokenizer, inputs["input_ids"])
            except Exception as e:
                st.error(f"An error occurred while processing the query: {str(e)}")

if __name__ == "__main__":
    main()