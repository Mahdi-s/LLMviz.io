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

def get_layer_details(model, inputs, outputs, attention_activations, mlp_activations):
    all_hidden_states = outputs.hidden_states

    # Get the positional encodings for the input positions
    position_ids = inputs['position_ids'] if 'position_ids' in inputs else torch.arange(0, inputs['input_ids'].size(-1), dtype=torch.long, device=inputs['input_ids'].device).unsqueeze(0)
    positional_encodings = model.transformer.wpe(position_ids)

    details = [
        ("1. Input Embedding Layer", all_hidden_states[0]),
        ("2. Positional Encoding", positional_encodings),
        ("3. Transformer Block 1", (attention_activations[0], mlp_activations[0])),
    ]

    return details

def create_stem_plot(layer_data, layer_name, tokens):
    data = layer_data.cpu().detach().numpy()
    # Remove batch dimension if present
    if data.shape[0] == 1:
        data = np.squeeze(data, axis=0)

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

def create_positional_encoding_plot(positional_encoding, tokens):
    data = positional_encoding.cpu().detach().numpy()
    # Remove batch dimension if present
    if data.shape[0] == 1:
        data = np.squeeze(data, axis=0)

    fig = go.Figure()

    for i, token in enumerate(tokens):
        fig.add_trace(go.Scatter(
            x=np.arange(data.shape[1]),
            y=data[i],
            mode='lines+markers',
            name=f'Position: {i}',
            line=dict(width=1),
            marker=dict(size=4)
        ))

    fig.update_layout(
        title="Stem Plot - Positional Encoding",
        xaxis_title="Embedding Dimension",
        yaxis_title="Encoding Value",
        height=500,
        showlegend=True
    )

    return fig

def create_combined_3d_stem_plot(attention_data, mlp_data, layer_name, tokens):
    attention_data = attention_data.cpu().detach().numpy()
    mlp_data = mlp_data.cpu().detach().numpy()

    # Remove batch dimension if present
    if attention_data.shape[0] == 1:
        attention_data = np.squeeze(attention_data, axis=0)
    if mlp_data.shape[0] == 1:
        mlp_data = np.squeeze(mlp_data, axis=0)

    seq_len, hidden_size = attention_data.shape

    x_attn = []
    y_attn = []
    z_attn = []

    x_mlp = []
    y_mlp = []
    z_mlp = []

    for i in range(seq_len):
        for j in range(hidden_size):
            x_attn.append(j)
            y_attn.append(i)
            z_attn.append(attention_data[i, j])

            x_mlp.append(j)
            y_mlp.append(i)
            z_mlp.append(mlp_data[i, j])

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x_attn,
        y=y_attn,
        z=z_attn,
        mode='markers',
        name='Attention',
        marker=dict(
            size=2,
            color='red',
            opacity=0.5
        )
    ))

    fig.add_trace(go.Scatter3d(
        x=x_mlp,
        y=y_mlp,
        z=z_mlp,
        mode='markers',
        name='MLP',
        marker=dict(
            size=2,
            color='blue',
            opacity=0.5
        )
    ))

    fig.update_layout(
        title=f"3D Stem Plot - {layer_name}",
        scene=dict(
            xaxis_title='Neuron Index',
            yaxis_title='Token Position',
            zaxis_title='Activation Value'
        ),
        height=700
    )

    return fig

def display_layer_details(details, tokenizer, input_ids):
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    for layer_name, layer_data in details:
        st.subheader(layer_name)

        if layer_name == "3. Transformer Block 1":
            attention_data, mlp_data = layer_data
            st.write("Attention Output:")
            st.write(f"Shape: {attention_data.shape}")
            st.write(f"Mean: {attention_data.mean().item():.4f}")
            st.write(f"Std: {attention_data.std().item():.4f}")

            st.write("MLP Output:")
            st.write(f"Shape: {mlp_data.shape}")
            st.write(f"Mean: {mlp_data.mean().item():.4f}")
            st.write(f"Std: {mlp_data.std().item():.4f}")

            stem_plot = create_combined_3d_stem_plot(attention_data, mlp_data, layer_name, tokens)

            st.plotly_chart(stem_plot, use_container_width=True)

            st.write("---")
        else:
            st.write(f"Shape: {layer_data.shape}")
            st.write(f"Mean: {layer_data.mean().item():.4f}")
            st.write(f"Std: {layer_data.std().item():.4f}")

            if layer_name == "2. Positional Encoding":
                stem_plot = create_positional_encoding_plot(layer_data, tokens)
            else:
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

                attention_activations = []
                mlp_activations = []

                # Define the hook functions inside main() so they can access the lists
                def attention_hook(module, input, output):
                    # Extract the tensor from the tuple if necessary
                    if isinstance(output, tuple):
                        attention_activations.append(output[0])
                    else:
                        attention_activations.append(output)

                def mlp_hook(module, input, output):
                    # Extract the tensor from the tuple if necessary
                    if isinstance(output, tuple):
                        mlp_activations.append(output[0])
                    else:
                        mlp_activations.append(output)

                first_block = model.transformer.h[0]

                attn_handle = first_block.attn.register_forward_hook(attention_hook)
                mlp_handle = first_block.mlp.register_forward_hook(mlp_hook)

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)

                # Remove hooks after use
                attn_handle.remove()
                mlp_handle.remove()

                details = get_layer_details(model, inputs, outputs, attention_activations, mlp_activations)

                # Generate text with temperature
                generated_text = model.generate(
                    **inputs,
                    max_length=50,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,  # Enable sampling to use temperature
                    pad_token_id=tokenizer.eos_token_id  # Avoid an error with padding
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
