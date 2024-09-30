import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import HfApi
import pandas as pd
import numpy as np
import pygame as pg
from OpenGL.GL import *
from OpenGL.GLU import *
import multiprocessing

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
    details.append(("Embedding Layer", all_hidden_states[0]))
    
    for idx, (hidden_state, attention) in enumerate(zip(all_hidden_states[1:], all_attentions)):
        details.append((f"Layer {idx + 1}", {
            "Hidden State": hidden_state,
            "Attention": attention
        }))
    
    details.append(("Final Output", outputs.logits))
    
    return details

class Camera:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, 10.0])
        self.front = np.array([0.0, 0.0, -1.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.right = np.array([1.0, 0.0, 0.0])
        self.yaw = -90.0
        self.pitch = 0.0
        self.move_speed = 0.1
        self.mouse_sensitivity = 0.1

    def update_camera_vectors(self):
        front = np.array([
            np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
            np.sin(np.radians(self.pitch)),
            np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        ])
        self.front = front / np.linalg.norm(front)
        self.right = np.cross(self.front, np.array([0.0, 1.0, 0.0]))
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.front)
        self.up /= np.linalg.norm(self.up)

class GUISlider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label):
        self.rect = pg.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.active = False

    def draw(self, surface):
        pg.draw.rect(surface, (200, 200, 200), self.rect)
        slider_pos = int((self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width)
        pg.draw.rect(surface, (100, 100, 100), (self.rect.left + slider_pos - 5, self.rect.top, 10, self.rect.height))
        font = pg.font.Font(None, 24)
        text = font.render(f"{self.label}: {self.value:.2f}", True, (0, 0, 0))
        surface.blit(text, (self.rect.left, self.rect.top - 25))

    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
        elif event.type == pg.MOUSEBUTTONUP:
            self.active = False
        elif event.type == pg.MOUSEMOTION and self.active:
            relative_x = max(0, min(event.pos[0] - self.rect.left, self.rect.width))
            self.value = self.min_val + (self.max_val - self.min_val) * (relative_x / self.rect.width)

class Renderer:
    def __init__(self, width, height):
        pg.init()
        self.width, self.height = width, height
        self.display = pg.display.set_mode((width, height), pg.OPENGL | pg.DOUBLEBUF | pg.RESIZABLE)
        self.gui_surface = pg.Surface((width, height), pg.SRCALPHA)
        self.clock = pg.time.Clock()
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 1, 0))
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        self.camera = Camera()
        self.setup_perspective()

        self.sliders = [
            GUISlider(10, height - 40, 200, 30, 0.1, 2.0, 1.0, "Sphere Size"),
            GUISlider(10, height - 80, 200, 30, 0.1, 2.0, 1.0, "Color Intensity")
        ]

        self.font = pg.font.Font(None, 36)

    def setup_perspective(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

    def render(self, layer_data):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(*self.camera.pos, *(self.camera.pos + self.camera.front), *self.camera.up)

        layers_3d = [layer for layer in layer_data if layer.ndim == 3]
        
        if not layers_3d:
            print("No 3D layers found in the data.")
            self.render_no_data_message()
        else:
            max_layers = len(layers_3d)
            max_seq_len = max(layer.shape[1] for layer in layers_3d)
            max_neurons = max(layer.shape[2] for layer in layers_3d)

            sphere_size = self.sliders[0].value * 0.02
            color_intensity = self.sliders[1].value

            for layer_idx, layer in enumerate(layers_3d):
                z = (layer_idx / max_layers) * 20 - 10
                seq_len, num_neurons = layer.shape[1], layer.shape[2]
                for seq_idx in range(seq_len):
                    x = (seq_idx / max_seq_len) * 20 - 10
                    for neuron_idx in range(num_neurons):
                        y = (neuron_idx / max_neurons) * 20 - 10
                        activation = layer[0, seq_idx, neuron_idx]

                        glPushMatrix()
                        glTranslatef(x, y, z)
                        
                        color = (activation * color_intensity, 0, (1 - activation) * color_intensity)
                        glColor3f(*color)
                        
                        quad = gluNewQuadric()
                        gluSphere(quad, sphere_size, 8, 8)
                        gluDeleteQuadric(quad)
                        
                        glPopMatrix()

        self.render_gui()
        pg.display.flip()

    def render_no_data_message(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        text_surface = self.font.render("No 3D layers found in the data.", True, (255, 255, 255))
        text_data = pg.image.tostring(text_surface, "RGBA", True)
        
        glRasterPos2f(-0.9, 0)
        glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def render_gui(self):
        # Switch to 2D rendering for GUI
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        # Render GUI
        self.gui_surface.fill((255, 255, 255, 100))
        for slider in self.sliders:
            slider.draw(self.gui_surface)
        
        # Convert Pygame surface to OpenGL texture
        texture_data = pg.image.tostring(self.gui_surface, "RGBA", 1)
        width = self.gui_surface.get_width()
        height = self.gui_surface.get_height()

        # Generate and bind texture
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

        # Draw textured quad
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(width, 0)
        glTexCoord2f(1, 1); glVertex2f(width, height)
        glTexCoord2f(0, 1); glVertex2f(0, height)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)

        # Restore 3D rendering state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

        # Delete the texture
        glDeleteTextures(1, [texture])

    def handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
            elif event.type == pg.VIDEORESIZE:
                self.width, self.height = event.size
                glViewport(0, 0, self.width, self.height)
                self.setup_perspective()
                self.gui_surface = pg.Surface((self.width, self.height), pg.SRCALPHA)
                for i, slider in enumerate(self.sliders):
                    slider.rect.y = self.height - 40 - i * 40
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    self.camera.pos += self.camera.front * 0.1
                elif event.button == 5:  # Scroll down
                    self.camera.pos -= self.camera.front * 0.1
            for slider in self.sliders:
                slider.handle_event(event)

        keys = pg.key.get_pressed()
        if keys[pg.K_w]:
            self.camera.pos += self.camera.front * self.camera.move_speed
        if keys[pg.K_s]:
            self.camera.pos -= self.camera.front * self.camera.move_speed
        if keys[pg.K_a]:
            self.camera.pos -= self.camera.right * self.camera.move_speed
        if keys[pg.K_d]:
            self.camera.pos += self.camera.right * self.camera.move_speed
        if keys[pg.K_SPACE]:
            self.camera.pos += self.camera.up * self.camera.move_speed
        if keys[pg.K_LSHIFT]:
            self.camera.pos -= self.camera.up * self.camera.move_speed

        mouse_buttons = pg.mouse.get_pressed()
        if mouse_buttons[0]:  # Left mouse button
            mouse_movement = pg.mouse.get_rel()
            self.camera.yaw += mouse_movement[0] * self.camera.mouse_sensitivity
            self.camera.pitch -= mouse_movement[1] * self.camera.mouse_sensitivity
            self.camera.pitch = max(-89.0, min(89.0, self.camera.pitch))
            self.camera.update_camera_vectors()

        return True

def create_network_data(details):
    all_activations = []
    for layer_name, layer_data in details:
        if isinstance(layer_data, dict):
            activations = layer_data["Hidden State"].cpu().detach().numpy()
        else:
            activations = layer_data.cpu().detach().numpy()
        # Preserve the sequence length dimension
        all_activations.append(np.squeeze(activations))
    
    # Normalize activations across all layers
    min_val = min(np.min(layer) for layer in all_activations)
    max_val = max(np.max(layer) for layer in all_activations)
    all_activations = [(layer - min_val) / (max_val - min_val) for layer in all_activations]
    
    return all_activations

def visualize_network(details):
    width, height = 1024, 768
    renderer = Renderer(width, height)
    layer_data = create_network_data(details)

    running = True
    while running:
        running = renderer.handle_events()
        renderer.render(layer_data)
        renderer.clock.tick(60)

    pg.quit()

def run_visualization(details):
    multiprocessing.set_start_method('spawn', force=True)
    process = multiprocessing.Process(target=visualize_network, args=(details,))
    process.start()
    process.join()

def main():
    st.title("3D LLM Architecture Visualization with Camera Controls")

    st.sidebar.header("Model Selection and Settings")

    try:
        model_df = get_model_list()
        if model_df.empty:
            st.sidebar.error("No suitable models found. Please try again later.")
            return

        selected_model = st.sidebar.selectbox("Choose a model:", model_df['name'])

        temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        prompt = st.sidebar.text_area("Enter your prompt:", value="Hello, how are you?")

        if st.sidebar.button("Run Query and Visualize"):
            with st.spinner("Loading model and processing query..."):
                try:
                    model, tokenizer, device = load_model_and_tokenizer(selected_model)
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

                    details = get_layer_details(outputs, tokenizer, inputs["input_ids"])

                    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
                    generated_text = generator(prompt, max_length=50, num_return_sequences=1, temperature=temperature)[0]['generated_text']

                    st.subheader("Generated Text")
                    st.write(generated_text)

                    st.subheader("Neural Network Visualization")
                    st.write("Opening 3D visualization in a new window...")
                    st.write("Use WASD keys to move, mouse to look around, and scroll wheel to zoom.")
                    st.write("Adjust visualization options using the sliders in the visualization window.")
                    run_visualization(details)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

        if selected_model in model_df['name'].values:
            model_info = model_df[model_df['name'] == selected_model].iloc[0]
            st.sidebar.subheader("Model Information")
            st.sidebar.write(f"Downloads: {model_info['downloads']}")
            st.sidebar.write(f"Likes: {model_info['likes']}")
            st.sidebar.write(f"Size: {model_info['size']} MB")

    except Exception as e:
        st.error(f"An error occurred while setting up the application: {str(e)}")

if __name__ == "__main__":
    main()