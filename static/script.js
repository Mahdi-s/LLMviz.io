import * as THREE from 'three';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'https://cdn.jsdelivr.net/npm/dat.gui@0.7.9/build/dat.gui.module.js';
import { CSS2DRenderer, CSS2DObject } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/renderers/CSS2DRenderer.js';



// import { OrbitControls } from './OrbitControls.js';
// import { GUI } from './dat.gui.module.js';

let scene, camera, renderer, controls;
let activationData = {};
let barMeshes = [];
let gui;
let visualizationParams = {
  barColor: '#00ff00',
  barHeightScale: 1,
  gridSpacing: 1,
  showLayers: true,
  shaderEffectIntensity: 1,
};

document.addEventListener('DOMContentLoaded', () => {
  initScene();
  initGUI();
  document.getElementById('prompt-form').addEventListener('submit', onFormSubmit);
});

let labelRenderer; // Declare this at the top with other variables


function initScene() {
  // Scene setup
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  // Camera setup
  camera = new THREE.PerspectiveCamera(
    75,
    (window.innerWidth - 300) / window.innerHeight,
    0.1,
    1000
  );
  camera.position.set(0, 20, 50);

  // Renderer setup
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth - 300, window.innerHeight);
  document.getElementById('visualization-panel').appendChild(renderer.domElement);

    // Initialize CSS2DRenderer
    labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(window.innerWidth - 300, window.innerHeight);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0px';
    labelRenderer.domElement.style.pointerEvents = 'none';
    document.getElementById('visualization-panel').appendChild(labelRenderer.domElement);
  

  // Controls setup
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableKeys = true;
  controls.keys = {
    LEFT: 37, // left arrow
    UP: 38, // up arrow
    RIGHT: 39, // right arrow
    BOTTOM: 40, // down arrow
  };

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambientLight);

  const pointLight = new THREE.PointLight(0xffffff, 0.7);
  camera.add(pointLight);
  scene.add(camera);

  // Handle window resize
  window.addEventListener('resize', onWindowResize, false);

  // Start animation loop
  animate();
}

function initGUI() {
  gui = new GUI({ autoPlace: false });
  const guiContainer = document.createElement('div');
  guiContainer.classList.add('gui-container');
  document.body.appendChild(guiContainer);
  guiContainer.appendChild(gui.domElement);

  gui.addColor(visualizationParams, 'barColor').onChange(updateBarColors);
  gui.add(visualizationParams, 'barHeightScale', 0.1, 5).onChange(updateBarHeights);
  gui.add(visualizationParams, 'gridSpacing', 0.5, 5).onChange(renderActivationBars);
  gui.add(visualizationParams, 'showLayers').onChange(toggleLayersVisibility);
  gui.add(visualizationParams, 'shaderEffectIntensity', 0, 2).onChange(updateShaderEffects);
}

function onFormSubmit(event) {
  event.preventDefault();
  const prompt = document.getElementById('prompt').value;
  fetchActivations(prompt);
}

function fetchActivations(prompt) {
  fetch('/compute', {
    method: 'POST',
    body: new URLSearchParams({ prompt }),
  })
    .then((response) => response.json())
    .then((data) => {
      activationData = data.activations;
      displayPredictedToken(data.predicted_token);
      renderActivationBars();
    })
    .catch((error) => {
      console.error('Error:', error);
    });
}

function displayPredictedToken(token) {
  const predictedTokenDiv = document.getElementById('predicted-token');
  predictedTokenDiv.textContent = `Predicted Token: ${token}`;
}

let layerLabels = []; // Add this at the top with other variables


function renderActivationBars() {
  // Clear existing bars
  barMeshes.forEach((mesh) => scene.remove(mesh));
  barMeshes = [];

  // Clear existing labels
  layerLabels.forEach((label) => scene.remove(label));
  layerLabels = [];

  const layerNames = Object.keys(activationData);
  const gridSpacing = visualizationParams.gridSpacing;
  const barHeightScale = visualizationParams.barHeightScale;

  let layerIndex = 0;

  layerNames.forEach((layerName) => {
    const activations = activationData[layerName];
    // Flatten activations if necessary
    const flatActivations = flattenActivations(activations);

    const numBars = flatActivations.length;
    const gridSize = Math.ceil(Math.sqrt(numBars));
    const bars = [];

    for (let i = 0; i < numBars; i++) {
      const value = flatActivations[i];
      const geometry = new THREE.BoxGeometry(
        gridSpacing,
        Math.abs(value) * barHeightScale,
        gridSpacing
      );

      // Shader material
      const material = new THREE.ShaderMaterial({
        uniforms: {
          time: { value: 0 },
          color: { value: new THREE.Color(visualizationParams.barColor) },
          intensity: { value: visualizationParams.shaderEffectIntensity },
        },
        vertexShader: vertexShaderSource,
        fragmentShader: fragmentShaderSource,
      });

      const bar = new THREE.Mesh(geometry, material);

      // Position the bars in a grid
      const row = Math.floor(i / gridSize);
      const col = i % gridSize;
      bar.position.set(
        col * gridSpacing - (gridSize * gridSpacing) / 2,
        (Math.abs(value) * barHeightScale) / 2,
        row * gridSpacing - (gridSize * gridSpacing) / 2
      );

      // Position the entire layer
      bar.position.y += layerIndex * 10; // Adjust layer spacing

      bars.push(bar);
      scene.add(bar);
    }

    barMeshes.push(...bars);

    // Add label for the layer
    const labelDiv = document.createElement('div');
    labelDiv.className = 'layer-label';
    labelDiv.textContent = layerName;
    labelDiv.style.marginTop = '-1em';
    labelDiv.style.color = 'white';
    labelDiv.style.fontSize = '14px';
    labelDiv.style.textAlign = 'center';

    const label = new CSS2DObject(labelDiv);
    label.position.set(0, layerIndex * 10 + 5, 0); // Adjust position as needed
    scene.add(label);
    layerLabels.push(label);

    layerIndex++;
  });
}


function flattenActivations(activations) {
  if (Array.isArray(activations)) {
    return activations.flat(Infinity);
  }
  return [];
}

function updateBarColors() {
  barMeshes.forEach((mesh) => {
    if (mesh.material.uniforms) {
      mesh.material.uniforms.color.value.set(visualizationParams.barColor);
    } else {
      mesh.material.color.set(visualizationParams.barColor);
    }
  });
}

function updateBarHeights() {
  renderActivationBars();
}

function toggleLayersVisibility() {
  barMeshes.forEach((mesh) => {
    mesh.visible = visualizationParams.showLayers;
  });
}

function updateShaderEffects() {
  barMeshes.forEach((mesh) => {
    if (mesh.material.uniforms && mesh.material.uniforms.intensity) {
      mesh.material.uniforms.intensity.value = visualizationParams.shaderEffectIntensity;
    }
  });
}

function onWindowResize() {
  camera.aspect = (window.innerWidth - 300) / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth - 300, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();

  // Update shader uniforms
  barMeshes.forEach((mesh) => {
    if (mesh.material.uniforms && mesh.material.uniforms.time) {
      mesh.material.uniforms.time.value = performance.now() / 1000;
    }
  });

  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}

// Shader sources
const vertexShaderSource = `
  varying vec3 vPosition;

  void main() {
    vPosition = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const fragmentShaderSource = `
  uniform float time;
  uniform vec3 color;
  uniform float intensity;
  varying vec3 vPosition;

  void main() {
    float pulsate = sin(time + vPosition.y * 0.5) * 0.5 + 0.5;
    gl_FragColor = vec4(color * pulsate * intensity, 1.0);
  }
`;
