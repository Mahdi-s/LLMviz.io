// script.js

// Use module imports with correct paths and CORS support
import * as THREE from 'three';
import { InstancedMesh } from 'three';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'https://cdn.jsdelivr.net/npm/dat.gui@0.7.9/build/dat.gui.module.js';
import { CSS2DRenderer, CSS2DObject } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/renderers/CSS2DRenderer.js';

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
  activationThreshold: 0.5,
};

let labelRenderer;

function initScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  camera = new THREE.PerspectiveCamera(
    75,
    (window.innerWidth - 300) / window.innerHeight,
    0.1,
    1000
  );
  camera.position.set(0, 20, 50);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth - 300, window.innerHeight);
  document.getElementById('visualization-panel').appendChild(renderer.domElement);

  labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(window.innerWidth - 300, window.innerHeight);
  labelRenderer.domElement.style.position = 'absolute';
  labelRenderer.domElement.style.top = '0px';
  labelRenderer.domElement.style.pointerEvents = 'none';
  document.getElementById('visualization-panel').appendChild(labelRenderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableKeys = true;
  controls.keys = {
    LEFT: 37,
    UP: 38,
    RIGHT: 39,
    BOTTOM: 40,
  };

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambientLight);

  const pointLight = new THREE.PointLight(0xffffff, 0.7);
  camera.add(pointLight);
  scene.add(camera);

  window.addEventListener('resize', onWindowResize, false);

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
  gui.add(visualizationParams, 'activationThreshold', 0, 1).name('Activation Threshold').onChange(applyActivationFilter);
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

let layerLabels = [];

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

function renderActivationBars() {
  // Remove previous meshes and labels
  barMeshes.forEach((mesh) => scene.remove(mesh));
  barMeshes = [];

  layerLabels.forEach((label) => scene.remove(label));
  layerLabels = [];

  const layerNames = Object.keys(activationData);
  const gridSpacing = visualizationParams.gridSpacing;
  const barHeightScale = visualizationParams.barHeightScale;

  let layerIndex = 0;

  layerNames.forEach((layerName) => {
    const activations = activationData[layerName];
    const flatActivations = flattenActivations(activations);

    const numBars = flatActivations.length;
    const gridSize = Math.ceil(Math.sqrt(numBars));
    const bars = [];

    for (let i = 0; i < numBars; i++) {
      const value = flatActivations[i];
      const activationValue = Math.abs(value) * barHeightScale || 0.1; // Avoid zero height

      const geometry = new THREE.BoxGeometry(
        gridSpacing,
        activationValue,
        gridSpacing
      );

      const material = new THREE.ShaderMaterial({
        uniforms: {
          time: { value: 0 },
          color: { value: new THREE.Color(getLayerColor(layerIndex)) },
          intensity: { value: visualizationParams.shaderEffectIntensity },
        },
        vertexShader: vertexShaderSource,
        fragmentShader: fragmentShaderSource,
      });

      const bar = new THREE.Mesh(geometry, material);

      const row = Math.floor(i / gridSize);
      const col = i % gridSize;
      bar.position.set(
        col * gridSpacing - (gridSize * gridSpacing) / 2,
        activationValue / 2,
        layerIndex * 5
      );

      // Apply activation threshold
      bar.visible = Math.abs(value) >= visualizationParams.activationThreshold;

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
    label.position.set(0, layerIndex * 5 + 2.5, 0);
    scene.add(label);
    layerLabels.push(label);

    layerIndex++;
  });
}

function getLayerColor(index) {
  const colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', '#ffffff'];
  return colors[index % colors.length];
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
  labelRenderer.setSize(window.innerWidth - 300, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();

  const time = performance.now() * 0.001;

  barMeshes.forEach((mesh) => {
    if (mesh.material.uniforms && mesh.material.uniforms.time) {
      mesh.material.uniforms.time.value = time;
    }
  });

  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}

document.addEventListener('DOMContentLoaded', () => {
  initScene();
  initGUI();
  document.getElementById('prompt-form').addEventListener('submit', onFormSubmit);
});

function applyActivationFilter() {
  barMeshes.forEach((mesh) => {
    // The activation value is embedded in the scale of the bar
    const activationValue = mesh.scale.y / visualizationParams.barHeightScale;
    mesh.visible = Math.abs(activationValue) >= visualizationParams.activationThreshold;
  });
}
