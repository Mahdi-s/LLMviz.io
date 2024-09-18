// script.js

import * as THREE from 'https://unpkg.com/three@0.152.2/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.152.2/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'https://cdn.jsdelivr.net/npm/dat.gui@0.7.9/build/dat.gui.module.js';
import { CSS2DRenderer, CSS2DObject } from 'https://unpkg.com/three@0.152.2/examples/jsm/renderers/CSS2DRenderer.js';

// Define constants
const VOCAB_SIZE = 50257;
const EMBEDDING_DIM = 768;
const SEQUENCE_LENGTH = 1024;
const NUM_LAYERS = 6;
const NUM_HEADS = 12;
const HIDDEN_SIZE = 768;
const OUTPUT_VOCAB_SIZE = 50257;

// Grid dimensions for mapping tokens
const GRID_SIZE_X = Math.ceil(Math.sqrt(SEQUENCE_LENGTH));
const GRID_SIZE_Y = Math.ceil(SEQUENCE_LENGTH / GRID_SIZE_X);

// Initialize variables
let scene, camera, renderer, controls;
let activationData = {};
let gui;
let visualizationParams = {
    lowColor: '#0000ff', // Blue
    highColor: '#ff0000', // Red
    minSphereSize: 0.5,
    maxSphereSize: 3.0,
    showLayers: true,
};
let labelRenderer;
let transformerLayers = [];
let clock;

// Initialize raycaster for interactivity
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let INTERSECTED;

// Custom Shader Material for InstancedMesh
const vertexShader = `
    attribute float activation;
    uniform float uMinSphereSize;
    uniform float uMaxSphereSize;
    varying float vActivation;
    void main() {
        vActivation = activation;
        // Compute scale based on activation
        float scale = mix(uMinSphereSize, uMaxSphereSize, activation);
        // Apply scaling to the vertex position
        vec3 scaledPosition = position * scale;
        // Transform the scaled position using the instance matrix
        gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(scaledPosition, 1.0);
    }
`;

const fragmentShader = `
    uniform vec3 uLowColor;
    uniform vec3 uHighColor;
    varying float vActivation;
    void main() {
        // Interpolate color based on activation
        vec3 color = mix(uLowColor, uHighColor, vActivation);
        gl_FragColor = vec4(color, 1.0);
    }
`;

// Initialize the Scene
function initScene() {
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x121212); // Dark background for better contrast

    // Camera setup with adjusted aspect ratio
    camera = new THREE.PerspectiveCamera(
        75,
        (window.innerWidth - 300) / window.innerHeight,
        0.1,
        20000 // Adjusted far plane to accommodate large Z-axis
    );
    camera.position.set(0, 500, 2000); // Elevated and pulled back to view the entire structure

    // Renderer setup
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth - 300, window.innerHeight);
    document.getElementById('visualization-panel').appendChild(renderer.domElement);

    // Initialize CSS2DRenderer for labels if needed
    labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(window.innerWidth - 300, window.innerHeight);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0px';
    labelRenderer.domElement.style.pointerEvents = 'none';
    document.getElementById('visualization-panel').appendChild(labelRenderer.domElement);

    // Controls setup
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; // For smoother controls
    controls.dampingFactor = 0.05;
    controls.enablePan = true;
    controls.enableZoom = true;
    controls.enableRotate = true;
    controls.maxPolarAngle = Math.PI; // Allow free vertical movement
    controls.minDistance = 100; // Set minimum zoom distance
    controls.maxDistance = 5000; // Set maximum zoom distance

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(0, 1, 1).normalize();
    scene.add(directionalLight);

    // Add helper objects for debugging and orientation
    addHelpers();

    // Initialize Visualization Layers
    initInputEmbeddingLayer();
    initTransformerLayers();
    initOutputLayer();

    // Handle window resize
    window.addEventListener('resize', onWindowResize, false);

    // Initialize GUI
    initGUI();

    // Event Listener for Mouse Click
    window.addEventListener('click', onClick, false);

    // Initialize Clock
    clock = new THREE.Clock();

    // Start animation loop
    animate();
}

/**
 * Add helper objects like AxesHelper and GridHelper to the scene
 * This helps in understanding the orientation and scale of the scene
 */
function addHelpers() {
    // Axes Helper: X - Red, Y - Green, Z - Blue
    const axesHelper = new THREE.AxesHelper(1000);
    scene.add(axesHelper);

    // Grid Helper: size, divisions
    const gridHelper = new THREE.GridHelper(2000, 40);
    gridHelper.position.y = -500; // Position below the main visualization
    scene.add(gridHelper);
}

/**
 * Initialize the Input Embedding Layer with Instanced Spheres
 */
function initInputEmbeddingLayer() {
    // Create a group to hold the input layer
    const inputGroup = new THREE.Group();
    inputGroup.name = 'InputEmbeddingLayer';
    scene.add(inputGroup);

    // Define sphere geometry
    const sphereGeometry = new THREE.SphereGeometry(1, 16, 16);

    // Define shader material
    const sphereMaterial = new THREE.ShaderMaterial({
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        uniforms: {
            uLowColor: { value: new THREE.Color(visualizationParams.lowColor) },
            uHighColor: { value: new THREE.Color(visualizationParams.highColor) },
            uMinSphereSize: { value: visualizationParams.minSphereSize },
            uMaxSphereSize: { value: visualizationParams.maxSphereSize },
        },
        transparent: true,
    });

    // Number of instances
    const numInstances = SEQUENCE_LENGTH * EMBEDDING_DIM; // 1024 * 768 = 786,432

    // To manage performance, visualize a subset (e.g., first 100 tokens and 100 dimensions)
    const subsetTokens = 100;
    const subsetDimensions = 100; // First 100 dimensions
    const actualInstances = subsetTokens * subsetDimensions; // 10,000 spheres

    const instancedMesh = new THREE.InstancedMesh(sphereGeometry, sphereMaterial, actualInstances);
    instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage); // Will be updated frequently

    // Attributes for activation and position
    const activationArray = new Float32Array(actualInstances);
    const dummy = new THREE.Object3D();

    let index = 0;
    for (let i = 0; i < subsetTokens; i++) {
        for (let j = 0; j < subsetDimensions; j++) {
            // Position in grid
            const x = (i - subsetTokens / 2) * 5;
            const y = (j - subsetDimensions / 2) * 5;
            const z = 0; // Input layer at Z=0

            dummy.position.set(x, y, z);
            dummy.updateMatrix();
            instancedMesh.setMatrixAt(index, dummy.matrix);

            // Initialize activation to 0
            activationArray[index] = 0.0;

            index++;
        }
    }

    // Set activation attribute
    instancedMesh.geometry.setAttribute('activation', new THREE.InstancedBufferAttribute(activationArray, 1));

    inputGroup.add(instancedMesh);

    // Add label for Input Layer
    const inputLabel = createLabel('Input Embedding Layer', 0, 0, -200);
    scene.add(inputLabel);
}

/**
 * Initialize Transformer Layers with Instanced Spheres
 */
function initTransformerLayers() {
    for (let layer = 0; layer < NUM_LAYERS; layer++) {
        const layerGroup = new THREE.Group();
        layerGroup.name = `TransformerLayer_${layer}`;
        scene.add(layerGroup);

        // Define sphere geometry
        const sphereGeometry = new THREE.SphereGeometry(1, 16, 16);

        // Define shader material
        const sphereMaterial = new THREE.ShaderMaterial({
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            uniforms: {
                uLowColor: { value: new THREE.Color(visualizationParams.lowColor) },
                uHighColor: { value: new THREE.Color(visualizationParams.highColor) },
                uMinSphereSize: { value: visualizationParams.minSphereSize },
                uMaxSphereSize: { value: visualizationParams.maxSphereSize },
            },
            transparent: true,
        });

        // Number of instances per layer
        const numInstances = SEQUENCE_LENGTH * NUM_HEADS; // 1024 * 12 = 12,288

        // To manage performance, visualize a subset (e.g., first 100 tokens)
        const subsetTokens = 100;
        const subsetHeads = NUM_HEADS;
        const actualInstances = subsetTokens * subsetHeads; // 1,200 spheres

        const instancedMesh = new THREE.InstancedMesh(sphereGeometry, sphereMaterial, actualInstances);
        instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage); // Will be updated frequently

        // Attributes for activation and position
        const activationArray = new Float32Array(actualInstances);
        const dummyObj = new THREE.Object3D();

        let index = 0;
        for (let i = 0; i < subsetTokens; i++) {
            for (let j = 0; j < subsetHeads; j++) {
                // Position in grid
                const x = (i - subsetTokens / 2) * 5;
                const y = (j - subsetHeads / 2) * 5;
                const z = (layer + 1) * 500; // Layers spaced along Z-axis

                dummyObj.position.set(x, y, z);
                dummyObj.updateMatrix();
                instancedMesh.setMatrixAt(index, dummyObj.matrix);

                // Initialize activation to 0
                activationArray[index] = 0.0;

                index++;
            }
        }

        // Set activation attribute
        instancedMesh.geometry.setAttribute('activation', new THREE.InstancedBufferAttribute(activationArray, 1));

        layerGroup.add(instancedMesh);

        // Add label for Transformer Layer
        const label = createLabel(`Transformer Layer ${layer + 1}`, 0, 0, (layer + 1) * 500 + 100);
        scene.add(label);

        transformerLayers.push(layerGroup);
    }
}

/**
 * Initialize the Output Layer with Instanced Spheres
 */
function initOutputLayer() {
    // Create a group to hold the output layer
    const outputGroup = new THREE.Group();
    outputGroup.name = 'OutputLayer';
    scene.add(outputGroup);

    // Define sphere geometry
    const sphereGeometry = new THREE.SphereGeometry(1, 16, 16);

    // Define shader material
    const sphereMaterial = new THREE.ShaderMaterial({
        vertexShader: vertexShader,
        fragmentShader: fragmentShader,
        uniforms: {
            uLowColor: { value: new THREE.Color(visualizationParams.lowColor) },
            uHighColor: { value: new THREE.Color(visualizationParams.highColor) },
            uMinSphereSize: { value: visualizationParams.minSphereSize },
            uMaxSphereSize: { value: visualizationParams.maxSphereSize },
        },
        transparent: true,
    });

    // Number of instances
    const numInstances = OUTPUT_VOCAB_SIZE; // 50,257

    // To manage performance, visualize a subset (e.g., top 1000 tokens)
    const subsetSize = 1000;
    const gridCols = 50;
    const gridRows = Math.ceil(subsetSize / gridCols);
    const actualInstances = subsetSize;

    const instancedMesh = new THREE.InstancedMesh(sphereGeometry, sphereMaterial, actualInstances);
    instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage); // Will be updated frequently

    // Attributes for activation and position
    const activationArray = new Float32Array(actualInstances);
    const dummy = new THREE.Object3D();

    for (let i = 0; i < actualInstances; i++) {
        const row = Math.floor(i / gridCols);
        const col = i % gridCols;
        const x = (col - gridCols / 2) * 5;
        const y = (row - gridRows / 2) * 5;
        const z = (NUM_LAYERS + 1) * 500; // Output layer spaced along Z-axis

        dummy.position.set(x, y, z);
        dummy.updateMatrix();
        instancedMesh.setMatrixAt(i, dummy.matrix);

        // Initialize activation to 0
        activationArray[i] = 0.0;
    }

    // Set activation attribute
    instancedMesh.geometry.setAttribute('activation', new THREE.InstancedBufferAttribute(activationArray, 1));

    outputGroup.add(instancedMesh);

    // Add label for Output Layer
    const outputLabel = createLabel('Output Layer', 0, 0, (NUM_LAYERS + 1) * 500 + 100);
    scene.add(outputLabel);
}

/**
 * Create a 2D Label for Layers
 */
function createLabel(text, x, y, z) {
    const labelDiv = document.createElement('div');
    labelDiv.className = 'layer-label';
    labelDiv.textContent = text;
    labelDiv.style.marginTop = '-1em';
    const label = new CSS2DObject(labelDiv);
    label.position.set(x, y, z);
    return label;
}

/**
 * Initialize GUI Controls
 */
function initGUI() {
    gui = new GUI({ autoPlace: false });
    const guiContainer = document.createElement('div');
    guiContainer.classList.add('gui-container');
    document.body.appendChild(guiContainer);
    guiContainer.appendChild(gui.domElement);

    gui.addColor(visualizationParams, 'lowColor').name('Low Activation Color').onChange(updateShaders);
    gui.addColor(visualizationParams, 'highColor').name('High Activation Color').onChange(updateShaders);
    gui.add(visualizationParams, 'minSphereSize', 0.1, 2.0).name('Min Sphere Size').onChange(updateShaders);
    gui.add(visualizationParams, 'maxSphereSize', 1.0, 10.0).name('Max Sphere Size').onChange(updateShaders);
    gui.add(visualizationParams, 'showLayers').name('Show Transformer Layers').onChange(toggleLayersVisibility);
}

/**
 * Update Shaders based on GUI Controls
 */
function updateShaders() {
    // Update uniform values for all shader materials
    scene.traverse((object) => {
        if (object.isGroup) return;
        if (object.isInstancedMesh && object.material.isShaderMaterial) {
            object.material.uniforms.uLowColor.value.set(visualizationParams.lowColor);
            object.material.uniforms.uHighColor.value.set(visualizationParams.highColor);
            object.material.uniforms.uMinSphereSize.value = visualizationParams.minSphereSize;
            object.material.uniforms.uMaxSphereSize.value = visualizationParams.maxSphereSize;
            object.material.needsUpdate = true;
        }
    });
}

/**
 * Toggle Transformer Layers Visibility
 */
function toggleLayersVisibility() {
    transformerLayers.forEach((layerGroup) => {
        layerGroup.visible = visualizationParams.showLayers;
    });
}

/**
 * Handle Window Resize
 */
function onWindowResize() {
    const width = window.innerWidth - 300;
    const height = window.innerHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();

    renderer.setSize(width, height);
    labelRenderer.setSize(width, height);
}

/**
 * Handle Mouse Clicks for Interactivity
 */
function onClick(event) {
    // Calculate mouse position in normalized device coordinates
    mouse.x = (event.clientX / (window.innerWidth - 300)) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    // Update the picking ray with the camera and mouse position
    raycaster.setFromCamera(mouse, camera);

    // Calculate objects intersecting the picking ray
    const intersects = raycaster.intersectObjects(scene.children, true);

    if (intersects.length > 0) {
        const intersected = intersects[0].object;
        // Handle interaction based on the object type
        if (intersected instanceof THREE.InstancedMesh) {
            const mesh = intersected;
            const instanceId = intersects[0].instanceId;
            if (instanceId !== undefined) {
                // Retrieve layer information
                const parentGroup = mesh.parent;
                const layerName = parentGroup.name;

                // Example: Display activation value
                const activation = mesh.geometry.attributes.activation.array[instanceId];
                alert(`Layer: ${layerName}\nActivation Value: ${activation.toFixed(3)}`);
            }
        }
    }
}

/**
 * Handle Form Submission
 */
function onFormSubmit(event) {
    event.preventDefault();
    const prompt = document.getElementById('prompt').value.trim();
    if (!prompt) {
        alert('Please enter a prompt.');
        return;
    }
    fetchActivations(prompt);
}

/**
 * Fetch Activation Data from Backend
 */
function fetchActivations(prompt) {
    fetch('/compute', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({ prompt }),
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.error) {
                alert(`Error: ${data.error}`);
                return;
            }
            activationData = data.activations;
            displayPredictedToken(data.predicted_token);
            renderActivationVisualization();
        })
        .catch((error) => {
            console.error('Error:', error);
            alert('An error occurred while processing your request.');
        });
}

/**
 * Display Predicted Token
 */
function displayPredictedToken(token) {
    const predictedTokenDiv = document.getElementById('predicted-token');
    predictedTokenDiv.textContent = `Predicted Token: ${token}`;
}

/**
 * Render the Complete Activation Visualization
 */
function renderActivationVisualization() {
    // Update Input Embedding Layer
    if (activationData['input_embedding']) {
        updateInputEmbeddingLayer(activationData['input_embedding']);
    }

    // Update Transformer Layers
    updateTransformerLayers(activationData);

    // Update Output Layer
    if (activationData['output_layer']) {
        updateOutputLayer(activationData['output_layer']);
    }
}

/**
 * Update Input Embedding Layer with Activation Data
 * @param {Array} embeddings - Array of activation values for input embeddings
 */
function updateInputEmbeddingLayer(embeddings) {
    const inputGroup = scene.getObjectByName('InputEmbeddingLayer');
    if (!inputGroup) return;
    const mesh = inputGroup.children[0];
    if (!mesh) return;

    const activationAttribute = mesh.geometry.attributes.activation;
    const subsetTokens = 100;
    const subsetDimensions = 100;
    const actualInstances = subsetTokens * subsetDimensions;

    for (let i = 0; i < actualInstances; i++) {
        // Example: Normalize activation (assumes embeddings array is flat)
        const activation = embeddings[i] || 0.0;
        const normalized = normalizeActivation(activation);
        activationAttribute.array[i] = normalized;
    }
    activationAttribute.needsUpdate = true;
}

/**
 * Update Transformer Layers with Activation Data
 * @param {Object} activations - Activation data per layer
 */
function updateTransformerLayers(activations) {
    transformerLayers.forEach((layerGroup, layerIndex) => {
        const layerName = `TransformerLayer_${layerIndex}`;
        const layerActivation = activations[layerName] || {};

        const mesh = layerGroup.children[0];
        if (!mesh) return;

        const activationAttribute = mesh.geometry.attributes.activation;
        const subsetTokens = 100;
        const subsetHeads = NUM_HEADS;
        const actualInstances = subsetTokens * subsetHeads;

        for (let i = 0; i < actualInstances; i++) {
            // Example: Retrieve activation based on token and head
            // This depends on how the backend structures the activation data
            // Assuming a flat array for simplicity
            const activation = layerActivation[i] || 0.0;
            const normalized = normalizeActivation(activation);
            activationAttribute.array[i] = normalized;
        }
        activationAttribute.needsUpdate = true;
    });
}

/**
 * Update Output Layer with Activation Data
 * @param {Array} outputActivations - Array of activation values for output layer
 */
function updateOutputLayer(outputActivations) {
    const outputGroup = scene.getObjectByName('OutputLayer');
    if (!outputGroup) return;
    const mesh = outputGroup.children[0];
    if (!mesh) return;

    const activationAttribute = mesh.geometry.attributes.activation;
    const subsetSize = 1000;
    const actualInstances = subsetSize;

    for (let i = 0; i < actualInstances; i++) {
        const activation = outputActivations[i] || 0.0;
        const normalized = normalizeActivation(activation);
        activationAttribute.array[i] = normalized;
    }
    activationAttribute.needsUpdate = true;
}

/**
 * Normalize Activation Values to [0, 1]
 * @param {number} activation - Raw activation value
 * @returns {number} - Normalized activation value
 */
function normalizeActivation(activation) {
    // Implement normalization logic based on expected activation ranges
    // Here, using a sigmoid-like normalization for demonstration
    return THREE.MathUtils.clamp((activation + 1) / 2, 0, 1);
}

/**
 * Animation Loop
 */
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
}

// Initialize the application once the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initScene();
    document.getElementById('prompt-form').addEventListener('submit', onFormSubmit);
});
