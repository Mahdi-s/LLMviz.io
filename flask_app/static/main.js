let scene, camera, renderer, controls;
let visualizations = [];
let shaderMaterial;

function init() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    camera.position.set(5, 10, 20); // Adjusted camera position
    controls.target.set(5, 0, 50); // Adjusted controls target
    controls.update();

    // Add AxesHelper for reference
    const axesHelper = new THREE.AxesHelper(50);
    scene.add(axesHelper);

    window.addEventListener('resize', onWindowResize, false);

    const submitBtn = document.getElementById('submit-btn');
    submitBtn.addEventListener('click', processInput);


    loadShaders();
}

function loadShaders() {
    fetch('/static/transformerWaveShaders.glsl')
        .then(response => response.text())
        .then(shaderText => {
            const shaderParts = shaderText.split('// Fragment Shader');
            const vertexShader = shaderParts[0].trim();
            const fragmentShader = shaderParts[1].trim();

            shaderMaterial = new THREE.ShaderMaterial({
                vertexShader: vertexShader,
                fragmentShader: fragmentShader,
                uniforms: {
                    minValue: { value: -1 },
                    maxValue: { value: 1 }
                },
                vertexColors: true
            });
        })
        .catch(error => console.error('Error loading shaders:', error));
}


function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('open');
}

function processInput() {
    const inputText = document.getElementById('input-text').value;
    fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
    })
    .then(response => response.json())
    .then(data => {
        console.log('Received data:', data); // Add this line
        clearVisualizations();
        createVisualizations(data);
        updatePredictions(data.predictions);
    })
    .catch(error => console.error('Error:', error));
}


function clearVisualizations() {
    for (let viz of visualizations) {
        scene.remove(viz);
    }
    visualizations = [];
}

function createVisualizations(data) {
    let xOffset = 0;
    
    // Step 1: Token embeddings
    createEmbeddingVisualization(data.token_embeddings, xOffset, 0, 0, 'Token Embeddings');
    xOffset += 2;

    // Step 2: Positional embeddings
    createEmbeddingVisualization(data.positional_embeddings, xOffset, 0, 0, 'Positional Embeddings');
    xOffset += 2;

    // Steps 3-7: Repeat for each block
    for (let i = 0; i < data.blocks.length; i++) {
        createBlockVisualization(data.blocks[i], xOffset, 0, 0, i);
        xOffset += 7; // Increase offset for each block
    }

    // Step 8: Model predictions
    createPredictionVisualization(data.predictions, xOffset, 0, 0);
}

function createEmbeddingVisualization(embeddings, x, y, z, label) {
    const group = new THREE.Group();
    group.position.set(x, y, z);

    const xScale = 0.05; // Reduced from 0.15
    const zScale = 0.05; // Reduced from 0.15

    for (let i = 0; i < embeddings.length; i++) {
        for (let j = 0; j < embeddings[i].length; j++) {
            const value = embeddings[i][j];
            if (!isNaN(value)) {
                const barHeight = Math.max(0.01, Math.abs(value));
                const geometry = new THREE.BoxGeometry(0.04, barHeight, 0.04);
                const material = new THREE.MeshBasicMaterial({ color: getColor(value) });
                const bar = new THREE.Mesh(geometry, material);
                bar.position.set(i * xScale, barHeight / 2, j * zScale);
                group.add(bar);
            }
        }
    }

    addLabel(group, label);
    scene.add(group);
    visualizations.push(group);
}


function createBlockVisualization(blockData, x, y, z, blockIndex) {
    const group = new THREE.Group();
    group.position.set(x, y, z);

    // Step 3: hook_attn_scores
    createMatrixVisualization(blockData.attn_scores, 0, 0, 0, `Block ${blockIndex} Attention Scores`);

    // Step 4: hook_attn
    createMatrixVisualization(blockData.attn, 1, 0, 0, `Block ${blockIndex} Attention`);

    // Step 5: attn.hook_z
    createMatrixVisualization(blockData.attn_z, 2, 0, 0, `Block ${blockIndex} Attention Z`);

    // Step 6: hook_attn_out
    createMatrixVisualization(blockData.attn_out, 3, 0, 0, `Block ${blockIndex} Attention Out`);

    // Step 7: hook_mlp_out
    createMatrixVisualization(blockData.mlp_out, 4, 0, 0, `Block ${blockIndex} MLP Out`);

    addLabel(group, `Block ${blockIndex}`);
    scene.add(group);
    visualizations.push(group);
}

function createMatrixVisualization(matrix, x, y, z, label) {
    const group = new THREE.Group();
    group.position.set(x, y, z);

    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[i].length; j++) {
            const value = matrix[i][j];
            if (!isNaN(value)) {
                const barHeight = Math.max(0.01, Math.abs(value));
                const geometry = new THREE.BoxGeometry(0.1, barHeight, 0.1);
                const material = new THREE.MeshBasicMaterial({ color: getColor(value) });
                const bar = new THREE.Mesh(geometry, material);
                bar.position.set(i * 0.15, barHeight / 2, j * 0.15);
                group.add(bar);
            }
        }
    }

    addLabel(group, label);
    scene.add(group);
    visualizations.push(group);
}

function createPredictionVisualization(predictions, x, y, z) {
    const group = new THREE.Group();
    group.position.set(x, y, z);

    for (let i = 0; i < predictions.length; i++) {
        const probability = predictions[i].probability;
        if (!isNaN(probability)) {
            const barHeight = Math.max(0.01, probability);
            const geometry = new THREE.BoxGeometry(0.1, barHeight, 0.1);
            const material = new THREE.MeshBasicMaterial({ color: getColor(barHeight) });
            const bar = new THREE.Mesh(geometry, material);
            bar.position.set(i * 0.15, barHeight / 2, 0);
            group.add(bar);

            // Add hover effect
            bar.userData = { token: predictions[i].token };
        }
    }

    addLabel(group, 'Predictions');
    scene.add(group);
    visualizations.push(group);
}

function getColor(value) {
    const minVal = -10; // Adjust based on your data range
    const maxVal = 10;  // Adjust based on your data range
    const t = (value - minVal) / (maxVal - minVal);
    return new THREE.Color().setHSL(t * 0.7, 1, 0.5);
}

function addLabel(group, text) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    context.font = '24px Arial';
    context.fillStyle = 'white';
    context.fillText(text, 0, 24);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(material);
    sprite.position.set(0, -0.5, 0);
    sprite.scale.set(2, 1, 1);
    group.add(sprite);
}

function updatePredictions(predictions) {
    const predictionsDiv = document.getElementById('predictions');
    predictionsDiv.innerHTML = '<h3>Top Predictions:</h3>';
    
    for (let pred of predictions) {
        predictionsDiv.innerHTML += `<p>${pred.token}: ${(pred.probability * 100).toFixed(2)}%</p>`;
    }
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

init();
animate();