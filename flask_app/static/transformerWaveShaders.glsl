// Vertex Shader
attribute float value;
varying vec3 vColor;
varying float vValue;

void main() {
    vValue = value;
    vColor = color;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    gl_PointSize = 5.0;
}

// Fragment Shader
precision mediump float;
varying vec3 vColor;
varying float vValue;

void main() {
    vec3 color = mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), vValue);
    gl_FragColor = vec4(color, 1.0);
}