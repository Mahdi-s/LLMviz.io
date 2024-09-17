// fragmentShader.glsl
uniform float time;
uniform vec3 color;
varying vec3 vPosition;

void main() {
  float intensity = sin(time + vPosition.y * 0.1);
  gl_FragColor = vec4(color * intensity, 1.0);
}