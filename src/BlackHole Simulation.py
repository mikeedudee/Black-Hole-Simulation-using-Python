'''
MIT License

Copyright (c) 2026 Francis Mike John Camogao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import math
import sys

# ==========================================
# SHADER DEFINITIONS
# ==========================================

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
void main() {
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

uniform vec2 u_resolution;
uniform vec3 u_camPos;
uniform vec3 u_camDir;
uniform vec3 u_camUp;
uniform vec3 u_camRight;
uniform float u_time;

#define MAX_STEPS 900 

// Hash function for pseudo-random noise
float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

// Generates the background starfield
vec3 getStarfield(vec3 dir) {
    vec3 color = vec3(0.0);
    vec3 gridPos = dir * 350.0;
    vec3 cellId = floor(gridPos);
    vec3 cellFract = fract(gridPos);
    float n = hash(cellId);
    
    if (n > 0.90) {
        float dist = length(cellFract - vec3(0.5));
        float size = (n - 0.90) * 5.0; 
        float intensity = smoothstep(size, 0.0, dist);
        vec3 starColor = mix(vec3(0.4, 0.6, 1.0), vec3(1.0, 0.7, 0.4), fract(n * 345.678));
        color += starColor * intensity * (n - 0.90) * 100.0;
    }
    
    // Ambient cosmic background / Milky Way band
    float equatorFade = smoothstep(0.5, 0.0, abs(dir.y));
    float smoothNoise = sin(dir.x * 12.0) * cos(dir.y * 6.0) * sin(dir.z * 12.0);
    smoothNoise = smoothstep(-0.2, 1.0, smoothNoise);
    color += vec3(0.015, 0.025, 0.04) * equatorFade * smoothNoise;
    
    return color;
}

// ACES Tone mapping for realistic exposure handling
vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
    
    vec3 rayPos = u_camPos;
    vec3 rayDir = normalize(u_camDir + uv.x * u_camRight + uv.y * u_camUp);
    
    vec3 accumulatedColor = vec3(0.0);
    float transmittance = 1.0;
    
    float GM = 0.5; 
    float eventHorizon = 2.0 * GM; 
    bool escaped = false;
    float r = length(rayPos);
    
    for(int i = 0; i < MAX_STEPS; ++i) {
        r = length(rayPos);
        
        // Singularity absorption limit
        if (r < eventHorizon * 0.98) {
            break; 
        }
        
        // Ray escapes local system
        if (r > 60.0) {
            escaped = true;
            break;
        }
        
        // Adaptive step sizing to preserve accuracy near the photon sphere
        float current_dt = mix(0.003, 0.1, smoothstep(eventHorizon, eventHorizon * 8.0, r));
        
        // ==========================================================
        // SYMPLECTIC GEODESIC INTEGRATION
        // ==========================================================
        vec3 h_vec = cross(rayPos, rayDir);
        float h2 = dot(h_vec, h_vec);
        vec3 acceleration = -rayPos * (1.5 * eventHorizon * h2 / pow(r, 5.0));
        
        // Semi-implicit update conserves orbital energy significantly better
        rayDir = normalize(rayDir + acceleration * current_dt);
        rayPos += rayDir * current_dt;
        
        // ==========================================================
        // VOLUMETRIC ACCRETION DISK INTEGRATION
        // ==========================================================
        float diskHeight = abs(rayPos.y);
        float diskRadius = length(rayPos.xz);
        
        if (diskHeight < 0.8 && diskRadius > eventHorizon * 1.15 && diskRadius < 18.0) {
            
            float verticalDensity = exp(-diskHeight * 10.0);
            float radialDensity = smoothstep(18.0, 7.0, diskRadius) * smoothstep(eventHorizon * 1.15, eventHorizon * 3.0, diskRadius);
            
            float angularVelocity = sqrt(GM / pow(diskRadius, 3.0));
            vec3 orbitalVelocity = normalize(vec3(-rayPos.z, 0.0, rayPos.x)) * (angularVelocity * diskRadius);
            
            float angle = atan(rayPos.z, rayPos.x);
            float flowNoise = sin(angle * 8.0 - diskRadius * 4.0 + u_time * 4.0) * 0.5 + 0.5;
            float density = verticalDensity * radialDensity * (0.2 + 0.8 * flowNoise);
            
            if (density > 0.001) {
                float beta = length(orbitalVelocity); 
                float gamma = 1.0 / sqrt(1.0 - beta * beta);
                float cosTheta = dot(rayDir, normalize(orbitalVelocity));
                float doppler = 1.0 / (gamma * (1.0 - beta * cosTheta));
                
                float temp = pow(eventHorizon * 3.0 / diskRadius, 1.8); 
                vec3 baseColor = mix(vec3(0.8, 0.15, 0.02), vec3(0.5, 0.7, 1.0), temp);
                
                vec3 shiftedColor = baseColor * pow(doppler, 4.0);
                
                float dTau = density * current_dt * 20.0; 
                vec3 radiance = shiftedColor * density * 18.0;
                
                accumulatedColor += radiance * transmittance * dTau;
                transmittance *= exp(-dTau);
                
                if (transmittance < 0.01) break; 
            }
        }
    }
    
    // If the ray did not hit the event horizon, it must sample the cosmic background.
    // This prevents artificial black clipping when MAX_STEPS is exhausted.
    if (escaped || (r >= eventHorizon * 1.05 && transmittance > 0.01)) {
        accumulatedColor += getStarfield(normalize(rayDir)) * transmittance;
    }
    
    // Apply Physical Camera Exposure & ACES Tonemapping
    accumulatedColor *= 1.2; 
    vec3 mapped = ACESFilm(accumulatedColor);
    
    // Gamma correction
    mapped = pow(mapped, vec3(1.0 / 2.2)); 
    
    FragColor = vec4(mapped, 1.0);
}
"""

HUD_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
out vec2 TexCoords;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoords = aPos * 0.5 + 0.5;
}
"""

HUD_FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoords;
out vec4 color;
uniform sampler2D textTexture;
void main() {
    color = texture(textTexture, TexCoords);
}
"""

# ==========================================
# CORE SYSTEM LOGIC
# ==========================================

def render_hud_surface(surface, font, cam_pos, yaw, pitch, current_time):
    surface.fill((0, 0, 0, 0))
    
    event_horizon_radius = 1.0 
    dist_to_singularity = np.linalg.norm(cam_pos)
    distance_to_eh = dist_to_singularity - event_horizon_radius
    
    if dist_to_singularity > event_horizon_radius:
        dilation_factor = math.sqrt(1.0 - (event_horizon_radius / dist_to_singularity))
    else:
        dilation_factor = 0.0 
        
    local_time = current_time * dilation_factor
    
    lines = [
        "VOLUMETRIC TELEMETRY - [Made by Francis Mike John Camogao]",
        "====================================",
        f"COORD TIME (T_f): {current_time:.2f} s",
        f"LOCAL TIME (T_0): {local_time:.2f} s",
        f"TIME DILATION   : {dilation_factor:.4f}x",
        "------------------------------------",
        f"POS X : {cam_pos[0]:.3f}",
        f"POS Y : {cam_pos[1]:.3f}",
        f"POS Z : {cam_pos[2]:.3f}",
        f"PITCH : {pitch:.2f} DEG",
        f"YAW   : {yaw:.2f} DEG",
        f"DIST TO HORIZON : {distance_to_eh:.3f} R_s",
        "====================================",
        "FLIGHT CONTROLS:",
        "MOUSE : PITCH / YAW",
        "WASD  : TRANSLATIONAL THRUST",
        "SPACE : ASCEND (+Y)",
        "LSHIFT: DESCEND (-Y)",
        "ESC   : DISENGAGE MOUSE LOCK",
        "STATUS: FREE FLIGHT"
    ]
    
    y_offset = 20
    for line in lines:
        text_surface = font.render(line, True, (0, 255, 150))
        surface.blit(text_surface, (20, y_offset))
        y_offset += 22

def main():
    pygame.init()
    pygame.font.init()
    width, height = 1280, 720
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)
    pygame.display.set_caption("Black Hole Simulation")
    
    hud_font = pygame.font.SysFont("monospace", 14, bold=True)
    hud_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    
    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )
    
    hud_shader = compileProgram(
        compileShader(HUD_VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(HUD_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )
    
    vertices = np.array([
        -1.0, -1.0,  1.0, -1.0,
        -1.0,  1.0,  1.0,  1.0
    ], dtype=np.float32)
    
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
    
    hud_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, hud_tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    
    cam_pos = np.array([0.0, 3.0, 20.0], dtype=np.float32)
    cam_front = np.array([0.0, -0.1, -1.0], dtype=np.float32)
    cam_up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    yaw = -90.0
    pitch = -5.0
    mouse_sensitivity = 0.1
    flight_speed = 4.0
    
    loc_resolution = glGetUniformLocation(shader, "u_resolution")
    loc_camPos = glGetUniformLocation(shader, "u_camPos")
    loc_camDir = glGetUniformLocation(shader, "u_camDir")
    loc_camUp = glGetUniformLocation(shader, "u_camUp")
    loc_camRight = glGetUniformLocation(shader, "u_camRight")
    loc_time = glGetUniformLocation(shader, "u_time")
    loc_hud_tex = glGetUniformLocation(hud_shader, "textTexture")
    
    clock = pygame.time.Clock()
    running = True
    
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    mouse_locked = True
    
    while running:
        dt = clock.tick(60) / 1000.0 
        current_time = pygame.time.get_ticks() / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                width, height = event.w, event.h
                glViewport(0, 0, width, height)
                hud_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                glBindTexture(GL_TEXTURE_2D, hud_tex)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    mouse_locked = False
                    pygame.mouse.set_visible(True)
                    pygame.event.set_grab(False)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not mouse_locked:
                    mouse_locked = True
                    pygame.mouse.set_visible(False)
                    pygame.event.set_grab(True)
            elif event.type == pygame.MOUSEMOTION:
                if mouse_locked:
                    dx, dy = event.rel
                    yaw += dx * mouse_sensitivity
                    pitch -= dy * mouse_sensitivity
                    pitch = max(-89.0, min(89.0, pitch))

        cam_front[0] = math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))
        cam_front[1] = math.sin(math.radians(pitch))
        cam_front[2] = math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))
        cam_front = cam_front / np.linalg.norm(cam_front)
        
        cam_right = np.cross(cam_front, cam_up_world)
        cam_right = cam_right / np.linalg.norm(cam_right)
        
        cam_up = np.cross(cam_right, cam_front)
        cam_up = cam_up / np.linalg.norm(cam_up)

        if mouse_locked:
            keys = pygame.key.get_pressed()
            velocity = flight_speed * dt
            
            if keys[pygame.K_w]: cam_pos += cam_front * velocity
            if keys[pygame.K_s]: cam_pos -= cam_front * velocity
            if keys[pygame.K_a]: cam_pos -= cam_right * velocity
            if keys[pygame.K_d]: cam_pos += cam_right * velocity
            if keys[pygame.K_SPACE]: cam_pos += cam_up_world * velocity
            if keys[pygame.K_LSHIFT]: cam_pos -= cam_up_world * velocity

        if np.linalg.norm(cam_pos) < 0.1:
            cam_pos = cam_pos / np.linalg.norm(cam_pos) * 0.1

        glUseProgram(shader)
        glUniform2f(loc_resolution, width, height)
        glUniform3f(loc_camPos, cam_pos[0], cam_pos[1], cam_pos[2])
        glUniform3f(loc_camDir, cam_front[0], cam_front[1], cam_front[2])
        glUniform3f(loc_camUp, cam_up[0], cam_up[1], cam_up[2])
        glUniform3f(loc_camRight, cam_right[0], cam_right[1], cam_right[2])
        glUniform1f(loc_time, current_time)

        glClear(GL_COLOR_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        
        render_hud_surface(hud_surface, hud_font, cam_pos, yaw, pitch, current_time)
        hud_data = pygame.image.tostring(hud_surface, "RGBA", True)
        
        glBindTexture(GL_TEXTURE_2D, hud_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, hud_data)
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glUseProgram(hud_shader)
        glUniform1i(loc_hud_tex, 0)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        
        glDisable(GL_BLEND)
        
        pygame.display.flip()

    glDeleteTextures(1, [hud_tex])
    glDeleteBuffers(1, [vbo])
    glDeleteVertexArrays(1, [vao])
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()