#ifndef OPENMW_FRAGMENT_H_GLSL
#define OPENMW_FRAGMENT_H_GLSL

@link "lib/core/fragment.glsl" if !@useOVR_multiview
@link "lib/core/fragment_multiview.glsl" if @useOVR_multiview
@link "lib/core/fragment_bindless.glsl" if !@legacyBindings

#include "lib/material/material.glsl"

vec4 sampleReflectionMap(vec2 uv);

#if @waterRefraction
vec4 sampleRefractionMap(vec2 uv);
float sampleRefractionDepthMap(vec2 uv);
#endif

vec4 samplerLastShader(vec2 uv);

#if @skyBlending
vec3 sampleSkyColor(vec2 uv);
#endif

vec4 sampleOpaqueDepthTex(vec2 uv);

Material getMaterial();

#if @diffuseMap
    vec4 sample_diffuse(in vec2 uv);
#endif

#if @darkMap
    vec4 sample_dark(in vec2 uv);
#endif

#if @detailMap
    vec4 sample_detail(in vec2 uv);
#endif

#if @decalMap
    vec4 sample_decal(in vec2 uv);
#endif

#if @emissiveMap
    vec4 sample_emissive(in vec2 uv);
#endif

#if @normalMap
    vec4 sample_normal(in vec2 uv);
#endif

#if @envMap
    vec4 sample_env(in vec2 uv);
#endif

#if @specularMap
    vec4 sample_specular(in vec2 uv);
#endif

#if @bumpMap
    vec4 sample_bump(in vec2 uv);
#endif

#if @glossMap
    vec4 sample_gloss(in vec2 uv);
#endif

#endif  // OPENMW_FRAGMENT_H_GLSL
