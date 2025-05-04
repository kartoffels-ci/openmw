#version 430 core

#extension GL_NV_bindless_texture : require
#extension GL_NV_gpu_shader5 : require

#include "lib/material/bindings.glsl"
#include "lib/material/material.glsl"
#include "lib/texture/bindings.glsl"

flat in int materialIndex;
flat in int diffuseMapIndex;
flat in int darkMapIndex;
flat in int detailMapIndex;
flat in int decalMapIndex;
flat in int emissiveMapIndex;
flat in int normalMapIndex;
flat in int envMapIndex;
flat in int specularMapIndex;
flat in int bumpMapIndex;
flat in int glossMapIndex;

Material getMaterial()
{
    return materials[materialIndex];
}

#if @diffuseMap
// uniform sampler2D diffuseMap;
vec4 sample_diffuse(in vec2 uv)
{
    // return texture(diffuseMap, uv);
    return texture(textures[diffuseMapIndex], uv);
}
#endif

#if @darkMap
// uniform sampler2D darkMap;
vec4 sample_dark(in vec2 uv)
{
    //  return texture(darkMap, uv);
    return texture(textures[darkMapIndex], uv);
}
#endif

#if @detailMap
// uniform sampler2D detailMap;
vec4 sample_detail(in vec2 uv)
{ 
    // return texture(detailMap, uv);
    return texture(textures[detailMapIndex], uv);
}
#endif

#if @decalMap
// uniform sampler2D decalMap;
vec4 sample_decal(in vec2 uv)
{ 
    // return texture(decalMap, uv);
    return texture(textures[decalMapIndex], uv);
}
#endif

#if @emissiveMap
// uniform sampler2D emissiveMap;
vec4 sample_emissive(in vec2 uv)
{ 
    // return texture(emissiveMap, uv);
    return texture(textures[emissiveMapIndex], uv);
}
#endif

#if @normalMap
// uniform sampler2D normalMap;
vec4 sample_normal(in vec2 uv)
{ 
    // return texture(normalMap, uv);
    return texture(textures[normalMapIndex], uv);
}
#endif

#if @envMap
// uniform sampler2D envMap;
vec4 sample_env(in vec2 uv)
{ 
    // return texture(envMap, uv);
    return texture(textures[envMapIndex], uv);
}
#endif

#if @specularMap
// uniform sampler2D specularMap;
vec4 sample_specular(in vec2 uv)
{ 
    // return texture(specularMap, uv);
    return texture(textures[specularMapIndex], uv);
}
#endif

#if @bumpMap
// uniform sampler2D bumpMap;
vec4 sample_bump(in vec2 uv){
    // return texture(bumpMap, uv);
    return texture(textures[bumpMapIndex], uv);
}
#endif

#if @glossMap
// uniform sampler2D glossMap;
vec4 sample_gloss(in vec2 uv)
{
    // return texture(glossMap, uv);
    return texture(textures[glossMapIndex], uv);
}
#endif
