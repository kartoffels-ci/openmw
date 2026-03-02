#version 120

#include "lib/material/colormodes.glsl"
#include "lib/material/material.glsl"

uniform float specStrength;
uniform float emissiveMult;
uniform int colorMode;

Material getMaterial()
{
    Material material;

    material.diffuse = gl_FrontMaterial.diffuse;
    material.ambient = gl_FrontMaterial.ambient;
    material.specular = gl_FrontMaterial.specular;
    material.emission = gl_FrontMaterial.emission;
    material.shininess = gl_FrontMaterial.shininess;
    material.emissiveMult = emissiveMult;
    material.specStrength = specStrength;
    material.colorMode = colorMode;

    return material;
}

#if @diffuseMap
uniform sampler2D diffuseMap;
vec4 sample_diffuse(in vec2 uv)
{
    return texture2D(diffuseMap, uv);
}
#endif

#if @darkMap
uniform sampler2D darkMap;
vec4 sample_dark(in vec2 uv)
{
    return texture2D(darkMap, uv);
}
#endif

#if @detailMap
uniform sampler2D detailMap;
vec4 sample_detail(in vec2 uv)
{
    return texture2D(detailMap, uv);
}
#endif

#if @decalMap
uniform sampler2D decalMap;
vec4 sample_decal(in vec2 uv)
{
    return texture2D(decalMap, uv);
}
#endif

#if @emissiveMap
uniform sampler2D emissiveMap;
vec4 sample_emissive(in vec2 uv)
{
    return texture2D(emissiveMap, uv);
}
#endif

#if @normalMap
uniform sampler2D normalMap;
vec4 sample_normal(in vec2 uv)
{
    return texture2D(normalMap, uv);
}
#endif

#if @envMap
uniform sampler2D envMap;
vec4 sample_env(in vec2 uv)
{
    return texture2D(envMap, uv);
}
#endif

#if @specularMap
uniform sampler2D specularMap;
vec4 sample_specular(in vec2 uv)
{
    return texture2D(specularMap, uv);
}
#endif

#if @bumpMap
uniform sampler2D bumpMap;
vec4 sample_bump(in vec2 uv)
{
    return texture2D(bumpMap, uv);
}
#endif

#if @glossMap
uniform sampler2D glossMap;
vec4 sample_gloss(in vec2 uv)
{
    return texture2D(glossMap, uv);
}
#endif
