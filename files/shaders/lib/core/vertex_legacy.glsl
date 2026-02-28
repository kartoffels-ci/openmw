#version 120

#include "lib/material/colormodes.glsl"
#include "lib/material/material.glsl"

uniform float specStrength;
uniform float emissiveMult;
uniform int colorMode;

void assignMaterials()
{
    // No-op in legacy path — materials come from gl_FrontMaterial uniforms
}

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
