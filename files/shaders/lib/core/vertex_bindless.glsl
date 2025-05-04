#version 430 core

#extension GL_NV_bindless_texture : require
#extension GL_NV_gpu_shader5 : require

#include "lib/material/bindings.glsl"
#include "lib/material/material.glsl"

layout(location = 1) in vec4 vMaterialIndex0;
layout(location = 6) in vec4 vMaterialIndex1;
layout(location = 7) in vec4 vMaterialIndex2;

flat out int materialIndex;

flat out int diffuseMapIndex;
flat out int darkMapIndex;
flat out int detailMapIndex;
flat out int decalMapIndex;
flat out int emissiveMapIndex;
flat out int normalMapIndex;
flat out int envMapIndex;
flat out int specularMapIndex;
flat out int bumpMapIndex;
flat out int glossMapIndex;

void assignMaterials()
{
    materialIndex = int(vMaterialIndex0.x);

    diffuseMapIndex = int(vMaterialIndex0.y);
    normalMapIndex = int(vMaterialIndex0.z);
    specularMapIndex = int(vMaterialIndex0.w);

    decalMapIndex = int(vMaterialIndex1.x);
    emissiveMapIndex = int(vMaterialIndex1.y);
    darkMapIndex = int(vMaterialIndex1.z);
    envMapIndex = int(vMaterialIndex1.w);

    detailMapIndex = int(vMaterialIndex2.x);
    bumpMapIndex = int(vMaterialIndex2.y);
    glossMapIndex = int(vMaterialIndex2.z);
}

Material getMaterial()
{
    return materials[materialIndex];
}
