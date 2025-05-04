#version 430 compatibility
#pragma import_defines(FORCE_OPAQUE, DISTORTION)

#if @useUBO
    #extension GL_ARB_uniform_buffer_object : require
#endif

#if @useGPUShader4
    #extension GL_EXT_gpu_shader4: require
#endif

#if @diffuseMap
varying vec2 diffuseMapUV;
#endif

#if @darkMap
varying vec2 darkMapUV;
#endif

#if @detailMap
varying vec2 detailMapUV;
#endif

#if @decalMap
varying vec2 decalMapUV;
#endif

#if @emissiveMap
varying vec2 emissiveMapUV;
#endif

#if @normalMap
varying vec2 normalMapUV;
#endif

#if @envMap
varying vec2 envMapUV;
uniform vec4 envMapColor;
#endif

#if @specularMap
varying vec2 specularMapUV;
#endif

#if @bumpMap
varying vec2 bumpMapUV;
uniform vec2 envMapLumaBias;
uniform mat2 bumpMapMatrix;
#endif

#if @glossMap
varying vec2 glossMapUV;
#endif

uniform vec2 screenRes;
uniform float near;
uniform float far;
uniform float alphaRef;
uniform float distortionStrength;

#define PER_PIXEL_LIGHTING (@normalMap || @specularMap || @forcePPL)

#if !PER_PIXEL_LIGHTING
centroid varying vec3 passLighting;
centroid varying vec3 passSpecular;
centroid varying vec3 shadowDiffuseLighting;
centroid varying vec3 shadowSpecularLighting;
#endif
varying vec3 passViewPos;
varying vec3 passNormal;
#if @normalMap || @diffuseParallax
varying vec4 passTangent;
#endif

#if @additiveBlending
#define ADDITIVE_BLENDING
#endif

#include "lib/core/fragment.h.glsl"
#include "lib/light/lighting.glsl"
#include "lib/material/parallax.glsl"
#include "lib/material/alpha.glsl"
#include "lib/util/distortion.glsl"
#include "lib/view/depth.glsl"
#include "lib/material/colormodes.glsl"
#include "lib/material/vertexcolors.glsl"
#include "lib/core/fragment.h.glsl"

#include "fog.glsl"
#include "shadows_fragment.glsl"
#include "compatibility/normals.glsl"

#if @softParticles
#include "lib/particle/soft.glsl"

uniform float particleSize;
uniform bool particleFade;
uniform float softFalloffDepth;
#endif

#if @particleOcclusion
#include "lib/particle/occlusion.glsl"
uniform sampler2D orthoDepthMap;
varying vec3 orthoDepthMapCoord;
#endif

uniform sampler2D opaqueDepthTex;
uniform vec4 debugcolor = vec4(-1.0);

void main()
{
    uint zTile = uint((log(abs(linearizeDepth(gl_FragCoord.z, near, far)) / near) * gridDepth) / log(8192 / near));
    vec3 colors[8] = vec3[](
    vec3(0, 0, 0),    vec3( 0,  0,  1), vec3( 0, 1, 0),  vec3(0, 1,  1),
    vec3(1,  0,  0),  vec3( 1,  0,  1), vec3( 1, 1, 0),  vec3(1, 1, 1)
    );
    gl_FragData[0] = vec4(colors[uint(mod(float(zTile), 8.0))], 1.0);
    // gl_FragData[0] = vec4(vec3(float(lightCount) / 64.0), 1.0);
    // return;

    if (debugcolor.x > 0) {
        gl_FragData[0] = debugcolor;
        return;
    }
    Material material = getMaterial();

#if @particleOcclusion
    applyOcclusionDiscard(orthoDepthMapCoord, texture2D(orthoDepthMap, orthoDepthMapCoord.xy * 0.5 + 0.5).r);
#endif

    // only offset diffuse and normal maps for now, other textures are more likely to be using a completely different UV set
    vec2 offset = vec2(0.0);

#if @parallax || @diffuseParallax
#if @parallax
    float height = sample_normal(normalMapUV).a;
    float flipY = (passTangent.w > 0.0) ? -1.f : 1.f;
#else
    float height = sample_diffuse(diffuseMapUV).a;
    // FIXME: shouldn't be necessary, but in this path false-positives are common
    float flipY = -1.f;
#endif
    offset = getParallaxOffset(transpose(normalToViewMatrix) * normalize(-passViewPos), height);
#endif

vec2 screenCoords = gl_FragCoord.xy / screenRes;

#if @diffuseMap
    gl_FragData[0] = sample_diffuse(diffuseMapUV + offset);

#if defined(DISTORTION) && DISTORTION
    gl_FragData[0].a *= getDiffuseColor(material).a;
    gl_FragData[0] = applyDistortion(gl_FragData[0], distortionStrength, gl_FragCoord.z, texture2D(opaqueDepthTex, screenCoords / @distorionRTRatio).x);
    return;
#endif

#if @diffuseParallax
    gl_FragData[0].a = 1.0;
#else
    // TOOD: FIXME
    // gl_FragData[0].a *= coveragePreservingAlphaScale(diffuseMap, diffuseMapUV + offset);
#endif
#else
    gl_FragData[0] = vec4(1.0);
#endif

    vec4 diffuseColor = getDiffuseColor(material);
    gl_FragData[0].a *= diffuseColor.a;

#if @darkMap
    gl_FragData[0] *= sample_dark(darkMapUV);
    // TODO: FIXME
    // gl_FragData[0].a *= coveragePreservingAlphaScale(darkMap, darkMapUV);
#endif

    gl_FragData[0].a = alphaTest(gl_FragData[0].a, alphaRef);

#if @normalMap
    vec4 normalTex = sample_normal(normalMapUV + offset);
    vec3 normal = normalTex.xyz * 2.0 - 1.0;
#if @reconstructNormalZ
    normal.z = sqrt(1.0 - dot(normal.xy, normal.xy));
#endif
    vec3 viewNormal = normalToView(normal);
#else
    vec3 viewNormal = normalize(gl_NormalMatrix * passNormal);
#endif

    vec3 viewVec = normalize(passViewPos);

#if @detailMap
    gl_FragData[0].xyz *= sample_detail(detailMapUV).xyz * 2.0;
#endif

#if @decalMap
    vec4 decalTex = sample_decal(decalMapUV);
    gl_FragData[0].xyz = mix(gl_FragData[0].xyz, decalTex.xyz, decalTex.a * diffuseColor.a);
#endif

#if @envMap

    vec2 envTexCoordGen = envMapUV;
    float envLuma = 1.0;

#if @normalMap
    // if using normal map + env map, take advantage of per-pixel normals for envTexCoordGen
    vec3 r = reflect( viewVec, viewNormal );
    float m = 2.0 * sqrt( r.x*r.x + r.y*r.y + (r.z+1.0)*(r.z+1.0) );
    envTexCoordGen = vec2(r.x/m + 0.5, r.y/m + 0.5);
#endif

#if @bumpMap
    vec4 bumpTex = sample_bump(bumpMapUV);
    envTexCoordGen += bumpTex.rg * bumpMapMatrix;
    envLuma = clamp(bumpTex.b * envMapLumaBias.x + envMapLumaBias.y, 0.0, 1.0);
#endif

    vec3 envEffect = sample_env(envTexCoordGen).xyz * envMapColor.xyz * envLuma;

#if @glossMap
    envEffect *= sample_gloss(glossMapUV).xyz;
#endif

#if @preLightEnv
    gl_FragData[0].xyz += envEffect;
#endif

#endif

    float shadowing = unshadowedLightRatio(-passViewPos.z);
    vec3 lighting, specular;
#if !PER_PIXEL_LIGHTING
    lighting = passLighting + shadowDiffuseLighting * shadowing;
    specular = passSpecular + shadowSpecularLighting * shadowing;
#else
#if @specularMap
    vec4 specTex = sample_specular(specularMapUV);
    float shininess = specTex.a * 255.0;
    vec3 specularColor = specTex.xyz;
#else
    float shininess = material.shininess;
    vec3 specularColor = getSpecularColor(material).xyz;
#endif
    vec3 diffuseLight, ambientLight, specularLight;
    doLighting(gl_FragCoord.xyz, passViewPos, viewNormal, shininess, shadowing, diffuseLight, ambientLight, specularLight);
    lighting = diffuseColor.xyz * diffuseLight + getAmbientColor(material).xyz * ambientLight + getEmissionColor(material).xyz * material.emissiveMult;
    specular = specularColor * specularLight * material.specStrength;
#endif

    clampLightingResult(lighting);
    gl_FragData[0].xyz = gl_FragData[0].xyz * lighting + specular;

#if @envMap && !@preLightEnv
    gl_FragData[0].xyz += envEffect;
#endif

#if @emissiveMap
    gl_FragData[0].xyz += sample_emissive(emissiveMapUV).xyz;
#endif

    gl_FragData[0] = applyFogAtPos(gl_FragData[0], passViewPos, far);

#if !defined(FORCE_OPAQUE) && @softParticles
    gl_FragData[0].a *= calcSoftParticleFade(
        viewVec,
        passViewPos,
        viewNormal,
        near,
        far,
        sampleOpaqueDepthTex(screenCoords).x,
        particleSize,
        particleFade,
        softFalloffDepth
    );
#endif

#if defined(FORCE_OPAQUE) && FORCE_OPAQUE
    // having testing & blending isn't enough - we need to write an opaque pixel to be opaque
    gl_FragData[0].a = 1.0;
#endif

#if !defined(FORCE_OPAQUE) && !@disableNormals
    gl_FragData[1].xyz = viewNormal * 0.5 + 0.5;
#endif

    applyShadowDebugOverlay();
}
