#define SHADOWS @shadows_enabled

#if SHADOWS

#ifndef PER_PIXEL_LIGHTING
#define PER_PIXEL_LIGHTING 0
#endif

    uniform float maximumShadowMapDistance;
    uniform float shadowFadeStart;
    @foreach shadow_texture_unit_index @shadow_texture_unit_list
        uniform sampler2DShadow shadowTexture@shadow_texture_unit_index;
        varying vec4 shadowSpaceCoords@shadow_texture_unit_index;

#if @softShadows
        uniform mat4 shadowSpaceMatrix@shadow_texture_unit_index;
#endif

#if @perspectiveShadowMaps
        varying vec4 shadowRegionCoords@shadow_texture_unit_index;
#endif
    @endforeach

// ============================================================================
// Soft Shadow Filtering (8-tap spiral disc with PCSS penumbra)
// ============================================================================
#if @softShadows

#define SHADOWMAP_RES (@shadowMapResolution * 0.5)

// Minimum texels the filter should span (prevents undersampling at distance)
#ifndef FILTER_MIN_TEXELS
#define FILTER_MIN_TEXELS 1.3
#endif

// Penumbra width scaling (pseudo-PCSS)
#ifndef PENUMBRA_NEAR
#define PENUMBRA_NEAR 1
#endif

#ifndef PENUMBRA_FAR
#define PENUMBRA_FAR 3.0
#endif

#ifndef PENUMBRA_SCALING
#define PENUMBRA_SCALING 1
#endif

// Interleaved Gradient Noise - smooth spatial variation for rotation
float getIGNRotation()
{
    float noise = fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y));
    return noise * 6.283185;
}

// Precomputed 8-tap spiral with alternating flip
// Designed so rotated copies interleave rather than overlap
const vec2 spiralDisc[8] = vec2[8](
    vec2(-0.7071,  0.7071),
    vec2( 0.0000, -0.8750),
    vec2( 0.5303,  0.5303),
    vec2(-0.6250,  0.0000),
    vec2( 0.3536, -0.3536),
    vec2( 0.0000,  0.3750),
    vec2(-0.1768, -0.1768),
    vec2( 0.1250,  0.0000)
);

float getFilteredShadowing(sampler2DShadow tex, vec4 coord, mat4 matrix)
{
    vec3 shadowUV = coord.xyz / coord.w;
    float texelScale = coord.w / SHADOWMAP_RES;

    // ---- Compute filter directions aligned to surface plane ----
#if PER_PIXEL_LIGHTING
    // Use surface normal to derive tangent frame in shadow space
    vec3 viewNormal = normalize(gl_NormalMatrix * passNormal);
    vec3 viewTangent = cross(viewNormal, normalize(vec3(0.9153, -0.0115, -0.5141)));
    vec3 viewBitangent = cross(viewNormal, viewTangent);

    // Transform tangents to shadow space and find resulting UV directions
    vec3 shadowTangent = normalize((matrix * vec4(viewTangent, 0.0)).xyz);
    vec3 shadowBitangent = normalize((matrix * vec4(viewBitangent, 0.0)).xyz);
    vec4 tangentCoord = coord + vec4(shadowTangent, 0.0);
    vec4 bitangentCoord = coord + vec4(shadowBitangent, 0.0);
    vec3 filterX = normalize((tangentCoord.xyz / tangentCoord.w) - shadowUV);
    vec3 filterY = normalize((bitangentCoord.xyz / bitangentCoord.w) - shadowUV);
#else
    // Fallback: use screen-space derivatives
    vec3 filterX = normalize(dFdx(shadowUV));
    vec3 filterY = normalize(dFdy(shadowUV));
#endif

    // Enforce orthogonality between filter axes
    filterY = normalize(cross(filterX, cross(filterX, filterY)));

    // Build offset vectors (scaled to shadow map texels)
    vec4 offs_x = vec4(filterX, 0.0) * texelScale;
    vec4 offs_y = vec4(filterY, 0.0) * texelScale;

    // ---- Ensure minimum texel coverage (prevents undersampling at distance) ----
    float minSize = texelScale * FILTER_MIN_TEXELS;
    offs_x *= max(1.0, minSize / length(offs_x.xy));
    offs_y *= max(1.0, minSize / length(offs_y.xy));

#if PENUMBRA_SCALING
    // ---- Pseudo-PCSS: widen penumbra based on depth in shadow ----
    float penumbra = mix(float(PENUMBRA_NEAR), float(PENUMBRA_FAR), clamp(shadowUV.z, 0.0, 1.0));
    offs_x *= penumbra;
    offs_y *= penumbra;
#endif

    // ---- Sample shadow map with rotated spiral disc ----
    float rotation = getIGNRotation();
    float c = cos(rotation);
    float s = sin(rotation);

    float shadow = 0.0;
    for (int i = 0; i < 8; i++)
    {
        vec2 p = spiralDisc[i];
        vec2 rotated = vec2(p.x * c - p.y * s, p.x * s + p.y * c);
        shadow += shadow2DProj(tex, coord + offs_x * rotated.x + offs_y * rotated.y).r;
    }

    return shadow * 0.125;
}

#endif // @softShadows

#endif // SHADOWS

// ============================================================================

float unshadowedLightRatio(float distance)
{
    float shadowing = 1.0;
#if SHADOWS
#if @limitShadowMapDistance
    float fade = clamp((distance - shadowFadeStart) / (maximumShadowMapDistance - shadowFadeStart), 0.0, 1.0);
    if (fade == 1.0)
        return shadowing;
#endif
    bool doneShadows = false;
    @foreach shadow_texture_unit_index @shadow_texture_unit_list
        if (!doneShadows)
        {
            vec3 shadowXYZ = shadowSpaceCoords@shadow_texture_unit_index.xyz / shadowSpaceCoords@shadow_texture_unit_index.w;
#if @perspectiveShadowMaps
            vec3 shadowRegionXYZ = shadowRegionCoords@shadow_texture_unit_index.xyz / shadowRegionCoords@shadow_texture_unit_index.w;
#endif
            if (all(lessThan(shadowXYZ.xy, vec2(1.0, 1.0))) && all(greaterThan(shadowXYZ.xy, vec2(0.0, 0.0))))
            {
#if @softShadows
                shadowing = min(getFilteredShadowing(
                    shadowTexture@shadow_texture_unit_index,
                    shadowSpaceCoords@shadow_texture_unit_index,
                    shadowSpaceMatrix@shadow_texture_unit_index
                ), shadowing);
#else
                shadowing = min(shadow2DProj(shadowTexture@shadow_texture_unit_index, shadowSpaceCoords@shadow_texture_unit_index).r, shadowing);
#endif

                doneShadows = all(lessThan(shadowXYZ, vec3(0.95, 0.95, 1.0))) && all(greaterThan(shadowXYZ, vec3(0.05, 0.05, 0.0)));
#if @perspectiveShadowMaps
                doneShadows = doneShadows && all(lessThan(shadowRegionXYZ, vec3(1.0, 1.0, 1.0))) && all(greaterThan(shadowRegionXYZ.xy, vec2(-1.0, -1.0)));
#endif
            }
        }
    @endforeach
#if @limitShadowMapDistance
    shadowing = mix(shadowing, 1.0, fade);
#endif
#endif // SHADOWS
    return shadowing;
}

void applyShadowDebugOverlay()
{
#if SHADOWS && @useShadowDebugOverlay
    bool doneOverlay = false;
    float colourIndex = 0.0;
    @foreach shadow_texture_unit_index @shadow_texture_unit_list
        if (!doneOverlay)
        {
            vec3 shadowXYZ = shadowSpaceCoords@shadow_texture_unit_index.xyz / shadowSpaceCoords@shadow_texture_unit_index.w;
#if @perspectiveShadowMaps
            vec3 shadowRegionXYZ = shadowRegionCoords@shadow_texture_unit_index.xyz / shadowRegionCoords@shadow_texture_unit_index.w;
#endif
            if (all(lessThan(shadowXYZ.xy, vec2(1.0, 1.0))) && all(greaterThan(shadowXYZ.xy, vec2(0.0, 0.0))))
            {
                colourIndex = mod(@shadow_texture_unit_index.0, 3.0);
                if (colourIndex < 1.0)
                    gl_FragData[0].x += 0.1;
                else if (colourIndex < 2.0)
                    gl_FragData[0].y += 0.1;
                else
                    gl_FragData[0].z += 0.1;

                doneOverlay = all(lessThan(shadowXYZ, vec3(0.95, 0.95, 1.0))) && all(greaterThan(shadowXYZ, vec3(0.05, 0.05, 0.0)));
#if @perspectiveShadowMaps
                doneOverlay = doneOverlay && all(lessThan(shadowRegionXYZ.xyz, vec3(1.0, 1.0, 1.0))) && all(greaterThan(shadowRegionXYZ.xy, vec2(-1.0, -1.0)));
#endif
            }
        }
    @endforeach
#endif // SHADOWS
}
