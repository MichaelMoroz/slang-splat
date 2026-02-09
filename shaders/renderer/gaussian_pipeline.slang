static const uint INVALID_START = 0xFFFFFFFFu;

uniform StructuredBuffer<float4> g_Positions;
uniform StructuredBuffer<float4> g_Scales;
uniform StructuredBuffer<float4> g_Rotations;
uniform StructuredBuffer<float4> g_ColorAlpha;

uniform RWStructuredBuffer<float4> g_ScreenCenterRadiusDepth;
uniform RWStructuredBuffer<float4> g_ScreenColorAlpha;
uniform RWStructuredBuffer<uint> g_ScreenValid;
uniform RWStructuredBuffer<float4> g_SplatPosLocal;
uniform RWStructuredBuffer<float4> g_SplatInvScale;
uniform RWStructuredBuffer<float4> g_SplatQuat;

uniform RWStructuredBuffer<uint> g_Keys;
uniform RWStructuredBuffer<uint> g_Values;
uniform RWStructuredBuffer<uint> g_ListCounter;
uniform RWStructuredBuffer<uint2> g_TileRanges;

uniform StructuredBuffer<uint> g_SortedKeys;
uniform StructuredBuffer<uint> g_SortedValues;

uniform RWTexture2D<float4> g_Output;

uniform uint g_splatCount;
uniform uint g_tileSize;
uniform uint g_tileWidth;
uniform uint g_tileHeight;
uniform uint g_tileCount;
uniform uint g_depthBits;
uniform uint g_sortedCount;
uniform uint g_maxListEntries;
uniform uint g_width;
uniform uint g_height;
uniform uint g_maxSplatSteps;

uniform float2 g_viewport;
uniform float3 g_camPos;
uniform float3 g_camRight;
uniform float3 g_camUp;
uniform float3 g_camForward;
uniform float3 g_background;

uniform float g_focalPixels;
uniform float g_nearDepth;
uniform float g_farDepth;
uniform float g_radiusScale;
uniform float g_maxSplatRadiusPx;
uniform float g_alphaCutoff;
uniform float g_transmittanceThreshold;

float3 rotate_by_quat(float3 v, float4 q)
{
    float3 qv = q.yzw;
    return v + 2.0 * cross(cross(v, qv) + q.x * v, qv);
}

float3 world_to_camera(float3 v)
{
    return float3(dot(v, g_camRight), dot(v, g_camUp), dot(v, g_camForward));
}

float2 project_cam(float3 p)
{
    float z = max(p.z, 1e-6);
    return float2(
        p.x * g_focalPixels / z + 0.5 * g_viewport.x,
        p.y * g_focalPixels / z + 0.5 * g_viewport.y
    );
}

uint quantizeDepth(float depthValue)
{
    float denom = max(g_farDepth - g_nearDepth, 1e-6);
    float t = saturate((depthValue - g_nearDepth) / denom);
    uint maxValue = (1u << g_depthBits) - 1u;
    return min((uint)round(t * float(maxValue)), maxValue);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csProjectAndBin(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_splatCount)
    {
        return;
    }

    float3 worldPos = g_Positions[splatId].xyz;
    float3 rel = worldPos - g_camPos;
    float3 camPos = world_to_camera(rel);
    float camDistance = length(rel);
    float depthValue = camPos.z;

    float invDepth = 1.0 / max(depthValue, 1e-6);
    float px = camPos.x * g_focalPixels * invDepth + 0.5 * g_viewport.x;
    float py = camPos.y * g_focalPixels * invDepth + 0.5 * g_viewport.y;

    float4 q = g_Rotations[splatId];
    float3 scale = g_Scales[splatId].xyz;
    float maxScale = max(scale.x, max(scale.y, scale.z));
    scale = max(scale, maxScale * 0.05);
    scale *= g_radiusScale;

    float3 axis0 = world_to_camera(rotate_by_quat(float3(1.0, 0.0, 0.0), q)) * scale.x;
    float3 axis1 = world_to_camera(rotate_by_quat(float3(0.0, 1.0, 0.0), q)) * scale.y;
    float3 axis2 = world_to_camera(rotate_by_quat(float3(0.0, 0.0, 1.0), q)) * scale.z;

    float2 center = float2(px, py);
    float2 extent = float2(0.0, 0.0);
    float2 p0p = project_cam(camPos + axis0);
    float2 p0m = project_cam(camPos - axis0);
    float2 p1p = project_cam(camPos + axis1);
    float2 p1m = project_cam(camPos - axis1);
    float2 p2p = project_cam(camPos + axis2);
    float2 p2m = project_cam(camPos - axis2);
    extent = max(extent, abs(p0p - center));
    extent = max(extent, abs(p0m - center));
    extent = max(extent, abs(p1p - center));
    extent = max(extent, abs(p1m - center));
    extent = max(extent, abs(p2p - center));
    extent = max(extent, abs(p2m - center));
    float radiusPx = clamp(max(extent.x, extent.y) + 1.0, 1.0, g_maxSplatRadiusPx);

    float3 invScale = 1.0 / max(scale, float3(1e-6, 1e-6, 1e-6));
    float3 roLocal = rotate_by_quat(g_camPos - worldPos, q) * invScale;

    g_ScreenCenterRadiusDepth[splatId] = float4(px, py, radiusPx, camDistance);
    g_ScreenColorAlpha[splatId] = g_ColorAlpha[splatId];
    g_SplatPosLocal[splatId] = float4(roLocal, 0.0);
    g_SplatInvScale[splatId] = float4(invScale, 0.0);
    g_SplatQuat[splatId] = q;

    bool visible = depthValue > 1e-4
        && camDistance > g_nearDepth
        && camDistance < g_farDepth
        && (px + radiusPx) >= 0.0
        && (px - radiusPx) < g_viewport.x
        && (py + radiusPx) >= 0.0
        && (py - radiusPx) < g_viewport.y;

    if (!visible)
    {
        g_ScreenValid[splatId] = 0;
        return;
    }

    g_ScreenValid[splatId] = 1;

    int minX = int(floor((px - radiusPx) / float(g_tileSize)));
    int maxX = int(floor((px + radiusPx) / float(g_tileSize)));
    int minY = int(floor((py - radiusPx) / float(g_tileSize)));
    int maxY = int(floor((py + radiusPx) / float(g_tileSize)));
    minX = clamp(minX, 0, int(g_tileWidth) - 1);
    maxX = clamp(maxX, 0, int(g_tileWidth) - 1);
    minY = clamp(minY, 0, int(g_tileHeight) - 1);
    maxY = clamp(maxY, 0, int(g_tileHeight) - 1);

    uint depthCode = quantizeDepth(camDistance);
    for (int tileY = minY; tileY <= maxY; tileY++)
    {
        for (int tileX = minX; tileX <= maxX; tileX++)
        {
            uint tileId = uint(tileY) * g_tileWidth + uint(tileX);
            uint key = (tileId << g_depthBits) | depthCode;
            uint writeIndex = 0;
            InterlockedAdd(g_ListCounter[0], 1, writeIndex);
            if (writeIndex < g_maxListEntries)
            {
                g_Keys[writeIndex] = key;
                g_Values[writeIndex] = splatId;
            }
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csClearTileRanges(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tileId = dispatchThreadID.x;
    if (tileId >= g_tileCount)
    {
        return;
    }
    g_TileRanges[tileId] = uint2(INVALID_START, 0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csBuildTileRanges(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= g_sortedCount)
    {
        return;
    }

    uint tile = g_SortedKeys[idx] >> g_depthBits;
    uint prevTile = (idx > 0) ? (g_SortedKeys[idx - 1] >> g_depthBits) : INVALID_START;
    uint nextTile = (idx + 1 < g_sortedCount) ? (g_SortedKeys[idx + 1] >> g_depthBits) : INVALID_START;

    if (idx == 0 || tile != prevTile)
    {
        g_TileRanges[tile].x = idx;
    }
    if (idx + 1 == g_sortedCount || tile != nextTile)
    {
        g_TileRanges[tile].y = idx + 1;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csRasterize(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_width || py >= g_height)
    {
        return;
    }

    uint tileX = px / g_tileSize;
    uint tileY = py / g_tileSize;
    uint tileId = tileY * g_tileWidth + tileX;
    uint2 range = g_TileRanges[tileId];
    if (range.x == INVALID_START || range.y <= range.x)
    {
        float3 bgLinear = pow(max(g_background, 0.0), 2.2);
        g_Output[uint2(px, py)] = float4(bgLinear, 1.0);
        return;
    }

    float3 accum = float3(0.0, 0.0, 0.0);
    float trans = 1.0;
    float2 rayUV = (float2(float(px) + 0.5, float(py) + 0.5) - 0.5 * g_viewport) / g_focalPixels;
    float3 rayDir = normalize(g_camForward + rayUV.x * g_camRight + rayUV.y * g_camUp);
    uint processed = 0;
    for (uint idx = range.x; idx < range.y; idx++)
    {
        if (processed >= g_maxSplatSteps)
        {
            break;
        }
        processed++;
        uint splatId = g_SortedValues[idx];
        if (g_ScreenValid[splatId] == 0)
        {
            continue;
        }

        float3 roLocal = g_SplatPosLocal[splatId].xyz;
        float3 invScale = g_SplatInvScale[splatId].xyz;
        float4 q = g_SplatQuat[splatId];
        float3 rdLocal = rotate_by_quat(rayDir, q) * invScale;
        float denom = dot(rdLocal, rdLocal);
        if (denom <= 1e-10)
        {
            continue;
        }
        float tClosest = dot(rdLocal, -roLocal) / denom;
        if (tClosest <= 0.0)
        {
            continue;
        }
        float3 closest = roLocal + rdLocal * tClosest;
        float rho = length(closest);
        if (rho >= 1.0)
        {
            continue;
        }

        float4 colorAlpha = g_ScreenColorAlpha[splatId];
        float coverage = saturate(1.0 - rho);
        coverage = coverage * coverage * (3.0 - 2.0 * coverage);
        float alpha = saturate(colorAlpha.w * coverage);
        if (alpha < g_alphaCutoff)
        {
            continue;
        }

        accum += trans * alpha * colorAlpha.xyz;
        trans *= 1.0 - alpha;
        if (trans < g_transmittanceThreshold)
        {
            break;
        }
    }

    float3 finalColor = accum + trans * g_background;
    float3 linearOut = pow(max(finalColor, 0.0), 2.2);
    g_Output[uint2(px, py)] = float4(linearOut, 1.0 - trans);
}
