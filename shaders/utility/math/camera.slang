import "utility/math/constants.slang";

static const uint CAMERA_UNDISTORT_NEWTON_ITERS = 6u;

struct CameraParams
{
    float2 viewport;
    float3 camPos;
    float3x3 camBasis;
    float2 focalPixels;
    float2 principalPoint;
    float nearDepth;
    float farDepth;
    float4 projDistortionK1K2P1P2;
    float4 projDistortionK3K4K5K6;
    float minCameraDistance;
};

interface ICamera
{
    float3 world_to_camera(float3 worldVector);
    float3 camera_to_world(float3 cameraVector);
    float3 world_point_to_camera(float3 worldPos);
    float3 camera_point_to_world(float3 cameraPos);
    float2 project_camera_to_screen_linear(float3 cameraPos);
    bool project_camera_to_screen(float3 cameraPos, out float2 screenPos);
    bool project_world_to_screen(float3 worldPos, out float2 screenPos);
    float3 screen_to_world(float2 screenPos, float depth);
    float3 screen_to_world_ray(float2 screenPos);
    float pixel_world_size_max(float depth);
    float camera_depth(float3 worldPos);
    bool is_depth_in_range(float depthValue);
    float3 position_world();
};

struct PinholeCamera : ICamera
{
    CameraParams params;

    static PinholeCamera from_params(CameraParams paramsIn)
    {
        PinholeCamera camera;
        camera.params = paramsIn;
        return camera;
    }

    float3 world_to_camera(float3 worldVector)
    {
        return mul(params.camBasis, worldVector);
    }

    float3 camera_to_world(float3 cameraVector)
    {
        return mul(transpose(params.camBasis), cameraVector);
    }

    float3 world_point_to_camera(float3 worldPos)
    {
        return world_to_camera(worldPos - params.camPos);
    }

    float3 camera_point_to_world(float3 cameraPos)
    {
        return params.camPos + camera_to_world(cameraPos);
    }

    float2 project_camera_to_screen_linear(float3 cameraPos)
    {
        float invDepth = 1.0 / max(cameraPos.z, SMALL_VALUE);
        return cameraPos.xy * (params.focalPixels * invDepth) + params.principalPoint;
    }

    float radial_distortion(float r2, out float dRadialDr2)
    {
        float r4 = r2 * r2;
        float r6 = r4 * r2;
        float numerator = 1.0 + params.projDistortionK1K2P1P2.x * r2 + params.projDistortionK1K2P1P2.y * r4 + params.projDistortionK3K4K5K6.x * r6;
        float denominatorRaw = 1.0 + params.projDistortionK3K4K5K6.y * r2 + params.projDistortionK3K4K5K6.z * r4 + params.projDistortionK3K4K5K6.w * r6;
        float denominator = abs(denominatorRaw) > SMALL_VALUE ? denominatorRaw : (denominatorRaw >= 0.0 ? SMALL_VALUE : -SMALL_VALUE);
        float dNumerator = params.projDistortionK1K2P1P2.x + 2.0 * params.projDistortionK1K2P1P2.y * r2 + 3.0 * params.projDistortionK3K4K5K6.x * r4;
        float dDenominator = params.projDistortionK3K4K5K6.y + 2.0 * params.projDistortionK3K4K5K6.z * r2 + 3.0 * params.projDistortionK3K4K5K6.w * r4;
        dRadialDr2 = (dNumerator * denominator - numerator * dDenominator) / (denominator * denominator);
        return numerator / denominator;
    }

    float2 tangential_distortion(float2 uv, float r2)
    {
        float x = uv.x;
        float y = uv.y;
        float p1 = params.projDistortionK1K2P1P2.z;
        float p2 = params.projDistortionK1K2P1P2.w;
        return float2(
            2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
            p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        );
    }

    float2 distort_normalized(float2 uv)
    {
        float r2 = dot(uv, uv);
        float dRadialDr2 = 0.0;
        float radial = radial_distortion(r2, dRadialDr2);
        return uv * radial + tangential_distortion(uv, r2);
    }

    float2 undistort_normalized(float2 uvDistorted)
    {
        if (
            all(abs(params.projDistortionK1K2P1P2) <= SMALL_VALUE)
            && all(abs(params.projDistortionK3K4K5K6) <= SMALL_VALUE)
        ) return uvDistorted;

        float2 estimate = uvDistorted;
        [unroll] for (uint iter = 0u; iter < CAMERA_UNDISTORT_NEWTON_ITERS; iter++)
        {
            float x = estimate.x;
            float y = estimate.y;
            float r2 = dot(estimate, estimate);
            float dRadialDr2 = 0.0;
            float radial = radial_distortion(r2, dRadialDr2);
            float2 distorted = estimate * radial + tangential_distortion(estimate, r2);
            float2 error = distorted - uvDistorted;
            if (dot(error, error) <= SMALL_VALUE * SMALL_VALUE) break;

            float dRadialDx = dRadialDr2 * 2.0 * x;
            float dRadialDy = dRadialDr2 * 2.0 * y;
            float p1 = params.projDistortionK1K2P1P2.z;
            float p2 = params.projDistortionK1K2P1P2.w;
            float2 jacobianRow0 = float2(radial + x * dRadialDx + 2.0 * p1 * y + 6.0 * p2 * x, x * dRadialDy + 2.0 * p1 * x + 2.0 * p2 * y);
            float2 jacobianRow1 = float2(y * dRadialDx + 2.0 * p1 * x + 2.0 * p2 * y, radial + y * dRadialDy + 6.0 * p1 * y + 2.0 * p2 * x);
            float det = jacobianRow0.x * jacobianRow1.y - jacobianRow0.y * jacobianRow1.x;
            if (abs(det) <= SMALL_VALUE || !isfinite(det)) break;

            float2 step = float2(
                (jacobianRow1.y * error.x - jacobianRow0.y * error.y) / det,
                (-jacobianRow1.x * error.x + jacobianRow0.x * error.y) / det
            );
            float2 nextEstimate = estimate - step;
            if (!all(isfinite(nextEstimate))) break;
            estimate = nextEstimate;
        }
        return estimate;
    }

    bool project_camera_to_screen(float3 cameraPos, out float2 screenPos)
    {
        if (cameraPos.z <= SMALL_VALUE)
        {
            screenPos = 0.0;
            return false;
        }

        float2 uv = distort_normalized(cameraPos.xy / cameraPos.z);
        screenPos = uv * params.focalPixels + params.principalPoint;
        return all(isfinite(screenPos));
    }

    bool project_world_to_screen(float3 worldPos, out float2 screenPos)
    {
        return project_camera_to_screen(world_point_to_camera(worldPos), screenPos);
    }

    float3 screen_to_world(float2 screenPos, float depth)
    {
        float2 focalSafe = max(params.focalPixels, float2(SMALL_VALUE));
        float depthSafe = max(depth, SMALL_VALUE);
        float2 uv = undistort_normalized((screenPos - params.principalPoint) / focalSafe);
        return camera_point_to_world(float3(uv * depthSafe, depthSafe));
    }

    float3 screen_to_world_ray(float2 screenPos)
    {
        float2 focalSafe = max(params.focalPixels, float2(SMALL_VALUE));
        float2 uv = undistort_normalized((screenPos - params.principalPoint) / focalSafe);
        return normalize(camera_to_world(float3(uv, 1.0)));
    }

    float pixel_world_size_max(float depth)
    {
        float depthSafe = max(depth, SMALL_VALUE);
        float minFocal = max(min(params.focalPixels.x, params.focalPixels.y), SMALL_VALUE);
        return SPLAT_PIXEL_CLAMP_PX * depthSafe / minFocal;
    }

    float camera_depth(float3 worldPos)
    {
        return world_point_to_camera(worldPos).z;
    }

    bool is_depth_in_range(float depthValue)
    {
        return depthValue > SMALL_VALUE && depthValue >= params.nearDepth && depthValue <= params.farDepth;
    }

    float3 position_world()
    {
        return params.camPos;
    }
};

float2 viewport_center(float2 viewport)
{
    return viewport * 0.5;
}

bool is_screen_position_inside_viewport(float2 screenPos, float2 viewport)
{
    return screenPos.x >= 0.0 && screenPos.x < viewport.x && screenPos.y >= 0.0 && screenPos.y < viewport.y;
}

float2 nearest_circle_sample_to_viewport_center(float2 circleCenterPx, float circleRadiusPx, float2 viewport)
{
    float2 toViewportCenter = viewport_center(viewport) - circleCenterPx;
    float distanceToViewportCenter = length(toViewportCenter);
    if (distanceToViewportCenter <= SMALL_VALUE) return circleCenterPx + float2(circleRadiusPx, 0.0);
    return circleCenterPx + toViewportCenter * (circleRadiusPx / distanceToViewportCenter);
}

bool world_support_radius_from_screen_circle(PinholeCamera camera, float3 worldPos, float circleRadiusPx, out float supportRadius)
{
    supportRadius = 0.0;
    if (!(circleRadiusPx > SMALL_VALUE)) return false;

    float2 centerPx;
    if (!camera.project_world_to_screen(worldPos, centerPx)) return false;
    if (!is_screen_position_inside_viewport(centerPx, camera.params.viewport)) return false;

    float2 samplePx = nearest_circle_sample_to_viewport_center(centerPx, circleRadiusPx, camera.params.viewport);
    float3 rayDir = camera.screen_to_world_ray(samplePx);
    float3 toCenter = worldPos - camera.position_world();
    supportRadius = length(cross(toCenter, rayDir));
    return isfinite(supportRadius) && supportRadius > SMALL_VALUE;
}
