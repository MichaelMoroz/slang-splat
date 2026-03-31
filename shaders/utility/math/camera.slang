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
    float projDistortionK1;
    float projDistortionK2;
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

    float2 distort_normalized(float2 uv)
    {
        float r2 = dot(uv, uv);
        return uv * (1.0 + params.projDistortionK1 * r2 + params.projDistortionK2 * r2 * r2);
    }

    float2 undistort_normalized(float2 uvDistorted)
    {
        if (all(abs(float2(params.projDistortionK1, params.projDistortionK2)) <= SMALL_VALUE)) return uvDistorted;

        float radiusDistorted = length(uvDistorted);
        if (radiusDistorted <= SMALL_VALUE) return uvDistorted;

        float radius = radiusDistorted;
        [unroll] for (uint iter = 0u; iter < CAMERA_UNDISTORT_NEWTON_ITERS; iter++)
        {
            float r2 = radius * radius;
            float r4 = r2 * r2;
            float deriv = 1.0 + 3.0 * params.projDistortionK1 * r2 + 5.0 * params.projDistortionK2 * r4;
            if (abs(deriv) <= SMALL_VALUE) break;

            float nextRadius = radius - (radius * (1.0 + params.projDistortionK1 * r2 + params.projDistortionK2 * r4) - radiusDistorted) / deriv;
            if (!isfinite(nextRadius) || nextRadius < 0.0) break;
            radius = nextRadius;
        }
        return uvDistorted * (radius / max(radiusDistorted, SMALL_VALUE));
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
