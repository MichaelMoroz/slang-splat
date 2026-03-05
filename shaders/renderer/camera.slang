static const float CAMERA_EPS_SMALL = 1e-6;

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
        float invDepth = 1.0 / max(cameraPos.z, CAMERA_EPS_SMALL);
        return cameraPos.xy * (params.focalPixels * invDepth) + params.principalPoint;
    }

    bool project_camera_to_screen(float3 cameraPos, out float2 screenPos)
    {
        if (cameraPos.z <= CAMERA_EPS_SMALL)
        {
            screenPos = 0.0;
            return false;
        }

        float2 uv = cameraPos.xy / cameraPos.z;
        float r2 = dot(uv, uv);
        float distortion = 1.0 + params.projDistortionK1 * r2 + params.projDistortionK2 * r2 * r2;
        uv *= distortion;
        screenPos = uv * params.focalPixels + params.principalPoint;
        return all(isfinite(screenPos));
    }

    bool project_world_to_screen(float3 worldPos, out float2 screenPos)
    {
        return project_camera_to_screen(world_point_to_camera(worldPos), screenPos);
    }

    float3 screen_to_world(float2 screenPos, float depth)
    {
        float2 focalSafe = max(params.focalPixels, float2(CAMERA_EPS_SMALL));
        float depthSafe = max(depth, CAMERA_EPS_SMALL);
        float2 uv = (screenPos - params.principalPoint) / focalSafe;
        return camera_point_to_world(float3(uv * depthSafe, depthSafe));
    }

    float3 screen_to_world_ray(float2 screenPos)
    {
        float2 focalSafe = max(params.focalPixels, float2(CAMERA_EPS_SMALL));
        float2 uv = (screenPos - params.principalPoint) / focalSafe;
        return normalize(camera_to_world(float3(uv, 1.0)));
    }

    float camera_depth(float3 worldPos)
    {
        return world_point_to_camera(worldPos).z;
    }

    bool is_depth_in_range(float depthValue)
    {
        return depthValue > CAMERA_EPS_SMALL && depthValue >= params.nearDepth && depthValue <= params.farDepth;
    }

    float3 position_world()
    {
        return params.camPos;
    }
};
