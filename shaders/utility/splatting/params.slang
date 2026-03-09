static const uint PARAM_POSITION_X = 0u;
static const uint PARAM_POSITION_Y = 1u;
static const uint PARAM_POSITION_Z = 2u;
static const uint PARAM_SCALE_X = 3u;
static const uint PARAM_SCALE_Y = 4u;
static const uint PARAM_SCALE_Z = 5u;
static const uint PARAM_ROTATION_X = 6u;
static const uint PARAM_ROTATION_Y = 7u;
static const uint PARAM_ROTATION_Z = 8u;
static const uint PARAM_ROTATION_W = 9u;
static const uint PARAM_COLOR_R = 10u;
static const uint PARAM_COLOR_G = 11u;
static const uint PARAM_COLOR_B = 12u;
static const uint PARAM_RAW_OPACITY = 13u;
static const uint TRAINABLE_PARAM_COUNT = 14u;

uint splat_param_index(uint splatCount, uint paramId, uint splatId)
{
    return paramId * splatCount + splatId;
}

float load_splat_param(StructuredBuffer<float> params, uint splatCount, uint paramId, uint splatId)
{
    return params[splat_param_index(splatCount, paramId, splatId)];
}

float load_splat_param(RWStructuredBuffer<float> params, uint splatCount, uint paramId, uint splatId)
{
    return params[splat_param_index(splatCount, paramId, splatId)];
}

void store_splat_param(RWStructuredBuffer<float> params, uint splatCount, uint paramId, uint splatId, float value)
{
    params[splat_param_index(splatCount, paramId, splatId)] = value;
}
