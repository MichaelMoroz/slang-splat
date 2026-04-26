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
static const uint PARAM_SH_FIRST = 10u;
static const uint SH_COEFF_COUNT = 16u;
static const uint SH_PARAM_STRIDE = 3u;
static const uint PARAM_SH0_R = PARAM_SH_FIRST + 0u;
static const uint PARAM_SH0_G = PARAM_SH_FIRST + 1u;
static const uint PARAM_SH0_B = PARAM_SH_FIRST + 2u;
static const uint PARAM_SH1_X_R = PARAM_SH_FIRST + 3u;
static const uint PARAM_SH1_X_G = PARAM_SH_FIRST + 4u;
static const uint PARAM_SH1_X_B = PARAM_SH_FIRST + 5u;
static const uint PARAM_SH1_Y_R = PARAM_SH_FIRST + 6u;
static const uint PARAM_SH1_Y_G = PARAM_SH_FIRST + 7u;
static const uint PARAM_SH1_Y_B = PARAM_SH_FIRST + 8u;
static const uint PARAM_SH1_Z_R = PARAM_SH_FIRST + 9u;
static const uint PARAM_SH1_Z_G = PARAM_SH_FIRST + 10u;
static const uint PARAM_SH1_Z_B = PARAM_SH_FIRST + 11u;
static const uint PARAM_RAW_OPACITY = PARAM_SH_FIRST + SH_COEFF_COUNT * SH_PARAM_STRIDE;
static const uint TRAINABLE_PARAM_COUNT = PARAM_RAW_OPACITY + 1u;

uint sh_param_id(uint coeffIndex, uint channelIndex)
{
    return PARAM_SH_FIRST + coeffIndex * SH_PARAM_STRIDE + channelIndex;
}

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

vector<float, N> load_splat_param_vector<let N : int>(StructuredBuffer<float> params, uint splatCount, uint firstParamId, uint splatId)
{
    vector<float, N> value = 0.0;
    [unroll] for (int i = 0; i < N; i++)
    {
        value[i] = load_splat_param(params, splatCount, firstParamId + uint(i), splatId);
    }
    return value;
}

vector<float, N> load_splat_param_vector<let N : int>(RWStructuredBuffer<float> params, uint splatCount, uint firstParamId, uint splatId)
{
    vector<float, N> value = 0.0;
    [unroll] for (int i = 0; i < N; i++)
    {
        value[i] = load_splat_param(params, splatCount, firstParamId + uint(i), splatId);
    }
    return value;
}

void store_splat_param(RWStructuredBuffer<float> params, uint splatCount, uint paramId, uint splatId, float value)
{
    params[splat_param_index(splatCount, paramId, splatId)] = value;
}

void store_splat_param_vector<let N : int>(RWStructuredBuffer<float> params, uint splatCount, uint firstParamId, uint splatId, vector<float, N> value)
{
    [unroll] for (int i = 0; i < N; i++)
    {
        store_splat_param(params, splatCount, firstParamId + uint(i), splatId, value[i]);
    }
}
