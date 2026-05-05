struct AdamParams
{
    float beta1;
    float beta2;
    uint stepIndex;
};

static const float ADAM_EPSILON = 1e-8;

struct OptimizerRuntimeParams
{
    float gradComponentClip;
    float gradNormClip;
    float maxUpdate;
    float hugeValue;
};

struct OptimizerParamSettings
{
    float lr;
    float gradClipAbs;
    float gradNormClip;
    float valueMin;
    float valueMax;
    uint groupStart;
    uint groupSize;
    uint reserved;
    float regularizeTowardValue;
    float regularizeTowardWeight;
    float regularizeL1Weight;
    uint reserved2;
};

static const uint ADAM_GROUP_SIZE = 256u;