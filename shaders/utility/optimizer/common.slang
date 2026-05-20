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
    float maxUpdate;
    float hugeValue;
};

struct OptimizerParamSettings
{
    float lr;
    float gradClipAbs;
    float valueMin;
    float valueMax;
    float regularizeTowardValue;
    float regularizeTowardWeight;
    float regularizeL1Weight;
};

static const uint ADAM_GROUP_SIZE = 256u;