import "gaussian_types.slang";
import "gaussian_math_utils.slang";

bool generate_silhouette_points_local5(float3 roLocal, float eps, out float3 points[MVEE_POINT_COUNT])
{
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) points[i] = 0.0;

    float d = length(roLocal);
    if (d <= 1.0 + eps) return false;

    float3 n = roLocal / d;
    float3 circleCenter = n / d;
    float circleRadius = sqrt(max(1.0 - 1.0 / (d * d), 0.0));
    float3 u;
    float3 v;
    if (n.z < MVEE_SOUTH_POLE_THRESHOLD)
    {
        u = float3(0.0, -1.0, 0.0);
        v = float3(-1.0, 0.0, 0.0);
    } else {
        float a = 1.0 / (1.0 + n.z);
        float b = -n.x * n.y * a;
        u = float3(1.0 - n.x * n.x * a, b, -n.x);
        v = float3(b, 1.0 - n.y * n.y * a, -n.y);
    }
    if (!all(isfinite(u)) || !all(isfinite(v))) return false;

    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        float theta = MVEE_FULL_TURN_RADIANS * (float(i) / float(MVEE_POINT_COUNT));
        float s, c;
        sincos(theta, s, c);
        points[i] = circleCenter + circleRadius * (c * u + s * v);
    }
    return true;
}

bool fit_mvee_5pt_fixed_iters(
    float2 points[MVEE_POINT_COUNT],
    uint mveeIters,
    float eps,
    out MveeFitResult fitOut
)
{
    fitOut.clear();
    float2 pointsNorm[MVEE_POINT_COUNT];
    float2 normCenter = 0.0;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) normCenter += points[i];
    normCenter *= (1.0 / float(MVEE_POINT_COUNT));

    float normScale = 0.0;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) normScale = max(normScale, length(points[i] - normCenter));
    if (normScale <= eps) return false;
    float invNormScale = 1.0 / normScale;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) pointsNorm[i] = (points[i] - normCenter) * invNormScale;

    float weights[MVEE_POINT_COUNT];
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) weights[i] = 1.0 / float(MVEE_POINT_COUNT);

    [unroll] for (uint iter = 0; iter < MVEE_MAX_ITERS; iter++)
    {
        if (iter >= mveeIters) break;

        float3x3 xMat = 0.0;
        [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
        {
            float3 q = float3(pointsNorm[i], 1.0);
            xMat += weights[i] * float3x3(
                q.x * q.x, q.x * q.y, q.x,
                q.y * q.x, q.y * q.y, q.y,
                q.x, q.y, 1.0
            );
        }

        float3x3 xInv = 0.0;
        if (!inverse3x3(xMat, eps, xInv)) return false;

        uint maxIdx = 0;
        float maxVal = -1.0;
        [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
        {
            float3 q = float3(pointsNorm[i], 1.0);
            float3 xq = mul(xInv, q);
            float m = dot(q, xq);
            if (m > maxVal)
            {
                maxVal = m;
                maxIdx = i;
            }
        }

        float denom = MVEE_AUGMENTED_DIMENSION * (maxVal - 1.0);
        if (denom <= eps) break;
        float alpha = saturate((maxVal - MVEE_AUGMENTED_DIMENSION) / denom);
        [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) weights[i] *= (1.0 - alpha);
        weights[maxIdx] += alpha;
    }

    float2 center = 0.0;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) center += weights[i] * pointsNorm[i];

    float3 secondMoment = 0.0;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        float2 d = pointsNorm[i] - center;
        secondMoment += weights[i] * float3(d.x * d.x, d.x * d.y, d.y * d.y);
    }
    secondMoment += float3(eps, 0.0, eps);

    float detS = secondMoment.x * secondMoment.z - secondMoment.y * secondMoment.y;
    if (detS <= eps) return false;

    float3 conicNorm = 0.5 * float3(secondMoment.z, -secondMoment.y, secondMoment.x) / detS;
    float trace = conicNorm.x + conicNorm.z;
    float detA = conicNorm.x * conicNorm.z - conicNorm.y * conicNorm.y;
    float disc = sqrt(max(0.25 * trace * trace - detA, 0.0));
    float l0 = max(0.5 * trace + disc, eps);
    float l1 = max(0.5 * trace - disc, eps);
    float axis0 = rsqrt(l0);
    float axis1 = rsqrt(l1);

    float maxQuad = 0.0;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        float2 d = pointsNorm[i] - center;
        float quad = dot(float3(d.x * d.x, d.x * d.y, d.y * d.y), float3(conicNorm.x, 2.0 * conicNorm.y, conicNorm.z));
        maxQuad = max(maxQuad, quad);
    }
    float inflate = (maxQuad > 1.0) ? sqrt(maxQuad) : 1.0;
    float invInflate2 = 1.0 / max(inflate * inflate, eps);
    fitOut.centerPx = normCenter + center * normScale;
    fitOut.radiusPx = max(axis0, axis1) * inflate * normScale;
    fitOut.conic = conicNorm * (invInflate2 / max(normScale * normScale, eps));
    return all(isfinite(fitOut.centerPx)) && isfinite(fitOut.radiusPx) && fitOut.radiusPx > 0.0;
}
