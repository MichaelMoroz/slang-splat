import "gaussian_types.slang";
import "gaussian_math_utils.slang";

void clear_silhouette_points_local5(out float3 points[MVEE_POINT_COUNT])
{
    [unroll]
    for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        points[i] = 0.0;
    }
}

bool generate_silhouette_points_local5(float3 roLocal, float eps, out float3 points[MVEE_POINT_COUNT])
{
    float d = length(roLocal);
    if (d <= 1.0 + eps)
    {
        clear_silhouette_points_local5(points);
        return false;
    }

    float3 n = roLocal / d;
    float3 circleCenter = n / d;
    float circleRadius = sqrt(max(1.0 - 1.0 / (d * d), 0.0));
    float3 u;
    float3 v;
    if (n.z < -0.9999999)
    {
        u = float3(0.0, -1.0, 0.0);
        v = float3(-1.0, 0.0, 0.0);
    }
    else
    {
        float a = 1.0 / (1.0 + n.z);
        float b = -n.x * n.y * a;
        u = float3(1.0 - n.x * n.x * a, b, -n.x);
        v = float3(b, 1.0 - n.y * n.y * a, -n.y);
    }
    if (!all(isfinite(u)) || !all(isfinite(v)))
    {
        clear_silhouette_points_local5(points);
        return false;
    }

    [unroll]
    for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        float theta = 6.28318530718 * (float(i) / float(MVEE_POINT_COUNT));
        float s;
        float c;
        sincos(theta, s, c);
        points[i] = circleCenter + circleRadius * (c * u + s * v);
    }
    return true;
}

bool fit_mvee_5pt_fixed_iters(
    float2 points[MVEE_POINT_COUNT],
    uint mveeIters,
    float eps,
    out float2 centerOut,
    out float radiusOut,
    out float3 conicOut
)
{
    centerOut = 0.0;
    radiusOut = 0.0;
    conicOut = 0.0;
    float2 pointsNorm[MVEE_POINT_COUNT];
    float2 normCenter = 0.0;
    [unroll]
    for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        normCenter += points[i];
    }
    normCenter *= (1.0 / float(MVEE_POINT_COUNT));

    float normScale = 0.0;
    [unroll]
    for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        float2 d = points[i] - normCenter;
        normScale = max(normScale, length(d));
    }
    if (normScale <= eps)
    {
        return false;
    }
    float invNormScale = 1.0 / normScale;
    [unroll]
    for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        pointsNorm[i] = (points[i] - normCenter) * invNormScale;
    }

    float weights[MVEE_POINT_COUNT];
    [unroll]
    for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        weights[i] = 1.0 / float(MVEE_POINT_COUNT);
    }

    [unroll]
    for (uint iter = 0; iter < MVEE_MAX_ITERS; iter++)
    {
        if (iter >= mveeIters)
        {
            break;
        }

        float3x3 xMat = 0.0;
        [unroll]
        for (uint i = 0; i < MVEE_POINT_COUNT; i++)
        {
            float3 q = float3(pointsNorm[i], 1.0);
            xMat += weights[i] * float3x3(
                q.x * q.x, q.x * q.y, q.x,
                q.y * q.x, q.y * q.y, q.y,
                q.x, q.y, 1.0
            );
        }

        float3x3 xInv = 0.0;
        if (!inverse3x3(xMat, eps, xInv))
        {
            return false;
        }

        uint maxIdx = 0;
        float maxVal = -1.0;
        [unroll]
        for (uint i = 0; i < MVEE_POINT_COUNT; i++)
        {
            float3 q = float3(pointsNorm[i], 1.0);
            float3 xq = mul3x3(xInv, q);
            float m = dot(q, xq);
            if (m > maxVal)
            {
                maxVal = m;
                maxIdx = i;
            }
        }

        float denom = 3.0 * (maxVal - 1.0);
        if (denom <= eps)
        {
            break;
        }
        float alpha = saturate((maxVal - 3.0) / denom);
        [unroll]
        for (uint i = 0; i < MVEE_POINT_COUNT; i++)
        {
            weights[i] *= (1.0 - alpha);
        }
        weights[maxIdx] += alpha;
    }

    float2 center = 0.0;
    [unroll]
    for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        center += weights[i] * pointsNorm[i];
    }

    float s00 = 0.0;
    float s01 = 0.0;
    float s11 = 0.0;
    [unroll]
    for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        float2 d = pointsNorm[i] - center;
        s00 += weights[i] * d.x * d.x;
        s01 += weights[i] * d.x * d.y;
        s11 += weights[i] * d.y * d.y;
    }
    s00 += eps;
    s11 += eps;

    float detS = s00 * s11 - s01 * s01;
    if (detS <= eps)
    {
        return false;
    }

    float a00 = 0.5 * s11 / detS;
    float a01 = -0.5 * s01 / detS;
    float a11 = 0.5 * s00 / detS;
    float trace = a00 + a11;
    float detA = a00 * a11 - a01 * a01;
    float disc = sqrt(max(0.25 * trace * trace - detA, 0.0));
    float l0 = max(0.5 * trace + disc, eps);
    float l1 = max(0.5 * trace - disc, eps);
    float axis0 = rsqrt(l0);
    float axis1 = rsqrt(l1);

    float maxQuad = 0.0;
    [unroll]
    for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        float2 d = pointsNorm[i] - center;
        float quad = a00 * d.x * d.x + 2.0 * a01 * d.x * d.y + a11 * d.y * d.y;
        maxQuad = max(maxQuad, quad);
    }
    float inflate = (maxQuad > 1.0) ? sqrt(maxQuad) : 1.0;
    float invInflate2 = 1.0 / max(inflate * inflate, eps);
    centerOut = normCenter + center * normScale;
    radiusOut = max(axis0, axis1) * inflate * normScale;
    conicOut = float3(a00, a01, a11) * (invInflate2 / max(normScale * normScale, eps));
    return all(isfinite(centerOut)) && isfinite(radiusOut) && radiusOut > 0.0;
}
