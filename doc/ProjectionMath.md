# Projection Math

This note defines the projection formulas used for finite-support anisotropic Gaussians and records the algebra used in the gradient experiments. The emphasis here is the forward geometry and the exact matrix relationships between the 3D ellipsoid, its projected 2D ellipse, and the alternative screen-space representations.

## 1. Objects and Coordinates

Let the Gaussian parameters be:

- center $p \in \mathbb{R}^3$
- log standard deviations $\ell \in \mathbb{R}^3$
- orientation quaternion $q$

The decoded standard deviations are

$$
\sigma = \exp(\ell).
$$

Let $r_s$ denote the total support multiplier used by the renderer,

$$
r_s = \text{radiusScale} \cdot \text{supportSigmaRadius},
$$

and let the support radii be

$$
s = r_s \, \sigma.
$$

After anisotropy clamping, the support ellipsoid in its local frame is

$$
E = \{x \in \mathbb{R}^3 : x^T S^{-2} x = 1\},
\qquad
S = \operatorname{diag}(s_x, s_y, s_z).
$$

Let $R \in SO(3)$ be the rotation induced by $q$. In world coordinates the ellipsoid is

$$
\{p + R S u : u^T u = 1\}.
$$

Two image coordinate systems are used below:

- normalized pinhole coordinates $u = (u, v)^T$
- pixel coordinates $x = (x, y)^T$

For an undistorted pinhole camera with focal lengths $(f_x, f_y)$ and principal point $p_0 = (c_x, c_y)^T$,

$$
x = F u + p_0,
\qquad
F = \operatorname{diag}(f_x, f_y).
$$

## 2. Sampled Silhouette Projector

The sampled projector computes the exact silhouette of the support ellipsoid under the full camera model. Distortion is handled by projecting sampled outline points through the same camera mapping used elsewhere in the renderer.

### 2.1 Reduction to the Unit Sphere

Let $c_w$ be the camera position in world space. Define the support-local camera origin by

$$
o = S^{-1} R^T (c_w - p).
$$

In this coordinate system the ellipsoid becomes the unit sphere

$$
\|z\|^2 = 1.
$$

If $\|o\| > 1$, the silhouette is the tangent circle of the sphere seen from $o$.

### 2.2 Tangent Circle Geometry

Write

$$
d = \|o\|,
\qquad
v = \frac{o}{d}.
$$

The tangent circle lies in the plane

$$
v^T z = \frac{1}{d}
$$

with center and radius

$$
c_t = \frac{v}{d},
\qquad
r_t = \sqrt{1 - \frac{1}{d^2}}.
$$

Choose orthonormal vectors $u_1, u_2$ such that

$$
u_1^T v = 0,
\qquad
u_2^T v = 0,
\qquad
u_1^T u_2 = 0.
$$

The sampled outline points on the unit sphere are

$$
z_i = c_t + r_t (\cos \theta_i \, u_1 + \sin \theta_i \, u_2),
\qquad
	heta_i = \frac{2\pi i}{5},
\quad
i = 0, \dots, 4.
$$

Mapping back to world space gives

$$
w_i = p + R S z_i.
$$

Each $w_i$ is then projected through the full camera model, including distortion, to produce five screen points.

### 2.3 Five-Point Conic Fit

The screen-space ellipse is represented by the quadratic equation

$$
A x^2 + 2 B x y + C y^2 + D x + E y = 1.
$$

Equivalently, in homogeneous coordinates $\tilde{x} = (x, y, 1)^T$,

$$
	ilde{x}^T C \tilde{x} = 0,
\qquad
C =
\begin{bmatrix}
A & B & D/2 \\
B & C & E/2 \\
D/2 & E/2 & -1
\end{bmatrix}.
$$

The implementation first computes a screen-space bounding box from the five outline samples,

$$
b = \frac{1}{2}(x_{\min} + x_{\max}),
\qquad
h = \max\!\left( \frac{1}{2}(x_{\max} - x_{\min}), \varepsilon \mathbf{1} \right),
$$

and then normalizes each outline point by

$$
\hat{x}_i = \frac{x_i - b}{h}.
$$

This is the first normalization stage. The second stage is the renormalized conic solve used by `solve_conic_renorm`, which does not solve the full $5 \times 5$ system directly. Instead, it maps two normalized points to canonical coordinates.

Let the normalized points be $p_i = (x_i, y_i)^T$ for $i = 0, \dots, 4$. The shader defines

$$
s_x = x_0 - x_1,
\qquad
s_y = y_1 - y_0,
\qquad
o_x = x_1,
\qquad
o_y = y_0,
$$

and uses the affine map

$$
u = \frac{x - o_x}{s_x},
\qquad
v = \frac{y - o_y}{s_y}.
$$

Under this map,

$$
p_0 \mapsto (1, 0),
\qquad
p_1 \mapsto (0, 1).
$$

Write the conic in renormalized coordinates as

$$
Au^2 + 2Buv + Cv^2 + Du + Ev = 1.
$$

Because $(u, v) = (1, 0)$ and $(u, v) = (0, 1)$ lie on the conic, the coefficients satisfy

$$
D = -(A + 1),
\qquad
E = -(C + 1).
$$

Therefore only $(A, B, C)$ remain unknown. For each of the remaining three points,

$$
(u_k, v_k) = \left( \frac{x_k - o_x}{s_x}, \frac{y_k - o_y}{s_y} \right),
\qquad
k \in \{2, 3, 4\},
$$

substitution gives

$$
(u_k^2 - u_k) A + 2 u_k v_k B + (v_k^2 - v_k) C = u_k + v_k - 1.
$$

The shader therefore solves the $3 \times 3$ system

$$
M
\begin{bmatrix}
A \\
B \\
C
\end{bmatrix}
=
r,
$$

where

$$
M =
\begin{bmatrix}
u_2^2 - u_2 & 2 u_2 v_2 & v_2^2 - v_2 \\
u_3^2 - u_3 & 2 u_3 v_3 & v_3^2 - v_3 \\
u_4^2 - u_4 & 2 u_4 v_4 & v_4^2 - v_4
\end{bmatrix}.
$$

and

$$
r =
\begin{bmatrix}
u_2 + v_2 - 1 \\
u_3 + v_3 - 1 \\
u_4 + v_4 - 1
\end{bmatrix}.
$$

`solve_conic_renorm` performs explicit forward elimination without pivoting on this matrix, followed by back-substitution. After recovering $(A, B, C)$, it reconstructs

$$
D = -(A + 1),
\qquad
E = -(C + 1).
$$

This yields the renormalized conic in $(u, v)$ coordinates.

The shader then back-converts this conic to the original normalized screen coordinates. Since

$$
u = \alpha_x x + \beta_x,
\qquad
v = \alpha_y y + \beta_y,
\qquad
\alpha_x = \frac{1}{s_x},
\quad
\alpha_y = \frac{1}{s_y},
\quad
\beta_x = -\frac{o_x}{s_x},
\quad
\beta_y = -\frac{o_y}{s_y},
$$

substitution gives the coefficients

$$
A' = A \alpha_x^2,
\qquad
B' = B \alpha_x \alpha_y,
\qquad
C' = C \alpha_y^2,
$$

$$
D' = -2 A' o_x - 2 B' o_y + D \alpha_x,
$$

$$
E' = -2 B' o_x - 2 C' o_y + E \alpha_y,
$$

$$
F' = A' o_x^2 + 2 B' o_x o_y + C' o_y^2 - D \alpha_x o_x - E \alpha_y o_y + 1.
$$

The internal solve produces a conic with right-hand side $-F'$. The production convention used by the rest of the code is

$$
A'' x^2 + 2 B'' x y + C'' y^2 + D'' x + E'' y = 1,
$$

so the final coefficients stored by `solve_conic_renorm` are

$$
(A'', B'', C'', D'', E'') = -\frac{1}{F'} (A', B', C', D', E').
$$

After solving in normalized coordinates, the pixel-space conic is recovered by affine transport. If the homogeneous transform is

$$
\hat{\tilde{x}} = H \tilde{x},
$$

then the conic transforms as

$$
C = H^T \hat{C} H.
$$

To recenter the quadratic form, write the pixel-space conic coefficients as

$$
q(x, y) = A x^2 + 2 B x y + C y^2 + D x + E y - 1.
$$

Let

$$
K =
\begin{bmatrix}
A & B \\
B & C
\end{bmatrix},
\qquad
d =
\frac{1}{2}
\begin{bmatrix}
D \\
E
\end{bmatrix}.
$$

For the sampled path, the implementation does not apply the generic $-K^{-1} d$ formula directly. With the convention

$$
q(x, y) = A x^2 + 2 B x y + C y^2 + D x + E y - 1,
$$

the stationary point is computed in the shader as

$$
c = \frac{1}{2}
\begin{bmatrix}
\dfrac{B E - C D}{A C - B^2} \\
\dfrac{B D - A E}{A C - B^2}
\end{bmatrix}.
$$

Define

$$
K c =
\begin{bmatrix}
A c_x + B c_y \\
B c_x + C c_y
\end{bmatrix},
$$

and then the centered normalization factor used by the shader is

$$
\gamma = 1 + c^T K c.
$$

The centered conic stored by the sampled path is

$$
\widetilde{K} = \frac{1}{\gamma}
\begin{bmatrix}
A & B \\
B & C
\end{bmatrix}.
$$

If $\tau = \widetilde{K}_{11} + \widetilde{K}_{22}$ and $\delta = \widetilde{K}_{11} \widetilde{K}_{22} - \widetilde{K}_{12}^2$, then the eigenvalues are

$$
\mu_{\pm} = \frac{\tau}{2} \pm \sqrt{\frac{\tau^2}{4} - \delta},
$$

and the sampled path stores

$$
	ext{radiusPx} = \max\!\left( \frac{1}{\sqrt{\mu_+}} h_x, \frac{1}{\sqrt{\mu_-}} h_y \right),
$$

where $(h_x, h_y)$ are the half-extents of the outer bounding-box normalization.

## 3. Direct Undistorted Projector

For an undistorted pinhole camera the projected silhouette can be constructed without outline sampling.

### 3.1 Camera-Space Ellipsoid

Let the ellipsoid center in camera space be $m_c \in \mathbb{R}^3$. Let $a_x, a_y, a_z$ denote the rotated support axes in camera space. The camera-space support covariance is

$$
\Sigma_c = s_x^2 a_x a_x^T + s_y^2 a_y a_y^T + s_z^2 a_z a_z^T.
$$

Equivalently, if $A = R_c S$ is the linear map from the unit sphere to the camera-space ellipsoid frame, then

$$
\Sigma_c = A A^T.
$$

The ellipsoid is

$$
(y - m_c)^T \Sigma_c^{-1} (y - m_c) = 1.
$$

### 3.2 Dual Conic in Normalized Image Coordinates

For central projection from the camera origin, the silhouette in normalized image homogeneous coordinates $\tilde{u} = (u, v, 1)^T$ is represented by a dual conic

$$
Q^* = \Sigma_c - m_c m_c^T.
$$

This matrix is defined only up to a nonzero scalar multiple. Multiplying $Q^*$ by any $\lambda \neq 0$ leaves the represented ellipse unchanged.

The implementation may apply a depth-dependent rescaling before constructing $Q^*$ to reduce dynamic range. Since the conic is projective, that rescaling does not change the represented ellipse.

### 3.3 Affine Extraction from the Dual Conic

Partition the symmetric $3 \times 3$ dual conic as

$$
Q^* =
\begin{bmatrix}
G & h \\
h^T & h_{33}
\end{bmatrix},
$$

where $G \in \mathbb{R}^{2 \times 2}$ and $h \in \mathbb{R}^2$.

The normalized-image ellipse can be written in covariance form as

$$
(u - c_{uv})^T \Sigma_{uv}^{-1} (u - c_{uv}) = 1.
$$

The corresponding dual conic is

$$
Q^* \sim
\begin{bmatrix}
\Sigma_{uv} + c_{uv} c_{uv}^T & c_{uv} \\
c_{uv}^T & 1
\end{bmatrix}.
$$

Matching the block entries gives, up to the common projective scale,

$$
c_{uv} = \frac{h}{h_{33}},
$$

and

$$
\Sigma_{uv} = G - \frac{h h^T}{h_{33}}.
$$

The implementation first enforces the sign convention $h_{33} < 0$ by negating the entire dual conic when necessary. Then it defines

$$
s = -h_{33},
$$

and uses the exact formulas

$$
c_{uv} = \frac{h}{h_{33}},
\qquad
\Sigma_{uv} = \frac{1}{s} \left( G - \frac{h h^T}{h_{33}} \right).
$$

This keeps the extraction in normalized image space and avoids mixing focal-length and principal-point terms into the homogeneous conic algebra prematurely.

### 3.4 Transport to Pixel Coordinates

Let

$$
F = \operatorname{diag}(f_x, f_y).
$$

The affine map from normalized image space to pixel space is

$$
x = F u + p_0.
$$

Hence the center and covariance transport as

$$
c_{px} = F c_{uv} + p_0,
$$

$$
\Sigma_{px} = F \Sigma_{uv} F.
$$

Writing

$$
\Sigma_{px} =
\begin{bmatrix}
\sigma_{xx} & \sigma_{xy} \\
\sigma_{xy} & \sigma_{yy}
\end{bmatrix},
$$

the shader computes

$$
\sigma_{xx} = f_x^2 \Sigma_{uv, 11},
\qquad
\sigma_{xy} = f_x f_y \Sigma_{uv, 12},
\qquad
\sigma_{yy} = f_y^2 \Sigma_{uv, 22}.
$$

The centered inverse conic used by the rasterizer is then formed directly from the $2 \times 2$ covariance determinant

$$
\det(\Sigma_{px}) = \sigma_{xx} \sigma_{yy} - \sigma_{xy}^2,
$$

namely

$$
\widetilde{K} = \Sigma_{px}^{-1} =
\frac{1}{\det(\Sigma_{px})}
\begin{bmatrix}
\sigma_{yy} & -\sigma_{xy} \\
-\sigma_{xy} & \sigma_{xx}
\end{bmatrix}.
$$

The principal values of $\widetilde{K}$ are

$$
\mu_{\pm} = \frac{\widetilde{K}_{11} + \widetilde{K}_{22}}{2}
\pm
\frac{1}{2}
\sqrt{(\widetilde{K}_{11} - \widetilde{K}_{22})^2 + 4 \widetilde{K}_{12}^2},
$$

and the direct path stores the screen-space radius as

$$
	ext{radiusPx} = \max\!\left( \frac{1}{\sqrt{\mu_+}}, \frac{1}{\sqrt{\mu_-}} \right).
$$

The center written to the ellipse is

$$
c_{px} = F c_{uv} + p_0.
$$

### 3.5 Inverse-Conic Representation

The quadratic ellipse representation used by the raster stage is formed from the inverse covariance. If

$$
\Sigma_{px} =
\begin{bmatrix}
a & b \\
b & c
\end{bmatrix},
$$

then

$$
\Sigma_{px}^{-1}
=
\frac{1}{ac - b^2}
\begin{bmatrix}
c & -b \\
-b & a
\end{bmatrix}.
$$

The stored conic coefficients are the entries of this inverse matrix, interpreted as the centered quadratic form

$$
(x - c_{px})^T \Sigma_{px}^{-1} (x - c_{px}) = 1.
$$

## 4. Relation Between the Representations

The direct path naturally passes through the following sequence:

$$
(p, \ell, q)
\;\longrightarrow\;
(m_c, \Sigma_c)
\;\longrightarrow\;
Q^*
\;\longrightarrow\;
(c_{uv}, \Sigma_{uv})
\;\longrightarrow\;
(c_{px}, \Sigma_{px})
\;\longrightarrow\;
(c_{px}, \Sigma_{px}^{-1}, \lambda_+).
$$

The sampled path reaches the same final centered-ellipse form through a different chain:

$$
(p, \ell, q)
\;\longrightarrow\;
	ext{tangent circle on unit sphere}
\;\longrightarrow\;
	ext{five projected outline points}
\;\longrightarrow\;
	ext{screen-space conic fit}.
$$

For distorted cameras, the sampled path remains exact because the five outline points are projected by the full camera model before the conic fit.

## 5. Conditioning Notes

The experiments in [temp/projection_grad_bench.slang](../temp/projection_grad_bench.slang) and [temp/projection_gradient_investigation.py](../temp/projection_gradient_investigation.py) compare autodiff gradients against float64 finite differences. The main representation-level observations are:

- Stages through $(c_{uv}, \Sigma_{uv})$ and $(c_{px}, \Sigma_{px})$ are comparatively well-conditioned.
- The map $\Sigma_{px} \mapsto \Sigma_{px}^{-1}$ introduces the factor $(ac - b^2)^{-1}$, so sensitivity grows when the covariance determinant is small.
- The major-variance formula $\lambda_+$ is less sensitive than the inverse-conic conversion, but still contains a square-root discriminant term.

In particular, for

$$
\Delta = (\sigma_{xx} - \sigma_{yy})^2 + 4 \sigma_{xy}^2,
$$

the derivative of $\lambda_+$ contains $\Delta^{-1/2}$ through the square-root term, while the inverse covariance contains $(ac - b^2)^{-2}$ in its derivatives. The experiments were set up to distinguish error entering before and after these operations.

## 6. Ray Density From Camera-Space Mean And Covariance

For raster evaluation along a camera ray, the splat can be represented by:

$$
m \in \mathbb{R}^3,
\qquad
\Sigma \in \mathbb{R}^{3 \times 3},
\qquad
\Sigma = \Sigma^T,
\qquad
\Sigma \succ 0,
$$

where $m$ is the splat center in camera coordinates and $\Sigma$ is the camera-space covariance. Let

$$
P = \Sigma^{-1}.
$$

For a ray through the camera origin with direction $d$, the ray is

$$
r(t) = t d.
$$

The Gaussian exponent along the ray is

$$
q(t) = (t d - m)^T P (t d - m).
$$

Expanding this quadratic in $t$ gives

$$
q(t) = a t^2 - 2 b t + c,
$$

with

$$
a = d^T P d,
\qquad
b = d^T P m,
\qquad
c = m^T P m.
$$

The minimizing depth on the ray is therefore

$$
t_* = \frac{b}{a} = \frac{d^T P m}{d^T P d}.
$$

The minimum Mahalanobis distance achieved by the ray is

$$
q_* = c - \frac{b^2}{a}
=
m^T P m - \frac{(d^T P m)^2}{d^T P d}.
$$

If the renderer evaluates the Gaussian at the closest point on the ray, the density is

$$
\rho_{\text{closest}}(d) = \rho_0 \exp\!\left(-\frac{1}{2} q_*\right),
$$

where $\rho_0$ is the splat amplitude or fused opacity term.

If instead the infinite Gaussian is integrated exactly along the ray, then

$$
\int_{-\infty}^{\infty} \rho_0 \exp\!\left(-\frac{1}{2} q(t)\right) \, dt
=
\rho_0 \sqrt{\frac{2\pi}{a}} \exp\!\left(-\frac{1}{2} q_*\right).
$$

For a finite-support splat with support radius $r$ in Mahalanobis units, the ray intersects the support if and only if

$$
q_* \le r^2.
$$

This representation depends only on $(m, \Sigma)$ and one scalar amplitude term. No explicit decomposition back into position, log-scale, and quaternion is required for evaluation along a ray.

### 6.1 Example Slang Code

The following example evaluates both the closest-point density and the exact line integral using only the camera-space mean, inverse covariance, and amplitude:

```slang
struct CameraGaussianSigma
{
	float3 mean;
	float3x3 invCovariance;
	float amplitude;
};

struct RayGaussianEval
{
	float closestDensity;
	float integratedDensity;
	float rayDepth;
	float mahalanobisMin;
};

RayGaussianEval eval_gaussian_from_sigma(CameraGaussianSigma gaussian, float3 rayDir)
{
	RayGaussianEval result;
	result.closestDensity = 0.0;
	result.integratedDensity = 0.0;
	result.rayDepth = 0.0;
	result.mahalanobisMin = 0.0;

	float3 invCovRay = mul(gaussian.invCovariance, rayDir);
	float3 invCovMean = mul(gaussian.invCovariance, gaussian.mean);
	float a = dot(rayDir, invCovRay);
	if (a <= SMALL_VALUE || !isfinite(a)) return result;

	float b = dot(rayDir, invCovMean);
	float c = dot(gaussian.mean, invCovMean);
	float tStar = b / a;
	float qStar = max(c - b * b / a, 0.0);
	float closestDensity = gaussian.amplitude * exp(-0.5 * qStar);
	float integratedDensity = closestDensity * sqrt(2.0 * PI / a);

	result.closestDensity = closestDensity;
	result.integratedDensity = integratedDensity;
	result.rayDepth = tStar;
	result.mahalanobisMin = qStar;
	return result;
}
```

If a finite support cutoff is used, then the result can be rejected with

```slang
bool ray_intersects_support(float mahalanobisMin, float supportRadius)
{
	return mahalanobisMin <= supportRadius * supportRadius;
}
```

If the ray direction is normalized, then `rayDepth` is the Euclidean distance from the camera origin to the closest point on the ray. If the direction is not normalized, `rayDepth` is the affine ray parameter in the chosen ray basis.

## 7. Camera-Center-Direction Decomposition

For a camera-space Gaussian, a natural exact decomposition is obtained by aligning coordinates with the center direction seen from the camera origin.

Let

$$
m \in \mathbb{R}^3,
\qquad
\Sigma \in \mathbb{R}^{3 \times 3},
\qquad
\Sigma = \Sigma^T,
\qquad
\Sigma \succ 0.
$$

Assume $m \neq 0$, and define the view-aligned longitudinal axis

$$
r = \|m\|,
\qquad
n = \frac{m}{r}.
$$

Choose any orthonormal basis $b_1, b_2$ for the plane orthogonal to $n$, and collect it as

$$
B =
\begin{bmatrix}
b_1 & b_2
\end{bmatrix}
\in \mathbb{R}^{3 \times 2},
$$

so that

$$
B^T B = I,
\qquad
B^T n = 0.
$$

Define the full orthonormal change of basis

$$
U =
\begin{bmatrix}
b_1 & b_2 & n
\end{bmatrix}
\in SO(3).
$$

Then the mean in this basis is

$$
U^T m =
\begin{bmatrix}
0 \\
0 \\
r
\end{bmatrix}.
$$

The covariance in the same basis becomes a $2+1$ block matrix,

$$
\Sigma_{\text{view}} = U^T \Sigma U =
\begin{bmatrix}
\Sigma_{\perp} & k \\
k^T & \sigma_{\parallel}
\end{bmatrix},
$$

with

$$
\Sigma_{\perp} = B^T \Sigma B \in \mathbb{R}^{2 \times 2},
$$

$$
k = B^T \Sigma n \in \mathbb{R}^2,
$$

$$
\sigma_{\parallel} = n^T \Sigma n \in \mathbb{R}.
$$

To factor out distance explicitly, introduce normalized camera coordinates

$$
y = \frac{x}{r}.
$$

In these coordinates the Gaussian mean and covariance become

$$
\bar{m} = \frac{m}{r} = n,
\qquad
\bar{\Sigma} = \frac{1}{r^2} \Sigma.
$$

The normalized covariance in the same view-aligned basis is therefore

$$
\bar{\Sigma}_{\text{view}} = U^T \bar{\Sigma} U =
\begin{bmatrix}
\bar{\Sigma}_{\perp} & \bar{k} \\
\bar{k}^T & \bar{\sigma}_{\parallel}
\end{bmatrix},
$$

with

$$
\bar{\Sigma}_{\perp} = \frac{1}{r^2} \Sigma_{\perp},
\qquad
\bar{k} = \frac{1}{r^2} k,
\qquad
\bar{\sigma}_{\parallel} = \frac{1}{r^2} \sigma_{\parallel}.
$$

This yields the exact distance-normalized parameter split

$$
m \leftrightarrow (r, n),
\qquad
\Sigma \leftrightarrow r^2 (\bar{\Sigma}_{\perp}, \bar{k}, \bar{\sigma}_{\parallel}),
$$

or, if the basis construction rule for $B$ is fixed deterministically from $n$,

$$
(m, \Sigma)
\leftrightarrow
(r, n, \bar{\Sigma}_{\perp}, \bar{k}, \bar{\sigma}_{\parallel}).
$$

The inverse map is explicit:

$$
m = r n,
$$

$$
\Sigma = r^2 \left(
B \bar{\Sigma}_{\perp} B^T + B \bar{k} n^T + n \bar{k}^T B^T + \bar{\sigma}_{\parallel} n n^T
\right).
$$

Equivalently, the unnormalized blocks are recovered by

$$
\Sigma_{\perp} = r^2 \bar{\Sigma}_{\perp},
\qquad
k = r^2 \bar{k},
\qquad
\sigma_{\parallel} = r^2 \bar{\sigma}_{\parallel}.
$$

This decomposition is exact and recomposable. No information is lost. The distance dependence is isolated into the single scalar $r$, while all remaining shape terms are dimensionless in the normalized camera coordinates.

### 7.1 Interpretation of the Blocks

The normalized transverse covariance $\bar{\Sigma}_{\perp}$ describes angular or normalized-image spread in the plane orthogonal to the center direction $n$.

The scalar $\bar{\sigma}_{\parallel}$ is the normalized variance along the center-direction axis.

The vector $\bar{k}$ contains the normalized coupling between transverse directions and the longitudinal axis. If $\bar{k} = 0$, then the covariance ellipsoid is block-diagonal in the view-aligned basis.

The center itself decomposes into:

$$
m = r n,
$$

so the distance from the camera is isolated as $r$, while the viewing direction is isolated as $n$.

The normalized Gaussian itself is therefore

$$
(\bar{m}, \bar{\Sigma})
\leftrightarrow
(n, \bar{\Sigma}_{\perp}, \bar{k}, \bar{\sigma}_{\parallel}),
$$

and the original Gaussian is obtained by the single global rescaling $x = r y$.

### 7.2 Relation to Projection Sensitivity

For a fixed camera view, the parameters that most directly control the projected footprint are the transverse quantities. In particular, the local image Jacobian acts primarily on the transverse plane orthogonal to $n$.

To first order near the center ray, the normalized-image covariance is governed by the distance-normalized transverse block,

$$
\Sigma_{uv} \approx \bar{\Sigma}_{\perp}.
$$

Equivalently, in pixel coordinates,

$$
\Sigma_{px} \approx F \bar{\Sigma}_{\perp} F.
$$

Thus raw $\Sigma_{\perp}$ is not the right small-footprint quantity. The quantity that remains approximately stable under depth changes is $\bar{\Sigma}_{\perp} = \Sigma_{\perp} / r^2$.

The normalized longitudinal variance $\bar{\sigma}_{\parallel}$ affects the image footprint much more weakly. The normalized coupling term $\bar{k}$ affects how longitudinal variation mixes into transverse motion, so it is not projection-null, but its effect enters through off-axis coupling rather than pure in-plane spread.

This motivates the view-aligned split:

$$
	ext{projection-active part} = (n, \bar{\Sigma}_{\perp}),
\qquad
	ext{projection-secondary part} = (r, \bar{k}, \bar{\sigma}_{\parallel}),
$$

with the understanding that this is an exact decomposition of the 3D Gaussian, not an exact decomposition into projection-invariant and projection-active parameters. It is exact as a reparameterization, and only approximate as a statement about which terms matter most to the current screen-space projection.

For a sufficiently small splat, an approximate projected density model therefore needs only

$$
(n, \bar{\Sigma}_{\perp}, \rho_0),
$$

where $\rho_0$ is the amplitude term. If exact line-integrated amplitude or depth variation is needed, the remaining normalized longitudinal terms $(\bar{k}, \bar{\sigma}_{\parallel})$ must also be retained.

### 7.3 Example Slang Code

The distance-normalized decomposition can be implemented directly from a camera-space mean and covariance. The minimal approximate state stores only the center direction and the normalized transverse covariance.

```slang
struct DirectionalGaussianApprox
{
	float3 centerDir;
	float2x2 sigmaOrtho;
	float amplitude;
};

float3 pick_tangent_axis(float3 direction)
{
	return abs(direction.z) < 0.999
		? float3(0.0, 0.0, 1.0)
		: float3(0.0, 1.0, 0.0);
}

void build_direction_frame(float3 centerDir, out float3 tangentX, out float3 tangentY)
{
	float3 helperAxis = pick_tangent_axis(centerDir);
	tangentX = normalize(cross(helperAxis, centerDir));
	tangentY = cross(centerDir, tangentX);
}

DirectionalGaussianApprox decompose_directional_gaussian(float3 mean, float3x3 covariance, float amplitude)
{
	DirectionalGaussianApprox result;
	result.centerDir = float3(0.0, 0.0, 1.0);
	result.sigmaOrtho = 0.0;
	result.amplitude = amplitude;

	float radius = length(mean);
	if (radius <= SMALL_VALUE || !isfinite(radius)) return result;

	float3 centerDir = mean / radius;
	float3 tangentX;
	float3 tangentY;
	build_direction_frame(centerDir, tangentX, tangentY);

	float invRadius2 = 1.0 / (radius * radius);
	float3 sigmaTangentX = mul(covariance, tangentX);
	float3 sigmaTangentY = mul(covariance, tangentY);
	result.centerDir = centerDir;
	result.sigmaOrtho = float2x2(
		dot(tangentX, sigmaTangentX), dot(tangentX, sigmaTangentY),
		dot(tangentY, sigmaTangentX), dot(tangentY, sigmaTangentY)
	) * invRadius2;
	return result;
}
```

This code computes

$$
\bar{\Sigma}_{\perp} = \frac{1}{r^2} B^T \Sigma B,
$$

with $B = [\,b_1\;b_2\,] = [\,\text{tangentX}\;\text{tangentY}\,]$.

The approximate directional evaluation then intersects the ray with the normalized plane

$$
\Pi_n = \{ y : n^T y = 1 \},
$$

extracts the transverse coordinates, and evaluates the resulting $2$D Gaussian.

```slang
float2 project_ray_to_direction_frame(float3 centerDir, float3 rayDir)
{
	float3 tangentX;
	float3 tangentY;
	build_direction_frame(centerDir, tangentX, tangentY);
	float denom = max(dot(centerDir, rayDir), SMALL_VALUE);
	return float2(dot(tangentX, rayDir), dot(tangentY, rayDir)) / denom;
}

float eval_directional_gaussian(float3 centerDir, float2x2 sigmaOrtho, float amplitude, float3 rayDir)
{
	float det = sigmaOrtho[0][0] * sigmaOrtho[1][1] - sigmaOrtho[0][1] * sigmaOrtho[1][0];
	if (det <= SMALL_VALUE || !isfinite(det)) return 0.0;

	float invDet = 1.0 / det;
	float2x2 invSigmaOrtho = float2x2(
		 sigmaOrtho[1][1], -sigmaOrtho[0][1],
		-sigmaOrtho[1][0],  sigmaOrtho[0][0]
	) * invDet;
	float2 eta = project_ray_to_direction_frame(centerDir, rayDir);
	float2 quadVec = mul(invSigmaOrtho, eta);
	float quad = dot(eta, quadVec);
	return amplitude * exp(-0.5 * quad);
}
```

In the notation above,

$$
\eta = \frac{B^T d}{n^T d},
$$

and the approximation is

$$
\rho(d) \approx \rho_0 \exp\!\left(-\frac{1}{2} \eta^T \bar{\Sigma}_{\perp}^{-1} \eta\right).
$$

### 7.4 Degenerate Case

If $\|m\|$ is very small, then the center-direction axis $n = m / \|m\|$ is ill-defined. In that case a fallback axis must be chosen, for example the camera forward direction, and the same block construction can then be applied relative to that fallback axis.

### 7.5 Exact Evaluation In Distance-Normalized Coordinates

Using normalized camera coordinates $y = x / r$, rays parallel to the center direction can be written as

$$
y(\tau) = B \eta + \tau n,
$$

where $\eta = \xi / r$ and $\tau = t / r$ are the normalized transverse and longitudinal coordinates.

In these variables, the Gaussian mean is always at

$$
\bar{m} = n,
$$

and the exponent depends only on the normalized covariance blocks

$$
\bar{\Sigma}_{\text{view}} =
\begin{bmatrix}
\bar{\Sigma}_{\perp} & \bar{k} \\
\bar{k}^T & \bar{\sigma}_{\parallel}
\end{bmatrix}.
$$

Minimizing over $\tau$ yields

$$
q_*(\eta) = \eta^T \bar{\Sigma}_{\perp}^{-1} \eta,
$$

and the normalized peak depth is

$$
	au_*(\eta) = 1 + \bar{k}^T \bar{\Sigma}_{\perp}^{-1} \eta.
$$

The conditional longitudinal variance is

$$
\bar{s}_{\text{cond}}^2 = \bar{\sigma}_{\parallel} - \bar{k}^T \bar{\Sigma}_{\perp}^{-1} \bar{k}.
$$

These formulas are scale-free. The original dimensional values are recovered by multiplying longitudinal coordinates by $r$ and covariance entries by $r^2$.

### 7.6 Exact Evaluation For Rays Parallel To The Center Direction

For the special case of a ray bundle parallel to the center-direction axis $n$, the decomposition above gives an exact reduction to a transverse $2$D Gaussian.

Write the camera-space Gaussian in the view-aligned basis as

$$
U^T m =
\begin{bmatrix}
0 \\
0 \\
r
\end{bmatrix},
\qquad
U^T \Sigma U =
\begin{bmatrix}
\Sigma_{\perp} & k \\
k^T & \sigma_{\parallel}
\end{bmatrix}.
$$

Consider rays parameterized by a transverse offset $\xi \in \mathbb{R}^2$ and running parallel to $n$:

$$
x(t) = B \xi + t n,
\qquad
B = [\,b_1\;b_2\,].
$$

In the $(b_1, b_2, n)$ basis this ray is simply

$$
\begin{bmatrix}
\xi \\
t
\end{bmatrix}.
$$

The Gaussian exponent is therefore

$$
q(\xi, t) =
\begin{bmatrix}
\xi \\
t - r
\end{bmatrix}^T
\Sigma_{\text{view}}^{-1}
\begin{bmatrix}
\xi \\
t - r
\end{bmatrix}.
$$

Minimizing over $t$ yields an exact transverse exponent

$$
q_*(\xi) = \xi^T \Sigma_{\perp}^{-1} \xi.
$$

So the closest-point density across the parallel-ray bundle is

$$
\rho_{\text{closest}}(\xi)
=
\rho_0 \exp\!\left(-\frac{1}{2} \xi^T \Sigma_{\perp}^{-1} \xi\right).
$$

This shows that for parallel rays, the transverse density shape depends only on $\Sigma_{\perp}$.

The depth of the maximum along each ray is shifted by the coupling term $k$:

$$
t_*(\xi) = r + k^T \Sigma_{\perp}^{-1} \xi.
$$

Thus $k$ does not change the transverse Gaussian profile, but it does move the location of the peak along the ray family.

If the infinite Gaussian is integrated exactly along each parallel ray, the line integral is

$$
\rho_{\text{int}}(\xi)
=
\rho_0 \sqrt{2 \pi s^2_{\text{cond}}}
\exp\!\left(-\frac{1}{2} \xi^T \Sigma_{\perp}^{-1} \xi\right),
$$

where the longitudinal conditional variance is

$$
s^2_{\text{cond}} = \sigma_{\parallel} - k^T \Sigma_{\perp}^{-1} k.
$$

So for this parallel-ray case:

- the transverse density shape is determined by $\Sigma_{\perp}$
- the peak depth depends on $(r, k)$
- the integrated amplitude depends on $(\sigma_{\parallel}, k)$ through $s^2_{\text{cond}}$

This is exact for a parallel ray family aligned with the center direction $n$. It is not exact for perspective projection, where the camera emits a fan of rays whose directions vary across the image.

## 8. Terminology

- $\Sigma_c$: camera-space support covariance of the finite-support ellipsoid
- $Q^*$: dual conic in normalized image homogeneous coordinates
- $c_{uv}$, $\Sigma_{uv}$: normalized-image ellipse center and covariance
- $c_{px}$, $\Sigma_{px}$: pixel-space ellipse center and covariance
- conic coefficients: entries of $\Sigma_{px}^{-1}$ used by the rasterizer's centered quadratic form

## 9. Relevant Files

- [shaders/utility/splatting/projection.slang](../shaders/utility/splatting/projection.slang)
- [temp/projection_grad_bench.slang](../temp/projection_grad_bench.slang)
- [temp/projection_gradient_investigation.py](../temp/projection_gradient_investigation.py)
- [doc/Rendering.md](./Rendering.md)
