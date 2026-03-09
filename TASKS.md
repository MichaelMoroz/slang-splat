1) Move all utility functionality to utility shader submodule. (`random`(rng, distributions),  `losses`(l1,l2,ssim,regularization),  `math`(inverse, quat, constants, ellipsoid-ellipse and camera projection), `splatting`(gaussians, raster, training related stuff), `radix_sort`, `prefix_sum`, `blur`, `optimizer`(general adam optimizer))
Test criteria: all passing.
STATUS: Done
Description on completion: Moved reusable shader logic into `shaders/utility` (`math`, `random`, `losses`, `optimizer`, `splatting`, `blur`, `prefix_sum`, `radix_sort`), slimmed renderer/training stage files to binding + entry logic, updated Python shader loaders for moved utility kernels, and documented the new layout in `doc/ShaderUtilities.md`. Verified with `pytest tests/test_prefix_sum.py tests/test_radix_sort.py tests/test_renderer_pipeline.py tests/test_training_kernels.py` (44 passed) plus a full `pytest` run after installing missing `lizard`; the only remaining failure is the existing `tests/test_training_garden_regression.py` benchmark, which also fails on committed `HEAD` (`d42842a`) at ~24.58 dB average PSNR, so task 1 does not introduce a new repo-wide regression.

2) Nuke all old densification as well as MCMC code, including the SSIM loss, keep basic L1 loss fused in the forwardbackward kernel, only basic optimization from initialization should be active. This includes the UI, python bindings, tests, and kernels related to densification, pruning, etc. That includes the PSNR regression tests.
Test criteria: all except deleted ones.
STATUS: Done
Description on completion: Removed densification, pruning, MCMC, DSSIM/SSIM, PSNR regression, and bicycle benchmark code from the active training path across shaders, Python bindings, UI, and tests. The trainer now does fixed-count COLMAP point initialization with nearest-neighbor scales, fused L1 loss gradient, raster backward replay, and fused ADAM updates with scale/opacity regularization. Updated docs to match the simplified pipeline and verified with `pytest tests/test_app_shared.py tests/test_viewer_presenter.py tests/test_training_cli_smoke.py tests/test_training_kernels.py tests/test_renderer_pipeline.py`.

3) For `optimizer` *all the trainable stuff* should be packed in a single float buffer (param major [param_id * SPLAT_COUNT + splat_id]), so that the optimizer will work on any kind of params. The optimizer will also take an array of LR's for each param_id.  This means only 4-5 buffers instead of 40 or however many we have now. Epsilon in adam is not a runtime parameter!
Test criteria: create a basic optimization problem test, like a test function, use slang autodiff to get gradients which we will pass into the optimizer module. Check for convergence. (test shaders should be in a `tests` folder)
STATUS: Done
Description on completion: Packed all trainable splat state into one param-major float buffer (`param_id * splat_count + splat_id`) shared by renderer, raster backward, and training. Replaced per-attribute grad/moment storage with packed param-major buffers, switched ADAM to a per-param learning-rate table, and removed runtime epsilon from the optimizer module. Added a standalone autodiff-driven optimizer regression in `tests/test_optimizer_module.py` backed by `tests/optimizer_test_stage.slang`, and updated renderer/training tests to validate logical groups through the packed layout. Verified with `pytest tests/test_renderer_pipeline.py tests/test_training_kernels.py tests/test_app_shared.py tests/test_training_cli_smoke.py tests/test_viewer_presenter.py tests/test_optimizer_module.py` (38 passed).

4) For `blur` it should be able to do the blur on N channels simultaneusly. Use structured buffers instead of textures for multichannel storage. (Only the dataset and rendered results should be as textures. Gradients of textures should be buffers. )
Test criteria: create a basic N channel blur test.
STATUS: Done
Description on completion: Reworked `blur` to use flat structured buffers with runtime channel count and dispatch channel work in parallel as `SV_DispatchThreadID.z`, so one blur pass can process arbitrary `N`-channel tensors instead of only `float4` textures. Moved training image gradients from textures to a flat `StructuredBuffer<float4>` / `RWStructuredBuffer<float4>` path across the loss kernels, renderer bindings, and raster backward pass while keeping only dataset/rendered images as textures. Added an `N=6` separable Gaussian blur regression and verified with `pytest tests/test_training_kernels.py tests/test_renderer_pipeline.py`.

5) Now that all splat data is packed in one buffer, focus on keeping only a basic backprop on an unchanging gaussian splat count. Use the MCMC loss exactly. Initialized from the COLMAP point cloud. By the way the initialization scales for the splats are not valid at the moment, they should use the neigbor distance, and not be 0.0 or near zero as they are now.
Implement SSIM MCMC Loss into the optimizer. Separate forward and backward kernels. Avoid readbacks of splats in the training loop, do everything on the GPU. SSIM stuff should be their own kernels in `losses` submodule. Keep the L1 and regularization part inline, only SSIM needs separate kernels.
Loss: 
```
# Inputs
#   G = {g_i}_{i=1..N}              # set of N Gaussians
#   Each Gaussian g_i has:
#       alpha_i                     # opacity
#       mu_i                        # 3D center
#       Sigma_i                     # covariance
#       s_i = [s_i1, s_i2, s_i3]    # principal-axis scales / stddevs
#       c_i                         # color params (SH coefficients)
#   D = {(cam_j, I_gt_j)}           # training views and ground-truth images
#
# Hyperparameters
#   lambda_dssim = 0.2
#   lambda_opacity                  # opacity L1 weight
#   lambda_scale                    # scale L1 weight

function TOTAL_LOSS(G, cam_j, I_gt_j):
    I_pred_j = RENDER(G, cam_j)

    # Standard 3DGS reconstruction term
    l1_term    = mean_absolute_error(I_pred_j, I_gt_j)
    dssim_term = DSSIM(I_pred_j, I_gt_j)      # SSIM-based term used in 3DGS

    L_recon = (1 - lambda_dssim) * l1_term
            + lambda_dssim       * dssim_term

    # Sparsity on opacity
    L_opacity = sum_i abs(alpha_i)

    # Sparsity on Gaussian size
    # Equivalent reading of the paper:
    # regularize the 3 principal-axis scales of each Gaussian.
    L_scale = sum_i sum_k abs(s_ik)

    # Full objective
    L_total = L_recon
            + lambda_opacity * L_opacity
            + lambda_scale   * L_scale

    return L_total
```
Defaults:
```
L_total =
    0.8 * L1(I_pred, I_gt)
  + 0.2 * (1 - SSIM(I_pred, I_gt))
  + 0.01  * sum_i |o_i|
  + 0.01  * sum_i sum_j |sqrt(eig_j(Sigma_i))|
  ```
SSIM evaluation:
```
# img_pred, img_gt: shape [C, H, W], values typically in [0, 1]

function GAUSSIAN_1D(window_size=11, sigma=1.5):
    center = floor(window_size / 2)
    g[x] = exp(- (x - center)^2 / (2 * sigma^2)) for x = 0..window_size-1
    return g / sum(g)

function CREATE_WINDOW(window_size=11, channels=C):
    g1 = GAUSSIAN_1D(window_size, sigma=1.5)          # [W]
    g2 = outer_product(g1, g1)                        # [W, W]
    window = repeat_for_each_channel(g2)              # [C, 1, W, W]
    return window

function SSIM_VALUE(img_pred, img_gt, window_size=11):
    C1 = 0.01^2
    C2 = 0.03^2
    window = CREATE_WINDOW(window_size, channels=num_channels(img_pred))

    # local means (one pass of `blur` with 6 channels, 3 for each, store the 6 channel structbuffer input directly in the forward kernel's output)
    mu_x = depthwise_conv2d(img_pred, window, padding=window_size // 2)
    mu_y = depthwise_conv2d(img_gt,   window, padding=window_size // 2)

    # separate kernel 6 chan -> 12 chan
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy   = mu_x * mu_y

    # local variances and covariance  (one pass of `blur` with 12 channels, 3 for each, then a final kernel to evaluate SSIM)
    sigma_x_sq = depthwise_conv2d(img_pred * img_pred, window, padding=window_size // 2) - mu_x_sq
    sigma_y_sq = depthwise_conv2d(img_gt   * img_gt,   window, padding=window_size // 2) - mu_y_sq
    sigma_xy   = depthwise_conv2d(img_pred * img_gt,   window, padding=window_size // 2) - mu_xy

    # per-pixel / per-channel SSIM map
    ssim_map =
        ((2 * mu_xy      + C1) * (2 * sigma_xy              + C2)) /
        ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    # scalar SSIM used in training
    return mean(ssim_map)

function DSSIM_LOSS_FOR_3DGS(img_pred, img_gt):
    ssim_val = SSIM_VALUE(img_pred, img_gt)
    return 1 - ssim_val

function RECON_LOSS_FOR_3DGS(img_pred, img_gt, lambda_dssim=0.2):
    l1 = mean(abs(img_pred - img_gt))
    dssim = DSSIM_LOSS_FOR_3DGS(img_pred, img_gt)
    return (1 - lambda_dssim) * l1 + lambda_dssim * dssim
```
Test criteria: create a dummy dataset from procedurally generated splats (random sizes exp(rand()), random colors, random rotations and positions within a box, 16k splats), render those from 64 views and save those renderes as a training dataset. Add a small random offset to the splats and run the optimizer on them as initialization. Ideally should converge nearly perfectly, at least 50dB I expect. The random offset needs to be visible, i.e. around 1-5% difference between groundtruth images and disorted splat renders.
STATUS: Not done
Description on completion: none
