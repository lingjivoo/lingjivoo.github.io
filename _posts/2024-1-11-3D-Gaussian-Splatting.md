---
title: 3D Gaussian Splatting
author: Cheng Luo
date: 2024-01-11 15:46:00 +1100
categories: [Study]
tags: [Technique]
pin: true
math: true
image:
  src: https://raw.githubusercontent.com/lingjivoo/lingjivoo.github.io/master/_posts/2024-1-11-3D-Gaussian-Splatting.assets/overview.png
---


# Code 
***

[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

Clone the github
```
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```

## Setup
```
conda env create --file environment.yml
conda activate gaussian_splatting
```

**Notes**
Theu cuda version must $\geq 11.8$
And then you need to install submodules

```
pip install -q gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -q /content/gaussian-splatting/submodules/simple-knn
```
## Dataset
```
wget https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip
unzip tandt_db.zip
```

## Train
```
python train.py -s gaussian-splatting/tandt/train
```


## Eval
```
python render.py -m <path to trained model> 
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>
python metrics.py -m <path to pre-trained model>
```

``` <path to trained model> ``` may be ```outputs/xxxx```


## Code Explanation

### Train.py
model: GaussianModel (variable: gaussians)
dataset: Scene (variable: scene)
render: render (function: render, defined in gaussian_renderer)

```
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
```

```
    viewpoint_stack = None
```

```
      # Every 1000 its we increase the levels of SH up to a maximum degree
      if iteration % 1000 == 0:
          gaussians.oneupSHdegree()
```

```
      render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
```


Training:

```
      gaussians.update_learning_rate(iteration) # adjust the learning rate
      viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # pick a random camera
      bg = torch.rand((3), device="cuda") if opt.random_background else background # set the background color
      render_pkg = render(viewpoint_cam, gaussians, pipe, bg) # render process
      image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"] # get rendered outputs

      loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) # loss (reconstruction loss + ssim loss)

```


> The directional appearance component (color) of the radiance field is represented via spherical harmonics (SH)

Spherical harmonic lighting (SH coefficients)

```
      # Every 1000 its we increase the levels of SH up to a maximum degree
      if iteration % 1000 == 0:
          gaussians.oneupSHdegree()
```


Densification: 

![](https://raw.githubusercontent.com/lingjivoo/lingjivoo.github.io/master/_posts/2024-1-11-3D-Gaussian-Splatting.assets/Densification.png)

```
      # Densification
      if iteration < opt.densify_until_iter:
          # Keep track of max radii in image-space for pruning
          gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
          gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

          if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
              size_threshold = 20 if iteration > opt.opacity_reset_interval else None
              gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
          
          if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
              gaussians.reset_opacity()
```



### Model
Arguments
```
      self._xyz,   # spatial position
      self._features_dc, 
      self._features_rest,
      self._scaling,  # scaling of ellipsoids
      self._rotation,  # rotation of ellipsoids
      self._opacity,   # opacity
      self.max_radii2D, 
      self.xyz_gradient_accum
      self.denom
```

Crucial funstions:

<!-- ![](https://raw.githubusercontent.com/lingjivoo/lingjivoo.github.io/master/_posts/2024-1-11-3D-Gaussian-Splatting.assets/actual_covariance.png) -->

![](https://raw.githubusercontent.com/lingjivoo/lingjivoo.github.io/master/_posts/2024-1-11-3D-Gaussian-Splatting.assets/covariance.png)
```
    # s: scaling matrix,  r: rotation matrix
    def build_scaling_rotation(s, r):
        L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
        R = build_rotation(r)

        L[:,0,0] = s[:,0]
        L[:,1,1] = s[:,1]
        L[:,2,2] = s[:,2]
        # L: 3x3, R: 3x3
        L = R @ L
        return L

    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation) # 
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)  3x3 -> 6 
        return symm
```
Actual_covariance is the symmetric matrix so that we get the upper right part.
```
    # get the upper triangular matrix 
    def strip_lowerdiag(L): 
        uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 0, 2]
        uncertainty[:, 3] = L[:, 1, 1]
        uncertainty[:, 4] = L[:, 1, 2]
        uncertainty[:, 5] = L[:, 2, 2]
        return uncertainty

    def strip_symmetric(sym):
        return strip_lowerdiag(sym)
```

Scaling_matrix: 
![](https://raw.githubusercontent.com/lingjivoo/lingjivoo.github.io/master/_posts/2024-1-11-3D-Gaussian-Splatting.assets/scaling_matrix.png)

A quaternion q to represent rotation
q with real part $q_r$ and imaginary parts $q_i, q_j, and q_K$ to a rotation matrix R:
![](https://raw.githubusercontent.com/lingjivoo/lingjivoo.github.io/master/_posts/2024-1-11-3D-Gaussian-Splatting.assets/rotation_matrix.png)
```
    def build_rotation(r):
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

        q = r / norm[:, None]

        R = torch.zeros((q.size(0), 3, 3), device='cuda')

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        return R

```


Point Cloud Data (pcd) processing:

```
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
```


densify_and_split: increase the number of points
```
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
```


```
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom # calculate the gradients of estimated density
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)  # density and clone for under reconstrution areas
        self.densify_and_split(grads, max_grad, extent) # density and split for over reconstrution areas

        prune_mask = (self.get_opacity < min_opacity).squeeze() # mask points with opacity smaller than threshold

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size # mark points with radius greater than threshold
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent  # mark points with scales greater than threshold
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws) # get the final prune mask
        self.prune_points(prune_mask) # prune parameters

        torch.cuda.empty_cache() # clear the GPU cache
 
```

```
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
```


Rendering:

```
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
```


GaussianRasterizationSettings:
```
class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

```

GaussianRasterizer:
```
class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible


    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings


        # check if the SH features and pre-computed colors are provided.
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp, 
            raster_settings, 
        )


```

Corresponding to this code segment with inputs of 3D positions, 2D positions, opacity, SH features (color), scales, rotations and pre-computed 3D Covariances.
```
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
```
rasterize_gaussians function:

```
def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings, 
    )
```


```
class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )


      # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii


  @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads
```

CUDA: Forward_Preprocess


```
void FORWARD::preprocess(int P, int D, int M,
  const float* means3D,
  const glm::vec3* scales,
  const float scale_modifier,
  const glm::vec4* rotations,
  const float* opacities,
  const float* shs,
  bool* clamped,
  const float* cov3D_precomp,
  const float* colors_precomp,
  const float* viewmatrix,
  const float* projmatrix,
  const glm::vec3* cam_pos,
  const int W, int H,
  const float focal_x, float focal_y,
  const float tan_fovx, float tan_fovy,
  int* radii,
  float2* means2D,
  float* depths,
  float* cov3Ds,
  float* rgb,
  float4* conic_opacity,
  const dim3 grid,
  uint32_t* tiles_touched,
  bool prefiltered)
{
  preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
    P, D, M,
    means3D,
    scales,
    scale_modifier,
    rotations,
    opacities,
    shs,
    clamped,
    cov3D_precomp,
    colors_precomp,
    viewmatrix, 
    projmatrix,
    cam_pos,
    W, H,
    tan_fovx, tan_fovy,
    focal_x, focal_y,
    radii,
    means2D,
    depths,
    cov3Ds,
    rgb,
    conic_opacity,
    grid,
    tiles_touched,
    prefiltered
    );
}
```

preprocessCUDA

---
作者：lingjivoo，github主页：[传送门](https://github.com/lingjivoo)
