# PixL2R


## Setup:
1) Create conda environment using the environment.yml file.
2) Clone Meta-World (https://github.com/rlworkgroup/metaworld)
3) Run the following commands:
```bash
mv src/rl/sawyer_random.xml <METAWORLD-ROOTDIR>/metaworld/envs/assets/sawyer_xyz/sawyer_random.xml
mv src/rl/sawyer_random.py <METAWORLD-ROOTDIR>/metaworld/envs/mujoco/sawyer_xyz/sawyer_random.py
```

## Generate videos to annotate: 
1) Update the metaworld root path in src/rl/generate_videos.py
2) Run the following command in src/rl:
```bash
python generate_videos.py --obj-id=6 --start=0 --end=100
```

## Supervised learning:
1) Set up data.
2) Run the following command in src/supervised
```bash
python model.py --save-path=<save-path>
```

## Policy training: 
1) Update the metaworld root path in src/rl/train_policy.py
2) Set CUDA_VISIBLE_DEVICES environment variable to the desired GPU.
3) Run the following command in src/rl:
```bash
python train_policy.py --obj-id=6 --env-id=1 --descr-id=1 --reward-type=<sparse|dense|lang>
```
If reward-type is 'lang', the argument '--model-file' must also be specified, which should point to the model trained using supervised learning.



## Note:
To use the GPU for rendering (required for faster policy training), dm_control library needs to be installed. By default, it uses gpu 0. Replace the function create_initialized_headless_egl_display in file <CONDA_PATH>/envs/pix2r/lib/python3.7/site-packages/dm_control/_render/pyopengl/egl_renderer.py with the following to make it use the gpu specified using CUDA_VISIBLE_DEVICES:

```bash
def create_initialized_headless_egl_display():
  """Creates an initialized EGL display directly on a device."""
  device = EGL.eglQueryDevicesEXT()[int(os.environ['CUDA_VISIBLE_DEVICES'])]
  display = EGL.eglGetPlatformDisplayEXT(
      EGL.EGL_PLATFORM_DEVICE_EXT, device, None)
  if display != EGL.EGL_NO_DISPLAY and EGL.eglGetError() == EGL.EGL_SUCCESS:
    # `eglInitialize` may or may not raise an exception on failure depending
    # on how PyOpenGL is configured. We therefore catch a `GLError` and also
    # manually check the output of `eglGetError()` here.
    try:
      initialized = EGL.eglInitialize(display, None, None)
    except error.GLError:
      pass 
    else:
      if initialized == EGL.EGL_TRUE and EGL.eglGetError() == EGL.EGL_SUCCESS:
        return display
  return EGL.EGL_NO_DISPLAY
```

