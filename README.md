This is a complete implementation of Neural Radiance Fields (NeRF) for 3D reconstruction.

Put your data in a folder (e.g. data/lego) with:

- transforms_train.json - NeRF-style metadata with:
  - camera_angle_x (float): horizontal field of view in radians
  - frames: list of objects, each with:
    - file_path: path to the image without extension (e.g. train/r_0)
    - transform_matrix: 4x4 camera-to-world matrix (list of 4 lists of 4 floats)

- Images - PNG files at the paths given in file_path plus .png (e.g. train/r_0.png).

You can use the same format as the original NeRF repo (e.g. their Lego or Fern datasets) or export from COLMAP/Blender.

Hardware requirements

- This requires CUDA so you need at least an NVIDIA GPU with at least 6GB of VRAM (e.g. a 1660). It also requires 16GB or above of RAM.
- If you do not have that, use external hardware acceleration (e.g. Thunder).
- I personally used Google's Colab notebook as it provides a few hours of free GPU usage. You can also get the pro version which gets you more access to the GPU, thus helping in training and running the model.
- Other options to run this could be cloud computing services.