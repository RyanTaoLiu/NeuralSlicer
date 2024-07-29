FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
RUN pip install --find-links wheels tqdm scipy numpy pymeshlab pyvista tetgen trimesh einops comet_ml
CMD ["/bin/sh", "-c", "bash"]
