# StreamVLN inference server for Jetson Thor (JetPack 7 / L4T R38.4, sm_110, CUDA 13).
#
# Layered on policy-server:thor purely to inherit its NVIDIA-built torch stack
# (torch 2.9.0a0+nv25.09, CUDA 13.0, sm_110, flash_attn 2.7.4.post1). That base
# image is NOT modified.
#
# The only real change is transformers 5.14.1 -> 4.45.1: StreamVLN's vendored
# LLaVA code targets the 4.45-era API (notably the `num_logits_to_keep` kwarg in
# prepare_inputs_for_generation, renamed after 4.46, which fails *silently* on 5.x).
FROM policy-server:thor

# --no-deps keeps pip from touching torch/numpy, which are unreplaceable Jetson builds.
# transformers' remaining runtime deps (filelock, packaging, pyyaml, regex, requests,
# safetensors, tqdm) are already present from the 5.x install in the base image.
# --ignore-installed blinker: the base image's blinker is a Debian package with no
# RECORD file, so pip cannot uninstall it when Flask pulls in a newer one.
RUN pip install --no-cache-dir --no-deps "transformers==4.45.1" \
 && pip install --no-cache-dir --ignore-installed blinker \
      "tokenizers==0.20.3" "huggingface_hub<1.0" flask

# streamvln_agent.py imports these two at module scope but never calls them.
# Both are painful to build on aarch64/numpy-1.x, so stub them out rather than
# drag in a compiler toolchain for dead code.
RUN printf 'def filter_depth(*a, **k):\n    raise NotImplementedError("unused on the real-world path")\n' \
      > /usr/local/lib/python3.12/dist-packages/depth_camera_filtering.py \
 && printf 'def __getattr__(name):\n    raise AttributeError(name)\n' \
      > /usr/local/lib/python3.12/dist-packages/quaternion.py

# http_realworld_server.py:28 does ImageFont.truetype("DejaVuSansMono.ttf") on every
# request. The font ships only inside matplotlib's private data dir, where PIL will not
# look, so publish it on the system font path instead of patching the server.
RUN mkdir -p /usr/share/fonts/truetype/dejavu \
 && cp /usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono.ttf \
       /usr/share/fonts/truetype/dejavu/ \
 && python3 -c "from PIL import ImageFont; ImageFont.truetype('DejaVuSansMono.ttf', 20); print('font OK')"

# Fail the build rather than the robot if the pin didn't take. No GPU is visible
# during docker build, so anything CUDA-dependent (device capability, arch list)
# is checked at runtime by the smoke test instead.
RUN python3 -c "import torch, transformers, flask, quaternion, depth_camera_filtering as d; \
assert transformers.__version__ == '4.45.1', transformers.__version__; \
print('OK torch', torch.__version__, '| transformers', transformers.__version__)"

ENV STREAMVLN_ATTN=sdpa \
    PYTHONPATH=/workspace/StreamVLN:/workspace/StreamVLN/streamvln \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HUB_OFFLINE=1

WORKDIR /workspace/StreamVLN/streamvln
