set -e
echo "=== clean (root-owned leftovers) + clone DROID-SLAM with submodules ==="
docker run --rm -v /tmp:/t alpine rm -rf /t/droid_src 2>/dev/null || true
git clone --recurse-submodules -j4 https://github.com/princeton-vl/DROID-SLAM.git /tmp/droid_src 2>&1 | tail -2
test -f /tmp/droid_src/thirdparty/lietorch/eigen/Eigen/Sparse && echo "eigen submodule OK" || { echo "eigen MISSING"; exit 1; }
echo "=== build lietorch + droid_backends in matched torch 1.10 env ==="
docker run --rm --gpus all -e TORCH_CUDA_ARCH_LIST="8.6" -e MAX_JOBS=4 -v /tmp/droid_src:/droid \
  pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel bash -c '
    pip install --no-cache-dir ninja 2>&1 | tail -1
    pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html 2>&1 | tail -1
    echo "--- lietorch (bundled thirdparty) ---"
    cd /droid/thirdparty/lietorch && python setup.py install 2>&1 | tail -5
    echo "--- droid_backends ---"
    cd /droid && python setup.py install 2>&1 | tail -5
    python -c "import torch, lietorch, droid_backends; print(\"DROID_EXTENSIONS_OK cuda=\" + str(torch.cuda.is_available()))"
  '
