import os
import ctypes
import argparse
import warnings


def preload_libstdcxx():
    """Preload conda's libstdc++ to avoid CXXABI version conflicts (for cv2, etc.)."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    libstdcxx = os.path.join(conda_prefix, "lib", "libstdc++.so.6")
    if os.path.exists(libstdcxx):
        try:
            ctypes.CDLL(libstdcxx)
        except OSError:
            # Fall back to environment variable if preload fails
            current = os.environ.get("LD_LIBRARY_PATH", "")
            lib_dir = os.path.dirname(libstdcxx)
            if lib_dir not in current:
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{lib_dir}:{current}" if current else lib_dir
                )


def main():
    # Suppress noisy but harmless Warp tape warnings about set_control_points
    warnings.filterwarnings(
        "ignore",
        message=(
            "Warp UserWarning: Running the tape backwards may produce incorrect "
            "gradients because recorded kernel set_control_points is configured "
            "with the option 'enable_backward=False'.*"
        ),
    )

    # Ensure correct C++ runtime is loaded before any C++ extensions
    preload_libstdcxx()

    # Import Warp and initialize once
    import warp as wp

    wp.init()

    # Now import the rest of the stack
    from qqtt import InvPhyTrainerWarp
    from qqtt.utils import logger, cfg
    import numpy as np
    import pickle
    import json
    import glob
    import torch
    import os

    from interactive_playground_gradio import (
        GradioInteractivePlayground,
        create_gradio_interface,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="./data/different_types",
    )
    parser.add_argument(
        "--gaussian_path",
        type=str,
        default="./gaussian_output",
    )
    parser.add_argument(
        "--bg_img_path",
        type=str,
        default="./data/bg.png",
    )
    parser.add_argument("--case_name", type=str, default="double_lift_cloth_3")
    parser.add_argument("--n_ctrl_parts", type=int, default=2)
    parser.add_argument(
        "--inv_ctrl", action="store_true", help="invert horizontal control direction"
    )
    parser.add_argument(
        "--server_name", type=str, default="0.0.0.0", help="Gradio server name"
    )
    parser.add_argument(
        "--server_port", type=int, default=7860, help="Gradio server port"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create public Gradio link"
    )
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    base_dir = f"./temp_experiments/{case_name}"

    # Read the first-stage optimized parameters to set the indifferentiable parameters
    optimal_path = f"./experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.bg_img_path = args.bg_img_path

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"

    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )

    best_model_path = glob.glob(f"experiments/{case_name}/train/best_*.pth")[0]

    # Create Gradio playground
    playground = GradioInteractivePlayground(
        trainer,
        best_model_path,
        gaussians_path,
        args.n_ctrl_parts,
        args.inv_ctrl,
    )

    # Create and launch Gradio interface
    demo = create_gradio_interface(playground)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()


