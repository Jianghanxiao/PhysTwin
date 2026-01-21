"""author: Can Li, 2025-12-27"""

import os
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import pickle
import json
import gradio as gr
import queue
import threading
import time
import cv2

from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg

# Set environment variables for headless mode before importing any OpenGL-dependent libraries
# (Actual libstdc++ preloading and Warp initialization are handled in the launcher script.)
os.environ.setdefault("PYGLET_HEADLESS", "1")
os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")  # Use OSMesa (software rendering) instead of hardware OpenGL


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GradioInteractivePlayground:
    """Gradio-based interactive playground wrapper"""
    
    def __init__(self, trainer, model_path, gs_path, n_ctrl_parts=1, inv_ctrl=False):
        self.trainer = trainer
        self.model_path = model_path
        self.gs_path = gs_path
        self.n_ctrl_parts = n_ctrl_parts
        self.inv_ctrl = inv_ctrl
        
        # Queues for communication between simulation thread and Gradio
        self.frame_queue = queue.Queue(maxsize=10)  # Increased queue size to reduce frame drops
        self.control_queue = queue.Queue()
        self.running = False
        self.sim_thread = None
        
        # Control state
        self.pressed_keys = set()
        
        # Cache for last frame to avoid showing placeholder when queue is temporarily empty
        self.last_frame = None
        
    def setup_simulation(self):
        """Initialize simulation (same as interactive_playground)"""
        logger.info(f"Load model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        assert (
            len(spring_Y) == self.trainer.simulator.n_springs
        ), "Check if the loaded checkpoint match the config file to connect the springs"

        self.trainer.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.trainer.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.trainer.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        logger.info("Party Time Start!!!!")
        self.trainer.simulator.set_init_state(
            self.trainer.simulator.wp_init_vertices, self.trainer.simulator.wp_init_velocities
        )
        
        vis_cam_idx = 0
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]

        current_target = self.trainer.simulator.controller_points[0]
        vis_controller_points = current_target.cpu().numpy()

        from gaussian_splatting.scene.gaussian_model import GaussianModel
        from gs_render import remove_gaussians_with_low_opacity
        
        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(self.gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation
        use_white_background = True
        bg_color = [1, 1, 1] if use_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        view = self.trainer._create_gs_view(w2c, intrinsic, height, width)
        
        image_path = cfg.bg_img_path
        overlay = cv2.imread(image_path)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay = torch.tensor(overlay, dtype=torch.float32, device=cfg.device)

        if self.n_ctrl_parts > 1:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_ctrl_parts, random_state=0, n_init=10)
            cluster_labels = kmeans.fit_predict(vis_controller_points)
            masks_ctrl_pts = []
            for i in range(self.n_ctrl_parts):
                mask = cluster_labels == i
                masks_ctrl_pts.append(torch.from_numpy(mask))
            center1 = np.mean(vis_controller_points[masks_ctrl_pts[0]], axis=0)
            center2 = np.mean(vis_controller_points[masks_ctrl_pts[1]], axis=0)
            center1 = np.concatenate([center1, [1]])
            center2 = np.concatenate([center2, [1]])
            proj_mat = intrinsic @ w2c[:3, :]
            center1 = proj_mat @ center1
            center2 = proj_mat @ center2
            center1 = center1 / center1[-1]
            center2 = center2 / center2[-1]
            if center1[0] > center2[0]:
                print("Switching the control parts")
                masks_ctrl_pts = [masks_ctrl_pts[1], masks_ctrl_pts[0]]
        else:
            masks_ctrl_pts = None
            
        self.trainer.n_ctrl_parts = self.n_ctrl_parts
        self.trainer.mask_ctrl_pts = masks_ctrl_pts
        self.trainer.scale_factors = 1.0
        assert self.n_ctrl_parts <= 2, "Only support 1 or 2 control parts"
        
        self.trainer.inv_ctrl = -1.0 if self.inv_ctrl else 1.0
        self.trainer.key_mappings = {
            "w": (0, np.array([0.005, 0, 0]) * self.trainer.inv_ctrl),
            "s": (0, np.array([-0.005, 0, 0]) * self.trainer.inv_ctrl),
            "a": (0, np.array([0, -0.005, 0]) * self.trainer.inv_ctrl),
            "d": (0, np.array([0, 0.005, 0]) * self.trainer.inv_ctrl),
            "e": (0, np.array([0, 0, 0.005])),
            "q": (0, np.array([0, 0, -0.005])),
            "i": (1, np.array([0.005, 0, 0]) * self.trainer.inv_ctrl),
            "k": (1, np.array([-0.005, 0, 0]) * self.trainer.inv_ctrl),
            "j": (1, np.array([0, -0.005, 0]) * self.trainer.inv_ctrl),
            "l": (1, np.array([0, 0.005, 0]) * self.trainer.inv_ctrl),
            "o": (1, np.array([0, 0, 0.005])),
            "u": (1, np.array([0, 0, -0.005])),
        }
        self.trainer.pressed_keys = set()
        self.trainer.w2c = w2c
        self.trainer.intrinsic = intrinsic
        self.trainer.init_control_ui()
        
        if self.n_ctrl_parts > 1:
            hand_positions = []
            for i in range(2):
                target_points = torch.from_numpy(
                    vis_controller_points[self.trainer.mask_ctrl_pts[i]]
                ).to("cuda")
                hand_positions.append(self.trainer._find_closest_point(target_points))
            self.trainer.hand_left_pos, self.trainer.hand_right_pos = hand_positions
        else:
            target_points = torch.from_numpy(vis_controller_points).to("cuda")
            self.trainer.hand_left_pos = self.trainer._find_closest_point(target_points)

        self.trainer.target_change = np.zeros((self.n_ctrl_parts, 3))
        
        # Store simulation state
        self.sim_state = {
            'gaussians': gaussians,
            'view': view,
            'background': background,
            'overlay': overlay,
            'current_target': current_target,
            'prev_target': current_target,
            'prev_x': None,
            'relations': None,
            'weights': None,
            'weights_indices': None,
            'masks_ctrl_pts': masks_ctrl_pts,
            'width': width,
            'height': height,
            'intrinsic': intrinsic,
            'w2c': w2c,
            'use_white_background': use_white_background,
        }
        
    def simulation_loop(self):
        """Main simulation loop running in background thread"""
        from gaussian_splatting.gaussian_renderer import render as render_gaussian
        from gaussian_splatting.dynamic_utils import (
            interpolate_motions_speedup,
            get_topk_indices,
            knn_weights_sparse,
            calc_weights_vals_from_indices,
        )
        from qqtt.engine.trainer_warp import get_simple_shadow
        import warp as wp
        
        state = self.sim_state
        prev_x = state['prev_x']
        current_target = state['current_target']
        prev_target = state['prev_target']
        
        while self.running:
            try:
                # Check for control commands
                while not self.control_queue.empty():
                    cmd = self.control_queue.get_nowait()
                    if cmd == 'stop':
                        self.running = False
                        break
                    elif cmd.startswith('key:'):
                        key = cmd.split(':')[1]
                        if key in self.trainer.key_mappings:
                            self.trainer.pressed_keys.add(key)
                    elif cmd.startswith('release:'):
                        key = cmd.split(':')[1]
                        self.trainer.pressed_keys.discard(key)
                
                if not self.running:
                    break
                
                # Simulator step
                self.trainer.simulator.set_controller_interactive(prev_target, current_target)
                if self.trainer.simulator.object_collision_flag:
                    self.trainer.simulator.update_collision_graph()
                wp.capture_launch(self.trainer.simulator.forward_graph)
                x = wp.to_torch(self.trainer.simulator.wp_states[-1].wp_x, requires_grad=False)
                self.trainer.simulator.set_init_state(
                    self.trainer.simulator.wp_states[-1].wp_x,
                    self.trainer.simulator.wp_states[-1].wp_v,
                )
                torch.cuda.synchronize()

                # Frame initialization
                frame = state['overlay'].clone()

                # Rendering
                results = render_gaussian(state['view'], state['gaussians'], None, state['background'])
                rendering = results["render"]
                image = rendering.permute(1, 2, 0).detach()

                image = image.clamp(0, 1)
                if state['use_white_background']:
                    image_mask = torch.logical_and(
                        (image != 1.0).any(dim=2), image[:, :, 3] > 100 / 255
                    )
                else:
                    image_mask = torch.logical_and(
                        (image != 0.0).any(dim=2), image[:, :, 3] > 100 / 255
                    )
                image[..., 3].masked_fill_(~image_mask, 0.0)

                alpha = image[..., 3:4]
                rgb = image[..., :3] * 255
                frame = alpha * rgb + (1 - alpha) * frame
                frame = frame.cpu().numpy()
                image_mask = image_mask.cpu().numpy()
                frame = frame.astype(np.uint8)

                frame = self.trainer.update_frame(frame, self.trainer.pressed_keys)

                # Add shadows
                final_shadow = get_simple_shadow(
                    x, state['intrinsic'], state['w2c'], state['width'], state['height'], 
                    image_mask, light_point=[0, 0, -3]
                )
                frame[final_shadow] = (frame[final_shadow] * 0.95).astype(np.uint8)
                final_shadow = get_simple_shadow(
                    x, state['intrinsic'], state['w2c'], state['width'], state['height'],
                    image_mask, light_point=[1, 0.5, -2]
                )
                frame[final_shadow] = (frame[final_shadow] * 0.97).astype(np.uint8)
                final_shadow = get_simple_shadow(
                    x, state['intrinsic'], state['w2c'], state['width'], state['height'],
                    image_mask, light_point=[-3, -0.5, -5]
                )
                frame[final_shadow] = (frame[final_shadow] * 0.98).astype(np.uint8)
                # Keep frame in RGB format for Gradio (don't convert to BGR)
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Not needed for Gradio

                # Put frame in queue (non-blocking, drop old frames if queue is full)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass

                # Motion interpolation
                if prev_x is not None:
                    with torch.no_grad():
                        prev_particle_pos = prev_x
                        cur_particle_pos = x

                        if state['relations'] is None:
                            state['relations'] = get_topk_indices(prev_x, K=16)

                        if state['weights'] is None:
                            state['weights'], state['weights_indices'] = knn_weights_sparse(
                                prev_particle_pos, state['gaussians'].get_xyz, K=16
                            )

                        state['weights'] = calc_weights_vals_from_indices(
                            prev_particle_pos, state['gaussians'].get_xyz, state['weights_indices']
                        )

                        current_pos, current_rot, _ = interpolate_motions_speedup(
                            bones=prev_particle_pos,
                            motions=cur_particle_pos - prev_particle_pos,
                            relations=state['relations'],
                            weights=state['weights'],
                            weights_indices=state['weights_indices'],
                            xyz=state['gaussians'].get_xyz,
                            quat=state['gaussians'].get_rotation,
                        )

                        state['gaussians']._xyz = current_pos
                        state['gaussians']._rotation = current_rot

                prev_x = x.clone()

                # Update target based on pressed keys
                prev_target = current_target
                target_change = self.trainer.get_target_change()
                if state['masks_ctrl_pts'] is not None:
                    for i in range(self.n_ctrl_parts):
                        if state['masks_ctrl_pts'][i].sum() > 0:
                            current_target[state['masks_ctrl_pts'][i]] += torch.tensor(
                                target_change[i], dtype=torch.float32, device=cfg.device
                            )
                            if i == 0:
                                self.trainer.hand_left_pos += torch.tensor(
                                    target_change[i], dtype=torch.float32, device=cfg.device
                                )
                            if i == 1:
                                self.trainer.hand_right_pos += torch.tensor(
                                    target_change[i], dtype=torch.float32, device=cfg.device
                                )
                else:
                    current_target += torch.tensor(
                        target_change, dtype=torch.float32, device=cfg.device
                    )
                    self.trainer.hand_left_pos += torch.tensor(
                        target_change, dtype=torch.float32, device=cfg.device
                    )
                
                state['current_target'] = current_target
                state['prev_target'] = prev_target
                state['prev_x'] = prev_x
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                import traceback
                traceback.print_exc()
                break
    
    def get_frame(self):
        """Get latest frame from queue for Gradio"""
        # Get all frames from queue, keeping only the latest one
        latest_frame = self.last_frame  # Default to cached last frame
        try:
            while True:
                frame = self.frame_queue.get_nowait()
                if frame is not None and isinstance(frame, np.ndarray):
                    latest_frame = frame
        except queue.Empty:
            pass
        
        # Update cache if we got a new frame
        if latest_frame is not None and isinstance(latest_frame, np.ndarray):
            self.last_frame = latest_frame
            return latest_frame
        
        # If queue is empty, return cached last frame (avoids showing placeholder)
        return self.last_frame
    
    def start_simulation(self):
        """Start the simulation thread"""
        if not self.running:
            self.running = True
            self.setup_simulation()
            self.sim_thread = threading.Thread(target=self.simulation_loop, daemon=True)
            self.sim_thread.start()
            return "Simulation started"
        return "Simulation already running"
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        self.control_queue.put('stop')
        if self.sim_thread:
            self.sim_thread.join(timeout=2.0)
        return "Simulation stopped"
    
    def send_key(self, key):
        """Send key press command"""
        if self.running:
            self.control_queue.put(f'key:{key}')
    
    def release_key(self, key):
        """Send key release command"""
        if self.running:
            self.control_queue.put(f'release:{key}')


def create_gradio_interface(playground):
    """Create Gradio interface"""
    
    with gr.Blocks(title="PhysTwin Interactive Playground") as demo:
        gr.Markdown("# PhysTwin Interactive Playground")
        gr.Markdown("Control the physics simulation using the buttons below.")
        
        # Layout: center the interactive playground image, put status text below
        with gr.Row():
            gr.Column(scale=1)  # left spacer
            with gr.Column(scale=2):
                image_output = gr.Image(
                    label="Interactive Playground",
                    type="numpy",
                )
            gr.Column(scale=1)  # right spacer
        
        status_text = gr.Textbox(
            label="Status",
            value="Not started",
            interactive=False,
        )
        
        with gr.Row():
            start_btn = gr.Button("Start Simulation", variant="primary")
            stop_btn = gr.Button("Stop Simulation", variant="stop")
        
        gr.Markdown("### Control Panel 1 (Left Hand)")
        with gr.Row():
            btn_w = gr.Button("W (Forward)", size="sm")
            btn_s = gr.Button("S (Backward)", size="sm")
            btn_a = gr.Button("A (Left)", size="sm")
            btn_d = gr.Button("D (Right)", size="sm")
            btn_q = gr.Button("Q (Down)", size="sm")
            btn_e = gr.Button("E (Up)", size="sm")
        
        if playground.n_ctrl_parts > 1:
            gr.Markdown("### Control Panel 2 (Right Hand)")
            with gr.Row():
                btn_i = gr.Button("I (Forward)", size="sm")
                btn_k = gr.Button("K (Backward)", size="sm")
                btn_j = gr.Button("J (Left)", size="sm")
                btn_l = gr.Button("L (Right)", size="sm")
                btn_u = gr.Button("U (Down)", size="sm")
                btn_o = gr.Button("O (Up)", size="sm")
        
        # Button event handlers
        def start_sim():
            result = playground.start_simulation()
            return result, result
        
        def stop_sim():
            result = playground.stop_simulation()
            return "Stopped", result
        
        def key_press(key):
            playground.send_key(key)
            return f"Key {key} pressed"
        
        def key_release(key):
            playground.release_key(key)
            return f"Key {key} released"
        
        def update_frame():
            frame = playground.get_frame()
            if frame is not None:
                # Ensure frame is in correct format
                # Gradio expects: numpy array, shape (H, W, 3), dtype uint8, RGB format
                if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
                    # Debug: print frame info (only occasionally to avoid spam)
                    import random
                    if random.random() < 0.01:  # 1% chance to print
                        print(f"[DEBUG] Frame shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
                    return frame
            
            # Only show placeholder if simulation hasn't started yet
            if not playground.running:
                # Return a placeholder image only when simulation hasn't started
                if hasattr(playground, 'sim_state') and playground.sim_state:
                    h, w = playground.sim_state.get('height', 480), playground.sim_state.get('width', 640)
                else:
                    h, w = 480, 640
                placeholder = np.zeros((h, w, 3), dtype=np.uint8)
                placeholder[:, :] = [30, 30, 30]  # Dark gray
                cv2.putText(placeholder, "Click 'Start Simulation' to begin", (w//6, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                return placeholder
            
            # If simulation is running but no frame available yet, return None
            # Gradio will keep displaying the last frame (from cache) or show nothing
            # This avoids flashing "waiting for frames..." message
            return None
        
        # Bind events
        start_btn.click(start_sim, outputs=[status_text, status_text])
        stop_btn.click(stop_sim, outputs=[status_text, status_text])
        
        # Control buttons - press on click, auto-release after delay
        def make_key_handler(key, duration=0.1):
            def press():
                playground.send_key(key)
                # Auto-release after duration
                def auto_release():
                    time.sleep(duration)
                    playground.release_key(key)
                threading.Thread(target=auto_release, daemon=True).start()
                return f"Key {key.upper()} pressed"
            return press
        
        # Set 1 controls
        btn_w.click(make_key_handler('w'), outputs=[status_text])
        btn_s.click(make_key_handler('s'), outputs=[status_text])
        btn_a.click(make_key_handler('a'), outputs=[status_text])
        btn_d.click(make_key_handler('d'), outputs=[status_text])
        btn_q.click(make_key_handler('q'), outputs=[status_text])
        btn_e.click(make_key_handler('e'), outputs=[status_text])
        
        if playground.n_ctrl_parts > 1:
            btn_i.click(make_key_handler('i'), outputs=[status_text])
            btn_k.click(make_key_handler('k'), outputs=[status_text])
            btn_j.click(make_key_handler('j'), outputs=[status_text])
            btn_l.click(make_key_handler('l'), outputs=[status_text])
            btn_u.click(make_key_handler('u'), outputs=[status_text])
            btn_o.click(make_key_handler('o'), outputs=[status_text])
        
        gr.Markdown("**Tip:** Click buttons to control. Each click applies movement for 0.1 seconds. Click repeatedly for continuous movement.")
        
        # Auto-refresh image
        # For Gradio 6.x, use the timer component or every parameter
        # First, update on page load
        demo.load(
            fn=update_frame,
            inputs=None,
            outputs=image_output,
        )
        
        # Then set up auto-refresh using timer
        # Create a timer that triggers updates
        try:
            # Try using every parameter (works in Gradio 4.x+)
            timer_comp = gr.Timer(value=0.033, active=True)
            timer_comp.tick(
                fn=update_frame,
                inputs=None,
                outputs=image_output,
            )
        except (AttributeError, TypeError):
            # Fallback: Use a hidden button with JavaScript auto-click
            refresh_btn = gr.Button("Refresh", visible=False, elem_id="auto_refresh_btn")
            
            def auto_refresh():
                """Auto-refresh function that returns the latest frame"""
                return update_frame()
            
            refresh_btn.click(
                fn=auto_refresh,
                inputs=None,
                outputs=image_output,
            )
            
            # Use JavaScript to auto-click the refresh button every 33ms
            demo.load(
                fn=lambda: None,
                inputs=None,
                outputs=None,
                js="""
                () => {
                    setTimeout(() => {
                        const btn = document.getElementById('auto_refresh_btn');
                        if (btn) {
                            setInterval(() => {
                                btn.click();
                            }, 33);
                        }
                    }, 1000);
                }
                """
            )
    
    return demo


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
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
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Gradio server name")
    parser.add_argument("--server_port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
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

    # NOTE: Temporarily disable launch to debug LLVM CommandLine error.
    # If running without launch works, the issue is likely triggered during Gradio server startup.
    # Uncomment the lines below once the environment-side LLVM conflict is resolved.
    #
    # demo.launch(
    #     server_name=args.server_name,
    #     server_port=args.server_port,
    #     share=args.share,
    # )

