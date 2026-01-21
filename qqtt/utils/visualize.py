import open3d as o3d
import numpy as np
import torch
import time
import cv2
from .config import cfg
# Optional imports for headless environments
try:
    import pyrender
except (ImportError, OSError):
    pyrender = None
try:
    import trimesh
except (ImportError, OSError):
    trimesh = None


def visualize_pc(
    object_points,
    object_colors=None,
    controller_points=None,
    object_visibilities=None,
    object_motions_valid=None,
    visualize=True,
    save_video=False,
    save_path=None,
    vis_cam_idx=0,
):
    # Deprecated function, use visualize_pc instead
    FPS = cfg.FPS
    width, height = cfg.WH
    intrinsic = cfg.intrinsics[vis_cam_idx]
    w2c = cfg.w2cs[vis_cam_idx]

    # Convert the stuffs to numpy if it's tensor
    if isinstance(object_points, torch.Tensor):
        object_points = object_points.cpu().numpy()
    if isinstance(object_colors, torch.Tensor):
        object_colors = object_colors.cpu().numpy()
    if isinstance(object_visibilities, torch.Tensor):
        object_visibilities = object_visibilities.cpu().numpy()
    if isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = object_motions_valid.cpu().numpy()
    if isinstance(controller_points, torch.Tensor):
        controller_points = controller_points.cpu().numpy()

    if object_colors is None:
        object_colors = np.tile(
            [1, 0, 0], (object_points.shape[0], object_points.shape[1], 1)
        )
    else:
        if object_colors.shape[1] < object_points.shape[1]:
            # If the object_colors is not the same as object_points, fill the colors with black
            object_colors = np.concatenate(
                [
                    object_colors,
                    np.ones(
                        (
                            object_colors.shape[0],
                            object_points.shape[1] - object_colors.shape[1],
                            3,
                        )
                    )
                    * 0.3,
                ],
                axis=1,
            )

    # The pcs is a 4d pcd numpy array with shape (n_frames, n_points, 3)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=visualize, width=width, height=height)

    if save_video and visualize:
        raise ValueError("Cannot save video and visualize at the same time.")

    # Initialize video writer if save_video is True
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for .mp4 file format
        video_writer = cv2.VideoWriter(save_path, fourcc, FPS, (width, height))

    if controller_points is not None:
        controller_meshes = []
        prev_center = []
    for i in range(object_points.shape[0]):
        object_pcd = o3d.geometry.PointCloud()
        if object_visibilities is None:
            object_pcd.points = o3d.utility.Vector3dVector(object_points[i])
            object_pcd.colors = o3d.utility.Vector3dVector(object_colors[i])
        else:
            object_pcd.points = o3d.utility.Vector3dVector(
                object_points[i, np.where(object_visibilities[i])[0], :]
            )
            object_pcd.colors = o3d.utility.Vector3dVector(
                object_colors[i, np.where(object_visibilities[i])[0], :]
            )
        if i == 0:
            render_object_pcd = object_pcd
            vis.add_geometry(render_object_pcd)
            if controller_points is not None:
                # Use sphere mesh for each controller point
                for j in range(controller_points.shape[1]):
                    origin = controller_points[i, j]
                    origin_color = [1, 0, 0]
                    controller_mesh = o3d.geometry.TriangleMesh.create_sphere(
                        radius=0.01
                    ).translate(origin)
                    controller_mesh.compute_vertex_normals()
                    controller_mesh.paint_uniform_color(origin_color)
                    controller_meshes.append(controller_mesh)
                    vis.add_geometry(controller_meshes[-1])
                    prev_center.append(origin)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            camera_params = o3d.camera.PinholeCameraParameters()
            intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
                width, height, intrinsic
            )
            camera_params.intrinsic = intrinsic_parameter
            camera_params.extrinsic = w2c
            view_control.convert_from_pinhole_camera_parameters(
                camera_params, allow_arbitrary=True
            )
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
            vis.update_geometry(render_object_pcd)
            if controller_points is not None:
                for j in range(controller_points.shape[1]):
                    origin = controller_points[i, j]
                    controller_meshes[j].translate(origin - prev_center[j])
                    vis.update_geometry(controller_meshes[j])
                    prev_center[j] = origin
        vis.poll_events()
        vis.update_renderer()

        # Capture frame and write to video file if save_video is True
        if save_video:
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frame = (frame * 255).astype(np.uint8)
            if cfg.overlay_path is not None:
                # Get the mask where the pixel is white
                mask = np.all(frame == [255, 255, 255], axis=-1)
                image_path = f"{cfg.overlay_path}/{vis_cam_idx}/{i}.png"
                overlay = cv2.imread(image_path)
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                frame[mask] = overlay[mask]
            # Convert RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        if visualize:
            time.sleep(1 / FPS)

    vis.destroy_window()
    if save_video:
        video_writer.release()
