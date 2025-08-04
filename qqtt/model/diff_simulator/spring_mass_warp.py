import torch
from qqtt.utils import logger, cfg
import warp as wp
import numpy as np

wp.init()
wp.set_device("cuda:0")
if not cfg.use_graph:
    wp.config.mode = "debug"
    wp.config.verbose = True
    wp.config.verify_autograd_array_access = True


class State:
    def __init__(self, wp_init_vertices, num_control_points):
        self.wp_x = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_collision = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_ground = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v = wp.zeros_like(self.wp_x, requires_grad=True)
        self.wp_vertice_forces = wp.zeros_like(self.wp_x, requires_grad=True)
        # No need to compute the gradient for the control points
        self.wp_control_x = wp.zeros(
            (num_control_points), dtype=wp.vec3, requires_grad=False
        )
        self.wp_control_v = wp.zeros_like(self.wp_control_x, requires_grad=False)

    def clear_forces(self):
        self.wp_vertice_forces.zero_()

    # This takes more time but not necessary, will be overwritten directly
    # def clear_control(self):
    #     self.wp_control_x.zero_()
    #     self.wp_control_v.zero_()

    # def clear_states(self):
    #     self.wp_x.zero_()
    #     self.wp_v_before_ground.zero_()
    #     self.wp_v.zero_()

    @property
    def requires_grad(self):
        """Indicates whether the state arrays have gradient computation enabled."""
        return self.wp_x.requires_grad


@wp.kernel(enable_backward=False)
def set_mesh_points(
    points: wp.array(dtype=wp.vec3),
    interpolated_points: wp.array2d(dtype=wp.vec3),
    num_dynamic: int,
    step: int,
):
    tid = wp.tid()
    if tid < num_dynamic:
        points[tid] = interpolated_points[step][tid]


@wp.kernel(enable_backward=False)
def copy_vec3(data: wp.array(dtype=wp.vec3), origin: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def copy_2dvec3(data: wp.array2d(dtype=wp.vec3), origin: wp.array2d(dtype=wp.vec3)):
    i, j = wp.tid()
    origin[i][j] = data[i][j]


@wp.kernel(enable_backward=False)
def copy_int(data: wp.array(dtype=wp.int32), origin: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def copy_float(data: wp.array(dtype=wp.float32), origin: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def set_control_points(
    num_substeps: int,
    original_control_point: wp.array(dtype=wp.vec3),
    target_control_point: wp.array(dtype=wp.vec3),
    step: int,
    control_x: wp.array(dtype=wp.vec3),
):
    # Set the control points in each substep
    tid = wp.tid()

    t = float(step + 1) / float(num_substeps)
    control_x[tid] = (
        original_control_point[tid]
        + (target_control_point[tid] - original_control_point[tid]) * t
    )


@wp.kernel
def eval_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    control_x: wp.array(dtype=wp.vec3),
    control_v: wp.array(dtype=wp.vec3),
    num_object_points: int,
    springs: wp.array(dtype=wp.vec2i),
    rest_lengths: wp.array(dtype=float),
    spring_Y: wp.array(dtype=float),
    dashpot_damping: float,
    spring_Y_min: float,
    spring_Y_max: float,
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    if wp.exp(spring_Y[tid]) > spring_Y_min:

        idx1 = springs[tid][0]
        idx2 = springs[tid][1]

        if idx1 >= num_object_points:
            x1 = control_x[idx1 - num_object_points]
            v1 = control_v[idx1 - num_object_points]
        else:
            x1 = x[idx1]
            v1 = v[idx1]
        if idx2 >= num_object_points:
            x2 = control_x[idx2 - num_object_points]
            v2 = control_v[idx2 - num_object_points]
        else:
            x2 = x[idx2]
            v2 = v[idx2]

        rest = rest_lengths[tid]

        dis = x2 - x1
        dis_len = wp.length(dis)

        d = dis / wp.max(dis_len, 1e-6)

        spring_force = (
            wp.clamp(wp.exp(spring_Y[tid]), low=spring_Y_min, high=spring_Y_max)
            * (dis_len / rest - 1.0)
            * d
        )

        v_rel = wp.dot(v2 - v1, d)
        dashpot_forces = dashpot_damping * v_rel * d

        overall_force = spring_force + dashpot_forces

        if idx1 < num_object_points:
            wp.atomic_add(f, idx1, overall_force)
        if idx2 < num_object_points:
            wp.atomic_sub(f, idx2, overall_force)


@wp.kernel
def update_vel_from_force(
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    dt: float,
    drag_damping: float,
    reverse_factor: float,
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    v0 = v[tid]
    f0 = f[tid]
    m0 = masses[tid]

    drag_damping_factor = wp.exp(-dt * drag_damping)
    all_force = f0 + m0 * wp.vec3(0.0, 0.0, -9.8) * reverse_factor
    a = all_force / m0
    v1 = v0 + a * dt
    v2 = v1 * drag_damping_factor

    v_new[tid] = v2


@wp.func
def loop(
    i: int,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    clamp_collide_object_elas: float,
    clamp_collide_object_fric: float,
):
    x1 = x[i]
    v1 = v[i]
    m1 = masses[i]
    mask1 = masks[i]

    valid_count = float(0.0)
    J_sum = wp.vec3(0.0, 0.0, 0.0)
    for k in range(collision_number[i]):
        index = collision_indices[i][k]
        x2 = x[index]
        v2 = v[index]
        m2 = masses[index]
        mask2 = masks[index]

        dis = x2 - x1
        dis_len = wp.length(dis)
        relative_v = v2 - v1
        # If the distance is less than the collision distance and the two points are moving towards each other
        if (
            mask1 != mask2
            and dis_len < collision_dist
            and wp.dot(dis, relative_v) < -1e-4
        ):
            valid_count += 1.0

            collision_normal = dis / wp.max(dis_len, 1e-6)
            v_rel_n = wp.dot(relative_v, collision_normal) * collision_normal
            impulse_n = (-(1.0 + clamp_collide_object_elas) * v_rel_n) / (
                1.0 / m1 + 1.0 / m2
            )
            v_rel_n_length = wp.length(v_rel_n)

            v_rel_t = relative_v - v_rel_n
            v_rel_t_length = wp.max(wp.length(v_rel_t), 1e-6)
            a = wp.max(
                0.0,
                1.0
                - clamp_collide_object_fric
                * (1.0 + clamp_collide_object_elas)
                * v_rel_n_length
                / v_rel_t_length,
            )
            impulse_t = (a - 1.0) * v_rel_t / (1.0 / m1 + 1.0 / m2)

            J = impulse_n + impulse_t

            J_sum += J

    return valid_count, J_sum


@wp.kernel(enable_backward=False)
def update_potential_collision(
    x: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    grid: wp.uint64,
    resting_collision_pairs: wp.array2d(dtype=wp.bool),
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x1 = x[i]
    mask1 = masks[i]

    neighbors = wp.hash_grid_query(grid, x1, collision_dist * 5.0)
    for index in neighbors:
        if index != i:
            if (
                resting_collision_pairs[i][index] == True
                or resting_collision_pairs[index][i] == True
            ):
                continue
            x2 = x[index]
            mask2 = masks[index]

            dis = x2 - x1
            dis_len = wp.length(dis)
            # If the distance is less than the collision distance and the two points are moving towards each other
            if mask1 != mask2 and dis_len < collision_dist:
                collision_indices[i][collision_number[i]] = index
                collision_number[i] += 1


@wp.kernel
def object_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collide_object_elas: wp.array(dtype=float),
    collide_object_fric: wp.array(dtype=float),
    collision_dist: float,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    v1 = v[tid]
    m1 = masses[tid]

    clamp_collide_object_elas = wp.clamp(collide_object_elas[0], low=0.0, high=1.0)
    clamp_collide_object_fric = wp.clamp(collide_object_fric[0], low=0.0, high=2.0)

    valid_count, J_sum = loop(
        tid,
        collision_indices,
        collision_number,
        x,
        v,
        masses,
        masks,
        collision_dist,
        clamp_collide_object_elas,
        clamp_collide_object_fric,
    )

    if valid_count > 0:
        J_average = J_sum / valid_count
        v_new[tid] = v1 - J_average / m1
    else:
        v_new[tid] = v1

# Calcualte the rest state pairs
@wp.kernel(enable_backward=False)
def build_resting_collision_pairs(
    x: wp.array(dtype=wp.vec3),
    collision_dist: float,
    grid: wp.uint64,
    resting_collision_pairs: wp.array2d(dtype=wp.bool),
):

    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x1 = x[i]

    neighbors = wp.hash_grid_query(grid, x1, collision_dist)
    for index in neighbors:
        if index < i:
            resting_collision_pairs[i][index] = wp.bool(1)
            resting_collision_pairs[index][i] = wp.bool(1)


# This function is not validated to be differentiable yet
@wp.kernel(enable_backward=False)
def mesh_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    collide_elas: wp.array(dtype=float),
    collide_fric: wp.array(dtype=float),
    dt: float,
    face_map: wp.array(dtype=int),
    dynamic_velocity: wp.array(dtype=wp.vec3),
    dynamic_omega: wp.array(dtype=wp.vec3),
    step: int,
    interpolated_center: wp.array(dtype=wp.vec3),
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
    collision_forces: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]

    next_x = x0 + v0 * dt
    query = wp.mesh_query_point_sign_winding_number(
        mesh, next_x, max_dist=0.02, accuracy=3.0, threshold=0.6
    )
    # query = wp.mesh_query_point_sign_normal(
    #     mesh, next_x, max_dist=0.01
    # )
    if query.result:
        # Judge if this is the gripper
        if face_map[query.face] == 0 or face_map[query.face] == 1:
            is_gripper = 1
        else:
            is_gripper = 0

        p = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        delta = next_x - p
        dist = wp.length(delta) * query.sign
        err = dist - 0.005

        if err < 0.0:
            normal = wp.normalize(delta) * query.sign

            if is_gripper == 1:
                real_dynamic_velocity = dynamic_velocity[0] + wp.cross(
                    dynamic_omega[0], (x0 - interpolated_center[step])
                )
                v0 = v0 - real_dynamic_velocity
                clamp_collide_elas = 0.0
                clamp_collide_fric = 2.0
            else:
                clamp_collide_elas = wp.clamp(collide_elas[0], low=0.0, high=1.0)
                clamp_collide_fric = wp.clamp(collide_fric[0], low=0.0, high=2.0)

            v_normal = wp.dot(v0, normal) * normal
            v_tao = v0 - v_normal
            v_normal_length = wp.length(v_normal)
            v_tao_length = wp.max(wp.length(v_tao), 1e-6)

            v_normal_new = -clamp_collide_elas * v_normal

            delta_v_normal = v_normal_new - v_normal

            a = wp.max(
                0.0,
                1.0
                - clamp_collide_fric
                * (1.0 + clamp_collide_elas)
                * v_normal_length
                / v_tao_length,
            )
            v_tao_new = a * v_tao

            next_v = v_normal_new + v_tao_new
            if is_gripper == 1:
                next_v += real_dynamic_velocity

            if is_gripper == 1:
                # Use new speed to judge the new collision position
                next_x = x0 + next_v * dt
                query = wp.mesh_query_point_sign_winding_number(
                    mesh, next_x, max_dist=0.02, accuracy=3.0, threshold=0.6
                )
                if query.result:
                    p = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
                    delta = next_x - p
                    dist = wp.length(delta) * query.sign
                    err = dist - 0.005

                    if err < 0.0:
                        normal = wp.normalize(delta) * query.sign
                        next_x = next_x - normal * err
            else:
                next_x = next_x - normal * err

            # Only calculate the force in the normal direction
            delta_v_normal = v_normal_new - v_normal
            wp.atomic_add(collision_forces, face_map[query.face], delta_v_normal / dt)
        else:
            next_v = v0
    else:
        next_v = v0

    x_new[tid] = next_x
    v_new[tid] = next_v


@wp.kernel
def integrate_ground_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    collide_elas: wp.array(dtype=float),
    collide_fric: wp.array(dtype=float),
    dt: float,
    reverse_factor: float,
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]

    normal = wp.vec3(0.0, 0.0, 1.0) * reverse_factor

    x_z = x0[2]
    v_z = v0[2]
    next_x_z = (x_z + v_z * dt) * reverse_factor

    if next_x_z < 0.0 and v_z * reverse_factor < -1e-4:
        # Ground Collision
        v_normal = wp.dot(v0, normal) * normal
        v_tao = v0 - v_normal
        v_normal_length = wp.length(v_normal)
        v_tao_length = wp.max(wp.length(v_tao), 1e-6)
        clamp_collide_elas = wp.clamp(collide_elas[0], low=0.0, high=1.0)
        clamp_collide_fric = wp.clamp(collide_fric[0], low=0.0, high=2.0)

        v_normal_new = -clamp_collide_elas * v_normal
        a = wp.max(
            0.0,
            1.0
            - clamp_collide_fric
            * (1.0 + clamp_collide_elas)
            * v_normal_length
            / v_tao_length,
        )
        v_tao_new = a * v_tao

        v1 = v_normal_new + v_tao_new
        toi = -x_z / v_z
    else:
        v1 = v0
        toi = 0.0

    x_new[tid] = x0 + v0 * toi + v1 * (dt - toi)
    v_new[tid] = v1


@wp.kernel(enable_backward=False)
def compute_distances(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    distances: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    if gt_mask[i] == 1:
        dist = wp.length(gt[i] - pred[j])
        distances[i, j] = dist
    else:
        distances[i, j] = 1e6


@wp.kernel(enable_backward=False)
def compute_neigh_indices(
    distances: wp.array2d(dtype=float),
    neigh_indices: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    min_dist = float(1e6)
    min_index = int(-1)
    for j in range(distances.shape[1]):
        if distances[i, j] < min_dist:
            min_dist = distances[i, j]
            min_index = j
    neigh_indices[i] = min_index


@wp.kernel
def compute_chamfer_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    num_valid: int,
    neigh_indices: wp.array(dtype=wp.int32),
    loss_weight: float,
    chamfer_loss: wp.array(dtype=float),
):
    i = wp.tid()
    if gt_mask[i] == 1:
        min_pred = pred[neigh_indices[i]]
        min_dist = wp.length(min_pred - gt[i])
        final_min_dist = loss_weight * min_dist * min_dist / float(num_valid)
        wp.atomic_add(chamfer_loss, 0, final_min_dist)


@wp.kernel
def compute_track_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    num_valid: int,
    loss_weight: float,
    track_loss: wp.array(dtype=float),
):
    i = wp.tid()
    if gt_mask[i] == 1:
        # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
        pred_x = pred[i][0]
        pred_y = pred[i][1]
        pred_z = pred[i][2]
        gt_x = gt[i][0]
        gt_y = gt[i][1]
        gt_z = gt[i][2]

        dist_x = wp.abs(pred_x - gt_x)
        dist_y = wp.abs(pred_y - gt_y)
        dist_z = wp.abs(pred_z - gt_z)

        if dist_x < 1.0:
            temp_track_loss_x = 0.5 * (dist_x**2.0)
        else:
            temp_track_loss_x = dist_x - 0.5

        if dist_y < 1.0:
            temp_track_loss_y = 0.5 * (dist_y**2.0)
        else:
            temp_track_loss_y = dist_y - 0.5

        if dist_z < 1.0:
            temp_track_loss_z = 0.5 * (dist_z**2.0)
        else:
            temp_track_loss_z = dist_z - 0.5

        temp_track_loss = temp_track_loss_x + temp_track_loss_y + temp_track_loss_z

        average_factor = float(num_valid) * 3.0

        final_track_loss = loss_weight * temp_track_loss / average_factor

        wp.atomic_add(track_loss, 0, final_track_loss)


@wp.kernel(enable_backward=False)
def set_int(input: int, output: wp.array(dtype=wp.int32)):
    output[0] = input


@wp.kernel(enable_backward=False)
def update_acc(
    v1: wp.array(dtype=wp.vec3),
    v2: wp.array(dtype=wp.vec3),
    prev_acc: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    prev_acc[tid] = v2[tid] - v1[tid]


@wp.kernel
def compute_acc_loss(
    v1: wp.array(dtype=wp.vec3),
    v2: wp.array(dtype=wp.vec3),
    prev_acc: wp.array(dtype=wp.vec3),
    num_object_points: int,
    acc_count: wp.array(dtype=wp.int32),
    acc_weight: float,
    acc_loss: wp.array(dtype=wp.float32),
):
    if acc_count[0] == 1:
        # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
        tid = wp.tid()
        cur_acc = v2[tid] - v1[tid]
        cur_x = cur_acc[0]
        cur_y = cur_acc[1]
        cur_z = cur_acc[2]

        prev_x = prev_acc[tid][0]
        prev_y = prev_acc[tid][1]
        prev_z = prev_acc[tid][2]

        dist_x = wp.abs(cur_x - prev_x)
        dist_y = wp.abs(cur_y - prev_y)
        dist_z = wp.abs(cur_z - prev_z)

        if dist_x < 1.0:
            temp_acc_loss_x = 0.5 * (dist_x**2.0)
        else:
            temp_acc_loss_x = dist_x - 0.5

        if dist_y < 1.0:
            temp_acc_loss_y = 0.5 * (dist_y**2.0)
        else:
            temp_acc_loss_y = dist_y - 0.5

        if dist_z < 1.0:
            temp_acc_loss_z = 0.5 * (dist_z**2.0)
        else:
            temp_acc_loss_z = dist_z - 0.5

        temp_acc_loss = temp_acc_loss_x + temp_acc_loss_y + temp_acc_loss_z

        average_factor = float(num_object_points) * 3.0

        final_acc_loss = acc_weight * temp_acc_loss / average_factor

        wp.atomic_add(acc_loss, 0, final_acc_loss)


@wp.kernel
def compute_final_loss(
    chamfer_loss: wp.array(dtype=wp.float32),
    track_loss: wp.array(dtype=wp.float32),
    acc_loss: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    loss[0] = chamfer_loss[0] + track_loss[0] + acc_loss[0]


@wp.kernel
def compute_simple_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    num_object_points: int,
    loss: wp.array(dtype=wp.float32),
):
    # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
    tid = wp.tid()
    pred_x = pred[tid][0]
    pred_y = pred[tid][1]
    pred_z = pred[tid][2]

    gt_x = gt[tid][0]
    gt_y = gt[tid][1]
    gt_z = gt[tid][2]

    dist_x = wp.abs(pred_x - gt_x)
    dist_y = wp.abs(pred_y - gt_y)
    dist_z = wp.abs(pred_z - gt_z)

    if dist_x < 1.0:
        temp_simple_loss_x = 0.5 * (dist_x**2.0)
    else:
        temp_simple_loss_x = dist_x - 0.5

    if dist_y < 1.0:
        temp_simple_loss_y = 0.5 * (dist_y**2.0)
    else:
        temp_simple_loss_y = dist_y - 0.5

    if dist_z < 1.0:
        temp_simple_loss_z = 0.5 * (dist_z**2.0)
    else:
        temp_simple_loss_z = dist_z - 0.5

    temp_simple_loss = temp_simple_loss_x + temp_simple_loss_y + temp_simple_loss_z

    average_factor = float(num_object_points) * 3.0

    final_simple_loss = temp_simple_loss / average_factor

    wp.atomic_add(loss, 0, final_simple_loss)


class SpringMassSystemWarp:
    def __init__(
        self,
        init_vertices,
        init_springs,
        init_rest_lengths,
        init_masses,
        dt,
        num_substeps,
        spring_Y,
        collide_elas,
        collide_fric,
        dashpot_damping,
        drag_damping,
        collide_object_elas=0.7,
        collide_object_fric=0.3,
        init_masks=None,
        collision_dist=0.02,
        init_velocities=None,
        num_object_points=None,
        num_surface_points=None,
        num_original_points=None,
        controller_points=None,
        reverse_z=False,
        spring_Y_min=1e3,
        spring_Y_max=1e5,
        gt_object_points=None,
        gt_object_visibilities=None,
        gt_object_motions_valid=None,
        self_collision=False,
        disable_backward=False,
        static_meshes=None,
        dynamic_points=None,
        controller_spring_Y=None,
        num_object_springs=None,
    ):
        logger.info(f"[SIMULATION]: Initialize the Spring-Mass System")
        self.device = cfg.device

        # Record the parameters
        self.wp_init_vertices = wp.from_torch(
            init_vertices[:num_object_points].contiguous(),
            dtype=wp.vec3,
            requires_grad=False,
        )
        if init_velocities is None:
            self.wp_init_velocities = wp.zeros_like(
                self.wp_init_vertices, requires_grad=False
            )
        else:
            self.wp_init_velocities = wp.from_torch(
                init_velocities[:num_object_points].contiguous(),
                dtype=wp.vec3,
                requires_grad=False,
            )

        self.n_vertices = init_vertices.shape[0]
        self.n_springs = init_springs.shape[0]

        self.dt = dt
        self.num_substeps = num_substeps
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping
        self.reverse_factor = 1.0 if not reverse_z else -1.0
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max

        if controller_points is None:
            assert num_object_points == self.n_vertices
        else:
            assert (controller_points.shape[1] + num_object_points) == self.n_vertices
        self.num_object_points = num_object_points
        self.num_control_points = (
            controller_points.shape[1] if not controller_points is None else 0
        )
        self.controller_points = controller_points

        # Deal with the any collision detection
        self.object_collision_flag = 0
        if init_masks is not None:
            if torch.unique(init_masks).shape[0] > 1:
                self.object_collision_flag = 1

        if self_collision:
            assert init_masks is None
            self.object_collision_flag = 1
            # Make all points as the collision points
            init_masks = torch.arange(
                self.n_vertices, dtype=torch.int32, device=self.device
            )

        if self.object_collision_flag:
            self.wp_masks = wp.from_torch(
                init_masks[:num_object_points].int(),
                dtype=wp.int32,
                requires_grad=False,
            )

            self.collision_grid = wp.HashGrid(128, 128, 128)
            self.collision_dist = collision_dist

            self.wp_collision_indices = wp.zeros(
                (self.wp_init_vertices.shape[0], 500),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_collision_number = wp.zeros(
                (self.wp_init_vertices.shape[0]), dtype=wp.int32, requires_grad=False
            )

            self.resting_collision_pairs = wp.zeros(
                (self.wp_init_vertices.shape[0], self.wp_init_vertices.shape[0]),
                dtype=wp.bool,
                requires_grad=False,
            )

        # Initialize the GT for calculating losses
        self.gt_object_points = gt_object_points
        if cfg.data_type == "real":
            self.gt_object_visibilities = gt_object_visibilities.int()
            self.gt_object_motions_valid = gt_object_motions_valid.int()

        self.num_surface_points = num_surface_points
        self.num_original_points = num_original_points
        if num_original_points is None:
            self.num_original_points = self.num_object_points

        # # Do some initialization to initialize the warp cuda graph
        self.wp_springs = wp.from_torch(
            init_springs, dtype=wp.vec2i, requires_grad=False
        )
        self.wp_rest_lengths = wp.from_torch(
            init_rest_lengths, dtype=wp.float32, requires_grad=False
        )
        self.wp_masses = wp.from_torch(
            init_masses[:num_object_points], dtype=wp.float32, requires_grad=False
        )
        if cfg.data_type == "real":
            self.prev_acc = wp.zeros_like(self.wp_init_vertices, requires_grad=False)
            self.acc_count = wp.zeros(1, dtype=wp.int32, requires_grad=False)

        self.wp_current_object_points = wp.from_torch(
            self.gt_object_points[1].clone(), dtype=wp.vec3, requires_grad=False
        )
        if cfg.data_type == "real":
            self.wp_current_object_visibilities = wp.from_torch(
                self.gt_object_visibilities[1].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_current_object_motions_valid = wp.from_torch(
                self.gt_object_motions_valid[0].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.num_valid_visibilities = int(self.gt_object_visibilities[1].sum())
            self.num_valid_motions = int(self.gt_object_motions_valid[0].sum())

            if self.controller_points is not None:
                self.wp_original_control_point = wp.from_torch(
                    self.controller_points[0].clone(),
                    dtype=wp.vec3,
                    requires_grad=False,
                )
                self.wp_target_control_point = wp.from_torch(
                    self.controller_points[1].clone(),
                    dtype=wp.vec3,
                    requires_grad=False,
                )

            self.chamfer_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.track_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.acc_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        # Initialize the warp parameters
        self.wp_states = []
        for i in range(self.num_substeps + 1):
            state = State(self.wp_init_velocities, self.num_control_points)
            self.wp_states.append(state)
        if cfg.data_type == "real":
            self.distance_matrix = wp.zeros(
                (self.num_original_points, self.num_surface_points), requires_grad=False
            )
            self.neigh_indices = wp.zeros(
                (self.num_original_points), dtype=wp.int32, requires_grad=False
            )

        spring_Y_array = torch.log(
            torch.tensor(spring_Y, dtype=torch.float32, device=self.device)
        ) * torch.ones(self.n_springs, dtype=torch.float32, device=self.device)

        if controller_spring_Y is not None:
            spring_Y_array[num_object_springs:] = torch.log(
                torch.tensor(controller_spring_Y, dtype=torch.float32, device=self.device)
            )
            
        # Parameter to be optimized
        self.wp_spring_Y = wp.from_torch(
            spring_Y_array,
            requires_grad=True,
        )
        self.wp_collide_elas = wp.from_torch(
            torch.tensor([collide_elas], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_fric = wp.from_torch(
            torch.tensor([collide_fric], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_elas = wp.from_torch(
            torch.tensor(
                [collide_object_elas], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_fric = wp.from_torch(
            torch.tensor(
                [collide_object_fric], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )

        # Load the static meshes
        self.static_mesh_warp = None
        if static_meshes is not None:
            vertices = []
            indices = []
            face_map = []
            mesh_index = 0
            offset = 0
            for static_mesh in static_meshes:
                vertex = np.array(static_mesh.vertices, dtype=np.float32)
                index = np.array(static_mesh.triangles, dtype=np.int32).flatten()
                vertices.append(vertex)
                indices.append(index + offset)
                offset += vertex.shape[0]
                face_map.append(
                    np.ones(int(index.shape[0] / 3), dtype=np.int32) * mesh_index
                )
                mesh_index += 1
            vertices = np.concatenate(vertices, axis=0)
            indices = np.concatenate(indices, axis=0)
            self.static_mesh_warp = wp.Mesh(
                points=wp.array(vertices, dtype=wp.vec3, device=self.device),
                indices=wp.array(indices, dtype=int, device=self.device),
            )
            self.face_map = wp.array(
                np.concatenate(face_map, axis=0),
                dtype=int,
                device=self.device,
                requires_grad=False,
            )
            self.collision_forces = wp.array(
                np.zeros([len(static_meshes), 3], dtype=np.float32),
                dtype=wp.vec3,
                device=self.device,
                requires_grad=False,
            )
            self.wp_interpolated_dynamic_points = wp.from_torch(
                dynamic_points.clone().repeat(self.num_substeps, 1, 1),
                dtype=wp.vec3,
                requires_grad=False,
            )
            self.wp_interpolated_center = wp.from_torch(
                torch.mean(dynamic_points, dim=0).clone().repeat(self.num_substeps, 1),
                dtype=wp.vec3,
                requires_grad=False,
            )
            self.num_dynamic = len(dynamic_points)

            self.wp_dynamic_velocity = wp.zeros((1), dtype=wp.vec3, requires_grad=False)
            self.wp_dynamic_omega = wp.zeros((1), dtype=wp.vec3, requires_grad=False)

        # Create the CUDA graph to acclerate
        if cfg.use_graph:
            if cfg.data_type == "real":
                if not disable_backward:
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        with self.tape:
                            self.step()
                            self.calculate_loss()
                        self.tape.backward(self.loss)
                else:
                    with wp.ScopedCapture() as capture:
                        self.step()
                        self.calculate_loss()
                self.graph = capture.graph
            elif cfg.data_type == "synthetic":
                if not disable_backward:
                    # For synthetic data, we compute simple loss
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        with self.tape:
                            self.step()
                            self.calculate_simple_loss()
                        self.tape.backward(self.loss)
                else:
                    with wp.ScopedCapture() as capture:
                        self.step()
                        self.calculate_simple_loss()
                self.graph = capture.graph
            else:
                raise NotImplementedError

            with wp.ScopedCapture() as forward_capture:
                self.step()
            self.forward_graph = forward_capture.graph
        else:
            self.tape = wp.Tape()

    # Create the rest map for self-collision in frame 0
    def create_resting_case(self):
        self.collision_grid.build(self.wp_states[0].wp_x, self.collision_dist * 5)
        wp.launch(
            build_resting_collision_pairs,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_x,
                self.collision_dist,
                self.collision_grid.id,
            ],
            outputs=[self.resting_collision_pairs],
        )

    def set_controller_target(self, frame_idx, pure_inference=False):
        if self.controller_points is not None:
            # Set the controller points
            wp.launch(
                copy_vec3,
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx - 1]],
                outputs=[self.wp_original_control_point],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx]],
                outputs=[self.wp_target_control_point],
            )

        if not pure_inference:
            # Set the target points
            wp.launch(
                copy_vec3,
                dim=self.num_original_points,
                inputs=[self.gt_object_points[frame_idx]],
                outputs=[self.wp_current_object_points],
            )

            if cfg.data_type == "real":
                wp.launch(
                    copy_int,
                    dim=self.num_original_points,
                    inputs=[self.gt_object_visibilities[frame_idx]],
                    outputs=[self.wp_current_object_visibilities],
                )
                wp.launch(
                    copy_int,
                    dim=self.num_original_points,
                    inputs=[self.gt_object_motions_valid[frame_idx - 1]],
                    outputs=[self.wp_current_object_motions_valid],
                )

                self.num_valid_visibilities = int(
                    self.gt_object_visibilities[frame_idx].sum()
                )
                self.num_valid_motions = int(
                    self.gt_object_motions_valid[frame_idx - 1].sum()
                )

    def set_controller_interactive(
        self, last_controller_interactive, controller_interactive
    ):
        # Set the controller points
        wp.launch(
            copy_vec3,
            dim=self.num_control_points,
            inputs=[last_controller_interactive],
            outputs=[self.wp_original_control_point],
        )
        wp.launch(
            copy_vec3,
            dim=self.num_control_points,
            inputs=[controller_interactive],
            outputs=[self.wp_target_control_point],
        )

    def set_init_state(self, wp_x, wp_v, pure_inference=False):
        # Detach and clone and set requires_grad=True
        assert (
            self.num_object_points == wp_x.shape[0]
            and self.num_object_points == self.wp_states[0].wp_x.shape[0]
        )

        if not pure_inference:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp.clone(wp_x, requires_grad=False)],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp.clone(wp_v, requires_grad=False)],
                outputs=[self.wp_states[0].wp_v],
            )
        else:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp_x],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp_v],
                outputs=[self.wp_states[0].wp_v],
            )

    def set_acc_count(self, acc_count):
        if acc_count:
            input = 1
        else:
            input = 0
        wp.launch(
            set_int,
            dim=1,
            inputs=[input],
            outputs=[self.acc_count],
        )

    def update_acc(self):
        wp.launch(
            update_acc,
            dim=self.num_object_points,
            inputs=[
                wp.clone(self.wp_states[0].wp_v, requires_grad=False),
                wp.clone(self.wp_states[-1].wp_v, requires_grad=False),
            ],
            outputs=[self.prev_acc],
        )

    def set_mesh_interactive(
        self,
        interpolated_dynamic_points,
        interpolated_center,
        dynamic_velocity,
        dynamic_omega,
    ):
        # Set the controller points
        wp.launch(
            copy_2dvec3,
            dim=(
                interpolated_dynamic_points.shape[0],
                interpolated_dynamic_points.shape[1],
            ),
            inputs=[interpolated_dynamic_points],
            outputs=[self.wp_interpolated_dynamic_points],
        )
        wp.launch(
            copy_vec3,
            dim=len(interpolated_center),
            inputs=[interpolated_center],
            outputs=[self.wp_interpolated_center],
        )

        wp.launch(
            copy_vec3,
            dim=1,
            inputs=[dynamic_velocity],
            outputs=[self.wp_dynamic_velocity],
        )

        wp.launch(
            copy_vec3,
            dim=1,
            inputs=[dynamic_omega],
            outputs=[self.wp_dynamic_omega],
        )

    def update_collision_graph(self):
        assert self.object_collision_flag
        self.collision_grid.build(self.wp_states[0].wp_x, self.collision_dist * 5.0)
        self.wp_collision_number.zero_()
        wp.launch(
            update_potential_collision,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_x,
                self.wp_masks,
                self.collision_dist,
                self.collision_grid.id,
                self.resting_collision_pairs,
            ],
            outputs=[self.wp_collision_indices, self.wp_collision_number],
        )

    def step(self):
        for i in range(self.num_substeps):
            self.wp_states[i].clear_forces()
            if not self.controller_points is None:
                # Set the control point
                wp.launch(
                    set_control_points,
                    dim=self.num_control_points,
                    inputs=[
                        self.num_substeps,
                        self.wp_original_control_point,
                        self.wp_target_control_point,
                        i,
                    ],
                    outputs=[self.wp_states[i].wp_control_x],
                )

            # Calculate the spring forces
            wp.launch(
                kernel=eval_springs,
                dim=self.n_springs,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_control_x,
                    self.wp_states[i].wp_control_v,
                    self.num_object_points,
                    self.wp_springs,
                    self.wp_rest_lengths,
                    self.wp_spring_Y,
                    self.dashpot_damping,
                    self.spring_Y_min,
                    self.spring_Y_max,
                ],
                outputs=[self.wp_states[i].wp_vertice_forces],
            )

            if self.object_collision_flag:
                output_v = self.wp_states[i].wp_v_before_collision
            else:
                output_v = self.wp_states[i].wp_v_before_ground

            # Update the output_v using the vertive_forces
            wp.launch(
                kernel=update_vel_from_force,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_vertice_forces,
                    self.wp_masses,
                    self.dt,
                    self.drag_damping,
                    self.reverse_factor,
                ],
                outputs=[output_v],
            )

            if self.object_collision_flag:
                # Update the wp_v_before_ground based on the collision handling
                wp.launch(
                    kernel=object_collision,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_v_before_collision,
                        self.wp_masses,
                        self.wp_masks,
                        self.wp_collide_object_elas,
                        self.wp_collide_object_fric,
                        self.collision_dist,
                        self.wp_collision_indices,
                        self.wp_collision_number,
                    ],
                    outputs=[self.wp_states[i].wp_v_before_ground],
                )

            # This function is not promised to be differentiable for now
            # Need to further check and update
            if self.static_mesh_warp is not None:
                wp.launch(
                    set_mesh_points,
                    dim=len(self.static_mesh_warp.points),
                    inputs=[
                        self.static_mesh_warp.points,
                        self.wp_interpolated_dynamic_points,
                        self.num_dynamic,
                        i,
                    ],
                )
                self.static_mesh_warp.refit()
                self.collision_forces.zero_()
                wp.launch(
                    kernel=mesh_collision,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_v_before_ground,
                        self.static_mesh_warp.id,
                        self.wp_collide_elas,
                        self.wp_collide_fric,
                        self.dt,
                        self.face_map,
                        self.wp_dynamic_velocity,
                        self.wp_dynamic_omega,
                        i,
                        self.wp_interpolated_center,
                    ],
                    outputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_v_before_ground,
                        self.collision_forces,
                    ],
                )

            # Update the x and v
            wp.launch(
                kernel=integrate_ground_collision,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v_before_ground,
                    self.wp_collide_elas,
                    self.wp_collide_fric,
                    self.dt,
                    self.reverse_factor,
                ],
                outputs=[self.wp_states[i + 1].wp_x, self.wp_states[i + 1].wp_v],
            )

    def calculate_loss(self):
        # Compute the chamfer loss
        # Precompute the distances matrix for the chamfer loss
        wp.launch(
            compute_distances,
            dim=(self.num_original_points, self.num_surface_points),
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
            ],
            outputs=[self.distance_matrix],
        )

        wp.launch(
            compute_neigh_indices,
            dim=self.num_original_points,
            inputs=[self.distance_matrix],
            outputs=[self.neigh_indices],
        )

        wp.launch(
            compute_chamfer_loss,
            dim=self.num_original_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
                self.num_valid_visibilities,
                self.neigh_indices,
                cfg.chamfer_weight,
            ],
            outputs=[self.chamfer_loss],
        )

        # Compute the tracking loss
        wp.launch(
            compute_track_loss,
            dim=self.num_original_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_motions_valid,
                self.num_valid_motions,
                cfg.track_weight,
            ],
            outputs=[self.track_loss],
        )

        wp.launch(
            compute_acc_loss,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_v,
                self.wp_states[-1].wp_v,
                self.prev_acc,
                self.num_object_points,
                self.acc_count,
                cfg.acc_weight,
            ],
            outputs=[self.acc_loss],
        )

        wp.launch(
            compute_final_loss,
            dim=1,
            inputs=[self.chamfer_loss, self.track_loss, self.acc_loss],
            outputs=[self.loss],
        )

    def calculate_simple_loss(self):
        wp.launch(
            compute_simple_loss,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.num_object_points,
            ],
            outputs=[self.loss],
        )

    def clear_loss(self):
        if cfg.data_type == "real":
            self.distance_matrix.zero_()
            self.neigh_indices.zero_()
            self.chamfer_loss.zero_()
            self.track_loss.zero_()
            self.acc_loss.zero_()
        self.loss.zero_()

    # Functions used to load the parmeters
    def set_spring_Y(self, spring_Y):
        # assert spring_Y.shape[0] == self.n_springs
        wp.launch(
            copy_float,
            dim=self.n_springs,
            inputs=[spring_Y],
            outputs=[self.wp_spring_Y],
        )

    def set_collide(self, collide_elas, collide_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_elas],
            outputs=[self.wp_collide_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_fric],
            outputs=[self.wp_collide_fric],
        )

    def set_collide_object(self, collide_object_elas, collide_object_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_elas],
            outputs=[self.wp_collide_object_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_fric],
            outputs=[self.wp_collide_object_fric],
        )
