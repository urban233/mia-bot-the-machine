import math
import os
import numpy as np
import torch
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

# RLGym v1 Default Normalization Constants
POS_STD = 2300.0
VEL_STD = 2300.0
ANG_VEL_STD = math.pi


def get_rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Computes orientation matrix from RLBot Euler angles (radians)."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)

    # 3x3 rotation matrix (Forward, Right, Up)
    rot = np.array([
        [cp * cy, cp * sy, sp],
        [cy * sp * sr - cr * sy, sy * sp * sr + cr * cy, -cp * sr],
        [-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr],
    ], dtype=np.float32)
    return rot.flatten()


class MLBot(BaseAgent):
    def initialize_agent(self):
        self.controller = SimpleControllerState()
        self.device = torch.device("cpu")

        model_path = os.path.join(os.path.dirname(__file__), "policy.pt")
        if not os.path.exists(model_path):
            try:
                from rules_python.python.runfiles import runfiles
                r = runfiles.Create()
                resolved = r.Rlocation("mia_bot/policy.pt")
                if resolved and os.path.exists(resolved):
                    model_path = resolved
            except Exception:
                pass

        if os.path.exists(model_path):
            self.policy = torch.jit.load(model_path, map_location=self.device)
            self.policy.eval()
        else:
            self.policy = None
            print(f"[MLBot] Warning: {model_path} not found.")

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        if not packet.game_info.is_round_active or self.policy is None:
            return self.controller

        # Invert coordinates if on Orange team (matches RLGym DefaultObs training)
        inv = -1.0 if self.team == 1 else 1.0

        # --- 1. Ball State (9 floats) ---
        b_phys = packet.game_ball.physics
        ball_obs = [
            b_phys.location.x * inv / POS_STD,
            b_phys.location.y * inv / POS_STD,
            b_phys.location.z / POS_STD,
            b_phys.velocity.x * inv / VEL_STD,
            b_phys.velocity.y * inv / VEL_STD,
            b_phys.velocity.z / VEL_STD,
            b_phys.angular_velocity.x * inv / ANG_VEL_STD,
            b_phys.angular_velocity.y * inv / ANG_VEL_STD,
            b_phys.angular_velocity.z / ANG_VEL_STD,
        ]

        # --- 2. Player States (Self + Opponents) ---
        # Sort players: self first, then teammates, then opponents
        self_car = packet.game_cars[self.index]
        other_cars = [packet.game_cars[i] for i in range(packet.num_cars) if i != self.index]

        player_features = []
        for car in [self_car] + other_cars:
            c_phys = car.physics
            rot_mat = get_rotation_matrix(c_phys.rotation.pitch, c_phys.rotation.yaw, c_phys.rotation.roll)
            if inv == -1.0:
                # Mirror pitch/yaw rotation for Orange team
                rot_mat[0:2] *= -1.0
                rot_mat[3:5] *= -1.0
                rot_mat[6:8] *= -1.0

            car_data = [
                c_phys.location.x * inv / POS_STD,
                c_phys.location.y * inv / POS_STD,
                c_phys.location.z / POS_STD,
                *rot_mat.tolist(),
                c_phys.velocity.x * inv / VEL_STD,
                c_phys.velocity.y * inv / VEL_STD,
                c_phys.velocity.z / VEL_STD,
                c_phys.angular_velocity.x * inv / ANG_VEL_STD,
                c_phys.angular_velocity.y * inv / ANG_VEL_STD,
                c_phys.angular_velocity.z / ANG_VEL_STD,
                car.boost / 100.0,
                float(car.has_wheel_contact),
                float(car.jumped),
                float(car.is_demolished),
            ]
            player_features.extend(car_data)

        # Combine into 89-dim feature vector
        obs = np.zeros(89, dtype=np.float32)
        full_vec = ball_obs + player_features
        obs[:min(len(full_vec), 89)] = full_vec[:89]

        # --- 3. Neural Network Forward Pass ---
        with torch.no_grad():
            tensor_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            actions = self.policy(tensor_obs).squeeze().cpu().numpy()

        # --- 4. Apply Controls ---
        self.controller.throttle = float(np.clip(actions[0], -1.0, 1.0))
        self.controller.steer = float(np.clip(actions[1], -1.0, 1.0))
        self.controller.pitch = float(np.clip(actions[2], -1.0, 1.0))
        self.controller.yaw = float(np.clip(actions[3], -1.0, 1.0))
        self.controller.roll = float(np.clip(actions[4], -1.0, 1.0))
        self.controller.jump = bool(actions[5] > 0)
        self.controller.boost = bool(actions[6] > 0)
        self.controller.handbrake = bool(actions[7] > 0)

        return self.controller

