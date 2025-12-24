#!/usr/bin/env python3
"""
MuJoCo WebSocket Server for OpenDuck Mini
Provides accurate physics simulation via WebSocket connection to browser.
Supports both ONNX inference and online training with PPO.
"""

import asyncio
import json
import numpy as np
import mujoco
import onnxruntime as ort
from pathlib import Path
import websockets
from dataclasses import dataclass
from typing import Optional
import argparse


@dataclass
class RLConfig:
    """RL configuration matching Open_Duck_Playground"""
    linear_velocity_scale: float = 1.0
    angular_velocity_scale: float = 1.0
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    action_scale: float = 0.25
    max_motor_velocity: float = 5.24  # rad/s
    sim_dt: float = 0.002
    decimation: int = 10
    nb_steps_in_period: int = 50
    phase_frequency: float = 1.0


class OpenDuckController:
    """Controller for OpenDuck Mini robot with RL policy"""

    def __init__(self, model_path: str, onnx_path: Optional[str] = None):
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Load ONNX policy
        self.session = None
        if onnx_path and Path(onnx_path).exists():
            self.session = ort.InferenceSession(onnx_path)
            print(f"Loaded ONNX policy: {onnx_path}")

        self.config = RLConfig()

        # Default actuator positions (from keyframe)
        self.default_actuator = np.array([
            0.002, 0.053, -0.63, 1.368, -0.784,  # left leg
            0, 0, 0, 0,                           # head
            -0.003, -0.065, 0.635, 1.379, -0.796  # right leg
        ], dtype=np.float32)

        # Action history
        self.last_action = np.zeros(14, dtype=np.float32)
        self.last_last_action = np.zeros(14, dtype=np.float32)
        self.last_last_last_action = np.zeros(14, dtype=np.float32)
        self.motor_targets = self.default_actuator.copy()
        self.prev_motor_targets = self.default_actuator.copy()

        # Phase for walking gait
        self.phase_step = 0.0
        self.imitation_phase = np.array([0.0, 0.0], dtype=np.float32)

        # Commands (lin_vel_x, lin_vel_y, ang_vel, neck, head_pitch, head_yaw, head_roll)
        self.commands = np.zeros(7, dtype=np.float32)

        # Reset to keyframe
        self.reset()

    def reset(self):
        """Reset simulation to standing pose"""
        mujoco.mj_resetData(self.model, self.data)

        # Load keyframe if available
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Reset action history
        self.last_action.fill(0)
        self.last_last_action.fill(0)
        self.last_last_last_action.fill(0)
        self.motor_targets = self.default_actuator.copy()
        self.prev_motor_targets = self.default_actuator.copy()
        self.phase_step = 0.0
        self.imitation_phase = np.array([0.0, 0.0], dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)

        return self.get_state()

    def get_observation(self) -> np.ndarray:
        """Get 101-dimensional observation vector"""
        # Gyro from sensor
        gyro = self.data.sensordata[0:3].copy()

        # Accelerometer (adjusted like in JS)
        accel = self.data.sensordata[6:9].copy()
        accel[0] += 1.3  # Proper acceleration adjustment

        # Joint angles relative to default
        joint_angles = np.zeros(14, dtype=np.float32)
        for i in range(14):
            joint_angles[i] = (self.data.qpos[7 + i] - self.default_actuator[i]) * self.config.dof_pos_scale

        # Joint velocities
        joint_vel = self.data.qvel[6:20].astype(np.float32) * self.config.dof_vel_scale

        # Contacts (simplified - could improve with actual contact detection)
        contacts = np.array([0.0, 0.0], dtype=np.float32)

        # Build observation (101 dimensions)
        obs = np.concatenate([
            gyro,                        # 3
            accel,                       # 3
            self.commands,               # 7
            joint_angles,                # 14
            joint_vel,                   # 14
            self.last_action,            # 14
            self.last_last_action,       # 14
            self.last_last_last_action,  # 14
            self.motor_targets,          # 14
            contacts,                    # 2
            self.imitation_phase         # 2
        ]).astype(np.float32)

        return obs

    def infer(self) -> np.ndarray:
        """Run ONNX inference and return motor targets"""
        if self.session is None:
            return self.default_actuator.copy()

        obs = self.get_observation()

        # Run inference
        input_tensor = obs.reshape(1, -1)
        results = self.session.run(None, {'obs': input_tensor})
        action = results[0][0]

        # Update action history
        self.last_last_last_action = self.last_last_action.copy()
        self.last_last_action = self.last_action.copy()
        self.last_action = action.copy()

        # Calculate motor targets with speed limiting
        max_change = self.config.max_motor_velocity * (self.config.sim_dt * self.config.decimation)

        for i in range(14):
            target = self.default_actuator[i] + action[i] * self.config.action_scale
            min_target = self.prev_motor_targets[i] - max_change
            max_target = self.prev_motor_targets[i] + max_change
            self.motor_targets[i] = np.clip(target, min_target, max_target)

        self.prev_motor_targets = self.motor_targets.copy()

        # Update imitation phase
        self.phase_step = (self.phase_step + self.config.phase_frequency) % self.config.nb_steps_in_period
        phase_angle = self.phase_step / self.config.nb_steps_in_period * 2 * np.pi
        self.imitation_phase = np.array([np.cos(phase_angle), np.sin(phase_angle)], dtype=np.float32)

        return self.motor_targets

    def step(self, use_rl: bool = True) -> dict:
        """Step simulation and return state"""
        if use_rl:
            targets = self.infer()
        else:
            targets = self.default_actuator

        # Apply control
        self.data.ctrl[:14] = targets

        # Step physics (decimation steps)
        for _ in range(self.config.decimation):
            mujoco.mj_step(self.model, self.data)

        return self.get_state()

    def get_state(self) -> dict:
        """Get current simulation state for rendering"""
        # Body positions and quaternions
        bodies = []
        for i in range(self.model.nbody):
            pos = self.data.xpos[i].tolist()
            quat = self.data.xquat[i].tolist()
            bodies.append({'pos': pos, 'quat': quat})

        return {
            'time': self.data.time,
            'qpos': self.data.qpos.tolist(),
            'qvel': self.data.qvel.tolist(),
            'ctrl': self.data.ctrl.tolist(),
            'bodies': bodies,
            'fallen': self.check_fallen()
        }

    def check_fallen(self) -> bool:
        """Check if robot has fallen"""
        trunk_z = self.data.qpos[2]
        qw, qx, qy, qz = self.data.qpos[3:7]

        # Up vector from quaternion
        up_z = 1 - 2 * (qx * qx + qy * qy)

        return trunk_z < 0.08 or up_z < 0.5

    def set_command(self, lin_vel_x: float, lin_vel_y: float, ang_vel: float):
        """Set velocity commands"""
        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y
        self.commands[2] = ang_vel


class MuJoCoServer:
    """WebSocket server for MuJoCo simulation"""

    def __init__(self, model_path: str, onnx_path: str, host: str = "localhost", port: int = 8765):
        self.model_path = model_path
        self.onnx_path = onnx_path
        self.host = host
        self.port = port
        self.controllers = {}  # Per-client controllers

    async def handle_client(self, websocket):
        """Handle a single WebSocket client"""
        client_id = id(websocket)
        print(f"Client connected: {client_id}")

        # Create controller for this client
        self.controllers[client_id] = OpenDuckController(self.model_path, self.onnx_path)
        controller = self.controllers[client_id]

        try:
            # Send initial state
            state = controller.get_state()
            await websocket.send(json.dumps({'type': 'state', 'data': state}))

            async for message in websocket:
                try:
                    msg = json.loads(message)
                    msg_type = msg.get('type')

                    if msg_type == 'step':
                        # Step simulation
                        use_rl = msg.get('use_rl', True)
                        state = controller.step(use_rl)
                        await websocket.send(json.dumps({'type': 'state', 'data': state}))

                    elif msg_type == 'reset':
                        # Reset simulation
                        state = controller.reset()
                        await websocket.send(json.dumps({'type': 'state', 'data': state}))

                    elif msg_type == 'command':
                        # Set velocity command
                        controller.set_command(
                            msg.get('lin_vel_x', 0),
                            msg.get('lin_vel_y', 0),
                            msg.get('ang_vel', 0)
                        )
                        await websocket.send(json.dumps({'type': 'ack'}))

                    elif msg_type == 'get_obs':
                        # Get observation (for training)
                        obs = controller.get_observation()
                        await websocket.send(json.dumps({
                            'type': 'observation',
                            'data': obs.tolist()
                        }))

                    elif msg_type == 'apply_action':
                        # Apply action directly (for training)
                        action = np.array(msg['action'], dtype=np.float32)
                        controller.motor_targets = controller.default_actuator + action * controller.config.action_scale
                        controller.data.ctrl[:14] = controller.motor_targets

                        for _ in range(controller.config.decimation):
                            mujoco.mj_step(controller.model, controller.data)

                        state = controller.get_state()
                        reward = self.compute_reward(controller)
                        await websocket.send(json.dumps({
                            'type': 'step_result',
                            'state': state,
                            'reward': reward,
                            'done': state['fallen']
                        }))

                except json.JSONDecodeError:
                    await websocket.send(json.dumps({'type': 'error', 'message': 'Invalid JSON'}))
                except Exception as e:
                    await websocket.send(json.dumps({'type': 'error', 'message': str(e)}))

        finally:
            del self.controllers[client_id]
            print(f"Client disconnected: {client_id}")

    def compute_reward(self, controller: OpenDuckController) -> float:
        """Compute reward for RL training"""
        # Simple reward: alive bonus + forward velocity + posture
        alive_bonus = 1.0

        # Forward velocity reward
        vel_x = controller.data.qvel[0]
        target_vel = controller.commands[0]
        vel_reward = -abs(vel_x - target_vel) * 2.0

        # Upright reward
        qw, qx, qy, qz = controller.data.qpos[3:7]
        up_z = 1 - 2 * (qx * qx + qy * qy)
        upright_reward = up_z * 0.5

        # Height reward
        height = controller.data.qpos[2]
        height_reward = min(height / 0.15, 1.0) * 0.5

        # Action smoothness penalty
        action_diff = np.sum(np.square(controller.last_action - controller.last_last_action))
        smooth_penalty = -action_diff * 0.01

        total_reward = alive_bonus + vel_reward + upright_reward + height_reward + smooth_penalty

        if controller.check_fallen():
            total_reward = -10.0

        return float(total_reward)

    async def run(self):
        """Run the WebSocket server"""
        print(f"Starting MuJoCo WebSocket server on ws://{self.host}:{self.port}")
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever


def main():
    parser = argparse.ArgumentParser(description='MuJoCo WebSocket Server for OpenDuck')
    parser.add_argument('--model', default='../assets/scenes/openduck/scene_flat_terrain.xml',
                        help='Path to MuJoCo XML model')
    parser.add_argument('--onnx', default='../assets/BEST_WALK_ONNX.onnx',
                        help='Path to ONNX policy')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8765, help='Server port')
    args = parser.parse_args()

    server = MuJoCoServer(args.model, args.onnx, args.host, args.port)
    asyncio.run(server.run())


if __name__ == '__main__':
    main()
