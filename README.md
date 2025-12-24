# OpenDuck Mini - MuJoCo Web Simulation

Browser-based physics simulation of the OpenDuck Mini bipedal robot using MuJoCo WASM and ONNX neural network control.

## Demo

**[Try it Live](https://hwkim3330.github.io/lab/)**

## Features

- **Real-time Physics**: MuJoCo physics engine running in WebAssembly
- **Neural Network Control**: ONNX-based walking policy trained with reinforcement learning
- **Interactive 3D**: Three.js rendering with orbit controls
- **Keyboard Control**: WASD/Arrow keys to command the robot

## Controls

| Action | Keys |
|--------|------|
| Forward | W / Arrow Up |
| Backward | S / Arrow Down |
| Strafe Left | A / Arrow Left |
| Strafe Right | D / Arrow Right |
| Rotate Left | Q |
| Rotate Right | E |
| Pause | Space |

## Tech Stack

- **Physics**: [MuJoCo WASM](https://github.com/google-deepmind/mujoco)
- **Rendering**: [Three.js](https://threejs.org/)
- **Neural Network**: [ONNX Runtime Web](https://onnxruntime.ai/)
- **Robot Model**: [OpenDuck Mini v2](https://github.com/apirrone/Open_Duck_Mini)

## Architecture

```
mujoco/
├── src/
│   ├── main.js           # Main application
│   ├── onnxController.js # ONNX neural network controller
│   ├── mujocoUtils.js    # MuJoCo utilities
│   └── utils/            # Helpers
├── assets/
│   ├── scenes/openduck/  # Robot MJCF model
│   └── models/           # ONNX model
└── node_modules/
    ├── mujoco-js/        # MuJoCo WASM
    └── three/            # Three.js
```

## Neural Network Details

The walking policy is a neural network trained using reinforcement learning:

- **Input**: 101-dimensional observation vector
  - Gyroscope (3)
  - Accelerometer (3)
  - Commands (7)
  - Joint angles (14)
  - Joint velocities (14)
  - Action history (42)
  - Motor targets (14)
  - Foot contacts (2)
  - Phase (2)

- **Output**: 14-dimensional action (motor targets)

- **Parameters**:
  - Action scale: 0.25
  - Control frequency: 50Hz (decimation=10, sim_dt=0.002)
  - Max motor velocity: 5.24 rad/s

## References

- [RoboPianist](https://kzakka.com/robopianist/) - Similar MuJoCo web demo
- [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground) - Training code

## License

MIT
