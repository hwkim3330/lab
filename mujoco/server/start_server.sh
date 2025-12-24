#!/bin/bash
# Start MuJoCo WebSocket Server for OpenDuck

cd "$(dirname "$0")"

# Install dependencies if needed
pip3 install -r requirements.txt --quiet 2>/dev/null || true

# Start server
echo "Starting MuJoCo WebSocket Server..."
echo "Connect from browser at ws://localhost:8765"
echo "Press Ctrl+C to stop"
echo ""

python3 mujoco_server.py --model ../assets/scenes/openduck/scene_flat_terrain.xml --onnx ../assets/BEST_WALK_ONNX.onnx "$@"
