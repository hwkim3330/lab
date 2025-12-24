# Mini Isaac - RL Roguelike

A lightweight Isaac-style roguelike game built for reinforcement learning experiments.

## Play Online

[Play Now](https://hwkim3330.github.io/lab/)

## Features

- **Procedural Dungeon Generation**: Each run generates a unique dungeon layout
- **4 Enemy Types**: Fly, Gaper, Shooter, Boss
- **Twin-stick Shooter Mechanics**: Move with WASD, shoot with IJKL
- **RL Interface**: Built-in observation/action space for training AI agents

## Controls

| Action | Keys |
|--------|------|
| Move | WASD |
| Shoot | IJKL (Up/Left/Down/Right) |
| Reset | R |

## Tech Stack

- **Game Engine**: Rust + macroquad
- **Web**: WebAssembly (WASM)
- **RL Training**: Python + stable-baselines3 (coming soon)

## Building from Source

```bash
# Native build
cd game
cargo build --release

# WASM build
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/mini-isaac.wasm ../web/
```

## RL Interface

The game exposes an observation/action interface for RL training:

### Observation Space
- Player position (normalized)
- Player HP
- Enemy positions and HP (up to 5 nearest)
- Tear positions
- Nearest enemy direction
- Room cleared status

### Action Space (13 discrete actions)
- 0: Noop
- 1-8: Move (8 directions)
- 9-12: Shoot (4 directions)

## Project Structure

```
lab/
├── game/              # Rust game source
│   ├── src/
│   │   └── main.rs
│   └── Cargo.toml
├── rl/                # Python RL (coming soon)
├── index.html         # GitHub Pages
└── mini-isaac.wasm    # WASM build
```

## License

MIT
