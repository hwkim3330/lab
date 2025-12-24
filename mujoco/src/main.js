
import * as THREE           from 'three';
import { GUI              } from '../node_modules/three/examples/jsm/libs/lil-gui.module.min.js';
import { OrbitControls    } from '../node_modules/three/examples/jsm/controls/OrbitControls.js';
import { DragStateManager } from './utils/DragStateManager.js';
import { setupGUI, downloadExampleScenesFolder, loadSceneFromURL, drawTendonsAndFlex, getPosition, getQuaternion, toMujocoPos, standardNormal } from './mujocoUtils.js';
import   load_mujoco        from '../node_modules/mujoco-js/dist/mujoco_wasm.js';
import { RLPolicy } from './rlPolicy.js';
import { PPOTrainer, computeReward } from './tfTraining.js';
import { MuJoCoWebSocket, SimulationManager } from './wsClient.js';

// Check WebGPU support
const hasWebGPU = navigator.gpu !== undefined;

// RL Policy for learned walking (ONNX inference)
const rlPolicy = new RLPolicy();

// PPO Trainer for online learning (TensorFlow.js)
const ppoTrainer = new PPOTrainer();

// Simulation manager for WASM/WebSocket mode switching
const simManager = new SimulationManager();

// Load the MuJoCo Module
const mujoco = await load_mujoco();

// Set up Emscripten's Virtual File System
mujoco.FS.mkdir('/working');
mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');

// Download all files first (including OpenDuck)
await downloadExampleScenesFolder(mujoco);

// Now load OpenDuck as default
var initialScene = "openduck/scene_flat_terrain.xml";

export class MuJoCoDemo {
  constructor() {
    this.mujoco = mujoco;

    // Load in the state from XML
    this.model = mujoco.MjModel.loadFromXML("/working/" + initialScene);
    this.data  = new mujoco.MjData(this.model);

    // Define Random State Variables
    this.params = {
      scene: initialScene,
      paused: false,
      help: false,
      ctrlnoiserate: 0.0,
      ctrlnoisestd: 0.0,
      keyframeNumber: 0,
      walking: true,
      autoReset: false,
      useRLPolicy: true,
      // Training mode
      training: false,
      trainBatchSize: 256,
      simMode: 'wasm'  // 'wasm' or 'websocket'
    };
    this.mujoco_time = 0.0;
    this.walkPhase = 0.0;
    this.fallCount = 0;
    this.rlPolicyLoaded = false;
    this.inferenceStep = 0;
    this.decimation = 10; // Run policy every N steps (matches Python)
    this.inferenceRunning = false; // Prevent overlapping inferences

    // Training state
    this.ppoReady = false;
    this.trainingEpisode = 0;
    this.trainingStep = 0;
    this.lastObs = null;
    this.lastAction = null;
    this.lastValue = null;
    this.lastLogProb = null;
    this.bodies  = {}, this.lights = {};
    this.tmpVec  = new THREE.Vector3();
    this.tmpQuat = new THREE.Quaternion();
    this.updateGUICallbacks = [];

    this.container = document.createElement( 'div' );
    document.body.appendChild( this.container );

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 0.001, 100 );
    this.camera.name = 'PerspectiveCamera';
    this.camera.position.set(0.5, 0.4, 0.8);
    this.scene.add(this.camera);

    this.scene.background = new THREE.Color(0.15, 0.25, 0.35);
    this.scene.fog = new THREE.Fog(this.scene.background, 15, 25.5 );

    this.ambientLight = new THREE.AmbientLight( 0xffffff, 0.1 * 3.14 );
    this.ambientLight.name = 'AmbientLight';
    this.scene.add( this.ambientLight );

    this.spotlight = new THREE.SpotLight();
    this.spotlight.angle = 1.11;
    this.spotlight.distance = 10000;
    this.spotlight.penumbra = 0.5;
    this.spotlight.castShadow = true;
    this.spotlight.intensity = this.spotlight.intensity * 3.14 * 10.0;
    this.spotlight.shadow.mapSize.width = 1024;
    this.spotlight.shadow.mapSize.height = 1024;
    this.spotlight.shadow.camera.near = 0.1;
    this.spotlight.shadow.camera.far = 100;
    this.spotlight.position.set(0, 3, 3);
    const targetObject = new THREE.Object3D();
    this.scene.add(targetObject);
    this.spotlight.target = targetObject;
    targetObject.position.set(0, 0.15, 0);
    this.scene.add( this.spotlight );

    this.renderer = new THREE.WebGLRenderer( { antialias: true } );
    this.renderer.setPixelRatio(1.0);
    this.renderer.setSize( window.innerWidth, window.innerHeight );
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    THREE.ColorManagement.enabled = false;
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.useLegacyLights = true;

    this.renderer.setAnimationLoop( this.render.bind(this) );

    this.container.appendChild( this.renderer.domElement );

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.15, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.10;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    window.addEventListener('resize', this.onWindowResize.bind(this));

    // Initialize the Drag State Manager.
    this.dragStateManager = new DragStateManager(this.scene, this.renderer, this.camera, this.container.parentElement, this.controls);
  }

  async init() {
    // Initialize the three.js Scene using the .xml Model in initialScene
    [this.model, this.data, this.bodies, this.lights] =
      await loadSceneFromURL(mujoco, initialScene, this);

    // Load keyframe to start standing
    if (this.model.nkey > 0) {
      this.data.qpos.set(this.model.key_qpos.slice(0, this.model.nq));
      this.data.ctrl.set(this.model.key_ctrl.slice(0, this.model.nu));
      mujoco.mj_forward(this.model, this.data);
      console.log('Initial qpos (joints):', Array.from(this.data.qpos).slice(7, 21).map(v => v.toFixed(3)));
      console.log('Initial ctrl:', Array.from(this.data.ctrl).map(v => v.toFixed(3)));
      console.log('Trunk height:', this.data.qpos[2].toFixed(3));
    }

    this.gui = new GUI();
    setupGUI(this);

    // Load RL policy
    const policyLoaded = await rlPolicy.load('./assets/BEST_WALK_ONNX.onnx');
    this.rlPolicyLoaded = policyLoaded;
    if (policyLoaded) {
      console.log('RL Policy ready!');
      console.log('Initial motor targets:', Array.from(rlPolicy.motorTargets).map(t => t.toFixed(3)));
      console.log('Default actuator:', Array.from(rlPolicy.defaultActuator).map(t => t.toFixed(3)));
    }

    // Add walking toggle
    this.gui.add(this.params, 'walking').name('Walking');
    this.gui.add(this.params, 'useRLPolicy').name('Use RL Policy');

    // Add quick reset button
    this.gui.add({
      reset: () => { this.resetRobot(); }
    }, 'reset').name('Reset Robot (R)');

    // Add auto-reset toggle
    this.gui.add(this.params, 'autoReset').name('Auto Reset on Fall');

    // Initialize PPO trainer for online learning
    try {
      this.ppoReady = await ppoTrainer.init();
      if (this.ppoReady) {
        console.log('PPO Trainer ready for online learning!');
        // Try to load saved model
        await ppoTrainer.load();
      }
    } catch (e) {
      console.log('PPO Trainer init failed (TensorFlow.js not loaded?):', e);
    }

    // Training controls folder
    const trainingFolder = this.gui.addFolder('Training');
    trainingFolder.add(this.params, 'training').name('Enable Training').onChange((val) => {
      if (val) {
        this.params.autoReset = true;  // Auto-reset during training
        this.trainingStep = 0;
        this.trainingEpisode++;
        console.log(`Training Episode ${this.trainingEpisode} started`);
      }
    });
    trainingFolder.add(this.params, 'trainBatchSize', 64, 1024, 64).name('Batch Size');
    trainingFolder.add({
      train: async () => {
        if (ppoTrainer.buffer.observations.length > 0) {
          const lastValue = ppoTrainer.getValue(rlPolicy.getObservation(this.data, this.model));
          await ppoTrainer.train(lastValue);
        }
      }
    }, 'train').name('Train Now');
    trainingFolder.add({
      save: async () => { await ppoTrainer.save(); }
    }, 'save').name('Save Model');
    trainingFolder.add({
      export: async () => { await ppoTrainer.export(); }
    }, 'export').name('Export Model');

    // Simulation mode selector
    const modeFolder = this.gui.addFolder('Simulation Mode');
    modeFolder.add(this.params, 'simMode', ['wasm', 'websocket']).name('Mode').onChange(async (mode) => {
      if (mode === 'websocket') {
        const connected = await simManager.connectWebSocket();
        if (!connected) {
          this.params.simMode = 'wasm';
          alert('WebSocket connection failed. Using WASM mode.\nStart the Python server: python server/mujoco_server.py');
        }
      }
    });
    modeFolder.add({
      connect: async () => {
        const connected = await simManager.connectWebSocket();
        if (connected) {
          this.params.simMode = 'websocket';
          console.log('Connected to Python backend');
        }
      }
    }, 'connect').name('Connect to Server');

    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.key === 'r' || e.key === 'R') {
        this.resetRobot();
      }
      // Walking direction controls (WASD or arrows)
      if (this.rlPolicyLoaded) {
        const SPEED = 0.15;
        const TURN = 1.0;
        if (e.key === 'ArrowUp' || e.key === 'w') {
          rlPolicy.setCommand(SPEED, 0, 0);
        } else if (e.key === 'ArrowDown' || e.key === 's') {
          rlPolicy.setCommand(-SPEED, 0, 0);
        } else if (e.key === 'ArrowLeft' || e.key === 'a') {
          rlPolicy.setCommand(0, SPEED, 0);
        } else if (e.key === 'ArrowRight' || e.key === 'd') {
          rlPolicy.setCommand(0, -SPEED, 0);
        } else if (e.key === 'q') {
          rlPolicy.setCommand(0, 0, TURN);
        } else if (e.key === 'e') {
          rlPolicy.setCommand(0, 0, -TURN);
        } else if (e.key === ' ') {
          rlPolicy.setCommand(0, 0, 0); // Stop
        }
      }
    });
  }

  resetRobot() {
    if (this.model.nkey > 0) {
      this.data.qpos.set(this.model.key_qpos.slice(0, this.model.nq));
      this.data.ctrl.set(this.model.key_ctrl.slice(0, this.model.nu));
      this.walkPhase = 0;
      this.inferenceStep = 0;
      this.inferenceRunning = false;
      rlPolicy.reset(); // Reset action history

      // Reset training state
      this.lastObs = null;
      this.lastAction = null;
      this.lastValue = null;
      this.lastLogProb = null;

      mujoco.mj_forward(this.model, this.data);
      this.fallCount++;
      console.log(`Reset #${this.fallCount}`);
    }
  }

  checkFallen() {
    // Check if robot has fallen (trunk height < 0.08m or tilted > 60 degrees)
    // qpos[0,1,2] = position, qpos[3,4,5,6] = quaternion
    const trunkZ = this.data.qpos[2];
    const qw = this.data.qpos[3];
    const qx = this.data.qpos[4];
    const qy = this.data.qpos[5];
    const qz = this.data.qpos[6];

    // Calculate up vector from quaternion
    const upX = 2 * (qx * qz + qw * qy);
    const upY = 2 * (qy * qz - qw * qx);
    const upZ = 1 - 2 * (qx * qx + qy * qy);

    // If trunk too low or tilted too much
    return trunkZ < 0.08 || upZ < 0.5;
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize( window.innerWidth, window.innerHeight );
  }

  // Walking control - either RL policy or simple sinusoidal
  applyWalkingControl() {
    const ctrl = this.data.ctrl;
    if (ctrl.length < 14) return;

    // Training mode: use PPO policy and collect experience
    if (this.params.training && this.ppoReady && this.params.walking) {
      this.inferenceStep++;
      if (this.inferenceStep >= this.decimation) {
        this.inferenceStep = 0;

        // Get observation
        const obs = rlPolicy.getObservation(this.data, this.model);

        // Store previous transition if we have one
        if (this.lastObs !== null) {
          const state = {
            qpos: Array.from(this.data.qpos),
            qvel: Array.from(this.data.qvel),
            fallen: this.checkFallen()
          };
          const reward = computeReward(state, rlPolicy.commands);
          const done = state.fallen;

          ppoTrainer.storeTransition(
            this.lastObs,
            this.lastAction,
            reward,
            this.lastValue,
            this.lastLogProb,
            done
          );

          this.trainingStep++;

          // Train when batch is full
          if (ppoTrainer.buffer.observations.length >= this.params.trainBatchSize) {
            const lastValue = ppoTrainer.getValue(obs);
            ppoTrainer.train(lastValue).then(stats => {
              if (stats) {
                console.log(`Training step ${this.trainingStep}: reward=${stats.avgReward.toFixed(3)}`);
              }
            });
          }
        }

        // Get action from PPO policy
        const { action, logProb } = ppoTrainer.getAction(Array.from(obs));
        const value = ppoTrainer.getValue(Array.from(obs));

        // Store for next step
        this.lastObs = Array.from(obs);
        this.lastAction = action;
        this.lastValue = value;
        this.lastLogProb = logProb;

        // Apply action as motor targets
        for (let i = 0; i < 14; i++) {
          const target = rlPolicy.defaultActuator[i] + action[i] * rlPolicy.actionScale;
          ctrl[i] = target;
        }
      }
      return;
    }

    // Use RL policy if loaded and enabled AND walking mode is on
    if (this.params.walking && this.params.useRLPolicy && this.rlPolicyLoaded) {
      this.inferenceStep++;
      if (this.inferenceStep >= this.decimation && !this.inferenceRunning) {
        this.inferenceStep = 0;
        this.inferenceRunning = true;
        rlPolicy.infer(this.data, this.model).then(motorTargets => {
          if (motorTargets) {
            for (let i = 0; i < 14; i++) {
              this.data.ctrl[i] = motorTargets[i];
            }
          }
          this.inferenceRunning = false;
        }).catch(e => {
          console.error('Inference error:', e);
          this.inferenceRunning = false;
        });
      }
      // Always apply current motor targets (keeps control active between inferences)
      const targets = rlPolicy.motorTargets;
      for (let i = 0; i < 14; i++) {
        ctrl[i] = targets[i];
      }
      return;
    }

    // Standing mode: just hold the keyframe pose (for testing)
    if (!this.params.walking) {
      // Hold default standing pose
      const standingPose = [
        0.002, 0.053, -0.63, 1.368, -0.784,  // left leg
        0, 0, 0, 0,                           // head
        -0.003, -0.065, 0.635, 1.379, -0.796  // right leg
      ];
      for (let i = 0; i < 14; i++) {
        ctrl[i] = standingPose[i];
      }
      return;
    }

    // Fallback: simple sinusoidal gait
    const freq = 1.5;
    this.walkPhase += this.model.opt.timestep * freq * 2 * Math.PI;
    const phase = this.walkPhase;

    const stand = {
      left_hip_yaw: 0.002, left_hip_roll: 0.053, left_hip_pitch: -0.63, left_knee: 1.368, left_ankle: -0.784,
      right_hip_yaw: -0.003, right_hip_roll: -0.065, right_hip_pitch: 0.635, right_knee: 1.379, right_ankle: -0.796
    };

    const amp_pitch = 0.25, amp_knee = 0.3, amp_ankle = 0.15;

    ctrl[0] = stand.left_hip_yaw;
    ctrl[1] = stand.left_hip_roll;
    ctrl[2] = stand.left_hip_pitch + amp_pitch * Math.sin(phase);
    ctrl[3] = stand.left_knee + amp_knee * Math.sin(phase);
    ctrl[4] = stand.left_ankle + amp_ankle * Math.sin(phase);
    ctrl[5] = 0; ctrl[6] = 0; ctrl[7] = 0; ctrl[8] = 0;
    ctrl[9] = stand.right_hip_yaw;
    ctrl[10] = stand.right_hip_roll;
    ctrl[11] = stand.right_hip_pitch + amp_pitch * Math.sin(phase + Math.PI);
    ctrl[12] = stand.right_knee + amp_knee * Math.sin(phase + Math.PI);
    ctrl[13] = stand.right_ankle + amp_ankle * Math.sin(phase + Math.PI);
  }

  render(timeMS) {
    this.controls.update();

    if (!this.params["paused"]) {
      let timestep = this.model.opt.timestep;
      if (timeMS - this.mujoco_time > 35.0) { this.mujoco_time = timeMS; }
      while (this.mujoco_time < timeMS) {

        // Apply control for OpenDuck robot
        if (this.params.scene.includes("openduck")) {
          this.applyWalkingControl();
        }

        // Jitter the control state with gaussian random noise
        if (this.params["ctrlnoisestd"] > 0.0) {
          let rate  = Math.exp(-timestep / Math.max(1e-10, this.params["ctrlnoiserate"]));
          let scale = this.params["ctrlnoisestd"] * Math.sqrt(1 - rate * rate);
          let currentCtrl = this.data.ctrl;
          for (let i = 0; i < currentCtrl.length; i++) {
            currentCtrl[i] = rate * currentCtrl[i] + scale * standardNormal();
            this.params["Actuator " + i] = currentCtrl[i];
          }
        }

        // Clear old perturbations, apply new ones.
        for (let i = 0; i < this.data.qfrc_applied.length; i++) { this.data.qfrc_applied[i] = 0.0; }
        let dragged = this.dragStateManager.physicsObject;
        if (dragged && dragged.bodyID) {
          for (let b = 0; b < this.model.nbody; b++) {
            if (this.bodies[b]) {
              getPosition  (this.data.xpos , b, this.bodies[b].position);
              getQuaternion(this.data.xquat, b, this.bodies[b].quaternion);
              this.bodies[b].updateWorldMatrix();
            }
          }
          let bodyID = dragged.bodyID;
          this.dragStateManager.update();
          let force = toMujocoPos(this.dragStateManager.currentWorld.clone().sub(this.dragStateManager.worldHit).multiplyScalar(this.model.body_mass[bodyID] * 250));
          let point = toMujocoPos(this.dragStateManager.worldHit.clone());
          mujoco.mj_applyFT(this.model, this.data, [force.x, force.y, force.z], [0, 0, 0], [point.x, point.y, point.z], bodyID, this.data.qfrc_applied);
        }

        mujoco.mj_step(this.model, this.data);

        // Check for fall and auto-reset
        if (this.params.autoReset && this.params.scene.includes("openduck") && this.checkFallen()) {
          this.resetRobot();
        }

        this.mujoco_time += timestep * 1000.0;
      }

    } else if (this.params["paused"]) {
      this.dragStateManager.update();
      let dragged = this.dragStateManager.physicsObject;
      if (dragged && dragged.bodyID) {
        let b = dragged.bodyID;
        getPosition  (this.data.xpos , b, this.tmpVec , false);
        getQuaternion(this.data.xquat, b, this.tmpQuat, false);

        let offset = toMujocoPos(this.dragStateManager.currentWorld.clone()
          .sub(this.dragStateManager.worldHit).multiplyScalar(0.3));
        if (this.model.body_mocapid[b] >= 0) {
          console.log("Trying to move mocap body", b);
          let addr = this.model.body_mocapid[b] * 3;
          let pos  = this.data.mocap_pos;
          pos[addr+0] += offset.x;
          pos[addr+1] += offset.y;
          pos[addr+2] += offset.z;
        } else {
          let root = this.model.body_rootid[b];
          let addr = this.model.jnt_qposadr[this.model.body_jntadr[root]];
          let pos  = this.data.qpos;
          pos[addr+0] += offset.x;
          pos[addr+1] += offset.y;
          pos[addr+2] += offset.z;
        }
      }

      mujoco.mj_forward(this.model, this.data);
    }

    // Update body transforms.
    for (let b = 0; b < this.model.nbody; b++) {
      if (this.bodies[b]) {
        getPosition  (this.data.xpos , b, this.bodies[b].position);
        getQuaternion(this.data.xquat, b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix();
      }
    }

    // Update light transforms.
    for (let l = 0; l < this.model.nlight; l++) {
      if (this.lights[l]) {
        getPosition(this.data.light_xpos, l, this.lights[l].position);
        getPosition(this.data.light_xdir, l, this.tmpVec);
        this.lights[l].lookAt(this.tmpVec.add(this.lights[l].position));
      }
    }

    // Draw Tendons and Flex verts
    drawTendonsAndFlex(this.mujocoRoot, this.model, this.data);

    // Render!
    this.renderer.render( this.scene, this.camera );
  }
}

let demo = new MuJoCoDemo();
await demo.init();
