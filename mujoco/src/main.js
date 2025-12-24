
import * as THREE           from 'three';
import { GUI              } from '../node_modules/three/examples/jsm/libs/lil-gui.module.min.js';
import { OrbitControls    } from '../node_modules/three/examples/jsm/controls/OrbitControls.js';
import { DragStateManager } from './utils/DragStateManager.js';
import { setupGUI, downloadExampleScenesFolder, loadSceneFromURL, drawTendonsAndFlex, getPosition, getQuaternion, toMujocoPos, standardNormal } from './mujocoUtils.js';
import   load_mujoco        from '../node_modules/mujoco-js/dist/mujoco_wasm.js';

// Check WebGPU support
const hasWebGPU = navigator.gpu !== undefined;

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
    this.params = { scene: initialScene, paused: false, help: false, ctrlnoiserate: 0.0, ctrlnoisestd: 0.0, keyframeNumber: 0, walking: true };
    this.mujoco_time = 0.0;
    this.walkPhase = 0.0;
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
    }

    this.gui = new GUI();
    setupGUI(this);

    // Add walking toggle
    this.gui.add(this.params, 'walking').name('Walking');

    // Add quick reset button
    this.gui.add({
      reset: () => {
        if (this.model.nkey > 0) {
          this.data.qpos.set(this.model.key_qpos.slice(0, this.model.nq));
          this.data.ctrl.set(this.model.key_ctrl.slice(0, this.model.nu));
          this.walkPhase = 0;
          mujoco.mj_forward(this.model, this.data);
        }
      }
    }, 'reset').name('Reset Robot (R)');

    // Add keyboard shortcut for reset
    document.addEventListener('keydown', (e) => {
      if (e.key === 'r' || e.key === 'R') {
        if (this.model.nkey > 0) {
          this.data.qpos.set(this.model.key_qpos.slice(0, this.model.nq));
          this.data.ctrl.set(this.model.key_ctrl.slice(0, this.model.nu));
          this.walkPhase = 0;
          mujoco.mj_forward(this.model, this.data);
        }
      }
    });
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize( window.innerWidth, window.innerHeight );
  }

  // Walking gait generator for OpenDuck Mini
  // Actuator order:
  // 0: left_hip_yaw, 1: left_hip_roll, 2: left_hip_pitch, 3: left_knee, 4: left_ankle
  // 5: neck_pitch, 6: head_pitch, 7: head_yaw, 8: head_roll
  // 9: right_hip_yaw, 10: right_hip_roll, 11: right_hip_pitch, 12: right_knee, 13: right_ankle
  applyWalkingControl() {
    const freq = 1.5; // Walking frequency Hz
    this.walkPhase += this.model.opt.timestep * freq * 2 * Math.PI;
    const phase = this.walkPhase;

    const ctrl = this.data.ctrl;
    if (ctrl.length < 14) return;

    // Standing pose (from keyframe)
    const stand = {
      left_hip_yaw: 0.002, left_hip_roll: 0.053, left_hip_pitch: -0.63, left_knee: 1.368, left_ankle: -0.784,
      right_hip_yaw: -0.003, right_hip_roll: -0.065, right_hip_pitch: 0.635, right_knee: 1.379, right_ankle: -0.796
    };

    // Walking amplitudes
    const amp_pitch = 0.25;
    const amp_knee = 0.3;
    const amp_ankle = 0.15;

    // Left leg (phase 0)
    ctrl[0] = stand.left_hip_yaw;
    ctrl[1] = stand.left_hip_roll;
    ctrl[2] = stand.left_hip_pitch + amp_pitch * Math.sin(phase);
    ctrl[3] = stand.left_knee + amp_knee * Math.sin(phase);
    ctrl[4] = stand.left_ankle + amp_ankle * Math.sin(phase);

    // Head stable
    ctrl[5] = 0; ctrl[6] = 0; ctrl[7] = 0; ctrl[8] = 0;

    // Right leg (phase PI - opposite)
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

        // Apply walking control if enabled
        if (this.params["walking"] && this.params.scene.includes("openduck")) {
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
