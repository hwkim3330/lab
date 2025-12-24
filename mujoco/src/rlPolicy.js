// OpenDuck Mini RL Policy using ONNX Runtime Web

export class RLPolicy {
  constructor() {
    this.session = null;
    this.loaded = false;

    // Scaling factors (from Python code)
    this.linearVelocityScale = 1.0;
    this.angularVelocityScale = 1.0;
    this.dofPosScale = 1.0;
    this.dofVelScale = 0.05;
    this.actionScale = 0.25;

    // Default actuator positions (from keyframe)
    this.defaultActuator = new Float32Array([
      0.002, 0.053, -0.63, 1.368, -0.784,  // left leg
      0, 0, 0, 0,                           // head
      -0.003, -0.065, 0.635, 1.379, -0.796  // right leg
    ]);

    // Action history
    this.lastAction = new Float32Array(14);
    this.lastLastAction = new Float32Array(14);
    this.lastLastLastAction = new Float32Array(14);
    this.motorTargets = new Float32Array(this.defaultActuator);
    this.prevMotorTargets = new Float32Array(this.defaultActuator);

    // Motor speed limits (from Python code)
    this.maxMotorVelocity = 5.24; // rad/s
    this.simDt = 0.002;

    // Phase for walking gait
    this.imitationPhase = [0, 0];
    this.phaseStep = 0;
    this.phaseFrequency = 1.0;

    // Commands (lin_vel_x, lin_vel_y, ang_vel, neck, head_pitch, head_yaw, head_roll)
    this.commands = new Float32Array([0, 0, 0, 0, 0, 0, 0]); // Standing
  }

  async load(modelPath) {
    try {
      this.session = await ort.InferenceSession.create(modelPath);
      this.loaded = true;
      console.log('RL Policy loaded successfully');
      return true;
    } catch (e) {
      console.error('Failed to load RL policy:', e);
      return false;
    }
  }

  getObservation(data, model) {
    // Extract sensor data from MuJoCo
    const qpos = data.qpos;
    const qvel = data.qvel;
    const sensordata = data.sensordata;

    let gyro, accelerometer;

    // Sensor data layout (from Python):
    // 0-2: gyro (3)
    // 3-5: local_linvel (3)
    // 6-8: accelerometer (3)
    // Check if sensor data is available and valid
    if (sensordata && sensordata.length >= 9 && !isNaN(sensordata[0])) {
      // Gyro from sensor data (sensor 0: gyro, 3 values)
      gyro = [sensordata[0], sensordata[1], sensordata[2]];
      // Accelerometer from sensor data (starts at index 6)
      // Python adds 1.3 to x-component as adjustment
      // After physics settles: accel ~ [1.3, 0, 9.8], after +1.3: [2.6, 0, 9.8]
      accelerometer = [sensordata[6] + 1.3, sensordata[7], sensordata[8]];
    } else {
      // Fallback: use qvel for angular velocity (gyro)
      gyro = [qvel[3], qvel[4], qvel[5]];
      // Fallback: accelerometer after +1.3 adjustment when standing
      // Raw ~ [1.3, 0, 9.8], after +1.3 ~ [2.6, 0, 9.8]
      accelerometer = [2.6, 0.0, 9.81];
    }

    // Joint angles (actuated joints: indices 7-20 in qpos for 14 DOF)
    // qpos layout: [x, y, z, qw, qx, qy, qz, joint0, joint1, ...]
    const jointAngles = new Float32Array(14);
    for (let i = 0; i < 14; i++) {
      jointAngles[i] = (qpos[7 + i] - this.defaultActuator[i]) * this.dofPosScale;
    }

    // Joint velocities (indices 6-19 in qvel for 14 DOF)
    // qvel layout: [vx, vy, vz, wx, wy, wz, joint0_vel, joint1_vel, ...]
    const jointVel = new Float32Array(14);
    for (let i = 0; i < 14; i++) {
      jointVel[i] = qvel[6 + i] * this.dofVelScale;
    }

    // Foot contacts - when standing, both feet should be in contact
    // Python shows (True, True) when standing on ground
    // TODO: implement proper contact detection from MuJoCo contact data
    const height = qpos[2];
    const contacts = height < 0.2 ? [1.0, 1.0] : [0.0, 0.0];

    // Imitation phase - starts at [0, 0] initially, then updates
    // Phase update happens AFTER observation is used (like in Python)
    const nbStepsInPeriod = 50;

    // Build observation vector (101 dimensions)
    const obs = new Float32Array(101);
    let idx = 0;

    // Gyro (3)
    obs[idx++] = gyro[0]; obs[idx++] = gyro[1]; obs[idx++] = gyro[2];

    // Accelerometer (3)
    obs[idx++] = accelerometer[0]; obs[idx++] = accelerometer[1]; obs[idx++] = accelerometer[2];

    // Commands (7)
    for (let i = 0; i < 7; i++) obs[idx++] = this.commands[i];

    // Joint angles - default (14)
    for (let i = 0; i < 14; i++) obs[idx++] = jointAngles[i];

    // Joint velocities (14)
    for (let i = 0; i < 14; i++) obs[idx++] = jointVel[i];

    // Last action (14)
    for (let i = 0; i < 14; i++) obs[idx++] = this.lastAction[i];

    // Last last action (14)
    for (let i = 0; i < 14; i++) obs[idx++] = this.lastLastAction[i];

    // Last last last action (14)
    for (let i = 0; i < 14; i++) obs[idx++] = this.lastLastLastAction[i];

    // Motor targets (14)
    for (let i = 0; i < 14; i++) obs[idx++] = this.motorTargets[i];

    // Contacts (2)
    obs[idx++] = contacts[0]; obs[idx++] = contacts[1];

    // Imitation phase (2)
    obs[idx++] = this.imitationPhase[0]; obs[idx++] = this.imitationPhase[1];

    return obs;
  }

  async infer(data, model) {
    if (!this.loaded) return null;

    const obs = this.getObservation(data, model);

    // Debug: log first observation
    if (this.debugCount === undefined) this.debugCount = 0;
    if (this.debugCount < 5) {
      console.log(`Observation #${this.debugCount}:`, {
        gyro: [obs[0].toFixed(4), obs[1].toFixed(4), obs[2].toFixed(4)],
        accel: [obs[3].toFixed(4), obs[4].toFixed(4), obs[5].toFixed(4)],
        commands: Array.from(obs.slice(6, 13)).map(v => v.toFixed(3)),
        jointAngles: Array.from(obs.slice(13, 27)).map(v => v.toFixed(3)),
        motorTargets: Array.from(obs.slice(69, 83)).map(v => v.toFixed(3)),
        contacts: [obs[83].toFixed(1), obs[84].toFixed(1)],
        phase: [obs[99].toFixed(3), obs[100].toFixed(3)],
        height: data.qpos[2].toFixed(4),
        sensorAvailable: data.sensordata && data.sensordata.length >= 9,
        rawSensor6_8: data.sensordata ? [data.sensordata[6], data.sensordata[7], data.sensordata[8]] : 'N/A'
      });
      this.debugCount++;
    }

    try {
      const inputTensor = new ort.Tensor('float32', obs, [1, 101]);
      const results = await this.session.run({ obs: inputTensor });
      const action = results.continuous_actions.data;

      // Update action history
      this.lastLastLastAction.set(this.lastLastAction);
      this.lastLastAction.set(this.lastAction);
      this.lastAction.set(action);

      // Calculate motor targets with speed limiting (from Python code)
      const decimation = 10;
      const maxChange = this.maxMotorVelocity * (this.simDt * decimation);

      for (let i = 0; i < 14; i++) {
        let target = this.defaultActuator[i] + action[i] * this.actionScale;
        // Clamp to prevent too fast motor movement
        const minTarget = this.prevMotorTargets[i] - maxChange;
        const maxTarget = this.prevMotorTargets[i] + maxChange;
        this.motorTargets[i] = Math.max(minTarget, Math.min(maxTarget, target));
      }

      // Update previous targets
      this.prevMotorTargets.set(this.motorTargets);

      // Update imitation phase AFTER using it (like Python)
      const nbStepsInPeriod = 50;
      this.phaseStep += this.phaseFrequency;
      this.phaseStep = this.phaseStep % nbStepsInPeriod;
      this.imitationPhase = [
        Math.cos(this.phaseStep / nbStepsInPeriod * 2 * Math.PI),
        Math.sin(this.phaseStep / nbStepsInPeriod * 2 * Math.PI)
      ];

      // Debug: log actions
      if (this.debugCount <= 3) {
        console.log('Actions:', Array.from(action).map(a => a.toFixed(3)));
        console.log('Motor targets:', Array.from(this.motorTargets).map(t => t.toFixed(3)));
      }

      return this.motorTargets;
    } catch (e) {
      console.error('Inference error:', e);
      return null;
    }
  }

  setCommand(linVelX, linVelY, angVel) {
    this.commands[0] = linVelX;
    this.commands[1] = linVelY;
    this.commands[2] = angVel;
  }

  reset() {
    this.lastAction.fill(0);
    this.lastLastAction.fill(0);
    this.lastLastLastAction.fill(0);
    this.motorTargets.set(this.defaultActuator);
    this.prevMotorTargets.set(this.defaultActuator);
    this.phaseStep = 0;
    this.imitationPhase = [0, 0];
    this.debugCount = 0;
  }
}
