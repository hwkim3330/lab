// TensorFlow.js Online Training for OpenDuck Mini
// PPO (Proximal Policy Optimization) implementation for browser-based RL

export class PPOTrainer {
  constructor(obsSize = 101, actionSize = 14) {
    this.obsSize = obsSize;
    this.actionSize = actionSize;

    // PPO hyperparameters
    this.gamma = 0.99;           // Discount factor
    this.lambda = 0.95;          // GAE lambda
    this.clipRatio = 0.2;        // PPO clip ratio
    this.entropyCoef = 0.01;     // Entropy bonus
    this.valueCoef = 0.5;        // Value loss coefficient
    this.learningRate = 3e-4;
    this.batchSize = 64;
    this.epochs = 4;

    // Networks
    this.policy = null;
    this.valueNet = null;
    this.optimizer = null;

    // Experience buffer
    this.buffer = {
      observations: [],
      actions: [],
      rewards: [],
      values: [],
      logProbs: [],
      dones: []
    };

    this.isTraining = false;
    this.episodeReward = 0;
    this.totalSteps = 0;
    this.trainStats = { policyLoss: 0, valueLoss: 0, entropy: 0, avgReward: 0 };
  }

  async init() {
    if (typeof tf === 'undefined') {
      console.error('TensorFlow.js not loaded!');
      return false;
    }

    // Policy network (Actor) - outputs mean and log_std of Gaussian
    this.policy = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [this.obsSize], units: 256, activation: 'relu', kernelInitializer: 'heNormal' }),
        tf.layers.dense({ units: 256, activation: 'relu', kernelInitializer: 'heNormal' }),
        tf.layers.dense({ units: 128, activation: 'relu', kernelInitializer: 'heNormal' }),
        tf.layers.dense({ units: this.actionSize * 2, activation: 'linear' }) // mean + log_std
      ]
    });

    // Value network (Critic)
    this.valueNet = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [this.obsSize], units: 256, activation: 'relu', kernelInitializer: 'heNormal' }),
        tf.layers.dense({ units: 256, activation: 'relu', kernelInitializer: 'heNormal' }),
        tf.layers.dense({ units: 128, activation: 'relu', kernelInitializer: 'heNormal' }),
        tf.layers.dense({ units: 1, activation: 'linear' })
      ]
    });

    this.optimizer = tf.train.adam(this.learningRate);

    console.log('PPO Trainer initialized');
    console.log('Policy params:', this.policy.countParams());
    console.log('Value params:', this.valueNet.countParams());

    return true;
  }

  // Get action from policy (for inference)
  getAction(observation, deterministic = false) {
    return tf.tidy(() => {
      const obs = tf.tensor2d([observation], [1, this.obsSize]);
      const output = this.policy.predict(obs);
      const outputArray = output.arraySync()[0];

      // Split into mean and log_std
      const mean = outputArray.slice(0, this.actionSize);
      const logStd = outputArray.slice(this.actionSize);

      if (deterministic) {
        return { action: mean, logProb: 0 };
      }

      // Sample from Gaussian
      const std = logStd.map(ls => Math.exp(Math.max(-5, Math.min(2, ls))));
      const action = mean.map((m, i) => m + std[i] * this.randn());

      // Compute log probability
      const logProb = this.computeLogProb(action, mean, std);

      return { action, logProb };
    });
  }

  // Standard normal random number
  randn() {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  // Compute log probability of Gaussian
  computeLogProb(action, mean, std) {
    let logProb = 0;
    for (let i = 0; i < action.length; i++) {
      const diff = action[i] - mean[i];
      logProb += -0.5 * Math.pow(diff / std[i], 2) - Math.log(std[i]) - 0.5 * Math.log(2 * Math.PI);
    }
    return logProb;
  }

  // Get value estimate
  getValue(observation) {
    return tf.tidy(() => {
      const obs = tf.tensor2d([observation], [1, this.obsSize]);
      const value = this.valueNet.predict(obs);
      return value.dataSync()[0];
    });
  }

  // Store transition in buffer
  storeTransition(obs, action, reward, value, logProb, done) {
    this.buffer.observations.push(obs);
    this.buffer.actions.push(action);
    this.buffer.rewards.push(reward);
    this.buffer.values.push(value);
    this.buffer.logProbs.push(logProb);
    this.buffer.dones.push(done);

    this.episodeReward += reward;
    this.totalSteps++;
  }

  // Compute GAE (Generalized Advantage Estimation)
  computeGAE(lastValue) {
    const rewards = this.buffer.rewards;
    const values = this.buffer.values;
    const dones = this.buffer.dones;
    const n = rewards.length;

    const advantages = new Array(n);
    const returns = new Array(n);

    let lastGaeLam = 0;
    for (let t = n - 1; t >= 0; t--) {
      const nextValue = t === n - 1 ? lastValue : values[t + 1];
      const nextNonTerminal = t === n - 1 ? 1 - dones[t] : 1 - dones[t];

      const delta = rewards[t] + this.gamma * nextValue * nextNonTerminal - values[t];
      lastGaeLam = delta + this.gamma * this.lambda * nextNonTerminal * lastGaeLam;
      advantages[t] = lastGaeLam;
      returns[t] = advantages[t] + values[t];
    }

    // Normalize advantages
    const advMean = advantages.reduce((a, b) => a + b, 0) / n;
    const advStd = Math.sqrt(advantages.reduce((a, b) => a + Math.pow(b - advMean, 2), 0) / n) + 1e-8;
    for (let i = 0; i < n; i++) {
      advantages[i] = (advantages[i] - advMean) / advStd;
    }

    return { advantages, returns };
  }

  // Train on collected experience
  async train(lastValue = 0) {
    if (this.buffer.observations.length < this.batchSize) {
      console.log('Not enough samples for training:', this.buffer.observations.length);
      return null;
    }

    this.isTraining = true;
    const { advantages, returns } = this.computeGAE(lastValue);

    // Prepare data
    const observations = tf.tensor2d(this.buffer.observations);
    const actions = tf.tensor2d(this.buffer.actions);
    const oldLogProbs = tf.tensor1d(this.buffer.logProbs);
    const advantagesTensor = tf.tensor1d(advantages);
    const returnsTensor = tf.tensor1d(returns);

    let totalPolicyLoss = 0;
    let totalValueLoss = 0;
    let totalEntropy = 0;
    let numUpdates = 0;

    // Training epochs
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      // Shuffle indices
      const indices = tf.util.createShuffledIndices(this.buffer.observations.length);
      const numBatches = Math.floor(indices.length / this.batchSize);

      for (let b = 0; b < numBatches; b++) {
        const batchIndices = Array.from(indices.slice(b * this.batchSize, (b + 1) * this.batchSize));

        const batchObs = tf.gather(observations, batchIndices);
        const batchActions = tf.gather(actions, batchIndices);
        const batchOldLogProbs = tf.gather(oldLogProbs, batchIndices);
        const batchAdvantages = tf.gather(advantagesTensor, batchIndices);
        const batchReturns = tf.gather(returnsTensor, batchIndices);

        // Compute loss and gradients
        const { policyLoss, valueLoss, entropy } = await this.trainStep(
          batchObs, batchActions, batchOldLogProbs, batchAdvantages, batchReturns
        );

        totalPolicyLoss += policyLoss;
        totalValueLoss += valueLoss;
        totalEntropy += entropy;
        numUpdates++;

        // Dispose batch tensors
        batchObs.dispose();
        batchActions.dispose();
        batchOldLogProbs.dispose();
        batchAdvantages.dispose();
        batchReturns.dispose();
      }
    }

    // Cleanup
    observations.dispose();
    actions.dispose();
    oldLogProbs.dispose();
    advantagesTensor.dispose();
    returnsTensor.dispose();

    // Update stats
    this.trainStats = {
      policyLoss: totalPolicyLoss / numUpdates,
      valueLoss: totalValueLoss / numUpdates,
      entropy: totalEntropy / numUpdates,
      avgReward: this.episodeReward / this.buffer.rewards.length
    };

    // Clear buffer
    this.clearBuffer();
    this.isTraining = false;

    console.log('Training stats:', this.trainStats);
    return this.trainStats;
  }

  async trainStep(obs, actions, oldLogProbs, advantages, returns) {
    let policyLoss = 0, valueLoss = 0, entropy = 0;

    const grads = tf.tidy(() => {
      // Policy forward pass
      const policyOutput = this.policy.predict(obs);
      const mean = policyOutput.slice([0, 0], [-1, this.actionSize]);
      const logStd = policyOutput.slice([0, this.actionSize], [-1, this.actionSize]);
      const std = tf.exp(tf.clipByValue(logStd, -5, 2));

      // New log probabilities
      const diff = tf.sub(actions, mean);
      const logProbComponents = tf.add(
        tf.mul(tf.scalar(-0.5), tf.square(tf.div(diff, std))),
        tf.add(tf.neg(tf.log(std)), tf.scalar(-0.5 * Math.log(2 * Math.PI)))
      );
      const newLogProbs = tf.sum(logProbComponents, 1);

      // Ratio and clipped ratio
      const ratio = tf.exp(tf.sub(newLogProbs, oldLogProbs));
      const clippedRatio = tf.clipByValue(ratio, 1 - this.clipRatio, 1 + this.clipRatio);

      // Policy loss (negative because we want to maximize)
      const surr1 = tf.mul(ratio, advantages);
      const surr2 = tf.mul(clippedRatio, advantages);
      const policyLossTensor = tf.neg(tf.mean(tf.minimum(surr1, surr2)));

      // Value loss
      const values = this.valueNet.predict(obs).squeeze();
      const valueLossTensor = tf.mean(tf.square(tf.sub(values, returns)));

      // Entropy bonus (encourage exploration)
      const entropyTensor = tf.mean(tf.sum(tf.add(logStd, tf.scalar(0.5 * Math.log(2 * Math.PI * Math.E))), 1));

      // Total loss
      const totalLoss = tf.add(
        policyLossTensor,
        tf.add(
          tf.mul(tf.scalar(this.valueCoef), valueLossTensor),
          tf.mul(tf.scalar(-this.entropyCoef), entropyTensor)
        )
      );

      policyLoss = policyLossTensor.dataSync()[0];
      valueLoss = valueLossTensor.dataSync()[0];
      entropy = entropyTensor.dataSync()[0];

      return totalLoss;
    });

    // Apply gradients
    const allVars = [...this.policy.trainableWeights, ...this.valueNet.trainableWeights];
    this.optimizer.minimize(() => grads, false, allVars);

    return { policyLoss, valueLoss, entropy };
  }

  clearBuffer() {
    this.buffer = {
      observations: [],
      actions: [],
      rewards: [],
      values: [],
      logProbs: [],
      dones: []
    };
    this.episodeReward = 0;
  }

  // Save model to browser storage
  async save(name = 'openduck-ppo') {
    await this.policy.save(`localstorage://${name}-policy`);
    await this.valueNet.save(`localstorage://${name}-value`);
    console.log(`Models saved as ${name}`);
  }

  // Load model from browser storage
  async load(name = 'openduck-ppo') {
    try {
      this.policy = await tf.loadLayersModel(`localstorage://${name}-policy`);
      this.valueNet = await tf.loadLayersModel(`localstorage://${name}-value`);
      console.log(`Models loaded from ${name}`);
      return true;
    } catch (e) {
      console.log('No saved model found, using initialized networks');
      return false;
    }
  }

  // Export model as downloadable file
  async export(name = 'openduck-ppo') {
    await this.policy.save(`downloads://${name}-policy`);
    await this.valueNet.save(`downloads://${name}-value`);
    console.log('Models exported to downloads');
  }

  getStats() {
    return {
      ...this.trainStats,
      totalSteps: this.totalSteps,
      bufferSize: this.buffer.observations.length
    };
  }
}

// Reward function for OpenDuck walking
export function computeReward(state, commands, prevState = null) {
  const qpos = state.qpos;
  const qvel = state.qvel;

  // Height reward
  const height = qpos[2];
  const heightReward = Math.min(height / 0.15, 1.0) * 0.5;

  // Upright reward (from quaternion)
  const qw = qpos[3], qx = qpos[4], qy = qpos[5], qz = qpos[6];
  const upZ = 1 - 2 * (qx * qx + qy * qy);
  const uprightReward = upZ * 0.5;

  // Velocity tracking reward
  const velX = qvel[0];
  const targetVelX = commands[0];
  const velReward = -Math.abs(velX - targetVelX) * 2.0;

  // Alive bonus
  const aliveBonus = 1.0;

  // Fall penalty
  if (state.fallen) {
    return -10.0;
  }

  return aliveBonus + heightReward + uprightReward + velReward;
}
