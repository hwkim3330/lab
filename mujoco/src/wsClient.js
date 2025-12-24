// WebSocket Client for Python MuJoCo Backend
// Provides accurate physics simulation via Python native MuJoCo

export class MuJoCoWebSocket {
  constructor() {
    this.ws = null;
    this.connected = false;
    this.pendingRequests = new Map();
    this.requestId = 0;
    this.stateCallback = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
  }

  async connect(url = 'ws://localhost:8765') {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
          console.log('Connected to MuJoCo server');
          this.connected = true;
          this.reconnectAttempts = 0;
          resolve(true);
        };

        this.ws.onclose = () => {
          console.log('Disconnected from MuJoCo server');
          this.connected = false;
          this.attemptReconnect(url);
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          if (!this.connected) {
            reject(error);
          }
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(JSON.parse(event.data));
        };

      } catch (e) {
        reject(e);
      }
    });
  }

  attemptReconnect(url) {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      setTimeout(() => {
        this.connect(url).catch(() => {});
      }, this.reconnectDelay * this.reconnectAttempts);
    }
  }

  handleMessage(msg) {
    const type = msg.type;

    if (type === 'state') {
      if (this.stateCallback) {
        this.stateCallback(msg.data);
      }
    } else if (type === 'observation') {
      const resolve = this.pendingRequests.get('obs');
      if (resolve) {
        resolve(msg.data);
        this.pendingRequests.delete('obs');
      }
    } else if (type === 'step_result') {
      const resolve = this.pendingRequests.get('step');
      if (resolve) {
        resolve(msg);
        this.pendingRequests.delete('step');
      }
    } else if (type === 'ack') {
      // Command acknowledged
    } else if (type === 'error') {
      console.error('Server error:', msg.message);
    }
  }

  onState(callback) {
    this.stateCallback = callback;
  }

  async step(useRL = true) {
    if (!this.connected) return null;

    return new Promise((resolve) => {
      this.pendingRequests.set('step', resolve);
      this.ws.send(JSON.stringify({
        type: 'step',
        use_rl: useRL
      }));

      // Timeout
      setTimeout(() => {
        if (this.pendingRequests.has('step')) {
          this.pendingRequests.delete('step');
          resolve(null);
        }
      }, 1000);
    });
  }

  async reset() {
    if (!this.connected) return null;

    return new Promise((resolve) => {
      this.ws.send(JSON.stringify({ type: 'reset' }));

      const handler = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'state') {
          resolve(msg.data);
        }
      };

      this.ws.addEventListener('message', handler, { once: true });

      // Timeout
      setTimeout(() => resolve(null), 1000);
    });
  }

  setCommand(linVelX, linVelY, angVel) {
    if (!this.connected) return;

    this.ws.send(JSON.stringify({
      type: 'command',
      lin_vel_x: linVelX,
      lin_vel_y: linVelY,
      ang_vel: angVel
    }));
  }

  async getObservation() {
    if (!this.connected) return null;

    return new Promise((resolve) => {
      this.pendingRequests.set('obs', resolve);
      this.ws.send(JSON.stringify({ type: 'get_obs' }));

      // Timeout
      setTimeout(() => {
        if (this.pendingRequests.has('obs')) {
          this.pendingRequests.delete('obs');
          resolve(null);
        }
      }, 1000);
    });
  }

  async applyAction(action) {
    if (!this.connected) return null;

    return new Promise((resolve) => {
      this.pendingRequests.set('step', resolve);
      this.ws.send(JSON.stringify({
        type: 'apply_action',
        action: Array.from(action)
      }));

      // Timeout
      setTimeout(() => {
        if (this.pendingRequests.has('step')) {
          this.pendingRequests.delete('step');
          resolve(null);
        }
      }, 1000);
    });
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
      this.connected = false;
    }
  }

  isConnected() {
    return this.connected;
  }
}

// Helper class to manage both WASM and WebSocket modes
export class SimulationManager {
  constructor() {
    this.mode = 'wasm';  // 'wasm' or 'websocket'
    this.wsClient = new MuJoCoWebSocket();
    this.wasmSim = null;
    this.lastState = null;
  }

  setMode(mode) {
    this.mode = mode;
    console.log(`Simulation mode: ${mode}`);
  }

  setWASMSimulation(sim) {
    this.wasmSim = sim;
  }

  async connectWebSocket(url = 'ws://localhost:8765') {
    try {
      await this.wsClient.connect(url);
      this.mode = 'websocket';
      return true;
    } catch (e) {
      console.warn('WebSocket connection failed, using WASM mode');
      this.mode = 'wasm';
      return false;
    }
  }

  getMode() {
    return this.mode;
  }

  isWebSocketConnected() {
    return this.wsClient.isConnected();
  }
}
