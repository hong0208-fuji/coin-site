const fs = require("fs");
const path = require("path");

const STATE_FILE = path.join(__dirname, "..", "data", "trading-state.json");
const DEFAULT_STATE = { status: "STOPPED", updatedAt: null };

function readState() {
  try {
    const raw = fs.readFileSync(STATE_FILE, "utf-8");
    return JSON.parse(raw);
  } catch (err) {
    return { ...DEFAULT_STATE };
  }
}

function writeState(state) {
  fs.mkdirSync(path.dirname(STATE_FILE), { recursive: true });
  fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
  return state;
}

function getState() {
  return readState();
}

function start() {
  const current = readState();
  if (current.status === "EMERGENCY_STOPPED") {
    return { ...current, error: "긴급 정지 상태입니다. 먼저 리셋해야 재시작할 수 있습니다." };
  }
  return writeState({ status: "RUNNING", updatedAt: new Date().toISOString() });
}

function stop() {
  return writeState({ status: "STOPPED", updatedAt: new Date().toISOString() });
}

function emergencyStop() {
  return writeState({ status: "EMERGENCY_STOPPED", updatedAt: new Date().toISOString() });
}

function reset() {
  return writeState({ status: "STOPPED", updatedAt: new Date().toISOString() });
}

module.exports = { getState, start, stop, emergencyStop, reset };
