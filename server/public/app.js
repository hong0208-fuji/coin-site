const statusBadge = document.getElementById("status-badge");
const statusText = document.getElementById("status-text");
const statusUpdated = document.getElementById("status-updated");
const controlError = document.getElementById("control-error");
const usernameLabel = document.getElementById("username-label");

const STATUS_LABEL = {
  STOPPED: "정지됨",
  RUNNING: "매매 중",
  EMERGENCY_STOPPED: "긴급 정지됨",
};

function renderStatus(state) {
  const label = STATUS_LABEL[state.status] || state.status;
  statusBadge.textContent = `현재 상태: ${label}`;
  statusBadge.classList.remove("is-running", "is-emergency");
  if (state.status === "RUNNING") statusBadge.classList.add("is-running");
  if (state.status === "EMERGENCY_STOPPED") statusBadge.classList.add("is-emergency");

  statusText.textContent = label;
  statusUpdated.textContent = state.updatedAt
    ? new Date(state.updatedAt).toLocaleString("ko-KR")
    : "-";
}

async function fetchStatus() {
  const res = await fetch("/api/trading/status");
  if (res.status === 401) {
    window.location.href = "/login";
    return;
  }
  const data = await res.json();
  renderStatus(data);
}

async function callTradingApi(path) {
  controlError.textContent = "";
  try {
    const res = await fetch(path, { method: "POST" });
    if (res.status === 401) {
      window.location.href = "/login";
      return;
    }
    const data = await res.json();
    if (data.error) {
      controlError.textContent = data.error;
    }
    renderStatus(data);
  } catch (err) {
    controlError.textContent = "서버에 연결할 수 없습니다.";
  }
}

document.getElementById("go-btn").addEventListener("click", () => callTradingApi("/api/trading/start"));
document.getElementById("stop-btn").addEventListener("click", () => callTradingApi("/api/trading/stop"));
document.getElementById("kill-btn").addEventListener("click", () => callTradingApi("/api/trading/kill"));

document.getElementById("logout-link").addEventListener("click", async (e) => {
  e.preventDefault();
  await fetch("/api/logout", { method: "POST" });
  window.location.href = "/login";
});

async function loadSession() {
  const res = await fetch("/api/session");
  const data = await res.json();
  if (data.authenticated) {
    usernameLabel.textContent = data.user;
  }
}

loadSession();
fetchStatus();
setInterval(fetchStatus, 5000);
