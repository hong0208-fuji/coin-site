require("dotenv").config();
const path = require("path");
const express = require("express");
const session = require("express-session");
const rateLimit = require("express-rate-limit");
const { verifyCredentials } = require("./lib/auth");
const tradingState = require("./lib/tradingState");

const app = express();
const PORT = process.env.PORT || 3000;
const isProduction = process.env.NODE_ENV === "production";

// 배포 시 리버스 프록시(Nginx/Caddy 등) 뒤에서 HTTPS를 쓸 경우 secure 쿠키 판별에 필요
if (isProduction) app.set("trust proxy", 1);

app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(
  session({
    secret: process.env.SESSION_SECRET || "change-me-in-env",
    resave: false,
    saveUninitialized: false,
    cookie: {
      httpOnly: true,
      secure: isProduction, // HTTPS 배포 시(NODE_ENV=production)에만 secure 쿠키 강제
      maxAge: 1000 * 60 * 60 * 12,
    },
  })
);

// 로그인 무차별 대입 공격 방어: 15분에 IP당 10회로 제한
const loginLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  limit: 10,
  standardHeaders: true,
  legacyHeaders: false,
  message: { ok: false, error: "로그인 시도가 너무 많습니다. 잠시 후 다시 시도해주세요." },
});

function requireAuthPage(req, res, next) {
  if (req.session && req.session.user) return next();
  return res.redirect("/login");
}

function requireAuthApi(req, res, next) {
  if (req.session && req.session.user) return next();
  return res.status(401).json({ error: "unauthorized" });
}

// 로그인 여부와 무관하게 필요한 정적 자산(css/js)
app.use(
  "/assets",
  express.static(path.join(__dirname, "public"), {
    index: false,
    extensions: [],
    setHeaders: (res) => res.setHeader("Cache-Control", "no-cache"),
  })
);

app.get("/", (req, res) => {
  res.redirect(req.session && req.session.user ? "/dashboard" : "/login");
});

app.get("/login", (req, res) => {
  if (req.session && req.session.user) return res.redirect("/dashboard");
  res.sendFile(path.join(__dirname, "public", "login.html"));
});

app.get("/dashboard", requireAuthPage, (req, res) => {
  res.sendFile(path.join(__dirname, "public", "dashboard.html"));
});

app.post("/api/login", loginLimiter, (req, res) => {
  const { username, password } = req.body || {};
  if (verifyCredentials(username, password)) {
    req.session.user = username;
    return res.json({ ok: true });
  }
  return res
    .status(401)
    .json({ ok: false, error: "아이디 또는 비밀번호가 올바르지 않습니다." });
});

app.post("/api/logout", (req, res) => {
  req.session.destroy(() => res.json({ ok: true }));
});

app.get("/api/session", (req, res) => {
  res.json({ authenticated: !!(req.session && req.session.user), user: req.session?.user || null });
});

app.get("/api/trading/status", requireAuthApi, (req, res) => {
  res.json(tradingState.getState());
});

app.post("/api/trading/start", requireAuthApi, (req, res) => {
  res.json(tradingState.start());
});

app.post("/api/trading/stop", requireAuthApi, (req, res) => {
  res.json(tradingState.stop());
});

app.post("/api/trading/kill", requireAuthApi, (req, res) => {
  res.json(tradingState.emergencyStop());
});

app.post("/api/trading/reset", requireAuthApi, (req, res) => {
  res.json(tradingState.reset());
});

app.listen(PORT, () => {
  console.log(`coin server listening on http://localhost:${PORT}`);
});
