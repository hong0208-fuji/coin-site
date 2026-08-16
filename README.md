# coin — 바이낸스 AI 자동매매 웹서비스

로그인 후 GO/STOP 버튼만으로 AI가 바이낸스에서 자동매매를 수행/중단할 수 있는 개인용 웹 서비스.

## 진행 단계

현재 **2단계 완료 + 3~4단계 선행 백테스트 1차 진행**.

- 1단계: 기획 문서 + 로그인/대시보드 디자인 목업 — 완료
- 2단계: Express 서버, 세션 로그인, GO/STOP 상태 API — 완료 (`server/`)
- 백테스트 선행 검증: 3년치 BTCUSDT 1시간봉 데이터로 지표 전략 그리드서치 + train/test 검증 — 1차 완료, 결과는 실전 투입 불가 수준 (`backtest/results/report.md` 참고)

## 실행 방법

### 웹서버
```
cd server
npm install
cp .env.example .env   # ADMIN_USERNAME, ADMIN_PASSWORD_HASH 채우기 (npm run hash-password -- <비밀번호>)
npm start
```
`http://localhost:3000` 접속 → 로그인 → 대시보드에서 GO/STOP 확인.

### 백테스트
```
cd backtest
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python fetch_data.py   # 바이낸스 공개 API에서 3년치 캔들 수집
python optimize.py     # 파라미터 그리드서치 + train/test 검증, results/report.md 생성
```

## 문서 및 코드 위치

- `docs/ARCHITECTURE.md` — 전체 시스템 설계, AI 판단 방식, 리스크 관리, 로드맵, 백테스트 결과 요약
- `mockup/` — 로그인/대시보드 정적 디자인 목업 (초기 디자인 참고용)
- `server/` — 실제 동작하는 Express 백엔드(로그인, GO/STOP API) + 프론트(`server/public/`)
- `backtest/` — Python 백테스트 엔진 (데이터 수집, 지표/전략, 시뮬레이션, 파라미터 최적화)

## 로드맵

1. 기획 문서 + 목업 — 완료
2. Express 서버 골격 + 실제 로그인 + GO/STOP 상태 API — 완료
3. 바이낸스 모의투자(paper trading) 연동 — 예정
4. AI 판단 로직 구현 + 백테스트 고도화 — 1차 백테스트 완료, 전략 개선 필요
5. 리스크 관리 강화 후 소액 실거래 전환 — 예정
