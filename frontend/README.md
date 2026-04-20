# Wafer dashboard (React + TypeScript + Vite)

Single-page dashboard for the **Wafer** stack: wafer map, yield trends, correlations, ML yield prediction, and **SPC + process capability**. Parent project docs: [../README.md](../README.md).

## Prerequisites

- Node 20+ recommended (matches `docker-compose` frontend image).
- FastAPI backend on port **8000** unless you override the client base URL (see below).

## Scripts

| Command | Purpose |
|---------|---------|
| `npm install` | Install dependencies |
| `npm run dev` | Vite dev server (default [http://localhost:5173](http://localhost:5173)) |
| `npm run build` | Typecheck + production build to `dist/` |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | ESLint |

## API base URL

By default the client uses `http://<browser-hostname>:8000` (see [`src/config.ts`](src/config.ts)). If the API runs elsewhere, set before `npm run dev`:

```bash
# Windows (cmd)
set VITE_API_URL=http://127.0.0.1:8000

# Windows (PowerShell) / macOS / Linux
export VITE_API_URL=http://127.0.0.1:8000
```

## Notable source locations

| Area | Path |
|------|------|
| Main dashboard layout | [`src/pages/Dashboard.tsx`](src/pages/Dashboard.tsx) |
| SPC chart + capability strip | [`src/components/SPCPanel.tsx`](src/components/SPCPanel.tsx) |
| Capability UI | [`src/components/CapabilityPanel.tsx`](src/components/CapabilityPanel.tsx) |
| Recipe LSL/USL / engineering limits | [`src/config/processSpecs.ts`](src/config/processSpecs.ts) |
| Cp / Cpk / Pp / Ppk math | [`src/utils/capability.ts`](src/utils/capability.ts) |

**Where to see capability in the UI:** open the right panel → **SPC mode** → scroll below the control chart and the “Total points / Violations” line to **Process capability**.

## Generic Vite template

This app was bootstrapped with Vite. For upstream Vite, React, and ESLint configuration docs, see [https://vite.dev](https://vite.dev) and the [React documentation](https://react.dev).
