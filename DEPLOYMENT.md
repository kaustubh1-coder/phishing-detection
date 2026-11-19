# Deployment Guide: Render (Backend) & Vercel (Frontend)

This guide provides step-by-step instructions for deploying the **Network Security – Phishing URL Detection** project to production using **Render** for the Python/FastAPI backend and **Vercel** for the React frontend.

---

## Architecture Overview

```
┌─────────────────────────────────┐
│     User Browser / Client       │
└───────┬─────────────────┬───────┘
        │                 │
        ▼                 ▼
┌──────────────────┐  ┌───────────────────────────────────────────────┐
│  Vercel Frontend │  │  Render Backend (FastAPI + ML + SHAP)         │
│  React 19 + Vite │  │  Container / Web Service                      │
│  (my-app.vercel) │──┼─► Port $PORT (Default: 8000 / 10000)          │
└──────────────────┘  └───────┬──────────────────────┬────────────────┘
                              │                      │
                              ▼                      ▼
                     ┌──────────────────┐  ┌───────────────────────┐
                     │  MongoDB Atlas   │  │   Google Gemini API   │
                     │  (Cloud Cluster) │  │   (Evidence Grounded) │
                     └──────────────────┘  └───────────────────────┘
```

---

## Prerequisites

1. **GitHub Repository**: Ensure the repository is pushed to GitHub:
   `https://github.com/kaustubh1-coder/phishing-detection`
2. **MongoDB Atlas Database**:
   - A free cloud MongoDB cluster from [mongodb.com/atlas](https://www.mongodb.com/atlas).
   - **Crucial**: In MongoDB Atlas **Network Access**, ensure IP Access List includes `0.0.0.0/0` (Allow Access from Anywhere) because cloud providers like Render rotate dynamic egress IPs.
   - Obtain your SRV connection URI: `mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority`.
3. **Render Account**: Sign up at [render.com](https://render.com).
4. **Vercel Account**: Sign up at [vercel.com](https://vercel.com).
5. **Google Gemini API Key (Optional)**: Obtain a free API key at [aistudio.google.com](https://aistudio.google.com).

---

## Part 1: Deploy Backend on Render

You can deploy the backend on Render using **Option A (Blueprint - Recommended)** or **Option B (Manual Web Service)**.

### Option A: Via Render Blueprint (Fastest)

1. Log in to your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** and select **Blueprint**.
3. Connect your GitHub repository: `kaustubh1-coder/phishing-detection`.
4. Render will detect the [`render.yaml`](./render.yaml) file automatically.
5. In the configuration prompt, provide the required environment variables:
   - `MONGODB_URL_KEY`: Your MongoDB Atlas connection URI.
   - `GEMINI_API_KEY`: Your Google Gemini API key (optional; if omitted, the backend falls back to local SHAP explainability).
6. Click **Apply**. Render will build the Docker container and deploy the service.

---

### Option B: Manual Web Service

1. On the Render Dashboard, click **New +** > **Web Service**.
2. Connect your GitHub repository.
3. Configure the service settings:
   - **Name**: `phishing-detection-api`
   - **Region**: Choose the region closest to you (e.g., Frankfurt, Oregon, Singapore).
   - **Branch**: `main`
   - **Runtime**: **Docker** (uses the preconfigured [`Dockerfile`](./Dockerfile))
     *(Alternatively, runtime: **Python**, with Build Command `pip install -r requirements.txt` and Start Command `uvicorn app:app --host 0.0.0.0 --port $PORT`)*
   - **Instance Type**: **Free**
4. Expand **Environment Variables** and add:

   | Key | Value | Notes |
   | :--- | :--- | :--- |
   | `MONGODB_URL_KEY` | `mongodb+srv://<user>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority` | Required |
   | `MONGO_DB_URL` | Same as `MONGODB_URL_KEY` | Fallback alias |
   | `GEMINI_API_KEY` | `AIzaSy...` | Optional |
   | `GEMINI_MODEL` | `gemini-3.5-flash` | Default |
   | `CORS_ORIGINS` | `*` | Or specify your Vercel URL |

5. Click **Create Web Service**.
6. Once deployed, note down your Render Web Service URL:
   `https://<your-service-name>.onrender.com`

### Verifying Backend Deployment

Test the following endpoints in your browser or with `curl`:
- Interactive Swagger API Docs: `https://<your-render-url>.onrender.com/docs`
- Robustness Evaluation API: `https://<your-render-url>.onrender.com/api/model-robustness`

---

## Part 2: Deploy Frontend on Vercel

1. Log in to your [Vercel Dashboard](https://vercel.com/).
2. Click **Add New...** > **Project**.
3. Import your GitHub repository: `kaustubh1-coder/phishing-detection`.
4. In the project configuration screen:
   - **Framework Preset**: `Vite` (auto-detected).
   - **Root Directory**: Click **Edit** and select **`frontend`** (or leave as root, as [`vercel.json`](./vercel.json) handles monorepo build).
   - **Build and Output Settings**:
     - Build Command: `npm run build`
     - Output Directory: `dist`
     - Install Command: `npm install`
5. Expand **Environment Variables** and add:

   | Key | Value | Description |
   | :--- | :--- | :--- |
   | `VITE_API_BASE_URL` | `https://<your-render-url>.onrender.com` | **Your Render backend URL without trailing slash** |

6. Click **Deploy**.
7. Vercel will install dependencies, build the TypeScript/Vite bundle, and deploy the application to a live URL:
   `https://<your-app-name>.vercel.app`

---

## Part 3: Verification & End-to-End Testing

1. Open your Vercel URL in your browser: `https://<your-app-name>.vercel.app`.
2. **Account Registration / Login**:
   - Go to the **Sign In** tab and create a new account.
   - Verify that your user is successfully created in MongoDB Atlas.
3. **Test Legitimate URL**:
   - Go to **Analyze** and test `https://www.google.com`.
   - Verify that:
     - The prediction returns `Legitimate`.
     - The **Top 5 Contributing Features (SHAP)** table renders with importance scores and directional badges.
     - Evidence-grounded AI explanation is shown.
4. **Test Synthetic Phishing URL**:
   - Test `http://192.168.1.10/login`.
   - Verify that:
     - Missing webpage signals are safely estimated via KNN imputation without false penalties.
     - The **Robustness Warning Alert** banner appears.
     - Prediction returns `Phishing` with high confidence.
5. **Inspect Model Robustness**:
   - Navigate to the **Models** page.
   - Review the 4-scenario robustness leaderboard (Full Baseline, Missing Webpage, Missing WHOIS, Lexical-Only).

---

## Troubleshooting & Key Configurations

### 1. Render Free Tier Spin-Down
- On Render's Free tier, services spin down after 15 minutes of inactivity. The first request after idle can take 40–50 seconds to warm up. Subsequent requests respond in milliseconds.
- Tip: Use an uptime monitor (e.g. UptimeRobot or Cron-job.org) pinging `https://<your-render-url>.onrender.com/api/model-robustness` every 10 minutes to keep it active.

### 2. MongoDB Atlas Connection Timeout
- If you see `ServerSelectionTimeoutError`, verify that **Network Access** in MongoDB Atlas has `0.0.0.0/0` allowed.
- Verify your database user credentials (username and password) have read/write access to the `MyPhishingDetection` database.

### 3. Cross-Origin (CORS) Configuration
- The backend in `app.py` has been configured with:
  `allow_origin_regex=r"https://.*\.vercel\.app"`
  This automatically allows all Vercel production and preview URLs without needing manual CORS changes.

### 4. Direct Page Refresh on Vercel (404 Avoidance)
- Single Page Applications require URL rewrites so subroutes (e.g. `/analyze`, `/dashboard`) redirect to `/index.html`.
- This is preconfigured via `frontend/vercel.json` and the root `vercel.json`.
