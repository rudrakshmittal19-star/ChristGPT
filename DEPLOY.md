# Deploy CampusGPT on Render

## Deploy the frontend-only demo on Vercel

This is the recommended first deployment: it serves the static browser demo and does not run the Python API.

1. Push the repository to GitHub.
2. Sign in at [vercel.com](https://vercel.com) and choose **Add New** → **Project**.
3. Import `rudrakshmittal19-star/ChristGPT` from GitHub.
4. Set **Framework Preset** to **Other**. Leave **Build Command** and **Output Directory** empty; do not add environment variables.
5. Click **Deploy**. `vercel.json` rewrites `/` to `frontend/standalone_demo.html`.
6. Open the assigned URL and check Home, Assistant retrieval, Attendance, Calendar, Official portals, and the calendar/handbook PDF downloads.

This version has no server-side authentication or live LLM generation. Student/faculty demo login is client-side only; do not enter real student information. Add the API as a separate phase before treating the site as a production service.

## Optional API deployment on Render

## Before deploying

1. Revoke the Groq API key that was exposed during setup and create a replacement in the Groq console.
2. Push this project to a private GitHub repository. Do not commit `.env`, student data, or credentials.
3. Keep only public university source documents in `data/raw/`. The public download list is allowlisted in the API.

## Render setup

1. Sign in at [render.com](https://render.com) and choose **New +** → **Blueprint**.
2. Connect the GitHub account and select the repository containing `render.yaml`.
3. Review the `campusgpt` service and choose **Apply**. The blueprint uses the free Docker plan and probes `/health`.
4. Open the service's **Environment** page. Set `GROQ_API_KEY` to the newly rotated secret. Keep `ENV=production` and `LLM_BACKEND=groq`.
5. After Render assigns the public URL, update `CORS_ORIGINS` to that exact origin, such as `https://campusgpt-abc.onrender.com` (no trailing slash), then save and redeploy.
6. Wait for the deploy status to become **Live**. Open `https://<your-service>.onrender.com/health`; expect JSON containing `"ok": true`.
7. Open the root URL, sign in with an account created for production, and check Events, CIA & MSE, and Official documents.

## Production account setup

Do not deploy with `ENV=dev`; that mode seeds the demo password `christ123`. The account CLI is:

```sh
python -m campusgpt.manage_users create <username> --name "Full Name" --role admin
```

Run it against the same persistent SQLite database used by the service. Render's free service does not provide a persistent disk or an interactive shell suitable for durable account provisioning. For a real public pilot, use a paid persistent disk or add an external database and provision the admin before opening the service publicly.

## Important free-tier limitation

The service can be deployed on Render's free plan, but its filesystem is ephemeral and the service may sleep. SQLite users, chat history, feedback, scraped event records, and any runtime-built Chroma index can be lost on restart or redeploy. The source PDFs and prebuilt files included in the repository are present in each image, but database mutations are not durable. Do not use this deployment for real student records or operational decisions. Use persistent storage before a real pilot.

## Before sharing the link publicly

- Replace the exposed Groq key; never put the replacement in source control.
- Keep `ENV=production` and use a unique random `JWT_SECRET` of at least 32 characters. Render's generated value is suitable; never reuse `change-me`.
- Set `CORS_ORIGINS` to only the final site origin; do not use `*`.
- Create non-demo users and remove/disable any demo accounts before sharing.
- Verify all PDFs and event sources are public and current. The downloaded practice sheet is synthetic, not an official PYQ.
- Do not upload student records, marks, attendance, or private notices. The app is a prototype and has no university SSO or durable free-tier storage.
- Confirm Groq model availability and rate limits in the Groq console; the LLM is optional for the extractive demo.

## Docker Compose locally

Set `.env` to `ENV=dev` with a blank Groq key for the extractive demo, then run:

```sh
docker compose up --build
```

Open `http://localhost:8000`. Compose persists `data/` and `index/` as host-mounted volumes.