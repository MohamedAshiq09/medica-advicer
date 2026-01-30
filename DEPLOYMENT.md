# Deployment Guide

## Backend Deployment (Railway)

### Step 1: Create Railway Project
1. Go to [Railway.app](https://railway.app)
2. Sign up/Login with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository

### Step 2: Configure Backend Service
1. Railway will detect your project structure
2. Click "Add Service" → "GitHub Repo"
3. Set **Root Directory**: `backend`
4. Railway will auto-detect Python and use `requirements.txt`

### Step 3: Environment Variables (Optional)
If you want LLM features:
- Add `OPENAI_API_KEY` in Railway dashboard

### Step 4: Get Backend URL
- After deployment, Railway gives you a URL like: `https://your-app.up.railway.app`
- Copy this URL for frontend configuration

---

## Frontend Deployment (Vercel)

### Step 1: Deploy to Vercel
1. Go to [Vercel.com](https://vercel.com)
2. Click "New Project" → Import your GitHub repo
3. Set **Root Directory**: `frontend`
4. Framework Preset: Next.js (auto-detected)

### Step 2: Environment Variables
Add this in Vercel dashboard:
- `NEXT_PUBLIC_API_URL` = Your Railway backend URL (e.g., `https://your-app.up.railway.app`)

### Step 3: Deploy
- Click "Deploy"
- Vercel will build and deploy your frontend

---

## Update CORS After Deployment

After getting your Vercel URL, update `backend/main.py`:

```python
allow_origins=[
    "http://localhost:3000",
    "https://your-vercel-app.vercel.app",  # Add your actual Vercel URL
]
```

Then redeploy backend on Railway (it auto-redeploys on git push).

---

## Testing

1. Visit your Vercel frontend URL
2. Try the symptom checker
3. Check Railway logs if issues occur: Railway Dashboard → Your Service → Logs

---

## Troubleshooting

### Railway Build Fails
- Check logs in Railway dashboard
- Ensure `requirements.txt` has all dependencies
- Check Python version compatibility

### Frontend Can't Connect to Backend
- Verify `NEXT_PUBLIC_API_URL` is set in Vercel
- Check CORS settings in `backend/main.py`
- Check Railway backend is running (green status)

### ML Models Not Loading
- Ensure all `.joblib` and `.pkl` files are committed to git
- Check Railway logs for model loading errors
