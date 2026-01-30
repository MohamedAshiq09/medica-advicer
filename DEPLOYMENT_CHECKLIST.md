# 🚀 Deployment Checklist

## ✅ Files Created/Updated

### Backend
- ✅ `backend/requirements.txt` - Python dependencies
- ✅ `backend/railway.toml` - Railway configuration
- ✅ `backend/nixpacks.toml` - Build configuration
- ✅ `backend/.env.example` - Environment variables template
- ✅ `backend/main.py` - Updated CORS for production

### Frontend
- ✅ `frontend/lib/utils.ts` - Missing utility file (FIXED)
- ✅ `frontend/.env.example` - Environment variables template
- ✅ `frontend/components/SymptomChecker.tsx` - Dynamic API URL

---

## 🎯 Deployment Steps

### 1️⃣ Backend on Railway

1. **Push to GitHub** (if not already)
   ```bash
   git add .
   git commit -m "Add deployment configs"
   git push
   ```

2. **Deploy on Railway**
   - Go to https://railway.app
   - New Project → Deploy from GitHub
   - Select your repo
   - **Important**: Set Root Directory to `backend`
   - Railway will auto-detect Python and deploy

3. **Copy Backend URL**
   - After deployment, copy the URL (e.g., `https://xxx.up.railway.app`)

4. **Update CORS** (Important!)
   - After getting Vercel URL, update `backend/main.py`:
   ```python
   allow_origins=[
       "http://localhost:3000",
       "https://your-vercel-url.vercel.app",  # Add your Vercel URL here
   ]
   ```
   - Push changes, Railway auto-redeploys

---

### 2️⃣ Frontend on Vercel

1. **Deploy on Vercel**
   - Go to https://vercel.com
   - New Project → Import from GitHub
   - Select your repo
   - **Important**: Set Root Directory to `frontend`
   - Framework: Next.js (auto-detected)

2. **Add Environment Variable**
   - In Vercel dashboard → Settings → Environment Variables
   - Add: `NEXT_PUBLIC_API_URL` = Your Railway backend URL
   - Example: `https://your-backend.up.railway.app`

3. **Redeploy**
   - After adding env var, trigger a redeploy

---

## 🧪 Testing

1. Visit your Vercel URL
2. Try symptom checker with: "I have fever and headache"
3. Should get triage results

---

## 🐛 Common Issues

### Railway: "Railpack could not determine how to build"
**Solution**: Make sure Root Directory is set to `backend` in Railway settings

### Frontend Build Error: "Module not found: @/lib/utils"
**Solution**: Already fixed! `frontend/lib/utils.ts` created

### Frontend can't connect to backend
**Solution**: 
- Check `NEXT_PUBLIC_API_URL` is set in Vercel
- Check CORS in `backend/main.py` includes your Vercel URL

### ML Models not loading
**Solution**: Ensure all `.joblib` and `.pkl` files in `backend/models/` are committed to git

---

## 📝 Notes

- Railway free tier: 500 hours/month (enough for testing)
- Vercel free tier: Unlimited for personal projects
- Both auto-deploy on git push
- Check logs in respective dashboards if issues occur
