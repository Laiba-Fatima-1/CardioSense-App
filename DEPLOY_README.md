# CardioSense — Live Web App (Task 4: Disease Prediction)

An interactive web app where anyone can enter clinical measurements and get an instant heart disease risk assessment, powered by the Random Forest model from the Task 4 project.

## 📁 Files
- `streamlit_app.py` — the full app (UI + prediction logic)
- `model.pkl` — trained Random Forest model
- `scaler.pkl` — feature scaler (must match the model's training preprocessing)
- `requirements.txt` — dependencies for deployment

## 🚀 How to deploy live (Streamlit Community Cloud — free)

1. **Create a new GitHub repo** (e.g., `CardioSense-App`) and upload these 4 files to it (or add them to your existing `CodeAlpha_DiseasePrediction` repo in a subfolder — either works).

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.

3. Click **"New app"**:
   - Repository: select the repo you just created
   - Branch: `main`
   - Main file path: `streamlit_app.py`

4. Click **Deploy**. It takes 1-2 minutes to build.

5. You'll get a live URL like:
   `https://cardiosense-yourname.streamlit.app`

That's it — anyone with the link can open it, enter their data, and get a live prediction. Every time you push changes to GitHub, the app auto-updates.

## 🧪 Test locally first (optional)
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`

## ⚠️ Important
Keep `model.pkl` and `scaler.pkl` in the **same folder** as `streamlit_app.py` — the app loads them by relative path.
