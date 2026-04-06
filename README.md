# 📈 StockOracle — AI Real-Time Stock Predictor

A full-stack Flask web app that fetches live stock data from Alpha Vantage and uses a Ridge Regression ML model (with polynomial features + technical indicators) to predict future stock prices.

---

## 🚀 Features

- **Real-time stock data** via Alpha Vantage free API
- **ML prediction model** using Ridge Regression with 12 technical indicators
- **7-day price forecast** with daily breakdown table
- **Technical indicators**: RSI, MACD, Bollinger Bands, Moving Averages, Momentum, Volatility
- **AI signal**: BUY / SELL / HOLD recommendation
- **Beautiful dark dashboard** with Chart.js visualizations
- **Live quote bar** with price, change %, and volume
- **5-minute data caching** to avoid API rate limits
- **Demo data fallback** when API limit is reached

---

## 📦 Project Structure

```
stock_predictor/
├── app.py              # Flask backend + ML model
├── templates/
│   └── index.html      # Frontend dashboard
├── requirements.txt    # Python dependencies
├── Procfile            # For Render/Heroku
├── render.yaml         # Render auto-deploy config
├── runtime.txt         # Python version
└── README.md
```

---

## 🔑 Get Your Free Alpha Vantage API Key

1. Go to: https://www.alphavantage.co/support/#api-key
2. Enter your email → click **Get Free API Key**
3. You get **25 API calls/day** for free (enough for ~10 stocks/day)
4. The app **auto-falls back to demo data** if the limit is hit

---

## 💻 Local Development

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key
**Windows:**
```cmd
set ALPHA_VANTAGE_API_KEY=your_key_here
```
**Mac/Linux:**
```bash
export ALPHA_VANTAGE_API_KEY=your_key_here
```

### 3. Run the app
```bash
python app.py
```
Open: http://localhost:5000

---

## ☁️ FREE Deployment on Render.com (Stays Online 24/7!)

> **Why Render?** Render's free tier keeps your web service running continuously.
> It does spin down after 15 mins of inactivity, but auto-wakes on next request.
> Use a free uptime monitor like UptimeRobot to ping it every 10 min → stays awake!

### Step-by-Step Deployment:

**1. Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/stockoracle.git
git push -u origin main
```

**2. Create Render Account**
- Go to https://render.com → Sign up free with GitHub

**3. Create New Web Service**
- Click **New** → **Web Service**
- Connect your GitHub repo
- Settings:
  - **Name**: stockoracle
  - **Runtime**: Python 3
  - **Build Command**: `pip install -r requirements.txt`
  - **Start Command**: `gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT --timeout 120`
  - **Plan**: Free

**4. Add Environment Variable**
- In Render dashboard → your service → **Environment**
- Add: `ALPHA_VANTAGE_API_KEY` = `your_key_here`

**5. Deploy!**
- Click **Deploy** — takes ~2 minutes
- Your app will be live at: `https://stockoracle.onrender.com`

---

## ⏰ Keep It Awake 24/7 (Free)

Render free tier sleeps after 15 minutes of inactivity. To prevent this:

1. Go to **https://uptimerobot.com** → Sign up free
2. Create a new monitor:
   - Type: HTTP(s)
   - URL: `https://your-app-name.onrender.com/health`
   - Interval: **5 minutes**
3. That's it! Your app stays awake permanently.

---

## 🤖 How the ML Model Works

```
Raw OHLCV Data (Alpha Vantage)
         ↓
Feature Engineering (12 indicators):
  • Open, High, Low, Volume
  • MA(5), MA(10), MA(20)
  • RSI(14), MACD
  • Bollinger Bands, Momentum, Volatility
         ↓
Polynomial Feature Expansion (degree 2)
  → Captures non-linear price relationships
         ↓
Ridge Regression (L2 regularization)
  → Prevents overfitting
         ↓
80/20 Train/Test Split
  → Accuracy measured on unseen data
         ↓
7-Day Forward Forecast
  → Trend extrapolation + mean reversion blend
```

### Buy/Sell/Hold Signal Logic:
| Indicator | BUY Signal | SELL Signal |
|-----------|------------|-------------|
| RSI | < 30 (oversold) | > 70 (overbought) |
| MACD | > 0 (bullish) | < 0 (bearish) |
| Price vs MA20 | Above | Below |
| Momentum | Positive | Negative |

---

## ⚠️ Disclaimer

This app is for **educational purposes only**. Stock predictions are inherently uncertain. Do not make real financial decisions based on this tool.

---

## 📄 License
MIT License — free to use and modify.
