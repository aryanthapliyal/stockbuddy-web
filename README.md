# StockBuddy - AI-Powered Stock Prediction Platform

A comprehensive stock prediction platform that combines machine learning models with sentiment analysis to provide accurate stock price forecasts.

## 🚀 Features

- **AI-Powered Predictions**: Uses LSTM and XGBoost models for accurate stock price forecasting
- **Real-time Sentiment Analysis**: Analyzes market sentiment from news sources
- **Interactive Charts**: Beautiful visualizations of predictions and trends
- **Stock Search**: Easy stock discovery with real-time price data
- **Progress Tracking**: Real-time monitoring of prediction progress
- **Responsive Design**: Modern UI that works on all devices

## 🏗️ Architecture

The application consists of three main components:

### 1. Frontend (React + Vite)
- **Location**: `/frontend`
- **Port**: 5173 (development)
- **Tech Stack**: React 18, Vite, Tailwind CSS, Recharts

### 2. Backend API (Node.js + Express)
- **Location**: `/Backend`
- **Port**: 5000
- **Tech Stack**: Node.js, Express, MongoDB, JWT

### 3. Model Backend (Python + Flask)
- **Location**: `/Model_Backend`
- **Port**: 5001
- **Tech Stack**: Python, Flask, TensorFlow, XGBoost, NLTK

## 📋 Prerequisites

- Node.js 16+
- Python 3.8+
- MongoDB
- API Keys for:
  - Alpha Vantage (stock data)
  - Finnhub (news sentiment)

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Stockbuddy-main
```

### 2. Backend Setup
```bash
cd Backend
npm install
```

Create a `.env` file:
```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/stockbuddy
SESSION_SECRET=your-session-secret
MODEL_API_URL=http://localhost:5001
```

### 3. Model Backend Setup
```bash
cd Model_Backend
pip install -r requirements.txt
```

Create a `.env` file:
```env
ALPHAVANTAGE_API_KEY=your-alpha-vantage-key
FINNHUB_API_KEY=your-finnhub-key
```

### 4. Frontend Setup
```bash
cd frontend
npm install
```

Create a `.env` file (optional):
```env
VITE_API_BASE_URL=http://localhost:5000
```

## 🚀 Running the Application

### 1. Start MongoDB
```bash
mongod
```

### 2. Start Model Backend
```bash
cd Model_Backend
python app.py
```

### 3. Start Backend API
```bash
cd Backend
npm run dev
```

### 4. Start Frontend
```bash
cd frontend
npm run dev
```

### 5. Access the Application
- Frontend: http://localhost:5173
- Backend API: http://localhost:5000
- Model API: http://localhost:5001

## 📊 API Endpoints

### Backend API (Port 5000)

#### Predictions
- `POST /api/predictions` - Start a new prediction
- `GET /api/predictions` - Get all predictions
- `GET /api/predictions/:id` - Get specific prediction
- `GET /api/predictions/status/:taskId` - Get prediction status
- `POST /api/predictions/stop/:taskId` - Stop prediction
- `DELETE /api/predictions/:id` - Delete prediction

### Model API (Port 5001)

#### Predictions
- `POST /api/predict` - Start prediction task
- `GET /api/predict/status/:taskId` - Get task status
- `POST /api/predict/stop/:taskId` - Stop task
- `GET /api/predict/sentiment/:symbol` - Get sentiment analysis
- `GET /api/diagnose` - System diagnostics

## 🤖 Machine Learning Models

### LSTM Model
- **Purpose**: Sequence prediction for stock prices
- **Features**: Historical price data, technical indicators
- **Architecture**: Multi-layer LSTM with dropout

### XGBoost Model
- **Purpose**: Residual prediction and ensemble learning
- **Features**: Technical indicators, market data
- **Benefits**: Handles non-linear relationships

### Sentiment Analysis
- **Source**: Finnhub news API
- **Model**: NLTK + Transformers
- **Output**: Positive, negative, neutral sentiment scores

## 📁 Project Structure

```
Stockbuddy-main/
├── Backend/                 # Node.js API server
│   ├── routes/             # API routes
│   ├── models/             # MongoDB models
│   ├── middleware/         # Express middleware
│   ├── config/             # Configuration files
│   └── server.js           # Main server file
├── Model_Backend/          # Python ML service
│   ├── app.py              # Flask application
│   ├── model.py            # ML models and utilities
│   └── requirements.txt    # Python dependencies
├── frontend/               # React application
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── config/         # Configuration
│   │   └── App.jsx         # Main app component
│   └── package.json        # Node dependencies
└── README.md               # This file
```

## 🔧 Configuration

### Environment Variables

#### Backend (.env)
```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/stockbuddy
SESSION_SECRET=your-secret-key
MODEL_API_URL=http://localhost:5001
```

#### Model Backend (.env)
```env
ALPHAVANTAGE_API_KEY=your-key
FINNHUB_API_KEY=your-key
```

#### Frontend (.env)
```env
VITE_API_BASE_URL=http://localhost:5000
```

## 🧪 Testing

### Backend Tests
```bash
cd Backend
npm test
```

### Model Backend Tests
```bash
cd Model_Backend
python -m pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 📈 Usage

1. **Search for Stocks**: Use the search bar to find stocks by symbol or company name
2. **Start Prediction**: Select a stock and prediction period (3-30 days)
3. **Monitor Progress**: Watch real-time progress updates
4. **View Results**: Explore interactive charts and detailed predictions
5. **Analyze Sentiment**: Review market sentiment analysis

## 🚨 Troubleshooting

### Common Issues

1. **Model Backend Not Starting**
   - Check Python version (3.8+ required)
   - Verify TensorFlow installation
   - Check API keys in .env file

2. **Backend Connection Issues**
   - Ensure MongoDB is running
   - Check port availability
   - Verify environment variables

3. **Frontend Build Issues**
   - Clear node_modules and reinstall
   - Check Node.js version
   - Verify Vite configuration

### Logs and Debugging

- Backend logs: Check console output
- Model logs: Check Python console
- Frontend logs: Browser developer tools

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Alpha Vantage for stock data
- Finnhub for news sentiment data
- TensorFlow and XGBoost communities
- React and Vite teams

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation


