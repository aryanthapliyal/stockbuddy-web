const mongoose = require('mongoose');

// Define the schema for a stock holding in the demo trading account
const holdingSchema = new mongoose.Schema({
  symbol: { 
    type: String, 
    required: true,
    trim: true
  },
  companyName: {
    type: String,
    required: true,
    trim: true
  },
  quantity: {
    type: Number,
    required: true,
    min: 0
  },
  averagePrice: {
    type: Number,
    required: true,
    min: 0
  },
  purchaseValue: {
    type: Number,
    required: true,
    min: 0
  },
  currentPrice: {
    type: Number,
    default: 0
  },
  currentValue: {
    type: Number,
    default: 0
  },
  profit: {
    type: Number,
    default: 0
  },
  profitPercentage: {
    type: Number,
    default: 0
  },
  lastUpdated: {
    type: Date,
    default: Date.now
  }
});

// Define the schema for transaction history
const transactionSchema = new mongoose.Schema({
  symbol: {
    type: String,
    required: true,
    trim: true
  },
  companyName: {
    type: String,
    required: true,
    trim: true
  },
  type: {
    type: String,
    enum: ['BUY', 'SELL'],
    required: true
  },
  quantity: {
    type: Number,
    required: true,
    min: 1
  },
  price: {
    type: Number,
    required: true,
    min: 0
  },
  totalAmount: {
    type: Number,
    required: true
  },
  date: {
    type: Date,
    default: Date.now
  }
});

// Define the main demo trading account schema
const demoTradingAccountSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    unique: true
  },
  balance: {
    type: Number,
    required: true,
    default: 100000, // Default starting balance of $100,000
    min: 0
  },
  initialBalance: {
    type: Number,
    required: true,
    default: 100000,
    immutable: true
  },
  equity: {
    type: Number,
    default: 0
  },
  totalProfitLoss: {
    type: Number,
    default: 0
  },
  totalProfitLossPercentage: {
    type: Number,
    default: 0
  },
  dayChange: {
    type: Number,
    default: 0
  },
  dayChangePercentage: {
    type: Number,
    default: 0
  },
  weekChange: {
    type: Number,
    default: 0
  },
  weekChangePercentage: {
    type: Number,
    default: 0
  },
  monthChange: {
    type: Number,
    default: 0
  },
  monthChangePercentage: {
    type: Number,
    default: 0
  },
  yearChange: {
    type: Number,
    default: 0
  },
  yearChangePercentage: {
    type: Number,
    default: 0
  },
  holdings: [holdingSchema],
  transactions: [transactionSchema],
  createdAt: {
    type: Date,
    default: Date.now,
    immutable: true
  },
  lastUpdated: {
    type: Date,
    default: Date.now
  }
});

// Add a pre-save hook to update calculation fields
demoTradingAccountSchema.pre('save', function(next) {
  // Calculate total equity from holdings
  if (this.holdings && this.holdings.length > 0) {
    const holdingsValue = this.holdings.reduce((total, holding) => {
      return total + (holding.currentValue || 0);
    }, 0);
    
    this.equity = holdingsValue + this.balance;
    
    // Calculate total profit/loss
    const initialValue = this.initialBalance;
    this.totalProfitLoss = this.equity - initialValue;
    this.totalProfitLossPercentage = (this.totalProfitLoss / initialValue) * 100;
  } else {
    this.equity = this.balance;
    this.totalProfitLoss = this.balance - this.initialBalance;
    this.totalProfitLossPercentage = (this.totalProfitLoss / this.initialBalance) * 100;
  }
  
  this.lastUpdated = new Date();
  next();
});

const DemoTradingAccount = mongoose.model('DemoTradingAccount', demoTradingAccountSchema);

module.exports = DemoTradingAccount; 