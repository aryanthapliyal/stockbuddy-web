const express = require('express');
const router = express.Router();
const DemoTradingAccount = require('../models/DemoTradingAccount');
const User = require('../models/User');
const tradingAuth = require('../middleware/tradingAuth');

const { check, validationResult } = require('express-validator');

// @route   GET /api/demotrading/account
// @desc    Get user's demo trading account
// @access  Private
router.get('/account', tradingAuth, async (req, res) => {
  try {
    console.log('GET /account - User ID:', req.user.id);
    
    // Find user's trading account or create a new one if it doesn't exist
    let tradingAccount = await DemoTradingAccount.findOne({ userId: req.user.id });
    
    if (!tradingAccount) {
      console.log('Creating new trading account for user:', req.user.id);
      // Create new account with default balance
      tradingAccount = new DemoTradingAccount({
        userId: req.user.id,
        balance: 100000,
        initialBalance: 100000
      });
      
      await tradingAccount.save();
    }
    
    res.json(tradingAccount);
  } catch (error) {
    console.error('Error fetching trading account:', error.message);
    res.status(500).json({ error: 'Server error', message: error.message });
  }
});

// @route   POST /api/demotrading/trade
// @desc    Execute a buy or sell trade
// @access  Private
router.post('/trade', [
  tradingAuth,
  check('symbol', 'Symbol is required').not().isEmpty(),
  check('companyName', 'Company name is required').not().isEmpty(),
  check('type', 'Trade type must be BUY or SELL').isIn(['BUY', 'SELL']),
  check('quantity', 'Quantity must be a positive number').isInt({ min: 1 }),
  check('price', 'Price must be a positive number').isFloat({ min: 0.01 })
], async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  
  const { symbol, companyName, type, quantity, price } = req.body;
  const totalAmount = quantity * price;
  
  try {
    console.log('POST /trade - User ID:', req.user.id);
    console.log('Trade request:', { symbol, type, quantity, price, totalAmount });
    
    // Find user's trading account
    let tradingAccount = await DemoTradingAccount.findOne({ userId: req.user.id });
    
    if (!tradingAccount) {
      // Create new account if it doesn't exist
      tradingAccount = new DemoTradingAccount({
        userId: req.user.id,
        balance: 100000,
        initialBalance: 100000
      });
    }
    
    // Handle BUY order
    if (type === 'BUY') {
      // Check if user has enough balance
      if (tradingAccount.balance < totalAmount) {
        return res.status(400).json({ 
          error: 'Insufficient funds', 
          available: tradingAccount.balance,
          required: totalAmount 
        });
      }
      
      // Subtract amount from balance
      tradingAccount.balance -= totalAmount;
      
      // Check if user already has this stock
      const existingHoldingIndex = tradingAccount.holdings.findIndex(
        holding => holding.symbol === symbol
      );
      
      if (existingHoldingIndex > -1) {
        // Update existing holding
        const existingHolding = tradingAccount.holdings[existingHoldingIndex];
        const newTotalQuantity = existingHolding.quantity + quantity;
        const newTotalValue = existingHolding.purchaseValue + totalAmount;
        
        // Calculate new average price
        const newAveragePrice = newTotalValue / newTotalQuantity;
        
        // Update the holding
        tradingAccount.holdings[existingHoldingIndex].quantity = newTotalQuantity;
        tradingAccount.holdings[existingHoldingIndex].averagePrice = newAveragePrice;
        tradingAccount.holdings[existingHoldingIndex].purchaseValue = newTotalValue;
        tradingAccount.holdings[existingHoldingIndex].currentPrice = price;
        tradingAccount.holdings[existingHoldingIndex].currentValue = newTotalQuantity * price;
        
        // Calculate profit/loss
        const profit = tradingAccount.holdings[existingHoldingIndex].currentValue - newTotalValue;
        tradingAccount.holdings[existingHoldingIndex].profit = profit;
        tradingAccount.holdings[existingHoldingIndex].profitPercentage = (profit / newTotalValue) * 100;
        tradingAccount.holdings[existingHoldingIndex].lastUpdated = new Date();
      } else {
        // Add new holding
        tradingAccount.holdings.push({
          symbol,
          companyName,
          quantity,
          averagePrice: price,
          purchaseValue: totalAmount,
          currentPrice: price,
          currentValue: totalAmount,
          profit: 0,
          profitPercentage: 0,
          lastUpdated: new Date()
        });
      }
    } 
    // Handle SELL order
    else if (type === 'SELL') {
      // Check if user has the stock
      const existingHoldingIndex = tradingAccount.holdings.findIndex(
        holding => holding.symbol === symbol
      );
      
      if (existingHoldingIndex === -1) {
        return res.status(400).json({ error: 'You do not own this stock' });
      }
      
      const existingHolding = tradingAccount.holdings[existingHoldingIndex];
      
      // Check if user has enough quantity
      if (existingHolding.quantity < quantity) {
        return res.status(400).json({ 
          error: 'Insufficient shares', 
          available: existingHolding.quantity,
          required: quantity 
        });
      }
      
      // Add amount to balance
      tradingAccount.balance += totalAmount;
      
      // Update or remove the holding
      const newQuantity = existingHolding.quantity - quantity;
      
      if (newQuantity === 0) {
        // Remove the holding if no shares left
        tradingAccount.holdings.splice(existingHoldingIndex, 1);
      } else {
        // Update the holding
        const soldValue = quantity * existingHolding.averagePrice;
        const newPurchaseValue = existingHolding.purchaseValue - soldValue;
        
        tradingAccount.holdings[existingHoldingIndex].quantity = newQuantity;
        tradingAccount.holdings[existingHoldingIndex].purchaseValue = newPurchaseValue;
        tradingAccount.holdings[existingHoldingIndex].currentPrice = price;
        tradingAccount.holdings[existingHoldingIndex].currentValue = newQuantity * price;
        
        // Calculate profit/loss
        const profit = tradingAccount.holdings[existingHoldingIndex].currentValue - newPurchaseValue;
        tradingAccount.holdings[existingHoldingIndex].profit = profit;
        tradingAccount.holdings[existingHoldingIndex].profitPercentage = (profit / newPurchaseValue) * 100;
        tradingAccount.holdings[existingHoldingIndex].lastUpdated = new Date();
      }
    }
    
    // Add transaction to history
    tradingAccount.transactions.push({
      symbol,
      companyName,
      type,
      quantity,
      price,
      totalAmount: type === 'BUY' ? -totalAmount : totalAmount,
      date: new Date()
    });
    
    // Save the updated account
    await tradingAccount.save();
    
    res.json(tradingAccount);
  } catch (error) {
    console.error('Error executing trade:', error.message);
    res.status(500).json({ error: 'Server error', message: error.message });
  }
});

// @route   GET /api/demotrading/transactions
// @desc    Get user's transaction history
// @access  Private
router.get('/transactions', tradingAuth, async (req, res) => {
  try {
    console.log('GET /transactions - User ID:', req.user.id);
    
    const tradingAccount = await DemoTradingAccount.findOne({ userId: req.user.id });
    
    if (!tradingAccount) {
      return res.status(404).json({ msg: 'No trading account found' });
    }
    
    // Return transactions sorted by date (newest first)
    const transactions = [...tradingAccount.transactions].sort((a, b) => b.date - a.date);
    
    res.json(transactions);
  } catch (error) {
    console.error('Error fetching transactions:', error.message);
    res.status(500).json({ error: 'Server error', message: error.message });
  }
});

// @route   PUT /api/demotrading/holdings/update
// @desc    Update current prices of holdings
// @access  Private
router.put('/holdings/update', [
  tradingAuth,
  check('holdings', 'Holdings data is required').isArray()
], async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  
  const { holdings } = req.body;
  
  try {
    console.log('PUT /holdings/update - User ID:', req.user.id);
    
    const tradingAccount = await DemoTradingAccount.findOne({ userId: req.user.id });
    
    if (!tradingAccount) {
      return res.status(404).json({ msg: 'No trading account found' });
    }
    
    // Update each holding with current price
    holdings.forEach(update => {
      const holdingIndex = tradingAccount.holdings.findIndex(h => h.symbol === update.symbol);
      
      if (holdingIndex !== -1) {
        const holding = tradingAccount.holdings[holdingIndex];
        
        // Update price and calculate new values
        holding.currentPrice = update.currentPrice;
        holding.currentValue = holding.quantity * update.currentPrice;
        holding.profit = holding.currentValue - holding.purchaseValue;
        holding.profitPercentage = (holding.profit / holding.purchaseValue) * 100;
        holding.lastUpdated = new Date();
      }
    });
    
    await tradingAccount.save();
    
    res.json(tradingAccount);
  } catch (error) {
    console.error('Error updating holdings:', error.message);
    res.status(500).json({ error: 'Server error', message: error.message });
  }
});

// @route   POST /api/demotrading/reset
// @desc    Reset demo trading account
// @access  Private
router.post('/reset', tradingAuth, async (req, res) => {
  try {
    console.log('POST /reset - User ID:', req.user.id);
    
    let tradingAccount = await DemoTradingAccount.findOne({ userId: req.user.id });
    
    if (tradingAccount) {
      // Reset the account to initial state
      tradingAccount.balance = tradingAccount.initialBalance;
      tradingAccount.holdings = [];
      tradingAccount.transactions = [];
      tradingAccount.equity = tradingAccount.initialBalance;
      tradingAccount.totalProfitLoss = 0;
      tradingAccount.totalProfitLossPercentage = 0;
      tradingAccount.lastUpdated = new Date();
      
      await tradingAccount.save();
    } else {
      // Create new account if it doesn't exist
      tradingAccount = new DemoTradingAccount({
        userId: req.user.id,
        balance: 100000,
        initialBalance: 100000
      });
      
      await tradingAccount.save();
    }
    
    res.json({ 
      success: true, 
      message: 'Trading account reset successfully',
      account: tradingAccount
    });
  } catch (error) {
    console.error('Error resetting trading account:', error.message);
    res.status(500).json({ error: 'Server error', message: error.message });
  }
});

// @route   GET /api/demotrading/portfolio-history
// @desc    Get portfolio history with performance data
// @access  Private
router.get('/portfolio-history', tradingAuth, async (req, res) => {
  try {
    console.log('GET /portfolio-history - User ID:', req.user.id);
    
    // Define now at the beginning
    const now = new Date();
    
    // Find user's trading account
    const tradingAccount = await DemoTradingAccount.findOne({ userId: req.user.id });
    
    if (!tradingAccount) {
      return res.status(404).json({ msg: 'No trading account found' });
    }
    
    // Get transactions to calculate history
    const transactions = tradingAccount.transactions;
    
    if (!transactions || transactions.length === 0) {
      // If no transactions, return single history point with current data
      return res.json([{
        date: now,
        equity: tradingAccount.equity || tradingAccount.initialBalance,
        balance: tradingAccount.balance,
        holdingsValue: tradingAccount.equity - tradingAccount.balance,
        dayChange: 0,
        dayChangePercentage: 0,
        weekChange: 0,
        weekChangePercentage: 0,
        monthChange: 0,
        monthChangePercentage: 0,
        yearChange: 0,
        yearChangePercentage: 0
      }]);
    }
    
    // Sort transactions by date (oldest first)
    const sortedTransactions = [...transactions].sort((a, b) => 
      new Date(a.date) - new Date(b.date)
    );
    
    // Create daily snapshots of portfolio
    const dailySnapshots = [];
    let currentBalance = tradingAccount.initialBalance;
    let currentHoldings = [];
    
    // Create initial snapshot at account creation
    dailySnapshots.push({
      date: tradingAccount.createdAt,
      equity: tradingAccount.initialBalance,
      balance: tradingAccount.initialBalance,
      holdingsValue: 0,
      holdings: []
    });
    
    // Process transactions to create historical snapshots
    let lastDate = new Date(tradingAccount.createdAt);
    
    sortedTransactions.forEach(transaction => {
      const transactionDate = new Date(transaction.date);
      
      // Update balance based on transaction
      if (transaction.type === 'BUY') {
        currentBalance += transaction.totalAmount; // totalAmount is negative for buys
        
        // Update holdings
        const existingIndex = currentHoldings.findIndex(h => h.symbol === transaction.symbol);
        if (existingIndex >= 0) {
          // Update existing holding
          const existing = currentHoldings[existingIndex];
          const newQuantity = existing.quantity + transaction.quantity;
          const newTotalCost = (existing.averageCost * existing.quantity) - transaction.totalAmount;
          
          currentHoldings[existingIndex] = {
            ...existing,
            quantity: newQuantity,
            averageCost: newTotalCost / newQuantity,
            currentPrice: transaction.price,
            currentValue: newQuantity * transaction.price
          };
        } else {
          // Add new holding
          currentHoldings.push({
            symbol: transaction.symbol,
            companyName: transaction.companyName,
            quantity: transaction.quantity,
            averageCost: transaction.price,
            currentPrice: transaction.price,
            currentValue: transaction.quantity * transaction.price
          });
        }
      } else if (transaction.type === 'SELL') {
        currentBalance += transaction.totalAmount; // totalAmount is positive for sells
        
        // Update holdings
        const existingIndex = currentHoldings.findIndex(h => h.symbol === transaction.symbol);
        if (existingIndex >= 0) {
          const existing = currentHoldings[existingIndex];
          const newQuantity = existing.quantity - transaction.quantity;
          
          if (newQuantity <= 0) {
            // Remove holding if all shares sold
            currentHoldings = currentHoldings.filter(h => h.symbol !== transaction.symbol);
          } else {
            // Update holding with reduced quantity
            currentHoldings[existingIndex] = {
              ...existing,
              quantity: newQuantity,
              currentPrice: transaction.price,
              currentValue: newQuantity * transaction.price
            };
          }
        }
      }
      
      // Calculate total holdings value
      const holdingsValue = currentHoldings.reduce((sum, h) => sum + h.currentValue, 0);
      const equity = currentBalance + holdingsValue;
      
      // Create snapshot for this transaction
      dailySnapshots.push({
        date: transactionDate,
        equity,
        balance: currentBalance,
        holdingsValue,
        holdings: JSON.parse(JSON.stringify(currentHoldings))
      });
      
      lastDate = transactionDate;
    });
    
    const daysSinceLastSnapshot = Math.floor((now - lastDate) / (1000 * 60 * 60 * 24));
    
    if (daysSinceLastSnapshot > 0) {
      // Calculate current holdings value from the account
      const holdingsValue = tradingAccount.holdings.reduce((sum, h) => sum + h.currentValue, 0);
      
      dailySnapshots.push({
        date: now,
        equity: tradingAccount.equity,
        balance: tradingAccount.balance,
        holdingsValue,
        holdings: tradingAccount.holdings
      });
    }
    
    // Calculate performance metrics
    const oneDayAgo = new Date(now);
    oneDayAgo.setDate(oneDayAgo.getDate() - 1);
    
    const oneWeekAgo = new Date(now);
    oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
    
    const oneMonthAgo = new Date(now);
    oneMonthAgo.setMonth(oneMonthAgo.getMonth() - 1);
    
    const oneYearAgo = new Date(now);
    oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);
    
    // Find closest snapshots for each time period
    const findClosestSnapshot = (targetDate) => {
      let closestSnapshot = dailySnapshots[0];
      let minDiff = Math.abs(new Date(dailySnapshots[0].date).getTime() - targetDate.getTime());
      
      for (let i = 1; i < dailySnapshots.length; i++) {
        const snapDate = new Date(dailySnapshots[i].date);
        const diff = Math.abs(snapDate.getTime() - targetDate.getTime());
        
        if (diff < minDiff) {
          minDiff = diff;
          closestSnapshot = dailySnapshots[i];
        }
      }
      
      return closestSnapshot;
    };
    
    const currentSnapshot = dailySnapshots[dailySnapshots.length - 1];
    const daySnapshot = findClosestSnapshot(oneDayAgo);
    const weekSnapshot = findClosestSnapshot(oneWeekAgo);
    const monthSnapshot = findClosestSnapshot(oneMonthAgo);
    const yearSnapshot = findClosestSnapshot(oneYearAgo);
    
    // Calculate changes and percentages
    const calculateChange = (oldValue, newValue) => {
      if (!oldValue || oldValue === 0) return { change: 0, percentage: 0 };
      const change = newValue - oldValue;
      const percentage = (change / oldValue) * 100;
      return { change, percentage };
    };
    
    const { change: dayChange, percentage: dayChangePercentage } = 
      calculateChange(daySnapshot.equity, currentSnapshot.equity);
      
    const { change: weekChange, percentage: weekChangePercentage } = 
      calculateChange(weekSnapshot.equity, currentSnapshot.equity);
      
    const { change: monthChange, percentage: monthChangePercentage } = 
      calculateChange(monthSnapshot.equity, currentSnapshot.equity);
      
    const { change: yearChange, percentage: yearChangePercentage } = 
      calculateChange(yearSnapshot.equity, currentSnapshot.equity);
    
    // Add performance metrics to the response
    const performance = {
      current: {
        date: currentSnapshot.date,
        equity: currentSnapshot.equity,
        balance: currentSnapshot.balance,
        holdingsValue: currentSnapshot.holdingsValue
      },
      day: {
        change: dayChange,
        percentage: dayChangePercentage
      },
      week: {
        change: weekChange,
        percentage: weekChangePercentage
      },
      month: {
        change: monthChange,
        percentage: monthChangePercentage
      },
      year: {
        change: yearChange,
        percentage: yearChangePercentage
      },
      total: {
        change: currentSnapshot.equity - tradingAccount.initialBalance,
        percentage: ((currentSnapshot.equity - tradingAccount.initialBalance) / tradingAccount.initialBalance) * 100
      }
    };
    
    // Add performance data to account in database
    tradingAccount.dayChange = dayChange;
    tradingAccount.dayChangePercentage = dayChangePercentage;
    await tradingAccount.save();
    
    res.json({
      history: dailySnapshots,
      performance
    });
  } catch (error) {
    console.error('Error fetching portfolio history:', error.message);
    res.status(500).json({ error: 'Server error', message: error.message });
  }
});

module.exports = router; 