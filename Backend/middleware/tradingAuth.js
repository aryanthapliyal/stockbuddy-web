const jwt = require('jsonwebtoken');

// Authentication middleware specifically for trading routes
const tradingAuth = (req, res, next) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    
    if (!token) {
      return res.status(401).json({ success: false, message: 'No token provided' });
    }
    
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-jwt-secret');
    
    // Debug log to see what's in the token
    console.log('Token decoded for trading route:', decoded);
    
    // Set up user object properly
    req.user = { id: decoded.userId };
    
    // Double check that we have a user ID
    if (!req.user.id) {
      return res.status(401).json({ success: false, message: 'Invalid user ID in token' });
    }
    
    next();
  } catch (error) {
    console.error('Trading auth error:', error);
    res.status(401).json({ success: false, message: 'Authentication failed: ' + error.message });
  }
};

module.exports = tradingAuth; 