const jwt = require('jsonwebtoken');

// Authentication middleware
const auth = (req, res, next) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    
    if (!token) {
      return res.status(401).json({ success: false, message: 'No token provided' });
    }
    
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-jwt-secret');
    
    // Set userId directly on req object
    req.userId = decoded.userId;
    
    // Also set user object with id property for compatibility
    req.user = { id: decoded.userId };
    
    next();
  } catch (error) {
    res.status(401).json({ success: false, message: 'Authentication required' });
  }
};

module.exports = auth; 