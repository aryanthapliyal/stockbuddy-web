const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const User = require('../models/User');
const auth = require('../middleware/auth');
const { deleteFromCloudinary } = require('../config/cloudinary');

const router = express.Router();

// Register User
router.post('/register', async (req, res) => {
  try {
    // Extract fields from the request body
    const { 
      name, 
      email, 
      password, 
      countryCode, 
      phoneNumber, 
      address, 
      dateOfBirth, 
      captchaVerified 
    } = req.body;
    
    // Log received data
    console.log('Registration data received:', {
      name,
      email,
      passwordProvided: !!password,
      countryCode,
      phoneNumber,
      address,
      dateOfBirth,
      captchaVerified
    });
    
    // Validate required fields
    if (!name || !email || !password) {
      return res.status(400).json({ 
        success: false, 
        message: 'Name, email and password are required',
        validationErrors: {
          name: !name ? 'Name is required' : null,
          email: !email ? 'Email is required' : null,
          password: !password ? 'Password is required' : null,
          address:!address ?'Aeddress is reqired':null,
          dateOfBirth:!dateOfBirth ? 'Date Of Birth is Required':null
        }
      });
    }

    // Validate password length
    if (password.length < 6) {
      return res.status(400).json({ 
        success: false, 
        message: 'Password must be at least 6 characters long',
        validationErrors: {
          password: 'Password must be at least 6 characters long'
        }
      });
    }
    
    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({
        success: false,
        message: 'Invalid email format',
        validationErrors: {
          email: 'Invalid email format'
        }
      });
    }
    
    // Check if user exists
    const existingUser = await User.findOne({ email: email.toLowerCase() });
    if (existingUser) {
      return res.status(400).json({ 
        success: false, 
        message: 'User already exists',
        validationErrors: {
          email: 'Email is already registered'
        }
      });
    }
    
    // Check CAPTCHA verification
    if (!captchaVerified) {
      return res.status(403).json({
        success: false,
        message: 'CAPTCHA verification required',
        requiresCaptcha: true
      });
    }
    
    // Hash password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);
    
    // Create current timestamp
    const currentTime = new Date();
    
    // Format date of birth
    let formattedDateOfBirth = null;
    if (dateOfBirth) {
      formattedDateOfBirth = new Date(dateOfBirth);
      if (isNaN(formattedDateOfBirth.getTime())) {
        return res.status(400).json({
          success: false,
          message: 'Invalid date of birth format',
          validationErrors: {
            dateOfBirth: 'Invalid date format'
          }
        });
      }
    }
    
    // Create new user with all fields
    const user = new User({
      name: name.trim(),
      email: email.toLowerCase().trim(),
      password: hashedPassword,
      countryCode: (countryCode || '+1').trim(),
      phoneNumber: (phoneNumber || '').trim(),
      address: (address || '').trim(),
      dateOfBirth: formattedDateOfBirth,
      captchaVerified: true,
      createdAt: currentTime,
      lastLogin: currentTime
    });
    
    // Log user object before saving
    console.log('User object before save:', user.toObject());
    
    // Save user to database
    await user.save();
    console.log('User saved successfully');
    
    // Generate JWT token
    const token = jwt.sign(
      { userId: user._id },
      process.env.JWT_SECRET || 'your-jwt-secret',
      { expiresIn: '7d' }
    );
    
    // Return user data and token
    res.status(201).json({
      success: true,
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        countryCode: user.countryCode,
        phoneNumber: user.phoneNumber,
        address: user.address,
        dateOfBirth: user.dateOfBirth,
        createdAt: user.createdAt,
        lastLogin: user.lastLogin,
        captchaVerified: user.captchaVerified
      }
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Server error during registration',
      error: error.message
    });
  }
});

// Login User
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    // Input validation
    if (!email || !password) {
      return res.status(400).json({ success: false, message: 'All fields are required' });
    }
    
    // Check if user exists
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({ success: false, message: 'Invalid credentials' });
    }
    
    // Verify password
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ success: false, message: 'Invalid credentials' });
    }
    
    // Check if CAPTCHA verification is required
    if (!user.captchaVerified) {
      return res.status(403).json({
        success: false,
        message: 'CAPTCHA verification required',
        requiresCaptcha: true,
      });
    }
    
    // Update last login time
    user.lastLogin = new Date();
    await user.save();
    
    // Get the updated user
    const updatedUser = await User.findById(user._id);
    if (!updatedUser) {
      throw new Error('Failed to retrieve updated user');
    }
    
    const userData = updatedUser.toObject();
    console.log('User data after login:', userData);
    
    // Generate JWT
    const token = jwt.sign(
      { userId: userData._id },
      process.env.JWT_SECRET || 'your-jwt-secret',
      { expiresIn: '7d' }
    );
    
    // Log token data for debugging
    console.log('Token payload:', { userId: userData._id });
    
    // Return ALL user data in response with consistent format
    res.json({
      success: true,
      token,
      user: {
        id: userData._id,
        name: userData.name || '',
        email: userData.email || '',
        countryCode: userData.countryCode || '+1',
        phoneNumber: userData.phoneNumber || '',
        address: userData.address || '',
        dateOfBirth: userData.dateOfBirth || '',
        createdAt: userData.createdAt || '',
        lastLogin: userData.lastLogin || '',
        captchaVerified: userData.captchaVerified || false
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ success: false, message: 'Server error' });
  }
});

// Change Password
router.post('/change-password', auth, async (req, res) => {
  try {
    const { currentPassword, newPassword } = req.body;
    const userId = req.userId;

    // Validate required fields
    if (!currentPassword || !newPassword) {
      return res.status(400).json({ 
        success: false, 
        message: 'Current password and new password are required'
      });
    }

    // Validate new password strength
    if (newPassword.length < 8) {
      return res.status(400).json({ 
        success: false, 
        message: 'New password must be at least 8 characters long'
      });
    }

    // Find user
    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ success: false, message: 'User not found' });
    }

    // Verify current password
    const isMatch = await bcrypt.compare(currentPassword, user.password);
    if (!isMatch) {
      return res.status(400).json({ success: false, message: 'Current password is incorrect' });
    }

    // Hash new password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(newPassword, salt);

    // Update password
    user.password = hashedPassword;
    await user.save();

    // Return success response
    res.json({ success: true, message: 'Password updated successfully' });
  } catch (error) {
    console.error('Change password error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Server error during password change',
      error: error.message 
    });
  }
});

// Admin route to delete a user account by email or name
router.delete('/admin/delete-user', auth, async (req, res) => {
  try {
    // Check if the requesting user exists
    const requestingUser = await User.findById(req.userId);
    if (!requestingUser) {
      return res.status(404).json({ success: false, message: 'Requesting user not found' });
    }
    
    // Require admin password for verification
    const { email, name, password, confirmationText } = req.body;
    
    if (!password) {
      return res.status(400).json({ 
        success: false, 
        message: 'Password is required for this operation' 
      });
    }
    
    // Validate confirmation text was provided and matches expected value
    const expectedConfirmation = `delete user ${email || name}`;
    if (!confirmationText || confirmationText.toLowerCase() !== expectedConfirmation) {
      return res.status(400).json({ 
        success: false, 
        message: `Please type "${expectedConfirmation}" to confirm user deletion`
      });
    }
    
    // Verify admin password
    const isMatch = await bcrypt.compare(password, requestingUser.password);
    if (!isMatch) {
      return res.status(403).json({ 
        success: false, 
        message: 'Authentication failed' 
      });
    }
    
    // Check if admin email is in the allowed admin list
    const adminEmails = ['admin@stockbuddy.com']; // Store this in environment variables in production
    if (!adminEmails.includes(requestingUser.email.toLowerCase())) {
      console.warn(`Unauthorized admin attempt from ${requestingUser.email}`);
      return res.status(403).json({ 
        success: false, 
        message: 'Not authorized to perform this action' 
      });
    }
    
    if (!email && !name) {
      return res.status(400).json({ success: false, message: 'Email or name is required to delete a user' });
    }
    
    // Find the user to delete
    let userToDelete;
    if (email) {
      userToDelete = await User.findOne({ email });
    } else if (name) {
      userToDelete = await User.findOne({ name });
    }
    
    if (!userToDelete) {
      return res.status(404).json({ success: false, message: 'User not found with the provided credentials' });
    }
    
    // Prevent admin from deleting themselves
    if (userToDelete._id.toString() === requestingUser._id.toString()) {
      return res.status(400).json({ 
        success: false, 
        message: 'Admin cannot delete their own account using this endpoint' 
      });
    }
    
    // Delete the user's profile picture from Cloudinary if it exists
    if (userToDelete.profilePicture) {
      try {
        console.log('Attempting to delete profile picture from Cloudinary before account deletion:', userToDelete.profilePicture);
        const deleteResult = await deleteFromCloudinary(userToDelete.profilePicture);
        console.log('Profile picture deletion result during account deletion:', deleteResult);
      } catch (cloudinaryError) {
        console.error('Error deleting profile picture during account deletion:', cloudinaryError);
        // Continue with account deletion even if profile picture deletion fails
      }
    }
    
    // Delete all predictions associated with the user
    const Prediction = require('../models/Prediction');
    try {
      const deletePredictionsResult = await Prediction.deleteMany({ userId: userToDelete._id });
      console.log(`Deleted ${deletePredictionsResult.deletedCount} predictions associated with the user`);
    } catch (predictionsError) {
      console.error('Error deleting user predictions:', predictionsError);
      // Continue with account deletion even if predictions deletion fails
    }
    
    // Delete the user account
    const deleteResult = await User.findByIdAndDelete(userToDelete._id);
    if (!deleteResult) {
      return res.status(500).json({ success: false, message: 'Account could not be deleted' });
    }
    
    // Log the action for audit purposes
    console.log(`Admin ${requestingUser.email} deleted user ${userToDelete.email} (${userToDelete._id})`);
    
    // Return success response
    return res.status(200).json({ 
      success: true, 
      message: `Account for ${email || name} successfully deleted with all associated data`
    });
  } catch (error) {
    console.error('Server error during admin account deletion:', error);
    return res.status(500).json({ success: false, message: 'Server error during account deletion' });
  }
});

module.exports = router; 