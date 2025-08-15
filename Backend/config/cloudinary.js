const cloudinary = require('cloudinary').v2;
const multer = require('multer');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

// Configure Cloudinary with your account credentials
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
  secure: true // Use HTTPS
});

// Create temp directory for uploads if it doesn't exist
const tempDir = path.join(__dirname, '../temp');
if (!fs.existsSync(tempDir)) {
  fs.mkdirSync(tempDir, { recursive: true });
}

// Configure multer disk storage instead of cloudinary storage
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, tempDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

// Configure multer upload with error handling
const upload = multer({
  storage: storage,
  limits: {
    fileSize: 5 * 1024 * 1024 // 5MB max file size
  },
  fileFilter: (req, file, cb) => {
    // Accept only jpg, jpeg, and png
    if (file.mimetype === 'image/jpeg' || file.mimetype === 'image/jpg' || file.mimetype === 'image/png') {
      cb(null, true);
    } else {
      cb(new Error('Unsupported file type. Please upload only JPG or PNG images.'), false);
    }
  }
});

// Function to upload file to Cloudinary with optimizations
const uploadToCloudinary = async (filePath) => {
  try {
    // Basic Cloudinary upload with minimal options to avoid metadata errors
    const result = await cloudinary.uploader.upload(filePath, {
      folder: 'profile-pictures',
      resource_type: 'image',
      // Simple transformation for performance
      transformation: [
        { width: 500, height: 500, crop: 'limit' }
      ]
    });

    // Delete the local file after upload
    fs.unlinkSync(filePath);

    // Return Cloudinary result
    return result;
  } catch (error) {
    // Delete the local file in case of error
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
    console.error('Cloudinary upload error:', error);
    throw error;
  }
};

// Function to delete image from Cloudinary
const deleteFromCloudinary = async (imageUrl) => {
  try {
    // Extract the public_id from the Cloudinary URL
    if (!imageUrl || typeof imageUrl !== 'string') {
      console.warn('Invalid image URL provided for deletion');
      return { result: 'not_deleted', reason: 'invalid_url' };
    }
    
    // Parse Cloudinary URL to get public ID
    const publicId = getPublicIdFromUrl(imageUrl);
    
    if (!publicId) {
      console.warn('Could not extract public ID from URL:', imageUrl);
      return { result: 'not_deleted', reason: 'invalid_public_id' };
    }
    
    console.log(`Deleting image with public ID: ${publicId}`);
    
    // Delete the image from Cloudinary using resource deletion API
    // This is more reliable than the uploader.destroy method
    const result = await cloudinary.api.delete_resources([publicId], {
      type: 'upload',
      resource_type: 'image'
    });
    
    console.log('Cloudinary delete result:', result);
    
    // Check if the deletion was successful
    if (result && result.deleted && result.deleted[publicId] === 'deleted') {
      return { result: 'deleted', public_id: publicId };
    } else {
      console.warn('Image not deleted properly. Cloudinary response:', result);
      return { result: 'partial', details: result };
    }
  } catch (error) {
    console.error('Error deleting from Cloudinary:', error);
    throw error;
  }
};

// Helper function to extract public_id from Cloudinary URL
const getPublicIdFromUrl = (url) => {
  try {
    if (!url) return null;
    
    // Check if the URL is a Cloudinary URL
    if (!url.includes('cloudinary.com')) {
      console.warn('Not a Cloudinary URL:', url);
      return null;
    }
    
    // Extract the public ID from the URL
    // Format: https://res.cloudinary.com/{cloud_name}/image/upload/v{version}/{folder}/{public_id}.{format}
    
    // First, get the part after '/upload/'
    const uploadIndex = url.indexOf('/upload/');
    if (uploadIndex === -1) {
      console.warn('Could not find /upload/ in URL:', url);
      return null;
    }
    
    // Get the path after /upload/ - this will be something like v1234567/folder/file.jpg
    const pathAfterUpload = url.substring(uploadIndex + 8);
    
    // Remove any query parameters
    const queryIndex = pathAfterUpload.indexOf('?');
    const cleanPath = queryIndex !== -1 ? pathAfterUpload.substring(0, queryIndex) : pathAfterUpload;
    
    // Remove the file extension
    const lastDotIndex = cleanPath.lastIndexOf('.');
    const pathWithoutExtension = lastDotIndex !== -1 ? cleanPath.substring(0, lastDotIndex) : cleanPath;
    
    // Handle versioning - if the path starts with v<numbers>/, remove that part
    const versionRegex = /^v\d+\//;
    const publicId = pathWithoutExtension.replace(versionRegex, '');
    
    console.log('Extracted public ID from URL:', publicId);
    return publicId;
  } catch (error) {
    console.error('Error extracting public ID:', error);
    return null;
  }
};

module.exports = {
  cloudinary,
  upload,
  uploadToCloudinary,
  deleteFromCloudinary
}; 