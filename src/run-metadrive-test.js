const axios = require('axios');

// Function to run MetaDrive simulation test in GitHub Actions
async function runMetaDriveTest() {
  try {
    const response = await axios.post('https://api.metadrive.io/simulate',
      { /* Simulation data */ },
      { headers: { 'Authorization': `Bearer ${process.env.METADRIVE_API_KEY}` } }
    );
    console.log('MetaDrive simulation test passed:', response.data);
  } catch (error) {
    console.error('Failed to run MetaDrive simulation test:', error.message);
    throw new Error(error.message);
  }
}

// Run the function
runMetaDriveTest();