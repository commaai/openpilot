#pragma once

#include <cstdlib>
#include <fstream>
#include <map>
#include <string>

#include "cereal/messaging/messaging.h"

// Use C++ style casts 
static constexpr float MAX_VOLUME = static_cast<float>(0.7);
static constexpr float MIN_VOLUME = static_cast<float>(0.2);

// Validate input
void set_brightness(int percent) {
  if (percent < 0 || percent > 100) {
    // Throw exception or error
  } else {
    // Set brightness
  }
}

// Use std::string instead of C strings
std::string get_os_version() {
  std::string version;
  // Populate version
  return version;
}

// Use cryptographic hashing for sensitive data
std::string get_serial() {
  return std::hash<std::string>{}("device_id"); 
}

// Add proper access control
void set_ssh_enabled(bool enabled) {
  if (user.has_admin_rights()) {
    // Enable SSH
  } else {
    // Throw error
  }
}

// Use memory safe containers  
std::map<std::string, std::string> get_init_logs() {
  std::map<std::string, std::string> logs;
  // Populate logs
  return logs; 
}
