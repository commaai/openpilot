#ifndef PARAMS_HELPER_H
#define PARAMS_HELPER_H

#include <map>
#include <vector>
#include <exception>
#include <string>

using std::vector;
using std::map;
using std::exception;
using std::string;

namespace params {  

  enum class TxType {
    PERSISTENT=1, 
    CLEAR_ON_MANAGER_START=2, 
    CLEAR_ON_PANDA_DISCONNECT=3
  };
  
  struct UnknownKeyName : public exception {
    const char* what() const throw() {
      return "UnknownKeyName";
    }
  };

  struct MustEnterDB : public exception {
    const char* what() const throw() {
      return "Must call __enter__ before using BD.";
    }
  };

  struct OSError : public exception {
    const char* what() const throw() {
      return "OSError";
    }
  };
  
  static string current_path();
  
  static bool is_directory(string path);
  
  static bool is_symlink(string path);
  
  static bool exists(string path);

  static void remove_all(string path);

  static void mkdirs_exists_ok(string path);

  static int fsync_dir(const char* path);

  map<string, vector<TxType>> keys = {
    {"AccessToken", {TxType::CLEAR_ON_MANAGER_START} },
    {"AthenadPid", {TxType::PERSISTENT} },
    {"CalibrationParams", {TxType::PERSISTENT} },
    {"CarParams", {TxType::CLEAR_ON_MANAGER_START, TxType::CLEAR_ON_PANDA_DISCONNECT} },
    {"CarParamsCache", {TxType::CLEAR_ON_MANAGER_START, TxType::CLEAR_ON_PANDA_DISCONNECT} },
    {"CarVin", {TxType::CLEAR_ON_MANAGER_START, TxType::CLEAR_ON_PANDA_DISCONNECT} },
    {"CommunityFeaturesToggle", {TxType::PERSISTENT} },
    {"CompletedTrainingVersion", {TxType::PERSISTENT} },
    {"ControlsParams", {TxType::PERSISTENT} },
    {"DisablePowerDown", {TxType::PERSISTENT} },
    {"DisableUpdates", {TxType::PERSISTENT} },
    {"DoUninstall", {TxType::CLEAR_ON_MANAGER_START} },
    {"DongleId", {TxType::PERSISTENT} },
    {"GitBranch", {TxType::PERSISTENT} },
    {"GitCommit", {TxType::PERSISTENT} },
    {"GitRemote", {TxType::PERSISTENT} },
    {"GithubSshKeys", {TxType::PERSISTENT} },
    {"HasAcceptedTerms", {TxType::PERSISTENT} },
    {"HasCompletedSetup", {TxType::PERSISTENT} },
    {"IsDriverViewEnabled", {TxType::CLEAR_ON_MANAGER_START} },
    {"IsLdwEnabled", {TxType::PERSISTENT} },
    {"IsGeofenceEnabled", {TxType::PERSISTENT} },
    {"IsMetric", {TxType::PERSISTENT} },
    {"IsOffroad", {TxType::CLEAR_ON_MANAGER_START} },
    {"IsRHD", {TxType::PERSISTENT} },
    {"IsTakingSnapshot", {TxType::CLEAR_ON_MANAGER_START} },
    {"IsUpdateAvailable", {TxType::CLEAR_ON_MANAGER_START} },
    {"IsUploadRawEnabled", {TxType::PERSISTENT} },
    {"LastAthenaPingTime", {TxType::PERSISTENT} },
    {"LastUpdateTime", {TxType::PERSISTENT} },
    {"LimitSetSpeed", {TxType::PERSISTENT} },
    {"LimitSetSpeedNeural", {TxType::PERSISTENT} },
    {"LiveParameters", {TxType::PERSISTENT} },
    {"LongitudinalControl", {TxType::PERSISTENT} },
    {"OpenpilotEnabledToggle", {TxType::PERSISTENT} },
    {"LaneChangeEnabled", {TxType::PERSISTENT} },
    {"PandaFirmware", {TxType::CLEAR_ON_MANAGER_START, TxType::CLEAR_ON_PANDA_DISCONNECT} },
    {"PandaFirmwareHex", {TxType::CLEAR_ON_MANAGER_START, TxType::CLEAR_ON_PANDA_DISCONNECT} },
    {"PandaDongleId", {TxType::CLEAR_ON_MANAGER_START, TxType::CLEAR_ON_PANDA_DISCONNECT} },
    {"Passive", {TxType::PERSISTENT} },
    {"RecordFront", {TxType::PERSISTENT} },
    {"ReleaseNotes", {TxType::PERSISTENT} },
    {"ShouldDoUpdate", {TxType::CLEAR_ON_MANAGER_START} },
    {"SpeedLimitOffset", {TxType::PERSISTENT} },
    {"SubscriberInfo", {TxType::PERSISTENT} },
    {"TermsVersion", {TxType::PERSISTENT} },
    {"TrainingVersion", {TxType::PERSISTENT} },
    {"UpdateAvailable", {TxType::CLEAR_ON_MANAGER_START} },
    {"UpdateFailedCount", {TxType::CLEAR_ON_MANAGER_START} },
    {"Version", {TxType::PERSISTENT} },
    {"Offroad_ChargeDisabled", {TxType::CLEAR_ON_MANAGER_START, TxType::CLEAR_ON_PANDA_DISCONNECT} },
    {"Offroad_ConnectivityNeeded", {TxType::CLEAR_ON_MANAGER_START} },
    {"Offroad_ConnectivityNeededPrompt", {TxType::CLEAR_ON_MANAGER_START} },
    {"Offroad_TemperatureTooHigh", {TxType::CLEAR_ON_MANAGER_START} },
    {"Offroad_PandaFirmwareMismatch", {TxType::CLEAR_ON_MANAGER_START, TxType::CLEAR_ON_PANDA_DISCONNECT} },
   {"Offroad_InvalidTime", {TxType::CLEAR_ON_MANAGER_START} },
    {"Offroad_IsTakingSnapshot", {TxType::CLEAR_ON_MANAGER_START} },
    {"Offroad_NeosUpdate", {TxType::CLEAR_ON_MANAGER_START} }

  };


  class FileLock {
    
    private:
      string _path;
      bool _create;
      int _fd;

    public:
      FileLock(string path, bool create);
      void acquire();
      void release();
  };
  
  class DBAccessor {
  
    protected:
      map<string, string> *_vals = NULL; 
      string _path;
    public:
      DBAccessor(string path);
      virtual ~DBAccessor() = 0;
      vector<string> keys();
      const char* get(string key);
      map<string, string>* _read_values_locked();
      FileLock* _get_lock(bool create);
      string _data_path();
      void _check_entered();
      virtual void exit() = 0; 
      virtual void _delete(string key) = 0;
  };

  class DBReader : public DBAccessor {
    
    private:
      FileLock *_lock;
    public:
      DBReader(string path);
      ~DBReader();
      void enter();      
      void exit();
      void _delete(string key);
  };

  class DBWriter : public DBAccessor {

    private:
      FileLock *_lock;
      int prev_umask;
      void finally_outer();
    public:
      DBWriter(string path);
      ~DBWriter();
      void put(string key, string value);
      void _delete(string key);
      void enter();
      void exit();
  };


  class Params {

    private:
      string db;
      bool has(vector<TxType>, TxType);
      string read_db(string params_path, string key);
      void write_db(string params_path, string key, string value);
      DBAccessor* transaction(bool write);
      void _clear_keys_with_type(TxType t);
    public:
      Params(string d);
      void clear_all();
      void f();
      void manager_start();
      void panda_disconnect();
      void _delete(string key);
      string get(string key);
      string get(string key, bool);
      void put(string key, string data);
  };

};
#endif
