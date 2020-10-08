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
      void manager_start();
      void panda_disconnect();
      void _delete(string key);
      string get(string key, bool);
      void put(string key, string data);
  };

};
#endif
