#include "tools/cabana/lua/lua_evaluator.h"

extern "C" {
#include "tools/cabana/lua/lua-src/lua.h"
#include "tools/cabana/lua/lua-src/lauxlib.h"
#include "tools/cabana/lua/lua-src/lualib.h"
}

#include <algorithm>

// Binary search for the value of a series at time t (linear interpolation).
static double interpAt(const std::vector<CerealPoint> &pts, double t) {
  if (pts.empty()) return 0.0;
  auto it = std::lower_bound(pts.begin(), pts.end(), t, [](const CerealPoint &p, double v) { return p.t < v; });
  if (it == pts.end()) return pts.back().v;
  if (it == pts.begin()) return pts.front().v;
  auto prev = std::prev(it);
  double dt = it->t - prev->t;
  if (dt < 1e-9) return prev->v;
  return prev->v + (it->v - prev->v) * (t - prev->t) / dt;
}

std::string luaEvaluate(const LuaSnippet &snippet, const CerealSeriesMap &map, CerealSeriesMap &out) {
  auto src_it = map.find(snippet.linked_source);
  if (src_it == map.end() || src_it->second.empty())
    return "linked source '" + snippet.linked_source + "' not found in loaded data";

  // Pre-fetch additional source pointers (null if not found)
  std::vector<const std::vector<CerealPoint> *> extra;
  for (const auto &path : snippet.additional_sources) {
    auto it = map.find(path);
    extra.push_back(it != map.end() ? &it->second : nullptr);
  }

  lua_State *L = luaL_newstate();
  luaL_openlibs(L); // gives math, string, table — no io/os file access needed but harmless

  // Run global init once (sets up persistent state like last_bad_time)
  if (!snippet.global_init.empty()) {
    if (luaL_dostring(L, snippet.global_init.c_str()) != LUA_OK) {
      std::string err = lua_tostring(L, -1);
      lua_close(L);
      return "global init error: " + err;
    }
  }

  // Wrap function body so it can `return` a value
  std::string func = "function __eval(value, v1, v2, v3, v4, time)\n" + snippet.function_body + "\nend";
  if (luaL_dostring(L, func.c_str()) != LUA_OK) {
    std::string err = lua_tostring(L, -1);
    lua_close(L);
    return "function compile error: " + err;
  }

  auto &result = out[snippet.name];
  result.reserve(src_it->second.size());

  for (const auto &pt : src_it->second) {
    lua_getglobal(L, "__eval");
    lua_pushnumber(L, pt.v);                                              // value
    for (int i = 0; i < 4; i++) {
      double v = (i < (int)extra.size() && extra[i]) ? interpAt(*extra[i], pt.t) : 0.0;
      lua_pushnumber(L, v);                                               // v1..v4
    }
    lua_pushnumber(L, pt.t);                                              // time

    if (lua_pcall(L, 6, 1, 0) != LUA_OK) {
      std::string err = lua_tostring(L, -1);
      lua_close(L);
      return "runtime error at t=" + std::to_string(pt.t) + ": " + err;
    }

    if (lua_isnumber(L, -1))
      result.push_back({pt.t, lua_tonumber(L, -1)});
    lua_pop(L, 1);
  }

  lua_close(L);
  return {};
}
