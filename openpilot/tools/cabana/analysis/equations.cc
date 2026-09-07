#include "tools/cabana/analysis/equations.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <stdexcept>

namespace cabana {
namespace {
struct lua_State;
struct lua_Debug;
using LuaFunction = int (*)(lua_State *);
using LuaContinuation = int (*)(lua_State *, int, intptr_t);

// Lua 5.3/5.4's stable C ABI. Resolve at runtime so ordinary Cabana charts need no Lua installation.
struct Lua {
  void *library = nullptr;
  lua_State *(*newstate)();
  void (*close)(lua_State *);
  void (*requiref)(lua_State *, const char *, LuaFunction, int);
  LuaFunction math;
  int (*loadstring)(lua_State *, const char *);
  int (*pcall)(lua_State *, int, int, int, intptr_t, LuaContinuation);
  int (*getglobal)(lua_State *, const char *);
  void (*pushnumber)(lua_State *, double);
  double (*tonumber)(lua_State *, int, int *);
  const char *(*tostring)(lua_State *, int, size_t *);
  void (*settop)(lua_State *, int);
  int (*gettop)(lua_State *);
  void (*sethook)(lua_State *, void (*)(lua_State *, lua_Debug *), int, int);
  int (*error)(lua_State *, const char *, ...);

  Lua() {
    for (const char *name : {"liblua5.4.so", "liblua5.4.so.0", "liblua5.3.so.0", "liblua.5.4.dylib",
                             "/opt/homebrew/opt/lua/lib/liblua.dylib", "/usr/local/opt/lua/lib/liblua.dylib"}) {
      if ((library = dlopen(name, RTLD_NOW | RTLD_LOCAL))) break;
    }
    if (!library) throw std::runtime_error("Lua 5.3 or 5.4 is required for this layout's equations (install the Lua shared library).");
#define LUA_SYMBOL(member, symbol) member = reinterpret_cast<decltype(member)>(dlsym(library, symbol)); if (!member) throw std::runtime_error("Incomplete Lua runtime: " symbol)
    LUA_SYMBOL(newstate, "luaL_newstate");
    LUA_SYMBOL(close, "lua_close");
    LUA_SYMBOL(requiref, "luaL_requiref");
    LUA_SYMBOL(math, "luaopen_math");
    LUA_SYMBOL(loadstring, "luaL_loadstring");
    LUA_SYMBOL(pcall, "lua_pcallk");
    LUA_SYMBOL(getglobal, "lua_getglobal");
    LUA_SYMBOL(pushnumber, "lua_pushnumber");
    LUA_SYMBOL(tonumber, "lua_tonumberx");
    LUA_SYMBOL(tostring, "lua_tolstring");
    LUA_SYMBOL(settop, "lua_settop");
    LUA_SYMBOL(gettop, "lua_gettop");
    LUA_SYMBOL(sethook, "lua_sethook");
    LUA_SYMBOL(error, "luaL_error");
#undef LUA_SYMBOL
  }
};
Lua &lua() { static Lua api; return api; }
void instructionLimit(lua_State *state, lua_Debug *) { lua().error(state, "Equation exceeded its instruction limit"); }
}  // namespace

double nearestValue(const std::vector<Sample> &samples, double time) {
  if (samples.empty()) return std::numeric_limits<double>::quiet_NaN();
  auto it = std::lower_bound(samples.begin(), samples.end(), time, [](const auto &p, double x) { return p.x < x; });
  if (it == samples.end()) return samples.back().y;
  if (it != samples.begin() && time - (it - 1)->x < it->x - time) --it;
  return it->y;
}

std::vector<Sample> evaluateEquation(const Equation &equation, const Telemetry &data) {
  auto source = data.find(equation.source);
  if (source == data.end() || source->second.empty()) throw std::runtime_error("Waiting for " + equation.source);
  std::vector<const std::vector<Sample> *> inputs;
  for (const auto &path : equation.additional) {
    auto it = data.find(path);
    if (it == data.end() || it->second.empty()) throw std::runtime_error("Waiting for " + path);
    inputs.push_back(&it->second);
  }
  auto &api = lua();
  std::unique_ptr<lua_State, decltype(api.close)> state(api.newstate(), api.close);
  if (!state) throw std::runtime_error("Could not create Lua state");
  auto *L = state.get();
  api.requiref(L, "math", api.math, 1);
  api.settop(L, 0);
  auto check = [&](int result) {
    if (result) {
      const char *message = api.tostring(L, -1, nullptr);
      throw std::runtime_error(message ? message : "Lua equation failed");
    }
  };
  std::string code = equation.globals + "\nfunction calc(time, value";
  for (size_t i = 0; i < inputs.size(); ++i) code += ", v" + std::to_string(i + 1);
  code += ")\n" + equation.function + "\nend";
  api.sethook(L, instructionLimit, 8, 100000);
  check(api.loadstring(L, code.c_str()));
  check(api.pcall(L, 0, 0, 0, 0, nullptr));
  std::vector<Sample> result;
  result.reserve(source->second.size());
  for (const auto &sample : source->second) {
    api.sethook(L, instructionLimit, 8, 100000);
    api.getglobal(L, "calc");
    api.pushnumber(L, sample.x);
    api.pushnumber(L, sample.y);
    for (auto input : inputs) api.pushnumber(L, nearestValue(*input, sample.x));
    check(api.pcall(L, inputs.size() + 2, -1, 0, 0, nullptr));
    int count = api.gettop(L), valid = 0;
    double time = sample.x;
    if (count == 2) {
      time = api.tonumber(L, 1, &valid);
      if (!valid) throw std::runtime_error("Equation time must be a number");
    }
    if (count != 1 && count != 2) throw std::runtime_error("Equation must return value or time, value");
    double value = api.tonumber(L, count, &valid);
    if (!valid) throw std::runtime_error("Equation value must be a number");
    if (std::isfinite(time) && std::isfinite(value)) result.emplace_back(time, value);
    api.settop(L, 0);
  }
  std::stable_sort(result.begin(), result.end(), [](const auto &a, const auto &b) { return a.x < b.x; });
  return result;
}
}  // namespace cabana
