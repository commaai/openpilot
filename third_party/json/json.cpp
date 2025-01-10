/* Copyright (c) 2013 Dropbox, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "json_helper.hpp"
#include "json.hpp"

namespace json {

// Constructors
Json::Json() noexcept : m_json(nullptr) {}
Json::Json(std::nullptr_t) noexcept : m_json(nullptr) {}
Json::Json(double value) : m_json(value) {}
Json::Json(int value) : m_json(value) {}
Json::Json(bool value) : m_json(value) {}
Json::Json(const std::string &value) : m_json(value) {}
Json::Json(std::string &&value) : m_json(std::move(value)) {}
Json::Json(const char * value) : m_json(value) {}
Json::Json(const array &values) : m_json(nlohmann::json::array()) {
    for (const auto& v : values) {
        m_json.push_back(v.m_json);
    }
}
Json::Json(array &&values) : m_json(nlohmann::json::array()) {
    for (auto& v : values) {
        m_json.push_back(std::move(v.m_json));
    }
}
Json::Json(const object &values) : m_json(nlohmann::json::object()) {
    for (const auto& kv : values) {
        m_json[kv.first] = kv.second.m_json;
    }
}
Json::Json(object &&values) : m_json(nlohmann::json::object()) {
    for (auto& kv : values) {
        m_json[kv.first] = std::move(kv.second.m_json);
    }
}

Json::Json(const nlohmann::json& j) : m_json(j) {}

// Type accessors
Json::Type Json::type() const {
    if (m_json.is_null()) return NUL;
    if (m_json.is_number()) return NUMBER;
    if (m_json.is_boolean()) return BOOL;
    if (m_json.is_string()) return STRING;
    if (m_json.is_array()) return ARRAY;
    if (m_json.is_object()) return OBJECT;
    return NUL;
}

// Accessors
double Json::number_value() const { return m_json.get<double>(); }
int Json::int_value() const { return m_json.get<int>(); }
bool Json::bool_value() const { return m_json.get<bool>(); }
const std::string& Json::string_value() const {
    static const std::string empty;
    return m_json.is_string() ? m_json.get_ref<const std::string&>() : empty;
}

const Json::array& Json::array_items() const {
    static const array empty;
    if (!is_array()) return empty;
    static thread_local array items;
    items.clear();
    for (const auto& item : m_json) {
        items.emplace_back(item);
    }
    return items;
}

const Json::object& Json::object_items() const {
    static const object empty;
    if (!is_object()) return empty;
    static thread_local object items;
    items.clear();
    for (auto it = m_json.begin(); it != m_json.end(); ++it) {
        items.emplace(it.key(), Json(it.value()));
    }
    return items;
}

// Operators
const Json& Json::operator[](size_t i) const {
    static const Json empty;
    if (!is_array() || i >= m_json.size()) return empty;
    static thread_local Json result;
    result = Json(m_json[i]);
    return result;
}

const Json& Json::operator[](const std::string& key) const {
    static const Json empty;
    if (!is_object()) return empty;
    auto it = m_json.find(key);
    if (it == m_json.end()) return empty;
    static thread_local Json result;
    result = Json(*it);
    return result;
}

bool Json::operator==(const Json& other) const {
    return m_json == other.m_json;
}

bool Json::operator<(const Json& other) const {
    return m_json < other.m_json;
}

// Parsing
Json Json::parse(const std::string& in, std::string& err, JsonParse strategy) {
    try {
        auto result = nlohmann::json::parse(in);
        err.clear();
        return Json(result);
    } catch (const nlohmann::json::parse_error& e) {
        err = e.what();
        return Json();
    }
}

std::vector<Json> Json::parse_multi(const std::string& in, std::string::size_type& parser_stop_pos, std::string& err, JsonParse strategy) {
    std::vector<Json> result;
    try {
        auto json_array = nlohmann::json::parse("[" + in + "]");
        for (const auto& element : json_array) {
            result.emplace_back(element);
        }
        parser_stop_pos = in.size();
        err.clear();
    } catch (const nlohmann::json::parse_error& e) {
        err = e.what();
    }
    return result;
}

void Json::dump(std::string& out) const {
    out = m_json.dump();
}

bool Json::has_shape(const shape& types, std::string& err) const {
    if (!is_object()) {
        err = "expected JSON object, got " + dump();
        return false;
    }

    for (const auto& item : types) {
        const auto it = m_json.find(item.first);
        if (it == m_json.end() || Json(*it).type() != item.second) {
            err = "bad type for " + item.first + " in " + dump();
            return false;
        }
    }
    return true;
}

} // namespace json
