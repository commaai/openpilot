/* json
 *
 * json is a tiny JSON library for C++11, providing JSON parsing and serialization.
 *
 * The core object provided by the library is json::Json. A Json object represents any JSON
 * value: null, bool, number (int or double), string (std::string), array (std::vector), or
 * object (std::map).
 *
 * Json objects act like values: they can be assigned, copied, moved, compared for equality or
 * order, etc. There are also helper methods Json::dump, to serialize a Json to a string, and
 * Json::parse (static) to parse a std::string as a Json object.
 *
 * Internally, the various types of Json object are represented by the JsonValue class
 * hierarchy.
 *
 * A note on numbers - JSON specifies the syntax of number formatting but not its semantics,
 * so some JSON implementations distinguish between integers and floating-point numbers, while
 * some don't. In json, we choose the latter. Because some JSON implementations (namely
 * Javascript itself) treat all numbers as the same type, distinguishing the two leads
 * to JSON that will be *silently* changed by a round-trip through those implementations.
 * Dangerous! To avoid that risk, json stores all numbers as double internally, but also
 * provides integer helpers.
 *
 * Fortunately, double-precision IEEE754 ('double') can precisely store any integer in the
 * range +/-2^53, which includes every 'int' on most systems. (Timestamps often use int64
 * or long long to avoid the Y2038K problem; a double storing microseconds since some epoch
 * will be exact for +/- 275 years.)
 */

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

#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <initializer_list>
#include "json.hpp"

namespace json {

enum JsonParse {
    STANDARD, COMMENTS
};

class JsonValue;

class Json final {
public:
    // Types
    enum Type {
        NUL, NUMBER, BOOL, STRING, ARRAY, OBJECT
    };

    typedef std::vector<Json> array;
    typedef std::map<std::string, Json> object;

    // Constructors
    Json() noexcept;
    Json(std::nullptr_t) noexcept;
    Json(double value);
    Json(int value);
    Json(bool value);
    Json(const std::string &value);
    Json(std::string &&value);
    Json(const char * value);
    Json(const array &values);
    Json(array &&values);
    Json(const object &values);
    Json(object &&values);

    // Implicit constructor for nlohmann::json
    Json(const nlohmann::json& j);

    // Accessors
    Type type() const;
    bool is_null() const { return type() == NUL; }
    bool is_number() const { return type() == NUMBER; }
    bool is_bool() const { return type() == BOOL; }
    bool is_string() const { return type() == STRING; }
    bool is_array() const { return type() == ARRAY; }
    bool is_object() const { return type() == OBJECT; }

    // Serialize to string. Return success.
    static Json parse(const std::string & in, std::string & err, JsonParse strategy = JsonParse::STANDARD);
    static std::vector<Json> parse_multi(const std::string & in, std::string::size_type & parser_stop_pos, std::string & err, JsonParse strategy = JsonParse::STANDARD);

    double number_value() const;
    int int_value() const;
    bool bool_value() const;
    const std::string & string_value() const;
    const array & array_items() const;
    const object & object_items() const;

    // Operators
    const Json & operator[](size_t i) const;
    const Json & operator[](const std::string &key) const;
    bool operator==(const Json &other) const;
    bool operator<(const Json &other) const;

    void dump(std::string &out) const;
    std::string dump() const {
        std::string out;
        dump(out);
        return out;
    }

    // Shape-checking
    typedef std::initializer_list<std::pair<std::string, Type>> shape;
    bool has_shape(const shape & types, std::string & err) const;

private:
    nlohmann::json m_json;
};
}
