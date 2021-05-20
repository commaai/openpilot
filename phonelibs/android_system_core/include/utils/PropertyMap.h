/*
 * Copyright (C) 2010 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _UTILS_PROPERTY_MAP_H
#define _UTILS_PROPERTY_MAP_H

#include <utils/KeyedVector.h>
#include <utils/String8.h>
#include <utils/Errors.h>
#include <utils/Tokenizer.h>

namespace android {

/*
 * Provides a mechanism for passing around string-based property key / value pairs
 * and loading them from property files.
 *
 * The property files have the following simple structure:
 *
 * # Comment
 * key = value
 *
 * Keys and values are any sequence of printable ASCII characters.
 * The '=' separates the key from the value.
 * The key and value may not contain whitespace.
 *
 * The '\' character is reserved for escape sequences and is not currently supported.
 * The '"" character is reserved for quoting and is not currently supported.
 * Files that contain the '\' or '"' character will fail to parse.
 *
 * The file must not contain duplicate keys.
 *
 * TODO Support escape sequences and quoted values when needed.
 */
class PropertyMap {
public:
    /* Creates an empty property map. */
    PropertyMap();
    ~PropertyMap();

    /* Clears the property map. */
    void clear();

    /* Adds a property.
     * Replaces the property with the same key if it is already present.
     */
    void addProperty(const String8& key, const String8& value);

    /* Returns true if the property map contains the specified key. */
    bool hasProperty(const String8& key) const;

    /* Gets the value of a property and parses it.
     * Returns true and sets outValue if the key was found and its value was parsed successfully.
     * Otherwise returns false and does not modify outValue.  (Also logs a warning.)
     */
    bool tryGetProperty(const String8& key, String8& outValue) const;
    bool tryGetProperty(const String8& key, bool& outValue) const;
    bool tryGetProperty(const String8& key, int32_t& outValue) const;
    bool tryGetProperty(const String8& key, float& outValue) const;

    /* Adds all values from the specified property map. */
    void addAll(const PropertyMap* map);

    /* Gets the underlying property map. */
    inline const KeyedVector<String8, String8>& getProperties() const { return mProperties; }

    /* Loads a property map from a file. */
    static status_t load(const String8& filename, PropertyMap** outMap);

private:
    class Parser {
        PropertyMap* mMap;
        Tokenizer* mTokenizer;

    public:
        Parser(PropertyMap* map, Tokenizer* tokenizer);
        ~Parser();
        status_t parse();

    private:
        status_t parseType();
        status_t parseKey();
        status_t parseKeyProperty();
        status_t parseModifier(const String8& token, int32_t* outMetaState);
        status_t parseCharacterLiteral(char16_t* outCharacter);
    };

    KeyedVector<String8, String8> mProperties;
};

} // namespace android

#endif // _UTILS_PROPERTY_MAP_H
