/*
 * Copyright (C) 2006 The Android Open Source Project
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

#ifndef __CUTILS_CONFIG_UTILS_H
#define __CUTILS_CONFIG_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif
    
typedef struct cnode cnode;


struct cnode
{
    cnode *next;
    cnode *first_child;
    cnode *last_child;
    const char *name;
    const char *value;
};

/* parse a text string into a config node tree */
void config_load(cnode *root, char *data);

/* parse a file into a config node tree */
void config_load_file(cnode *root, const char *fn);

/* create a single config node */
cnode* config_node(const char *name, const char *value);

/* locate a named child of a config node */
cnode* config_find(cnode *root, const char *name);

/* look up a child by name and return the boolean value */
int config_bool(cnode *root, const char *name, int _default);

/* look up a child by name and return the string value */
const char* config_str(cnode *root, const char *name, const char *_default);

/* add a named child to a config node (or modify it if it already exists) */
void config_set(cnode *root, const char *name, const char *value);

/* free a config node tree */
void config_free(cnode *root);

#ifdef __cplusplus
}
#endif

#endif
