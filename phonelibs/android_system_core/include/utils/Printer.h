/*
 * Copyright (C) 2013 The Android Open Source Project
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

#ifndef ANDROID_PRINTER_H
#define ANDROID_PRINTER_H

#include <android/log.h>

namespace android {

// Interface for printing to an arbitrary data stream
class Printer {
public:
    // Print a new line specified by 'string'. \n is appended automatically.
    // -- Assumes that the string has no new line in it.
    virtual void printLine(const char* string = "") = 0;

    // Print a new line specified by the format string. \n is appended automatically.
    // -- Assumes that the resulting string has no new line in it.
    virtual void printFormatLine(const char* format, ...) __attribute__((format (printf, 2, 3)));

protected:
    Printer();
    virtual ~Printer();
}; // class Printer

// Print to logcat
class LogPrinter : public Printer {
public:
    // Create a printer using the specified logcat and log priority
    // - Unless ignoreBlankLines is false, print blank lines to logcat
    // (Note that the default ALOG behavior is to ignore blank lines)
    LogPrinter(const char* logtag,
               android_LogPriority priority = ANDROID_LOG_DEBUG,
               const char* prefix = 0,
               bool ignoreBlankLines = false);

    // Print the specified line to logcat. No \n at the end is necessary.
    virtual void printLine(const char* string);

private:
    void printRaw(const char* string);

    const char* mLogTag;
    android_LogPriority mPriority;
    const char* mPrefix;
    bool mIgnoreBlankLines;
}; // class LogPrinter

// Print to a file descriptor
class FdPrinter : public Printer {
public:
    // Create a printer using the specified file descriptor.
    // - Each line will be prefixed with 'indent' number of blank spaces.
    // - In addition, each line will be prefixed with the 'prefix' string.
    FdPrinter(int fd, unsigned int indent = 0, const char* prefix = 0);

    // Print the specified line to the file descriptor. \n is appended automatically.
    virtual void printLine(const char* string);

private:
    enum {
        MAX_FORMAT_STRING = 20,
    };

    int mFd;
    unsigned int mIndent;
    const char* mPrefix;
    char mFormatString[MAX_FORMAT_STRING];
}; // class FdPrinter

class String8;

// Print to a String8
class String8Printer : public Printer {
public:
    // Create a printer using the specified String8 as the target.
    // - In addition, each line will be prefixed with the 'prefix' string.
    // - target's memory lifetime must be a superset of this String8Printer.
    String8Printer(String8* target, const char* prefix = 0);

    // Append the specified line to the String8. \n is appended automatically.
    virtual void printLine(const char* string);

private:
    String8* mTarget;
    const char* mPrefix;
}; // class String8Printer

// Print to an existing Printer by adding a prefix to each line
class PrefixPrinter : public Printer {
public:
    // Create a printer using the specified printer as the target.
    PrefixPrinter(Printer& printer, const char* prefix);

    // Print the line (prefixed with prefix) using the printer.
    virtual void printLine(const char* string);

private:
    Printer& mPrinter;
    const char* mPrefix;
};

}; // namespace android

#endif // ANDROID_PRINTER_H
