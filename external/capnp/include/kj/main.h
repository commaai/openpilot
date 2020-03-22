// Copyright (c) 2013-2014 Sandstorm Development Group, Inc. and contributors
// Licensed under the MIT License:
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef KJ_MAIN_H_
#define KJ_MAIN_H_

#if defined(__GNUC__) && !KJ_HEADER_WARNINGS
#pragma GCC system_header
#endif

#include "array.h"
#include "string.h"
#include "vector.h"
#include "function.h"

namespace kj {

class ProcessContext {
  // Context for command-line programs.

public:
  virtual StringPtr getProgramName() = 0;
  // Get argv[0] as passed to main().

  KJ_NORETURN(virtual void exit()) = 0;
  // Indicates program completion.  The program is considered successful unless `error()` was
  // called.  Typically this exits with _Exit(), meaning that the stack is not unwound, buffers
  // are not flushed, etc. -- it is the responsibility of the caller to flush any buffers that
  // matter.  However, an alternate context implementation e.g. for unit testing purposes could
  // choose to throw an exception instead.
  //
  // At first this approach may sound crazy.  Isn't it much better to shut down cleanly?  What if
  // you lose data?  However, it turns out that if you look at each common class of program, _Exit()
  // is almost always preferable.  Let's break it down:
  //
  // * Commands:  A typical program you might run from the command line is single-threaded and
  //   exits quickly and deterministically.  Commands often use buffered I/O and need to flush
  //   those buffers before exit.  However, most of the work performed by destructors is not
  //   flushing buffers, but rather freeing up memory, placing objects into freelists, and closing
  //   file descriptors.  All of this is irrelevant if the process is about to exit anyway, and
  //   for a command that runs quickly, time wasted freeing heap space may make a real difference
  //   in the overall runtime of a script.  Meanwhile, it is usually easy to determine exactly what
  //   resources need to be flushed before exit, and easy to tell if they are not being flushed
  //   (because the command fails to produce the expected output).  Therefore, it is reasonably
  //   easy for commands to explicitly ensure all output is flushed before exiting, and it is
  //   probably a good idea for them to do so anyway, because write failures should be detected
  //   and handled.  For commands, a good strategy is to allocate any objects that require clean
  //   destruction on the stack, and allow them to go out of scope before the command exits.
  //   Meanwhile, any resources which do not need to be cleaned up should be allocated as members
  //   of the command's main class, whose destructor normally will not be called.
  //
  // * Interactive apps:  Programs that interact with the user (whether they be graphical apps
  //   with windows or console-based apps like emacs) generally exit only when the user asks them
  //   to.  Such applications may store large data structures in memory which need to be synced
  //   to disk, such as documents or user preferences.  However, relying on stack unwind or global
  //   destructors as the mechanism for ensuring such syncing occurs is probably wrong.  First of
  //   all, it's 2013, and applications ought to be actively syncing changes to non-volatile
  //   storage the moment those changes are made.  Applications can crash at any time and a crash
  //   should never lose data that is more than half a second old.  Meanwhile, if a user actually
  //   does try to close an application while unsaved changes exist, the application UI should
  //   prompt the user to decide what to do.  Such a UI mechanism is obviously too high level to
  //   be implemented via destructors, so KJ's use of _Exit() shouldn't make a difference here.
  //
  // * Servers:  A good server is fault-tolerant, prepared for the possibility that at any time
  //   it could crash, the OS could decide to kill it off, or the machine it is running on could
  //   just die.  So, using _Exit() should be no problem.  In fact, servers generally never even
  //   call exit anyway; they are killed externally.
  //
  // * Batch jobs:  A long-running batch job is something between a command and a server.  It
  //   probably knows exactly what needs to be flushed before exiting, and it probably should be
  //   fault-tolerant.
  //
  // Meanwhile, regardless of program type, if you are adhering to KJ style, then the use of
  // _Exit() shouldn't be a problem anyway:
  //
  // * KJ style forbids global mutable state (singletons) in general and global constructors and
  //   destructors in particular.  Therefore, everything that could possibly need cleanup either
  //   lives on the stack or is transitively owned by something living on the stack.
  //
  // * Calling exit() simply means "Don't clean up anything older than this stack frame.".  If you
  //   have resources that require cleanup before exit, make sure they are owned by stack frames
  //   beyond the one that eventually calls exit().  To be as safe as possible, don't place any
  //   state in your program's main class, and don't call exit() yourself.  Then, runMainAndExit()
  //   will do it, and the only thing on the stack at that time will be your main class, which
  //   has no state anyway.
  //
  // TODO(someday):  Perhaps we should use the new std::quick_exit(), so that at_quick_exit() is
  //   available for those who really think they need it.  Unfortunately, it is not yet available
  //   on many platforms.

  virtual void warning(StringPtr message) = 0;
  // Print the given message to standard error.  A newline is printed after the message if it
  // doesn't already have one.

  virtual void error(StringPtr message) = 0;
  // Like `warning()`, but also sets a flag indicating that the process has failed, and that when
  // it eventually exits it should indicate an error status.

  KJ_NORETURN(virtual void exitError(StringPtr message)) = 0;
  // Equivalent to `error(message)` followed by `exit()`.

  KJ_NORETURN(virtual void exitInfo(StringPtr message)) = 0;
  // Displays the given non-error message to the user and then calls `exit()`.  This is used to
  // implement things like --help.

  virtual void increaseLoggingVerbosity() = 0;
  // Increase the level of detail produced by the debug logging system.  `MainBuilder` invokes
  // this if the caller uses the -v flag.

  // TODO(someday):  Add interfaces representing standard OS resources like the filesystem, so that
  //   these things can be mocked out.
};

class TopLevelProcessContext final: public ProcessContext {
  // A ProcessContext implementation appropriate for use at the actual entry point of a process
  // (as opposed to when you are trying to call a program's main function from within some other
  // program).  This implementation writes errors to stderr, and its `exit()` method actually
  // calls the C `quick_exit()` function.

public:
  explicit TopLevelProcessContext(StringPtr programName);

  struct CleanShutdownException { int exitCode; };
  // If the environment variable KJ_CLEAN_SHUTDOWN is set, then exit() will actually throw this
  // exception rather than exiting.  `kj::runMain()` catches this exception and returns normally.
  // This is useful primarily for testing purposes, to assist tools like memory leak checkers that
  // are easily confused by quick_exit().

  StringPtr getProgramName() override;
  KJ_NORETURN(void exit() override);
  void warning(StringPtr message) override;
  void error(StringPtr message) override;
  KJ_NORETURN(void exitError(StringPtr message) override);
  KJ_NORETURN(void exitInfo(StringPtr message) override);
  void increaseLoggingVerbosity() override;

private:
  StringPtr programName;
  bool cleanShutdown;
  bool hadErrors = false;
};

typedef Function<void(StringPtr programName, ArrayPtr<const StringPtr> params)> MainFunc;

int runMainAndExit(ProcessContext& context, MainFunc&& func, int argc, char* argv[]);
// Runs the given main function and then exits using the given context.  If an exception is thrown,
// this will catch it, report it via the context and exit with an error code.
//
// Normally this function does not return, because returning would probably lead to wasting time
// on cleanup when the process is just going to exit anyway.  However, to facilitate memory leak
// checkers and other tools that require a clean shutdown to do their job, if the environment
// variable KJ_CLEAN_SHUTDOWN is set, the function will in fact return an exit code, which should
// then be returned from main().
//
// Most users will use the KJ_MAIN() macro rather than call this function directly.

#define KJ_MAIN(MainClass) \
  int main(int argc, char* argv[]) { \
    ::kj::TopLevelProcessContext context(argv[0]); \
    MainClass mainObject(context); \
    return ::kj::runMainAndExit(context, mainObject.getMain(), argc, argv); \
  }
// Convenience macro for declaring a main function based on the given class.  The class must have
// a constructor that accepts a ProcessContext& and a method getMain() which returns
// kj::MainFunc (probably building it using a MainBuilder).

class MainBuilder {
  // Builds a main() function with nice argument parsing.  As options and arguments are parsed,
  // corresponding callbacks are called, so that you never have to write a massive switch()
  // statement to interpret arguments.  Additionally, this approach encourages you to write
  // main classes that have a reasonable API that can be used as an alternative to their
  // command-line interface.
  //
  // All StringPtrs passed to MainBuilder must remain valid until option parsing completes.  The
  // assumption is that these strings will all be literals, making this an easy requirement.  If
  // not, consider allocating them in an Arena.
  //
  // Some flags are automatically recognized by the main functions built by this class:
  //     --help:  Prints help text and exits.  The help text is constructed based on the
  //       information you provide to the builder as you define each flag.
  //     --verbose:  Increase logging verbosity.
  //     --version:  Print version information and exit.
  //
  // Example usage:
  //
  //     class FooMain {
  //     public:
  //       FooMain(kj::ProcessContext& context): context(context) {}
  //
  //       bool setAll() { all = true; return true; }
  //       // Enable the --all flag.
  //
  //       kj::MainBuilder::Validity setOutput(kj::StringPtr name) {
  //         // Set the output file.
  //
  //         if (name.endsWith(".foo")) {
  //           outputFile = name;
  //           return true;
  //         } else {
  //           return "Output file must have extension .foo.";
  //         }
  //       }
  //
  //       kj::MainBuilder::Validity processInput(kj::StringPtr name) {
  //         // Process an input file.
  //
  //         if (!exists(name)) {
  //           return kj::str(name, ": file not found");
  //         }
  //         // ... process the input file ...
  //         return true;
  //       }
  //
  //       kj::MainFunc getMain() {
  //         return MainBuilder(context, "Foo Builder v1.5", "Reads <source>s and builds a Foo.")
  //             .addOption({'a', "all"}, KJ_BIND_METHOD(*this, setAll),
  //                 "Frob all the widgets.  Otherwise, only some widgets are frobbed.")
  //             .addOptionWithArg({'o', "output"}, KJ_BIND_METHOD(*this, setOutput),
  //                 "<filename>", "Output to <filename>.  Must be a .foo file.")
  //             .expectOneOrMoreArgs("<source>", KJ_BIND_METHOD(*this, processInput))
  //             .build();
  //       }
  //
  //     private:
  //       bool all = false;
  //       kj::StringPtr outputFile;
  //       kj::ProcessContext& context;
  //     };

public:
  MainBuilder(ProcessContext& context, StringPtr version,
              StringPtr briefDescription, StringPtr extendedDescription = nullptr);
  ~MainBuilder() noexcept(false);

  class OptionName {
  public:
    OptionName() = default;
    inline OptionName(char shortName): isLong(false), shortName(shortName) {}
    inline OptionName(const char* longName): isLong(true), longName(longName) {}

  private:
    bool isLong;
    union {
      char shortName;
      const char* longName;
    };
    friend class MainBuilder;
  };

  class Validity {
  public:
    inline Validity(bool valid) {
      if (!valid) errorMessage = heapString("invalid argument");
    }
    inline Validity(const char* errorMessage)
        : errorMessage(heapString(errorMessage)) {}
    inline Validity(String&& errorMessage)
        : errorMessage(kj::mv(errorMessage)) {}

    inline const Maybe<String>& getError() const { return errorMessage; }
    inline Maybe<String> releaseError() { return kj::mv(errorMessage); }

  private:
    Maybe<String> errorMessage;
    friend class MainBuilder;
  };

  MainBuilder& addOption(std::initializer_list<OptionName> names, Function<Validity()> callback,
                         StringPtr helpText);
  // Defines a new option (flag).  `names` is a list of characters and strings that can be used to
  // specify the option on the command line.  Single-character names are used with "-" while string
  // names are used with "--".  `helpText` is a natural-language description of the flag.
  //
  // `callback` is called when the option is seen.  Its return value indicates whether the option
  // was accepted.  If not, further option processing stops, and error is written, and the process
  // exits.
  //
  // Example:
  //
  //     builder.addOption({'a', "all"}, KJ_BIND_METHOD(*this, showAll), "Show all files.");
  //
  // This option could be specified in the following ways:
  //
  //     -a
  //     --all
  //
  // Note that single-character option names can be combined into a single argument.  For example,
  // `-abcd` is equivalent to `-a -b -c -d`.
  //
  // The help text for this option would look like:
  //
  //     -a, --all
  //         Show all files.
  //
  // Note that help text is automatically word-wrapped.

  MainBuilder& addOptionWithArg(std::initializer_list<OptionName> names,
                                Function<Validity(StringPtr)> callback,
                                StringPtr argumentTitle, StringPtr helpText);
  // Like `addOption()`, but adds an option which accepts an argument.  `argumentTitle` is used in
  // the help text.  The argument text is passed to the callback.
  //
  // Example:
  //
  //     builder.addOptionWithArg({'o', "output"}, KJ_BIND_METHOD(*this, setOutput),
  //                              "<filename>", "Output to <filename>.");
  //
  // This option could be specified with an argument of "foo" in the following ways:
  //
  //     -ofoo
  //     -o foo
  //     --output=foo
  //     --output foo
  //
  // Note that single-character option names can be combined, but only the last option can have an
  // argument, since the characters after the option letter are interpreted as the argument.  E.g.
  // `-abofoo` would be equivalent to `-a -b -o foo`.
  //
  // The help text for this option would look like:
  //
  //     -o FILENAME, --output=FILENAME
  //         Output to FILENAME.

  MainBuilder& addSubCommand(StringPtr name, Function<MainFunc()> getSubParser,
                             StringPtr briefHelpText);
  // If exactly the given name is seen as an argument, invoke getSubParser() and then pass all
  // remaining arguments to the parser it returns.  This is useful for implementing commands which
  // have lots of sub-commands, like "git" (which has sub-commands "checkout", "branch", "pull",
  // etc.).
  //
  // `getSubParser` is only called if the command is seen.  This avoids building main functions
  // for commands that aren't used.
  //
  // `briefHelpText` should be brief enough to show immediately after the command name on a single
  // line.  It will not be wrapped.  Users can use the built-in "help" command to get extended
  // help on a particular command.

  MainBuilder& expectArg(StringPtr title, Function<Validity(StringPtr)> callback);
  MainBuilder& expectOptionalArg(StringPtr title, Function<Validity(StringPtr)> callback);
  MainBuilder& expectZeroOrMoreArgs(StringPtr title, Function<Validity(StringPtr)> callback);
  MainBuilder& expectOneOrMoreArgs(StringPtr title, Function<Validity(StringPtr)> callback);
  // Set callbacks to handle arguments.  `expectArg()` and `expectOptionalArg()` specify positional
  // arguments with special handling, while `expect{Zero,One}OrMoreArgs()` specifies a handler for
  // an argument list (the handler is called once for each argument in the list).  `title`
  // specifies how the argument should be represented in the usage text.
  //
  // All options callbacks are called before argument callbacks, regardless of their ordering on
  // the command line.  This matches GNU getopt's behavior of permuting non-flag arguments to the
  // end of the argument list.  Also matching getopt, the special option "--" indicates that the
  // rest of the command line is all arguments, not options, even if they start with '-'.
  //
  // The interpretation of positional arguments is fairly flexible.  The non-optional arguments can
  // be expected at the beginning, end, or in the middle.  If more arguments are specified than
  // the number of non-optional args, they are assigned to the optional argument handlers in the
  // order of registration.
  //
  // For example, say you called:
  //     builder.expectArg("<foo>", ...);
  //     builder.expectOptionalArg("<bar>", ...);
  //     builder.expectArg("<baz>", ...);
  //     builder.expectZeroOrMoreArgs("<qux>", ...);
  //     builder.expectArg("<corge>", ...);
  //
  // This command requires at least three arguments: foo, baz, and corge.  If four arguments are
  // given, the second is assigned to bar.  If five or more arguments are specified, then the
  // arguments between the third and last are assigned to qux.  Note that it never makes sense
  // to call `expect*OrMoreArgs()` more than once since only the first call would ever be used.
  //
  // In practice, you probably shouldn't create such complicated commands as in the above example.
  // But, this flexibility seems necessary to support commands where the first argument is special
  // as well as commands (like `cp`) where the last argument is special.

  MainBuilder& callAfterParsing(Function<Validity()> callback);
  // Call the given function after all arguments have been parsed.

  MainFunc build();
  // Build the "main" function, which simply parses the arguments.  Once this returns, the
  // `MainBuilder` is no longer valid.

private:
  struct Impl;
  Own<Impl> impl;

  class MainImpl;
};

}  // namespace kj

#endif  // KJ_MAIN_H_
