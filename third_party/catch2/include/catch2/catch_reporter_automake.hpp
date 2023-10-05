/*
 *  Created by Justin R. Wilson on 2/19/2017.
 *  Copyright 2017 Justin R. Wilson. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#ifndef TWOBLUECUBES_CATCH_REPORTER_AUTOMAKE_HPP_INCLUDED
#define TWOBLUECUBES_CATCH_REPORTER_AUTOMAKE_HPP_INCLUDED

// Don't #include any Catch headers here - we can assume they are already
// included before this header.
// This is not good practice in general but is necessary in this case so this
// file can be distributed as a single header that works with the main
// Catch single header.

namespace Catch {

    struct AutomakeReporter : StreamingReporterBase<AutomakeReporter> {
        AutomakeReporter( ReporterConfig const& _config )
          :   StreamingReporterBase( _config )
        {}

        ~AutomakeReporter() override;

        static std::string getDescription() {
            return "Reports test results in the format of Automake .trs files";
        }

        void assertionStarting( AssertionInfo const& ) override {}

        bool assertionEnded( AssertionStats const& /*_assertionStats*/ ) override { return true; }

        void testCaseEnded( TestCaseStats const& _testCaseStats ) override {
            // Possible values to emit are PASS, XFAIL, SKIP, FAIL, XPASS and ERROR.
            stream << ":test-result: ";
            if (_testCaseStats.totals.assertions.allPassed()) {
                stream << "PASS";
            } else if (_testCaseStats.totals.assertions.allOk()) {
                stream << "XFAIL";
            } else {
                stream << "FAIL";
            }
            stream << ' ' << _testCaseStats.testInfo.name << '\n';
            StreamingReporterBase::testCaseEnded( _testCaseStats );
        }

        void skipTest( TestCaseInfo const& testInfo ) override {
            stream << ":test-result: SKIP " << testInfo.name << '\n';
        }

    };

#ifdef CATCH_IMPL
    AutomakeReporter::~AutomakeReporter() {}
#endif

    CATCH_REGISTER_REPORTER( "automake", AutomakeReporter)

} // end namespace Catch

#endif // TWOBLUECUBES_CATCH_REPORTER_AUTOMAKE_HPP_INCLUDED
