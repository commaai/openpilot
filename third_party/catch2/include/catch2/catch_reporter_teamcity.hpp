/*
 *  Created by Phil Nash on 19th December 2014
 *  Copyright 2014 Two Blue Cubes Ltd. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#ifndef TWOBLUECUBES_CATCH_REPORTER_TEAMCITY_HPP_INCLUDED
#define TWOBLUECUBES_CATCH_REPORTER_TEAMCITY_HPP_INCLUDED

// Don't #include any Catch headers here - we can assume they are already
// included before this header.
// This is not good practice in general but is necessary in this case so this
// file can be distributed as a single header that works with the main
// Catch single header.

#include <cstring>

#ifdef __clang__
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wpadded"
#endif

namespace Catch {

    struct TeamCityReporter : StreamingReporterBase<TeamCityReporter> {
        TeamCityReporter( ReporterConfig const& _config )
        :   StreamingReporterBase( _config )
        {
            m_reporterPrefs.shouldRedirectStdOut = true;
        }

        static std::string escape( std::string const& str ) {
            std::string escaped = str;
            replaceInPlace( escaped, "|", "||" );
            replaceInPlace( escaped, "'", "|'" );
            replaceInPlace( escaped, "\n", "|n" );
            replaceInPlace( escaped, "\r", "|r" );
            replaceInPlace( escaped, "[", "|[" );
            replaceInPlace( escaped, "]", "|]" );
            return escaped;
        }
        ~TeamCityReporter() override;

        static std::string getDescription() {
            return "Reports test results as TeamCity service messages";
        }

        void skipTest( TestCaseInfo const& /* testInfo */ ) override {
        }

        void noMatchingTestCases( std::string const& /* spec */ ) override {}

        void testGroupStarting( GroupInfo const& groupInfo ) override {
            StreamingReporterBase::testGroupStarting( groupInfo );
            stream << "##teamcity[testSuiteStarted name='"
                << escape( groupInfo.name ) << "']\n";
        }
        void testGroupEnded( TestGroupStats const& testGroupStats ) override {
            StreamingReporterBase::testGroupEnded( testGroupStats );
            stream << "##teamcity[testSuiteFinished name='"
                << escape( testGroupStats.groupInfo.name ) << "']\n";
        }


        void assertionStarting( AssertionInfo const& ) override {}

        bool assertionEnded( AssertionStats const& assertionStats ) override {
            AssertionResult const& result = assertionStats.assertionResult;
            if( !result.isOk() ) {

                ReusableStringStream msg;
                if( !m_headerPrintedForThisSection )
                    printSectionHeader( msg.get() );
                m_headerPrintedForThisSection = true;

                msg << result.getSourceInfo() << "\n";

                switch( result.getResultType() ) {
                    case ResultWas::ExpressionFailed:
                        msg << "expression failed";
                        break;
                    case ResultWas::ThrewException:
                        msg << "unexpected exception";
                        break;
                    case ResultWas::FatalErrorCondition:
                        msg << "fatal error condition";
                        break;
                    case ResultWas::DidntThrowException:
                        msg << "no exception was thrown where one was expected";
                        break;
                    case ResultWas::ExplicitFailure:
                        msg << "explicit failure";
                        break;

                    // We shouldn't get here because of the isOk() test
                    case ResultWas::Ok:
                    case ResultWas::Info:
                    case ResultWas::Warning:
                        CATCH_ERROR( "Internal error in TeamCity reporter" );
                    // These cases are here to prevent compiler warnings
                    case ResultWas::Unknown:
                    case ResultWas::FailureBit:
                    case ResultWas::Exception:
                        CATCH_ERROR( "Not implemented" );
                }
                if( assertionStats.infoMessages.size() == 1 )
                    msg << " with message:";
                if( assertionStats.infoMessages.size() > 1 )
                    msg << " with messages:";
                for( auto const& messageInfo : assertionStats.infoMessages )
                    msg << "\n  \"" << messageInfo.message << "\"";


                if( result.hasExpression() ) {
                    msg <<
                        "\n  " << result.getExpressionInMacro() << "\n"
                        "with expansion:\n" <<
                        "  " << result.getExpandedExpression() << "\n";
                }

                if( currentTestCaseInfo->okToFail() ) {
                    msg << "- failure ignore as test marked as 'ok to fail'\n";
                    stream << "##teamcity[testIgnored"
                           << " name='" << escape( currentTestCaseInfo->name )<< "'"
                           << " message='" << escape( msg.str() ) << "'"
                           << "]\n";
                }
                else {
                    stream << "##teamcity[testFailed"
                           << " name='" << escape( currentTestCaseInfo->name )<< "'"
                           << " message='" << escape( msg.str() ) << "'"
                           << "]\n";
                }
            }
            stream.flush();
            return true;
        }

        void sectionStarting( SectionInfo const& sectionInfo ) override {
            m_headerPrintedForThisSection = false;
            StreamingReporterBase::sectionStarting( sectionInfo );
        }

        void testCaseStarting( TestCaseInfo const& testInfo ) override {
            m_testTimer.start();
            StreamingReporterBase::testCaseStarting( testInfo );
            stream << "##teamcity[testStarted name='"
                << escape( testInfo.name ) << "']\n";
            stream.flush();
        }

        void testCaseEnded( TestCaseStats const& testCaseStats ) override {
            StreamingReporterBase::testCaseEnded( testCaseStats );
            if( !testCaseStats.stdOut.empty() )
                stream << "##teamcity[testStdOut name='"
                    << escape( testCaseStats.testInfo.name )
                    << "' out='" << escape( testCaseStats.stdOut ) << "']\n";
            if( !testCaseStats.stdErr.empty() )
                stream << "##teamcity[testStdErr name='"
                    << escape( testCaseStats.testInfo.name )
                    << "' out='" << escape( testCaseStats.stdErr ) << "']\n";
            stream << "##teamcity[testFinished name='"
                    << escape( testCaseStats.testInfo.name ) << "' duration='"
                    << m_testTimer.getElapsedMilliseconds() << "']\n";
            stream.flush();
        }

    private:
        void printSectionHeader( std::ostream& os ) {
            assert( !m_sectionStack.empty() );

            if( m_sectionStack.size() > 1 ) {
                os << getLineOfChars<'-'>() << "\n";

                std::vector<SectionInfo>::const_iterator
                it = m_sectionStack.begin()+1, // Skip first section (test case)
                itEnd = m_sectionStack.end();
                for( ; it != itEnd; ++it )
                    printHeaderString( os, it->name );
                os << getLineOfChars<'-'>() << "\n";
            }

            SourceLineInfo lineInfo = m_sectionStack.front().lineInfo;

            os << lineInfo << "\n";
            os << getLineOfChars<'.'>() << "\n\n";
        }

        // if string has a : in first line will set indent to follow it on
        // subsequent lines
        static void printHeaderString( std::ostream& os, std::string const& _string, std::size_t indent = 0 ) {
            std::size_t i = _string.find( ": " );
            if( i != std::string::npos )
                i+=2;
            else
                i = 0;
            os << Column( _string )
                           .indent( indent+i)
                           .initialIndent( indent ) << "\n";
        }
    private:
        bool m_headerPrintedForThisSection = false;
        Timer m_testTimer;
    };

#ifdef CATCH_IMPL
    TeamCityReporter::~TeamCityReporter() {}
#endif

    CATCH_REGISTER_REPORTER( "teamcity", TeamCityReporter )

} // end namespace Catch

#ifdef __clang__
#   pragma clang diagnostic pop
#endif

#endif // TWOBLUECUBES_CATCH_REPORTER_TEAMCITY_HPP_INCLUDED
