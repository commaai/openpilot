/*
 *  Created by Daniel Garcia on 2018-12-04.
 *  Copyright Social Point SL. All rights reserved.
 *
 *  Distributed under the Boost Software License, Version 1.0. (See accompanying
 *  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#ifndef CATCH_REPORTER_SONARQUBE_HPP_INCLUDED
#define CATCH_REPORTER_SONARQUBE_HPP_INCLUDED


// Don't #include any Catch headers here - we can assume they are already
// included before this header.
// This is not good practice in general but is necessary in this case so this
// file can be distributed as a single header that works with the main
// Catch single header.

#include <map>

namespace Catch {

    struct SonarQubeReporter : CumulativeReporterBase<SonarQubeReporter> {

        SonarQubeReporter(ReporterConfig const& config)
        : CumulativeReporterBase(config)
        , xml(config.stream()) {
            m_reporterPrefs.shouldRedirectStdOut = true;
            m_reporterPrefs.shouldReportAllAssertions = true;
        }

        ~SonarQubeReporter() override;

        static std::string getDescription() {
            return "Reports test results in the Generic Test Data SonarQube XML format";
        }

        static std::set<Verbosity> getSupportedVerbosities() {
            return { Verbosity::Normal };
        }

        void noMatchingTestCases(std::string const& /*spec*/) override {}

        void testRunStarting(TestRunInfo const& testRunInfo) override {
            CumulativeReporterBase::testRunStarting(testRunInfo);
            xml.startElement("testExecutions");
            xml.writeAttribute("version", "1");
        }

        void testGroupEnded(TestGroupStats const& testGroupStats) override {
            CumulativeReporterBase::testGroupEnded(testGroupStats);
            writeGroup(*m_testGroups.back());
        }

        void testRunEndedCumulative() override {
            xml.endElement();
        }

        void writeGroup(TestGroupNode const& groupNode) {
            std::map<std::string, TestGroupNode::ChildNodes> testsPerFile;
            for(auto const& child : groupNode.children)
                testsPerFile[child->value.testInfo.lineInfo.file].push_back(child);

            for(auto const& kv : testsPerFile)
                writeTestFile(kv.first.c_str(), kv.second);
        }

        void writeTestFile(const char* filename, TestGroupNode::ChildNodes const& testCaseNodes) {
            XmlWriter::ScopedElement e = xml.scopedElement("file");
            xml.writeAttribute("path", filename);

            for(auto const& child : testCaseNodes)
                writeTestCase(*child);
        }

        void writeTestCase(TestCaseNode const& testCaseNode) {
            // All test cases have exactly one section - which represents the
            // test case itself. That section may have 0-n nested sections
            assert(testCaseNode.children.size() == 1);
            SectionNode const& rootSection = *testCaseNode.children.front();
            writeSection("", rootSection, testCaseNode.value.testInfo.okToFail());
        }

        void writeSection(std::string const& rootName, SectionNode const& sectionNode, bool okToFail) {
            std::string name = trim(sectionNode.stats.sectionInfo.name);
            if(!rootName.empty())
                name = rootName + '/' + name;

            if(!sectionNode.assertions.empty() || !sectionNode.stdOut.empty() || !sectionNode.stdErr.empty()) {
                XmlWriter::ScopedElement e = xml.scopedElement("testCase");
                xml.writeAttribute("name", name);
                xml.writeAttribute("duration", static_cast<long>(sectionNode.stats.durationInSeconds * 1000));

                writeAssertions(sectionNode, okToFail);
            }

            for(auto const& childNode : sectionNode.childSections)
                writeSection(name, *childNode, okToFail);
        }

        void writeAssertions(SectionNode const& sectionNode, bool okToFail) {
            for(auto const& assertion : sectionNode.assertions)
                writeAssertion( assertion, okToFail);
        }

        void writeAssertion(AssertionStats const& stats, bool okToFail) {
            AssertionResult const& result = stats.assertionResult;
            if(!result.isOk()) {
                std::string elementName;
                if(okToFail) {
                    elementName = "skipped";
                }
                else {
                    switch(result.getResultType()) {
                        case ResultWas::ThrewException:
                        case ResultWas::FatalErrorCondition:
                            elementName = "error";
                            break;
                        case ResultWas::ExplicitFailure:
                            elementName = "failure";
                            break;
                        case ResultWas::ExpressionFailed:
                            elementName = "failure";
                            break;
                        case ResultWas::DidntThrowException:
                            elementName = "failure";
                            break;

                            // We should never see these here:
                        case ResultWas::Info:
                        case ResultWas::Warning:
                        case ResultWas::Ok:
                        case ResultWas::Unknown:
                        case ResultWas::FailureBit:
                        case ResultWas::Exception:
                            elementName = "internalError";
                            break;
                    }
                }

                XmlWriter::ScopedElement e = xml.scopedElement(elementName);

                ReusableStringStream messageRss;
                messageRss << result.getTestMacroName() << "(" << result.getExpression() << ")";
                xml.writeAttribute("message", messageRss.str());

                ReusableStringStream textRss;
                if (stats.totals.assertions.total() > 0) {
                    textRss << "FAILED:\n";
                    if (result.hasExpression()) {
                        textRss << "\t" << result.getExpressionInMacro() << "\n";
                    }
                    if (result.hasExpandedExpression()) {
                        textRss << "with expansion:\n\t" << result.getExpandedExpression() << "\n";
                    }
                }

                if(!result.getMessage().empty())
                    textRss << result.getMessage() << "\n";

                for(auto const& msg : stats.infoMessages)
                    if(msg.type == ResultWas::Info)
                        textRss << msg.message << "\n";

                textRss << "at " << result.getSourceInfo();
                xml.writeText(textRss.str(), XmlFormatting::Newline);
            }
        }

    private:
        XmlWriter xml;
    };

#ifdef CATCH_IMPL
    SonarQubeReporter::~SonarQubeReporter() {}
#endif

    CATCH_REGISTER_REPORTER( "sonarqube", SonarQubeReporter )

} // end namespace Catch

#endif // CATCH_REPORTER_SONARQUBE_HPP_INCLUDED