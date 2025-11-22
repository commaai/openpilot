Version 0.2.1 (2025-02-10)
========================
* Fix missing files making car/ package not importable

Version 0.2.0 (2025-02-10)
========================
* Moved car/ directory from openpilot to opendbc. It comprises the APIs necessary to communicate with 275+ car models
  * opendbc is moving towards being a complete self-contained car API package
  * Soon all opendbc-related tests from openpilot will be migrated as well

Version 0.1.0 (2024-08-01)
========================
* Initial pre-release package with can/ and dbc/ directories
