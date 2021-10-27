# Minimal makefile for Sphinx documentation
#

OPENPILOT_ROOT = `git rev-parse --show-toplevel`

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
DOCSDIR       = "$(OPENPILOT_ROOT)/docs"
SOURCEDIR     = "$(OPENPILOT_ROOT)/build/docs"
DOCSBUILDDIR  = "$(OPENPILOT_ROOT)/build/docs"
BUILDDIR      = "$(OPENPILOT_ROOT)/build"

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(DOCSBUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@echo "Cleaning build folder..."
	rm -rf "$(BUILDDIR)"

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@echo "Cleaning build folder..."
	rm -rf "$(BUILDDIR)"
	mkdir -p "$(DOCSBUILDDIR)"

	@echo "Copying docs & config to build folder..."
	cp -a "$(DOCSDIR)" "$(BUILDDIR)"
	cd "$(OPENPILOT_ROOT)" && \
		find . -type f \( -name "*.md" -o -name "*.rst" \) \
			-not -path "*/.*" \
			-not -path "./build/*" \
			-not -path "./docs/*" \
			-not -path "./xx/*" \
			-exec cp --parents "{}" ./build/docs/ \;

	@echo "Building rst files..."
	sphinx-apidoc -o "$(DOCSBUILDDIR)" ../ \
		../xx ../laika_repo ../rednose_repo ../pyextra ../notebooks ../panda_jungle \
		../third_party \
		../panda/examples \
		../scripts \
		../selfdrive/modeld \
		../selfdrive/debug \
		$(shell find .. -type d -name "*test*")

	@echo "Building html files..."
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(DOCSBUILDDIR)" $(SPHINXOPTS) $(O)
