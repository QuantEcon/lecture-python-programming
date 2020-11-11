SHELL := bash
#
# Makefile for Sphinx Extension Test Cases
#

# You can set these variables from the command line.
SPHINXOPTS    = -c "./"
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = lecture-python-programming
SOURCEDIR     = source/rst
BUILDDIR      = _build
BUILDWEBSITE  = _build/website
BUILDCOVERAGE = _build/coverage
BUILDPDF      = _build/pdf
PORT          = 8890
FILES         = 
THEMEPATH     = theme/minimal
TEMPLATEPATH  = $(THEMEPATH)/templates

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(FILES) $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Install requiremenets for building lectures.
setup:
	pip install -r requirements.txt

preview:
ifeq (,$(filter $(target),website Website))
	cd $(BUILDWEBSITE)/jupyter_html && python -m http.server $(PORT)
else
ifdef lecture
	cd $(BUILDDIR)/jupyter/ && jupyter notebook --port $(PORT) --port-retries=0 $(basename $(lecture)).ipynb
else
	cd $(BUILDDIR)/jupyter/ && jupyter notebook --port $(PORT) --port-retries=0
endif
endif

clean-coverage:
	rm -rf $(BUILDCOVERAGE)

clean-website:
	rm -rf $(BUILDWEBSITE)

clean-pdf:
	rm -rf $(BUILDDIR)/jupyterpdf

coverage:
ifneq ($(strip $(parallel)),)
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDCOVERAGE)" $(FILES) $(SPHINXOPTS) $(O) -D tojupyter_make_coverage=1 -D tojupyter_execute_notebooks=1 -D tojupyter_ignore_skip_test=0 -D tojupyter_theme_path="$(THEMEPATH)" -D tojupyter_template_path="$(TEMPLATEPATH)" -D tojupyter_template_coverage_file_path="error_report_template.html" -D tojupyter_number_workers=$(parallel) 
else
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDCOVERAGE)" $(FILES) $(SPHINXOPTS) $(O) -D tojupyter_make_coverage=1 -D tojupyter_execute_notebooks=1 -D tojupyter_ignore_skip_test=0 -D tojupyter_theme_path="$(THEMEPATH)" -D tojupyter_template_path="$(TEMPLATEPATH)" -D tojupyter_template_coverage_file_path="error_report_template.html"
endif
 
website:
	echo "Theme: $(THEMEPATH)"
ifneq ($(strip $(parallel)),)
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDWEBSITE)" $(FILES) $(SPHINXOPTS) $(O) -D tojupyter_make_site=1 -D tojupyter_generate_html=1 -D tojupyter_download_nb=1 -D tojupyter_execute_notebooks=1 -D tojupyter_target_html=1 -D tojupyter_download_nb_image_urlpath="https://s3-ap-southeast-2.amazonaws.com/python-programming.quantecon.org/_static/" -D tojupyter_images_markdown=0 -D tojupyter_theme_path="$(THEMEPATH)" -D tojupyter_template_path="$(TEMPLATEPATH)" -D tojupyter_html_template="html.tpl" -D tojupyter_download_nb_urlpath="https://python-programming.quantecon.org/" -D tojupyter_coverage_dir=$(BUILDCOVERAGE) -D tojupyter_number_workers=$(parallel)

else
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDWEBSITE)" $(FILES) $(SPHINXOPTS) $(O) -D tojupyter_make_site=1 -D tojupyter_generate_html=1 -D tojupyter_download_nb=1 -D tojupyter_execute_notebooks=1 -D tojupyter_target_html=1 -D tojupyter_download_nb_image_urlpath="https://s3-ap-southeast-2.amazonaws.com/python-programming.quantecon.org/_static/" -D tojupyter_images_markdown=0 -D tojupyter_theme_path="$(THEMEPATH)" -D tojupyter_template_path="$(TEMPLATEPATH)" -D tojupyter_html_template="html.tpl" -D tojupyter_download_nb_urlpath="https://python-programming.quantecon.org/" -D tojupyter_coverage_dir=$(BUILDCOVERAGE)
endif

pdf:
ifneq ($(strip $(parallel)),)
	@$(SPHINXBUILD) -M jupyterpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(FILES) $(SPHINXOPTS) $(O) -D tojupyter_latex_template="latex.tpl" -D tojupyter_theme_path="$(THEMEPATH)" -D tojupyter_template_path="$(TEMPLATEPATH)" -D tojupyter_latex_template_book="latex_book.tpl" -D tojupyter_images_markdown=1 -D tojupyter_execute_notebooks=1 -D tojupyter_pdf_book=1 -D tojupyter_target_pdf=1 -D tojupyter_number_workers=$(parallel)

else
	@$(SPHINXBUILD) -M jupyterpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(FILES) $(SPHINXOPTS) $(O) -D tojupyter_theme_path="$(THEMEPATH)" -D tojupyter_template_path="$(TEMPLATEPATH)" -D tojupyter_latex_template="latex.tpl" -D tojupyter_latex_template_book="latex_book.tpl" -D tojupyter_images_markdown=1 -D tojupyter_execute_notebooks=1 -D tojupyter_pdf_book=1 -D tojupyter_target_pdf=1
endif

constructor-pdf:
ifneq ($(strip $(parallel)),)
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDPDF)" $(FILES) $(SPHINXOPTS) $(O) -D tojupyter_images_markdown=1 -D tojupyter_execute_notebooks=1 -D tojupyter_number_workers=$(parallel)

else
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDPDF)" $(FILES) $(SPHINXOPTS) $(O) -D tojupyter_images_markdown=1 -D tojupyter_execute_notebooks=1
endif

notebooks:
	make jupyter

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(FILES) $(SPHINXOPTS) $(O) -D tojupyter_allow_html_only=1
