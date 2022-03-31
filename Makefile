submissions: \
    pollos_petrel/example_python_submission.csv \
    pollos_petrel/example_r_submission.csv \
	pollos_petrel/mvb_submission.csv

pollos_petrel/example_python_submission.csv: setup_python src/example_submission.py
	@echo "Creating Python submission..."
	src/example_submission.py

pollos_petrel/example_r_submission.csv: setup_r src/example_submission.R
	@echo "Creating R submission..."
	src/example_submission.R

pollos_petrel/mvb_submission.csv: setup_python src/mvb_submission.py
	@echo "Creating MVB submission..."
	src/mvb_submission.py

module = pollos_petrel

define lint
	pylint \
        --disable=bad-continuation \
        --disable=missing-class-docstring \
        --disable=missing-function-docstring \
        --disable=missing-module-docstring \
        ${1}
endef

.PHONY: \
    check \
    clean \
    coverage \
    coverage_python \
    coverage_r \
    format \
    linter \
    mutants \
    mutants_python \
    mutants_r \
    setup \
    setup_python \
    setup_r \
    submissions \
    tests \
    tests_python \
    tests_r

check:
	black --check --line-length 100 ${module}
	black --check --line-length 100 src
	black --check --line-length 100 tests
	flake8 --max-line-length 100 ${module}
	flake8 --max-line-length 100 src
	flake8 --max-line-length 100 tests
	mypy ${module}
	mypy src
	mypy tests

clean:
	rm --force --recursive ${module}.egg-info
	rm --force --recursive ${module}/__pycache__
	rm --force --recursive .*_cache
	rm --force --recursive SeleccionAnalista2022.Rcheck
	rm --force --recursive tests/__pycache__
	rm --force --recursive tests/testthat/_snaps
	rm --force .mutmut-cache
	rm --force NAMESPACE
	rm --force SeleccionAnalista2022_*.tar.gz
	rm --force coverage.xml
	rm --force pollos_petrel/example_*_submission.csv

coverage: coverage_python

coverage_python: setup_python
	pytest --cov=${module} --cov-report=term-missing --verbose

coverage_r: setup_r
	Rscript tests/testthat/coverage.R

format:
	black --line-length 100 ${module}
	black --line-length 100 src
	black --line-length 100 tests

linter:
	$(call lint, ${module})
	$(call lint, tests)

mutants: mutants_python

mutants_python: setup_python tests_python
	mutmut run --paths-to-mutate ${module}
	mutmut run --paths-to-mutate src

mutants_r: setup_r tests_r
	@echo "🙁🏹 No mutation testing on R 👾🎉👾"

setup: setup_python

setup_python: clean
	pip install --editable .

setup_r: clean
	R -e "devtools::document()" && \
    R CMD build . && \
    R CMD check SeleccionAnalista2022_0.1.0.tar.gz && \
    R CMD INSTALL SeleccionAnalista2022_0.1.0.tar.gz

tests: tests_python

tests_python:
	pytest --verbose

tests_r:
	Rscript -e "devtools::test(stop_on_failure = TRUE)"
