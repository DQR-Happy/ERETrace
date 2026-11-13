PYTHON ?= python
.PHONY: test check tables

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v
check:
	$(PYTHON) scripts/check_repository.py
tables:
	$(PYTHON) scripts/summarize_paper.py
