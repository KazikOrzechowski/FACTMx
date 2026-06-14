.PHONY: docs clean-docs typecheck

docs:
	python -m sphinx -b html docs docs/_build/html

clean-docs:
	rm -rf docs/_build

typecheck:
	python -m mypy FACTMx
