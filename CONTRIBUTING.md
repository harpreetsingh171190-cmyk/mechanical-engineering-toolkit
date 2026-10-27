# Contributing

1. Create a virtual environment and run `python -m pip install -r requirements-dev.txt`.
2. Keep calculations in small, documented functions. State units, sign conventions and physical assumptions in the docstring.
3. Add tests for known analytical cases (textbook formulas) and for invalid inputs whenever you change calculation code.
4. Run `python -m pytest` before opening a pull request.
5. For notebook changes, install `requirements-notebooks.txt`, restart the kernel and run the notebook top to bottom before committing.
