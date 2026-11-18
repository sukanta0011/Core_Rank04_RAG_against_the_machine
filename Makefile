PYTHON = python3
PIP = pip
ENV = uv
MAIN = src
ENV_NAME = venv
DEBUGGER = pdb
ARGS = 
GENERATED = data/cache data/chunks data/processed data/output data/temp
PY_CACHE = $(shell find . \( -path "./.venv" -o -path "./data" \) -prune -o \
	\( -name "__pycache__" -o -name ".mypy_cache" -o -name ".pytest_cache" -o -name ".vscode" \) -print)
VENV_PATH = $(shell find . -type d -name ".$(ENV_NAME)" -print)

install:
	uv sync

run:
	@if [ ! -d ".$(ENV_NAME)" ]; then \
		make install;\
	fi
	$(ENV) run $(PYTHON) -m $(MAIN) $(ARGS)

debug:
	@if [ ! -d ".$(ENV_NAME)" ]; then \
		make install;\
	fi
#	pdb debugger controls: n(ext), c(ont(inue)), exit
	$(ENV) run $(PYTHON) -m $(DEBUGGER) -m $(MAIN) $(ARGS)

clean:
	@echo "Cleaning workspace..."
	@for cache in $(PY_CACHE); do \
		echo "Removing $$cache"; \
		rm -rf "$$cache"; \
	done
	@for dir in $(GENERATED); do \
		if [ -e "$$dir" ]; then \
			echo "Removing $$dir"; \
			rm -rf "$$dir"; \
		fi; \
	done
	@echo "Workspace cleaned"

fclean: clean
	@if [ -n "$(VENV_PATH)" ]; then \
		echo "Removing virtual environment..."; \
		rm -rf "$(VENV_PATH)";\
		echo "Virtual environment removed"; \
	else \
		echo "There is no virtual environment";\
	fi

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores \
			--ignore-missing-imports --disallow-untyped-defs \
			--check-untyped-defs

.PHONY: install run debug clean fclean lint lint-strict