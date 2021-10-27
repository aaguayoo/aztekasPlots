PROJECT_PATH=${PWD}
SHELL_PROFILE=${SHELL_PROFILE_PATH}

init:
	@echo ""
	@echo "Installing dependencies in poetry environment..."
	@poetry install
	@echo "Poetry dependencies installed."
	@echo ""
	@echo "Installing pre-commit..."
	@poetry run pre-commit install
	@echo "Pre-commit installed. Pre-commit will run by default when running `git commit`."
	@echo ""
	@echo "Installing kernel..."
	@poetry run python -m ipykernel install --name Aztekasplot --prefix=/home/aaguayoo/.local
	@echo "Aztekasplot kernel installed."
	@echo ""
	@echo "Running shell..."
	@poetry shell

shell:
	@poetry shell

pre-commit:
	@cd ${PROJECT_PATH}/
	@git add .
	@poetry run pre-commit run

profile:
	@poetry run mprof run ${PROJECT_PATH}/profiling/aztekasplot_profiling.py > ${PROJECT_PATH}/profiling/memory_profiler.log && echo "Se creó el archivo profiling/memory_profiler.log" || echo "Error al correr memory-profiler.";
	@poetry run mprof plot -t "Recorded memory usage" -o ${PROJECT_PATH}/profiling/memory_profiler_plot.png && echo "Se creo la gráfica profiling/memory_profiler_plot.png" || echo "Error al correr memory-profiler."
	@poetry run mprof clean

test:
	@pytest ${PROJECT_PATH}/tests
