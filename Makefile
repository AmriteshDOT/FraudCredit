# Makefile
run-step1:
	python -m src.step1_ingestion_sla.pipeline

test:
	pytest -v