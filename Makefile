PYTHON ?= python
.PHONY: pipeline generate train evaluate predict test dashboard api
pipeline:
	$(PYTHON) scripts/run_pipeline.py
generate:
	$(PYTHON) scripts/generate_data.py
train:
	$(PYTHON) scripts/train_models.py
evaluate:
	$(PYTHON) scripts/evaluate_models.py
predict:
	$(PYTHON) scripts/generate_predictions.py
test:
	$(PYTHON) -m pytest
dashboard:
	$(PYTHON) -m streamlit run app/streamlit_app.py
api:
	$(PYTHON) -m uvicorn app.api:app --reload
