# HARAYA 2.0

Rule-guided contextual classification using multilingual BERT.

HARAYA 2.0 keeps **mBERT as the contextual language-understanding component**. A **rule-based legal scoring engine** then checks whether the conversation satisfies the operational elements of each gender-based online harassment category mapped to the Safe Spaces Act (RA 11313) and its IRR.

The system does **not** ask “does this message contain a threatening word?” It asks whether the **entire conversation** supplies the legal indicators required for a category.

This is a research prototype, not a legal finding. Indicator weights in `backend/data/legal_rules.json` are drafts and should be finalized with a legal expert.

## Workflow

User input (text or screenshot) → preprocessing / Taglish, slang, emoji, and Gay Lingo normalization → mBERT context analysis → legal indicator detection → weighted legal scoring → prediction, confidence, and legal explanation.

## Setup

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy ..\.env.example ..\.env
uvicorn app.main:app --reload --port 8000
```

Check `http://localhost:8000/health`. The first mBERT download (`google-bert/bert-base-multilingual-cased`) can take several minutes. Set `HARAYA_REQUIRE_MBERT=true` if the API must refuse to start without the encoder.

Optional: point `HARAYA_CLASSIFIER_ID` at a HARAYA 1.0 fine-tuned checkpoint. That classifier is **contextual evidence only**; it does not make the legal decision.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

### Image-to-text

Set `GEMINI_API_KEY` in `.env`. Screenshots are transcribed only; they are not classified by Gemini.

### Fine-tune and evaluate with a labeled dataset

The web app analyzes one conversation at a time. To fine-tune the optional contextual classifier, prepare a UTF-8 CSV with one row per conversation. The current HARAYA dataset schema is:

```text
conversation_id,conversation,normalized_text,language,gender_target,sexual_content,physical_harm,intimidation,repeated_contact,after_refusal,monitoring_pursuit,misogynistic,sexist,homophobic,transphobic,actual_disclosure,threatened_disclosure,final_label,source_type
```

`conversation_id`, `conversation`, and `final_label` are the required training fields: the model learns to classify the full conversation into `final_label`. `normalized_text` and `language` are descriptive columns; preprocessing is performed from `conversation`. The indicator columns (`gender_target` through `threatened_disclosure`) are annotation/scoring fields retained in the dataset, but the current classifier does **not** train on or use them for its prediction. They are useful for indicator-level analysis and can be used later if HARAYA adds a separate indicator-scoring training objective. `source_type` records dataset provenance. The evaluation metrics (accuracy, precision, recall, and F1) are calculated by comparing held-out predictions with `final_label`; they are not read from the indicator columns.

For example:

```csv
conversation_id,conversation,normalized_text,language,gender_target,sexual_content,physical_harm,intimidation,repeated_contact,after_refusal,monitoring_pursuit,misogynistic,sexist,homophobic,transphobic,actual_disclosure,threatened_disclosure,final_label,source_type
chat-001,"A: Please stop messaging me. B: I will keep contacting you.",a please stop messaging me b i will keep contacting you,English,0,0,0,1,3,3,0,0,0,0,0,0,0,Incessant Messaging,expert_reviewed
chat-002,"A: Good luck on your exam! B: Thanks!",a good luck on your exam b thanks,English,0,0,0,0,0,0,0,0,0,0,0,0,0,Non-Harassment,expert_reviewed
```

Start with `backend/data/dataset_template.csv`. Indicator annotations use numeric scores (the current dataset uses values from 0 to 3); retain the same annotation scale consistently across the dataset. Optional `data_origin` and `validation_status` provenance columns are also recorded in the evaluation report when present. Labels must match a category in `backend/data/legal_rules.json` (including `Non-Harassment`). The trainer also accepts the alternate columns `text` and `label`, or a message-per-row dataset with `conversation_id`, `speaker`, `text`, and `label` (plus optional `order`). Provide at least two distinct conversations for every label so each label can appear in both the training and held-out test sets. The split is stratified and keeps each conversation wholly in one partition.

From the `backend` directory, run:

```powershell
.\.venv\Scripts\python.exe -m app.train --dataset .\csv\your_dataset.csv --output-dir .\models\haraya-classifier
```

The command fine-tunes mBERT, prints accuracy and per-label precision/recall/F1 (including macro F1), and saves the model plus `evaluation.json`. The held-out metrics evaluate the classifier, not the overall rule-based legal decision. The trained classifier is contextual evidence only; the rule-based legal scoring engine still makes the final prediction. To use the saved classifier in the app, set `HARAYA_CLASSIFIER_ID=backend/models/haraya-classifier` in the root `.env` and restart the backend. Keep the evaluation set separate from any later model or rule tuning to avoid optimistic metrics.

### Tests

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest
```

## Architecture map

| Stage | Code |
| --- | --- |
| Text / image input | `frontend/src/App.tsx`, `backend/app/vision.py` |
| Preprocessing and normalization | `backend/app/preprocess.py`, `backend/data/lexicons.json` |
| mBERT context | `backend/app/mbert.py` |
| Legal indicator detection | `backend/app/indicators.py` |
| Rule-based legal scoring engine | `backend/app/scoring.py`, `backend/data/legal_rules.json` |
| Explainable output | `POST /analyze`, frontend detection panel |

## Design notes

- Analysis is **conversation-level**, so repeated contact after refusal can support Cyberstalking even when no single message is decisive.
- mBERT embeddings are compared to legal-indicator definitions (contextual similarity), in addition to conversation structure and cue evidence.
- Output includes prediction, confidence, checked legal indicators, rule score, and a short legal explanation.
