## API Usage

The project aims to expose its functionalities via a RESTful API (or gRPC).

**Current Status:**
- API controllers are defined in `api/controllers.py` (currently stubs).
- API data models (request/response schemas) will be defined in `api/models.py` using Pydantic.
- The API will be served using a framework like FastAPI or Flask.

For detailed API endpoint documentation, see [docs/modules/api_reference.md](./docs/modules/api_reference.md) (once implemented).

**Example (Conceptual):**
```bash
# Example: Train a model via API
# POST /api/v1/train
# Body: { "model_type": "XGBoost", "data_config_ref": "...", "feature_config_ref": "..." }
```
