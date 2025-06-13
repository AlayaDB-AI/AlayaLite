# AlayaLite - Standalone App

## Development Setup

```bash
cd <path_to_your_project>/app
pip install -r requirements.txt
```

## Running Tests

```bash
cd <path_to_your_project>/app
pytest
```

## Running the Application

```bash
cd <path_to_your_project>
python -m uvicorn app.main:app --reload   # For development with hot reload
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000  # For production
```

