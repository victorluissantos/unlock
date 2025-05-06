# 🧠 CAPTCHA Solver API (FastAPI + Docker)

A simple, containerized FastAPI service that uses a trained SVM model to solve 6‑character uppercase CAPTCHA images.

---

## 📁 Project Structure

```
unlock/
├── datasource/
│   └── zeta/
│       └── run.py          # decrypt logic for 'zeta' model
├── routes/
│   └── decripto.py         # API route definitions
├── schemas/
│   └── decripto.py         # Pydantic request schemas
├── services/
│   └── decripto_service.py # dynamic loader for run.py
├── resources/
│   └── tmp/                # saved uploads, renamed as <decoded>.png
├── postman_decripto.json   # Postman collection for /decripto
├── main.py                 # FastAPI app entrypoint
├── docker-compose.yml
└── IaC/
    ├── fastAPI/
    │   └── Dockerfile      # FastAPI service image
    ├── python/
    │   └── Dockerfile      # Python helper container image
    └── requirements.txt    # all Python dependencies
```

---

## 🚀 Quick Start

1. **Build & run** all services:

   ```bash
   docker compose up --build
   ```
2. **API docs**:
   Visit `http://localhost:8000/docs` for the interactive Swagger UI.
3. **MongoDB ping**:
   `GET http://localhost:8000/ping-db`

---

## 🔗 `/decripto` Endpoint

**POST** `/decripto`
Accepts JSON with one of:

* \`\` (Base64 payload)
* \`\` (link to Base64 string or PNG image)

| Field     | Type   | Description                                           |
| --------- | ------ | ----------------------------------------------------- |
| `modelo`  | string | Model folder name under `datasource/` (e.g. `"zeta"`) |
| `type`    | string | `"base64"` or `"image"`                               |
| `content` | string | Optional Base64 string (`data:image/png;base64,...`)  |
| `url`     | string | Optional URL returning Base64 or PNG image            |

### Behavior

* If \`\` is provided, it is used directly.
* If \`\` is provided:

  * `type="base64"` → response body text treated as Base64.
  * `type="image"`  → response bytes converted to Base64 automatically.

**Success response**:

```json
{ "resultado": "ABCXYZ" }
```

---

## 📦 Postman Collection

Import the `postman_decripto.json` file in Postman:

1. File → Import
2. Select `postman_decripto.json`
3. Run the provided examples for each scenario.

---

## 🔧 Tech Stack

* **Python 3** & **FastAPI**
* **Docker** & **docker-compose**
* **Machine Learning**: scikit‑learn, scikit‑image, joblib
* **CV**: opencv‑python‑headless, numpy

---

## 📚 References

* FastAPI: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
* Pydantic: [https://docs.pydantic.dev/](https://docs.pydantic.dev/)
* Docker Compose: [https://docs.docker.com/compose/](https://docs.docker.com/compose/)
* scikit‑image HOG: [https://scikit-image.org/](https://scikit-image.org/)
