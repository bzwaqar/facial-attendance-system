# AI Facial Recognition Attendance System

An experimental, resume-focused AI Engineering project implementing an end-to-end biometric attendance pipeline. The system combines modern deep facial representation learning (**InsightFace SCRFD + ArcFace**), sub-millisecond vector indexing (**FAISS**), multi-frame tracking with temporal verification, and a modular liveness/anti-spoofing layer backed by a **FastAPI** application, **PostgreSQL / SQLAlchemy** persistence, and an interactive frontend dashboard.

---

## The Problem

Traditional attendance collection—such as paper rosters, manual roll calls, and shared badge scanners—is vulnerable to buddy-punching, error-prone, and inefficient in large classrooms or corporate environments. Classical computer vision approaches (e.g., Eigenfaces, Haar cascades, or raw pixel classifiers like KNN) struggle with real-world illumination changes, subtle pose variations, and scale shifts.

This project addresses these challenges by developing a modern, modular biometric pipeline that enforces pre-flight image quality checks, performs real-time face tracking, applies multi-frame temporal confirmation, filters static presentation attacks via a liveness interface, and logs verified attendance with sub-second latency.

---

## Key Features

- **Pretrained Deep Feature Extraction**: 512-dimensional unit-normalized embeddings generated via ArcFace (ResNet-50 backbone), avoiding the sample inefficiency and computational expense of training deep metric learning architectures from scratch.
- **Robust Face Detection & Alignment**: Fast, scale-invariant face localization and 5-point facial landmark alignment via SCRFD.
- **Sub-Millisecond Vector Search**: Cosine similarity matching across enrolled student galleries using FAISS (`IndexFlatIP`).
- **Real-Time Video Multi-Object Tracking**: Frame-to-frame identity persistence using IoU and spatial tracking to maintain consistent track histories.
- **Temporal Verification Engine**: Multi-frame accumulation requiring consistent identity predictions across consecutive frames before confirming presence, eliminating single-frame false positives.
- **Modular Anti-Spoofing & Liveness Layer**: Pluggable interface (`BaseLivenessDetector`) combining high-frequency texture analysis (Laplacian / FFT spectrum), natural facial micro-dynamics, and interactive challenge-response verification to reject static photo presentation attacks.
- **Automated Biometric Quality Filtering**: Enforces face count, Laplacian sharpness ($\ge 50.0$), mean luminance ($[40.0, 220.0]$), and minimum resolution prior to embedding enrollment.
- **Relational & Vector Data Architecture**: ACID-compliant persistence using PostgreSQL and SQLAlchemy 2.0 with Alembic schema versioning.
- **RESTful Backend & Interactive Dashboard**: Asynchronous FastAPI endpoints for student registration, enrollment, recognition, session management, and reporting, accompanied by an interactive web dashboard.
- **Docker-Ready Architecture**: Multi-container setup with `docker-compose` managing PostgreSQL 16 and containerized FastAPI with native OpenCV shared libraries.

---

## System Architecture

The following flow diagram illustrates the lifecycle of a video frame from capture to database persistence:

```mermaid
flowchart TD
    A[Camera Feed / Video Stream] --> B[Face Detection: SCRFD]
    B --> C[Face Tracking: IoU Tracker]
    C --> D{Quality Check: Sharpness & Luminance}
    D -- Rejected --> X1[Discard Frame / Log Quality Failure]
    D -- Passed --> E{Liveness / Anti-Spoof Layer}
    E -- Spoof Detected --> X2[Reject Attendance / Flag Attack]
    E -- Live Confirmed --> F[ArcFace Feature Extraction: 512-d Embedding]
    F --> G[FAISS Vector Search: IndexFlatIP]
    G --> H{Similarity >= Threshold tau?}
    H -- No --> X3[Flag as UNKNOWN Person]
    H -- Yes --> I[Temporal Verification: N Consecutive Frames]
    I -- Pending --> X4[Accumulate Track History]
    I -- Confirmed --> J[Attendance Engine: Cooldown & Deduplication]
    J --> K[(PostgreSQL Database)]
```

---

## AI Pipeline & Biometric Foundations

### Why Use Pretrained ArcFace Instead of Training from Scratch?

Training a state-of-the-art face recognition network from scratch requires millions of identities (e.g., MS-Celeb-1M, Glint360k), multi-GPU clusters, and extensive compute. More importantly, training on a small classroom dataset inevitably causes catastrophic overfitting, where the network memorizes background artifacts instead of learning generalizable facial manifold geometry.

By leveraging **pretrained ArcFace weights** (trained with Additive Angular Margin Loss), the model functions as a generalized, zero-shot biometric feature extractor. It maps any aligned face image into a compact 512-dimensional hypersphere where intra-class distance is minimized and inter-class distance is maximized.

### Component Explanations

1. **SCRFD (Sample and Computation Redistribution for Efficient Face Detection)**:
   A lightweight, highly accurate face detector designed for edge and CPU inference. It outputs precise bounding boxes and 5 facial landmark keypoints (left eye, right eye, nose tip, left mouth corner, right mouth corner).
2. **Facial Alignment**:
   Using the 5 detected landmarks, an affine transformation maps the face into a canonical orientation and normalized crop size ($112 \times 112$), neutralizing 2D in-plane rotation.
3. **ArcFace Embeddings**:
   The aligned crop is passed through a ResNet backbone to produce a 512-dimensional feature vector $\mathbf{v} \in \mathbb{R}^{512}$.
4. **L2 Normalization & Cosine Similarity**:
   Each vector is normalized to unit length:
   $$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}, \quad \|\mathbf{v}_{\text{norm}}\|_2 = 1.0$$
   On unit-norm vectors, cosine similarity simplifies to the Euclidean inner product:
   $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{512} u_i v_i$$
5. **FAISS (`IndexFlatIP`)**:
   Facebook AI Similarity Search performs exhaustive inner-product search over the gallery of enrolled vectors in less than 1 millisecond on CPU, retrieving the top-$k$ closest identity matches.
6. **Face Tracking**:
   Associates face bounding boxes across successive video frames using spatial overlap (IoU) and motion continuity, assigning persistent track IDs to prevent repeated redundant face processing.
7. **Temporal Verification**:
   Rather than marking attendance on a single fleeting match, the temporal engine requires a candidate identity to be recognized in $N$ out of $M$ consecutive frames (e.g., 5 frames within a 1.5-second window) with consistent liveness scores. This mitigates transient misidentifications and occlusion anomalies.

---

## Student Enrollment Workflow

1. **Student Registration**: An administrator creates a student profile (Name, Student ID, Department, Semester, Section) via `POST /api/v1/students`.
2. **Multi-Angle Image Capture**: The student submits 5 to 10 portrait images reflecting slight natural variations in expression and head pose.
3. **Pre-flight Quality Audits**: Each image is evaluated for:
   - *Face Count*: Exactly one face must be present.
   - *Sharpness*: Laplacian variance must exceed $\ge 50.0$ to reject motion blur.
   - *Illumination*: Average luminance must reside within $[40.0, 220.0]$ to reject extreme shadows or glare.
   - *Bounding Box Size*: Face region must be at least $80 \times 80$ pixels.
4. **Embedding Extraction & L2 Normalization**: Each accepted image is passed through SCRFD and ArcFace to generate a 512-dimensional normalized embedding.
5. **Database Persistence & Index Sync**: The vectors are written to the `face_embeddings` table in PostgreSQL and immediately synced into the active FAISS index.

---

## Real-Time Attendance Workflow

1. **Session Creation**: An instructor starts an attendance session for a course/classroom via `POST /api/v1/attendance/sessions`, specifying an active verification threshold (default $\tau = 0.60$).
2. **Frame Ingestion & Tracking**: The video capture pipeline acquires frames from an RTSP stream, webcam, or video file, updating the spatial multi-face tracker.
3. **Liveness Evaluation**: Each active track is evaluated by the anti-spoofing engine for natural micro-movements, eye blinks, and high-frequency texture cues. Static print attacks are flagged and rejected.
4. **FAISS Vector Query**: Aligned faces are embedded and queried against the FAISS index.
5. **Threshold Verification**: If $\text{similarity} \ge \tau$, the candidate student ID is passed to the temporal verifier. If $\text{similarity} < \tau$, the face is labeled `UNKNOWN`.
6. **Temporal Confirmation & Cooldown**: Once confirmed across consecutive frames, the attendance service records the event (`PRESENT`) in PostgreSQL. A cooldown mechanism prevents duplicate log entries for the remainder of the session.

---

## PostgreSQL Database Architecture

The persistence layer is structured using PostgreSQL 16 with SQLAlchemy 2.0 ORM:

```
┌───────────────────────────────┐         ┌───────────────────────────────┐
│           students            │         │        face_embeddings        │
├───────────────────────────────┤         ├───────────────────────────────┤
│ id (UUID, PK)                 │1       *│ id (UUID, PK)                 │
│ student_id (VARCHAR, UNIQUE)  ├─────────┤ student_id (UUID, FK)         │
│ name (VARCHAR)                │         │ embedding (FLOAT[512])        │
│ department (VARCHAR)          │         │ source_image_path (VARCHAR)   │
│ semester (VARCHAR)            │         │ created_at (TIMESTAMP)        │
│ section (VARCHAR)             │         └───────────────────────────────┘
│ is_active (BOOLEAN)           │
│ created_at (TIMESTAMP)        │
└──────────────┬────────────────┘
               │ 1
               │
               │ *
┌──────────────┴────────────────┐         ┌───────────────────────────────┐
│          attendance           │*       1│      attendance_sessions      │
├───────────────────────────────┤         ├───────────────────────────────┤
│ id (UUID, PK)                 │         │ id (UUID, PK)                 │
│ student_id (UUID, FK)         │         │ session_id (VARCHAR, UNIQUE)  │
│ session_id (UUID, FK)         ├─────────┤ course_name (VARCHAR)         │
│ marked_at (TIMESTAMP)         │         │ department (VARCHAR)          │
│ confidence (FLOAT)            │         │ room_number (VARCHAR)         │
│ status (VARCHAR)              │         │ similarity_threshold (FLOAT)  │
│ is_live (BOOLEAN)             │         │ is_active (BOOLEAN)           │
│ liveness_score (FLOAT)        │         │ created_at (TIMESTAMP)        │
└───────────────────────────────┘         └───────────────────────────────┘
```

### Core Entities:
- **`Student`**: Holds primary student metadata, department, semester, and active status.
- **`FaceEmbedding`**: Stores individual 512-dimensional biometric feature vectors linked to a student (5–10 vectors per identity).
- **`AttendanceSession`**: Represents a scheduled lecture or event with configurable similarity thresholds and room identifiers.
- **`Attendance`**: Immutable ledger of attendance records containing verification timestamps, similarity confidence scores, and liveness verification telemetry.

---

## Project Structure

```text
face-detection-attendance-system/
├── docker-compose.yml              # Multi-container orchestration (PostgreSQL 16 + FastAPI)
├── README.md                       # Technical system documentation
├── backend/
│   ├── Dockerfile                  # Container definition with native vision shared libraries
│   ├── requirements.txt            # Python dependencies
│   ├── alembic.ini                 # Database migration configuration
│   ├── .env.example                # Environment configuration template
│   ├── alembic/                    # Alembic schema versioning scripts
│   │   ├── env.py
│   │   └── versions/
│   ├── app/
│   │   ├── main.py                 # FastAPI application factory, static mounts, error handlers
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── attendance.py   # Session lifecycle and attendance marking endpoints
│   │   │   │   ├── health.py       # Health check and diagnostic routes
│   │   │   │   ├── recognition.py  # Standalone face predict and base64 recognition endpoints
│   │   │   │   ├── reports.py      # Aggregated system overview and session analytics
│   │   │   │   └── students.py     # Student registration and multi-image enrollment
│   │   │   └── v1/api.py           # Centralized API v1 route aggregator
│   │   ├── camera/                 # Video stream capture, tracking, and temporal verification
│   │   │   ├── annotator.py        # Visual HUD overlays for liveness and identity status
│   │   │   ├── capture.py          # OpenCV VideoCapture wrapper
│   │   │   ├── pipeline.py         # Frame processing pipeline
│   │   │   ├── temporal.py         # Multi-frame temporal verifier
│   │   │   └── tracker.py          # Spatial IoU multi-face tracker
│   │   ├── core/
│   │   │   ├── config.py           # Pydantic Settings environment configuration
│   │   │   └── logging.py          # Structured console and rotating file logger
│   │   ├── db/
│   │   │   ├── base.py             # SQLAlchemy DeclarativeBase
│   │   │   ├── session.py          # Asynchronous and synchronous session factories
│   │   │   └── models/             # Declarative ORM models (Student, Attendance, etc.)
│   │   ├── evaluation/             # Biometric evaluation and benchmarking suite
│   │   │   └── benchmark.py        # Latency profiling, threshold sweeps, distributions
│   │   ├── liveness/               # Modular anti-spoofing layer
│   │   │   ├── base.py             # Abstract BaseLivenessDetector and result contracts
│   │   │   ├── cues.py             # Head pose proxies, eye blink deltas, texture analysis
│   │   │   ├── detector.py         # Heuristic, challenge-response, and passthrough detectors
│   │   │   ├── evaluator.py        # Structured JSONL liveness audit logging
│   │   │   └── factory.py          # Runtime detector factory resolver
│   │   ├── ml/                     # Deep learning biometric inference and vector search
│   │   │   ├── engine.py           # FaceEngine coordinating SCRFD and ArcFace
│   │   │   ├── loader.py           # Thread-safe singleton model loader (CPU/GPU)
│   │   │   ├── models.py           # Dataclasses for detected faces and landmarks
│   │   │   └── vector_search.py    # FAISS IndexFlatIP index wrapper
│   │   ├── schemas/                # Pydantic validation schemas (requests and responses)
│   │   ├── services/               # Core business logic (Enrollment, Attendance, Matching)
│   │   └── static/                 # Embedded Web Dashboard
│   │       ├── app.js              # Asynchronous frontend controller
│   │       ├── index.html          # Responsive dark-theme dashboard UI
│   │       └── style.css           # Vanilla CSS design system
│   ├── logs/                       # Application logs and evaluation benchmark JSON reports
│   ├── scripts/                    # Command-line runners and demonstration utilities
│   │   ├── demo_attendance_engine.py
│   │   ├── demo_enrollment_pipeline.py
│   │   ├── demo_face_pipeline.py
│   │   ├── demo_identity_matching.py
│   │   ├── demo_liveness_detection.py
│   │   ├── evaluate_system.py      # Automated benchmark harness runner
│   │   └── run_camera_recognition.py # Real-time camera attendance runner
│   └── tests/                      # Automated pytest suite (105 unit and integration tests)
```

---

## Local Setup Instructions

### Prerequisites
- Python 3.10 to 3.12
- PostgreSQL 14+ (or Docker)
- Git and C++ Build Tools (for native extensions)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/face-detection-attendance-system.git
cd face-detection-attendance-system
```

### 2. Configure Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 4. PostgreSQL Configuration via `.env`
Copy the environment template in `backend/`:
```bash
cp backend/.env.example backend/.env
```

Configure your local database credentials in `backend/.env`:
```env
# Application Settings
ENVIRONMENT="development"
DEBUG=true
LOG_LEVEL="INFO"
PORT=8000

# PostgreSQL Configuration
POSTGRES_SERVER="localhost"
POSTGRES_PORT=5432
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="your_postgres_password"
POSTGRES_DB="attendance_db"

# Explicit Database URL
DATABASE_URL="postgresql+asyncpg://postgres:your_postgres_password@localhost:5432/attendance_db"

# Biometric Model & Verification Settings
INSIGHTFACE_MODEL_NAME="buffalo_sc"
FACE_SIMILARITY_THRESHOLD=0.60
DETECTION_CONFIDENCE_THRESHOLD=0.70

# Anti-Spoofing Settings
LIVENESS_ENABLED=True
LIVENESS_METHOD="heuristic"
LIVENESS_MIN_SCORE=0.50
```

### 5. Run Database Migrations
Initialize the schema and tables in PostgreSQL:
```bash
cd backend
alembic upgrade head
```

### 6. Start the Backend & Dashboard
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Web Dashboard**: `http://localhost:8000/dashboard` (or `http://localhost:8000/`)
- **Interactive Swagger API Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

---

## Docker Deployment

To launch the complete multi-container stack (PostgreSQL 16 and FastAPI) with a single command:

```bash
docker compose up -d --build
```
The compose file configures internal networking, persistent Docker volumes for PostgreSQL data and logs, and a PostgreSQL readiness healthcheck before starting the application service.

---

## How to Enroll Students & Start Attendance

### Option A: Using the Interactive Web Dashboard
1. Open `http://localhost:8000/dashboard`.
2. Click **"+ New Session"** to create an active attendance session (e.g., `CS101-LAB`, threshold `0.60`).
3. Drag and drop or upload a face portrait image into the **Biometric Probe Tester**.
4. Click **"Run Biometric Verification"** to inspect detected face coordinates, cosine similarity progress, and liveness status.
5. Click **"Mark Session Attendance"** to record the student into the active session ledger in PostgreSQL.

### Option B: Using the Python Scripts & CLI

1. **Enroll Students via Demonstration Script**:
   ```bash
   cd backend
   python scripts/demo_enrollment_pipeline.py
   ```
2. **Run Real-Time Camera Attendance**:
   ```bash
   cd backend
   # Start camera stream linked to session "CS-2026-LEC1"
   python scripts/run_camera_recognition.py --session-id "CS-2026-LEC1" --camera 0
   ```
3. **Run Liveness & Attack Simulation**:
   ```bash
   cd backend
   python scripts/demo_liveness_detection.py
   ```

---

## Empirical Evaluation

An automated benchmark harness (`backend/scripts/evaluate_system.py`) was executed on the local host to profile biometric recognition accuracy, error rates, and inference latency.

The evaluation protocol **strictly separates the Gallery Enrollment Set from independent Test Probe Sets** (including genuine probe variations, cross-identity impostors, and unregistered unknown subjects).

### 1. Hardware Test Environment
- **Operating System**: Windows 11 (x86_64)
- **Execution Provider**: ONNX Runtime `CPUExecutionProvider` (Intel64)
- **Detection Model**: InsightFace SCRFD-500M (`buffalo_sc`)
- **Recognition Model**: ArcFace ResNet-50 (`buffalo_sc`, 512-d)
- **Vector Search Engine**: FAISS `IndexFlatIP` (CPU)

### 2. Measured Verification Metrics (Threshold $\tau = 0.60$)

| Evaluation Metric | Measured Result | Interpretation |
| :--- | :---: | :--- |
| **Recognition Accuracy** | **100.0%** | All genuine test probes correctly attributed to enrolled identity. |
| **False Acceptance Rate (FAR)** | **0.00%** | Zero impostor pairs incorrectly accepted above threshold. |
| **False Rejection Rate (FRR)** | **0.00%** | Zero genuine probes incorrectly rejected below threshold. |
| **Unknown-Person Rejection** | **100.0%** | 100% of unregistered probe faces rejected as `UNKNOWN`. |

### 3. Cosine Similarity Distributions

| Distribution Pair | Mean | Std Dev | Min | Median | Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Genuine Pairs** | **0.770** | 0.012 | 0.742 | 0.769 | **0.797** |
| **Impostor Pairs** | **0.107** | 0.023 | 0.064 | 0.102 | **0.170** |

$$\text{Separation Margin} = \min(\text{Genuine}) - \max(\text{Impostor}) = 0.742 - 0.170 = \mathbf{0.572}$$

### 4. Measured Latency & Processing Throughput (Host CPU)

| Pipeline Stage | Average Latency | Throughput (FPS) | % of Total Time |
| :--- | :---: | :---: | :---: |
| **SCRFD Face Detection** | **54.36 ms** | 18.4 FPS | 61.2% |
| **ArcFace Embedding Extraction** | **27.98 ms** | 35.7 FPS | 31.5% |
| **FAISS Vector Search (`IndexFlatIP`)** | **0.51 ms** | **1960.8 FPS** | 0.6% |
| **Anti-Spoofing Liveness Verification** | **6.00 ms** | 166.7 FPS | 6.7% |
| **Total End-to-End Pipeline** | **88.86 ms** | **11.3 FPS** | **100.0%** |

*(Note: On systems with dedicated NVIDIA GPUs using `CUDAExecutionProvider`, end-to-end latency typically measures under 15 ms / >65 FPS).*

### Custom Environment Benchmark Placeholder
To run and measure performance on your specific hardware deployment:
```bash
cd backend
python scripts/evaluate_system.py
```
Results will be output to console and persisted in `backend/logs/evaluation_report.json`.

---

## Limitations

As an experimental portfolio system, several real-world limitations apply:

1. **Extreme Lighting & Shadows**: Severe backlighting, heavy underexposure ($< 40$ mean luminance), or harsh directional glare can cause detection dropouts or degrade cosine similarity below the operating threshold.
2. **Extreme Head Pose**: While ArcFace handles moderate yaw/pitch ($\pm 30^\circ$), extreme profile angles ($> 60^\circ$) distort facial landmark geometry and degrade feature alignment.
3. **Camera Quality & Compression**: Low-resolution webcams ($< 720\text{p}$) or aggressive video compression introduce artifact noise that impairs Laplacian sharpness metrics.
4. **Threshold Calibration**: A static threshold $\tau = 0.60$ is calibrated for standard indoor lighting. Variable environmental deployments may require adaptive per-camera calibration.
5. **2D Heuristic Liveness Constraints**: The default anti-spoofing layer uses monocular 2D cues (Laplacian frequency, eye blinks, pose micro-motion). While effective against printed paper photos and basic screen replays, it does not provide defense against advanced presentation attacks such as realistic 3D silicone masks or deepfake synthetic injections. Hardware-based infrared (IR) or structured-light depth sensors are required for high-security physical access control.

---

## Future Improvements

- **Neural Anti-Spoofing Backbones**: Integrate lightweight convolutional PAD models (e.g., MiniFASNet or Silent-Face-Anti-Spoofing) into the `BaseLivenessDetector` interface.
- **GPU Batching & TensorRT**: Accelerate inference throughput via TensorRT execution providers and batch video frames for multi-camera streams.
- **Edge Deployment**: Optimize models to INT8 ONNX or OpenVINO for deployment on embedded devices (e.g., Raspberry Pi 5, NVIDIA Jetson Orin Nano).
- **Automated Re-Enrollment**: Automatically update student vector clusters with high-confidence recognition frames to account for aging and appearance changes over semesters.
- **Enterprise RBAC**: Implement OAuth2 / JWT authentication with role-based access control (Admin, Instructor, Student) on all API endpoints.

---

## Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.12 | Core backend and machine learning pipeline |
| **Computer Vision** | OpenCV (`opencv-python-headless`) | Image decoding, color transformations, affine landmark alignment |
| **Detection** | InsightFace SCRFD-500M | Fast, scale-invariant face and 5-point landmark detection |
| **Embeddings** | InsightFace ArcFace (ResNet-50) | 512-dimensional deep facial representations |
| **Vector Search** | FAISS (`faiss-cpu`) | High-speed inner product (cosine similarity) vector retrieval |
| **API Framework** | FastAPI | Asynchronous REST APIs with automatic OpenAPI / Swagger generation |
| **Web Server** | Uvicorn (ASGI) | Asynchronous event loop application server |
| **Database** | PostgreSQL 16 | Relational storage for students, sessions, embeddings, and attendance |
| **ORM & Migrations** | SQLAlchemy 2.0 & Alembic | Declarative async mapping and schema migrations |
| **Frontend** | Vanilla HTML5 / CSS3 / ES6+ | Real-time dark-theme dashboard UI with zero bundle build dependencies |
| **Containerization** | Docker & Docker Compose | Container runtime and multi-service orchestration |
| **Testing** | Pytest & pytest-asyncio | Automated unit, regression, and integration testing (105 tests) |

---

## Project Development & Evolution

This system originated as an exploratory classical computer vision experiment evaluating local feature descriptors (Haar Cascades, HOG, and raw pixel flattenings with K-Nearest Neighbors). While instructive, classical approaches proved fragile against realistic lighting variances, occlusions, and scale differences, and scaling to hundreds of students was constrained by linear search overhead.

The repository was systematically re-architected into an enterprise-grade AI Engineering system:
1. Replaced classical descriptors with **deep metric representations** (InsightFace ArcFace) producing robust 512-dimensional embeddings.
2. Replaced brute-force distance comparisons with **FAISS vector indexing**, reducing query time to sub-millisecond durations.
3. Introduced **spatial tracking** and a **multi-frame temporal verification engine** to eliminate single-frame false triggers.
4. Added a **modular anti-spoofing layer** to mitigate presentation attacks.
5. Replaced flat-file storage with **PostgreSQL** using SQLAlchemy 2.0 and Alembic.
6. Built clean **FastAPI REST endpoints** and an interactive **web dashboard**.

---

## 🔒 Responsible Use & Privacy Notice

> **Biometric Privacy Disclaimer**: Facial embeddings are sensitive biometric data. Deployments of this software should strictly comply with applicable privacy regulations (e.g., GDPR, FERPA, BIPA):
> - **Informed Consent**: Individuals should be informed and provide explicit consent prior to biometric enrollment and image capture.
> - **Data Minimization**: Raw facial photos should not be stored permanently if only numerical embeddings are required for matching.
> - **Access Control & Encryption**: Database instances storing biometric embeddings must be restricted via network security groups, firewalls, and encryption at rest.
> - **Retention Policies**: Clear retention and deletion schedules should be defined to permanently purge student embeddings upon graduation or course withdrawal.
> - **Non-Surveillance Context**: This project is engineered exclusively for authorized, scheduled classroom/session attendance verification and must not be used for non-consensual public surveillance.

---

## 🧪 Testing

Execute the automated test suite covering database models, ML inference, vector search, API routes, and evaluation harnesses:

```bash
cd backend
pytest tests -v
```
*(All 105 unit and integration tests pass).*

---

## 📄 License
This project is licensed under the MIT License.
