/**
 * Biometric Attendance System - Frontend Controller
 * Interacts with FastAPI /api/v1 endpoints for sessions, recognition, students, and reports.
 */

const API_BASE = '/api/v1';

// State
let activeSessionId = null;
let currentProbeFile = null;
let latestRecognitionResult = null;

// DOM Elements
const sessionSelect = document.getElementById('sessionSelect');
const refreshBtn = document.getElementById('refreshBtn');
const openNewSessionModalBtn = document.getElementById('openNewSessionModalBtn');
const closeSessionModalBtn = document.getElementById('closeSessionModalBtn');
const cancelSessionModalBtn = document.getElementById('cancelSessionModalBtn');
const newSessionModal = document.getElementById('newSessionModal');
const createSessionForm = document.getElementById('createSessionForm');

// Enrollment Modal
const openRegisterStudentModalBtn = document.getElementById('openRegisterStudentModalBtn');
const registerStudentModal = document.getElementById('registerStudentModal');
const closeRegisterModalBtn = document.getElementById('closeRegisterModalBtn');
const cancelRegisterModalBtn = document.getElementById('cancelRegisterModalBtn');
const registerStudentForm = document.getElementById('registerStudentForm');
const enrollStep1 = document.getElementById('enrollStep1');
const enrollStep2 = document.getElementById('enrollStep2');
const stepIndicator1 = document.getElementById('stepIndicator1');
const stepIndicator2 = document.getElementById('stepIndicator2');
const stepConnector = document.querySelector('.step-connector');
const enrollDropZone = document.getElementById('enrollDropZone');
const enrollFileInput = document.getElementById('enrollFileInput');
const enrollDropPrompt = document.getElementById('enrollDropPrompt');
const enrollPreviewGrid = document.getElementById('enrollPreviewGrid');
const enrollPhotoCountText = document.getElementById('enrollPhotoCountText');
const enrollPhotoCountBadge = document.getElementById('enrollPhotoCountBadge');
const enrollSubmitBtn = document.getElementById('enrollSubmitBtn');
const enrollBackBtn = document.getElementById('enrollBackBtn');
const enrollProgress = document.getElementById('enrollProgress');
const enrollProgressFill = document.getElementById('enrollProgressFill');
const enrollProgressText = document.getElementById('enrollProgressText');
const enrollProgressPct = document.getElementById('enrollProgressPct');
const enrollStatusMsg = document.getElementById('enrollStatusMsg');

// Enrollment state
let enrollSelectedFiles = []; // Array of File objects
let pendingStudentId = null;  // student_id from step 1 after creation

// KPIs
const kpiTotalStudents = document.getElementById('kpiTotalStudents');
const kpiEnrolledCount = document.getElementById('kpiEnrolledCount');
const kpiSessionTitle = document.getElementById('kpiSessionTitle');
const kpiSessionMeta = document.getElementById('kpiSessionMeta');
const kpiPresentStudents = document.getElementById('kpiPresentStudents');
const kpiAttendanceRate = document.getElementById('kpiAttendanceRate');

// Probe / Dropzone
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const dropZonePrompt = document.getElementById('dropZonePrompt');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const clearImageBtn = document.getElementById('clearImageBtn');
const verifyBtn = document.getElementById('verifyBtn');
const markFromProbeBtn = document.getElementById('markFromProbeBtn');
const probeStatusBadge = document.getElementById('probeStatusBadge');

// Recognition Result Card
const recognitionResultCard = document.getElementById('recognitionResultCard');
const matchStatusPill = document.getElementById('matchStatusPill');
const resStudentId = document.getElementById('resStudentId');
const resStudentName = document.getElementById('resStudentName');
const resSimilarity = document.getElementById('resSimilarity');
const resLiveness = document.getElementById('resLiveness');
const resSimilarityPercent = document.getElementById('resSimilarityPercent');
const resSimilarityFill = document.getElementById('resSimilarityFill');
const alertBox = document.getElementById('alertBox');
const alertIcon = document.getElementById('alertIcon');
const alertText = document.getElementById('alertText');

// Tables
const attendanceTableBody = document.getElementById('attendanceTableBody');
const rosterCountBadge = document.getElementById('rosterCountBadge');
const studentsTableBody = document.getElementById('studentsTableBody');

// ==============================================================================
// Initialization
// ==============================================================================
document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  await loadOverview();
  await loadSessions();
  await loadStudents();
});

function setupEventListeners() {
  // Session switching
  sessionSelect.addEventListener('change', (e) => {
    activeSessionId = e.target.value;
    if (activeSessionId) {
      loadSessionDetails(activeSessionId);
    }
  });

  // Manual refresh
  refreshBtn.addEventListener('click', async () => {
    refreshBtn.classList.add('rotating');
    await loadOverview();
    if (activeSessionId) {
      await loadSessionDetails(activeSessionId);
    }
    await loadStudents();
    setTimeout(() => refreshBtn.classList.remove('rotating'), 600);
  });

  // Modal handlers
  openNewSessionModalBtn.addEventListener('click', () => {
    newSessionModal.classList.remove('hidden');
  });

  const closeModal = () => newSessionModal.classList.add('hidden');
  closeSessionModalBtn.addEventListener('click', closeModal);
  cancelSessionModalBtn.addEventListener('click', closeModal);

  createSessionForm.addEventListener('submit', handleCreateSession);

  // File Dropzone handlers
  dropZone.addEventListener('click', (e) => {
    if (e.target !== clearImageBtn) fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  clearImageBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetProbeState();
  });

  // Verification actions
  verifyBtn.addEventListener('click', runProbeVerification);
  markFromProbeBtn.addEventListener('click', markAttendanceFromProbe);

  // ── Enrollment Modal ────────────────────────────────────────────────────────
  openRegisterStudentModalBtn.addEventListener('click', () => {
    resetEnrollmentModal();
    registerStudentModal.classList.remove('hidden');
  });

  const closeEnrollModal = () => {
    registerStudentModal.classList.add('hidden');
    resetEnrollmentModal();
  };

  closeRegisterModalBtn.addEventListener('click', closeEnrollModal);
  cancelRegisterModalBtn.addEventListener('click', closeEnrollModal);

  // Close on backdrop click
  registerStudentModal.addEventListener('click', (e) => {
    if (e.target === registerStudentModal) closeEnrollModal();
  });
  newSessionModal.addEventListener('click', (e) => {
    if (e.target === newSessionModal) {
      newSessionModal.classList.add('hidden');
    }
  });

  // Step 1 form submit → go to step 2
  registerStudentForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const sid = document.getElementById('enrollStudentId').value.trim();
    const sname = document.getElementById('enrollName').value.trim();
    if (!sid || sid.length < 2) {
      alert('Student ID must be at least 2 characters long (e.g. STU-2026-001).');
      return;
    }
    if (!sname) {
      alert('Please enter student full name.');
      return;
    }
    goToEnrollStep2();
  });

  // Back button → return to step 1
  enrollBackBtn.addEventListener('click', () => {
    enrollStep2.classList.add('hidden');
    enrollStep1.classList.remove('hidden');
    stepIndicator1.classList.add('active');
    stepIndicator1.classList.remove('done');
    stepIndicator2.classList.remove('active', 'done');
    if (stepConnector) stepConnector.classList.remove('done');
    hideEnrollStatus();
  });

  // Enroll submit
  enrollSubmitBtn.addEventListener('click', handleEnrollSubmit);

  // Drop zone click
  enrollDropZone.addEventListener('click', () => enrollFileInput.click());

  enrollFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      addEnrollFiles(Array.from(e.target.files));
      enrollFileInput.value = ''; // reset so same files can be re-selected
    }
  });

  enrollDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    enrollDropZone.classList.add('dragover');
  });

  enrollDropZone.addEventListener('dragleave', () => {
    enrollDropZone.classList.remove('dragover');
  });

  enrollDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    enrollDropZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addEnrollFiles(Array.from(e.dataTransfer.files));
    }
  });
}

// ==============================================================================
// API Data Fetchers
// ==============================================================================
async function loadOverview() {
  try {
    const res = await fetch(`${API_BASE}/reports/overview`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    kpiTotalStudents.textContent = data.total_students;
    kpiEnrolledCount.textContent = `${data.enrolled_students} biometrically enrolled`;
  } catch (err) {
    console.error('Failed to load system overview:', err);
  }
}

async function loadSessions() {
  try {
    const res = await fetch(`${API_BASE}/attendance/sessions?limit=50`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const sessions = await res.json();

    sessionSelect.innerHTML = '';
    if (sessions.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No sessions available';
      sessionSelect.appendChild(opt);
      kpiSessionTitle.textContent = 'None Created';
      kpiSessionMeta.textContent = 'Click "+ New Session" to begin';
      return;
    }

    sessions.forEach((s) => {
      const opt = document.createElement('option');
      opt.value = s.session_id;
      opt.textContent = `${s.session_id} - ${s.course_name} (${s.is_active ? 'Active' : 'Closed'})`;
      sessionSelect.appendChild(opt);
    });

    // Select the first active session or first session
    const active = sessions.find((s) => s.is_active) || sessions[0];
    sessionSelect.value = active.session_id;
    activeSessionId = active.session_id;

    await loadSessionDetails(activeSessionId);
  } catch (err) {
    console.error('Failed to load sessions:', err);
  }
}

async function loadSessionDetails(sessionId) {
  if (!sessionId) return;
  try {
    const res = await fetch(`${API_BASE}/reports/sessions/${encodeURIComponent(sessionId)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    kpiSessionTitle.textContent = data.course_name || data.session_id;
    kpiSessionMeta.textContent = `ID: ${data.session_id} • Location: ${data.instructor_name || data.room_number || 'N/A'}`;

    const attendees = data.present_students || data.attendees || data.records || [];
    const presentCount = data.total_present ?? data.present_count ?? attendees.length;
    const rate = data.attendance_rate_percent ?? data.attendance_rate ?? 0;

    kpiPresentStudents.textContent = presentCount;
    kpiAttendanceRate.textContent = `${Number(rate).toFixed(1)}% of registered students`;

    renderAttendanceTable(attendees);
  } catch (err) {
    console.error(`Failed to load session ${sessionId}:`, err);
  }
}

async function loadStudents() {
  try {
    const res = await fetch(`${API_BASE}/students?limit=50`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const students = await res.json();

    if (students.length === 0) {
      studentsTableBody.innerHTML = `
        <tr><td colspan="6" class="table-empty">No students registered yet.</td></tr>
      `;
      return;
    }

    studentsTableBody.innerHTML = students.map((s) => `
      <tr>
        <td class="mono"><strong>${escapeHtml(s.student_id)}</strong></td>
        <td>${escapeHtml(s.name)}</td>
        <td>${escapeHtml(s.department || '-')}</td>
        <td>${escapeHtml(s.semester || '-')}/${escapeHtml(s.section || '-')}</td>
        <td>
          <span class="badge ${s.face_embeddings_count >= 5 ? 'badge-emerald' : 'badge-warning'}">
            ${s.face_embeddings_count} vectors
          </span>
        </td>
        <td>
          <span class="badge ${s.is_active ? 'badge-info' : 'badge-neutral'}">
            ${s.is_active ? 'Active' : 'Inactive'}
          </span>
        </td>
      </tr>
    `).join('');
  } catch (err) {
    console.error('Failed to load students:', err);
  }
}

// ==============================================================================
// Session Attendance Table Renderer
// ==============================================================================
function renderAttendanceTable(attendees) {
  rosterCountBadge.textContent = `${attendees.length} Verified`;

  if (!attendees || attendees.length === 0) {
    attendanceTableBody.innerHTML = `
      <tr>
        <td colspan="6" class="table-empty">
          <p>No students have marked attendance for this session yet.</p>
        </td>
      </tr>
    `;
    return;
  }

  attendanceTableBody.innerHTML = attendees.map((att) => {
    const rawDate = att.timestamp || att.marked_at;
    const dateStr = rawDate ? new Date(rawDate).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '--';
    const scoreVal = att.similarity_score != null ? att.similarity_score : att.confidence;
    const simScore = scoreVal != null ? Number(scoreVal).toFixed(4) : 'N/A';
    const livenessBadge = att.liveness_score != null && att.liveness_score >= 0.5
      ? `<span class="badge badge-emerald">Live (${(att.liveness_score * 100).toFixed(0)}%)</span>`
      : (att.is_live ? `<span class="badge badge-emerald">Live</span>` : `<span class="badge badge-warning">Unchecked</span>`);

    return `
      <tr>
        <td class="mono"><strong>${escapeHtml(att.student_id)}</strong></td>
        <td>${escapeHtml(att.student_name)}</td>
        <td class="mono">${dateStr}</td>
        <td class="mono">${simScore}</td>
        <td>${livenessBadge}</td>
        <td><span class="badge badge-emerald">${escapeHtml(att.status || 'Present')}</span></td>
      </tr>
    `;
  }).join('');
}

// ==============================================================================
// Image Probe & Biometric Recognition Flow
// ==============================================================================
function handleFileSelected(file) {
  currentProbeFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    dropZonePrompt.classList.add('hidden');
    previewContainer.classList.remove('hidden');
    verifyBtn.disabled = false;
    probeStatusBadge.textContent = 'Image Loaded';
    probeStatusBadge.className = 'badge badge-info';
    hideResultCard();
  };
  reader.readAsDataURL(file);
}

function resetProbeState() {
  currentProbeFile = null;
  fileInput.value = '';
  previewImage.src = '';
  previewContainer.classList.add('hidden');
  dropZonePrompt.classList.remove('hidden');
  verifyBtn.disabled = true;
  markFromProbeBtn.disabled = true;
  probeStatusBadge.textContent = 'Ready';
  probeStatusBadge.className = 'badge badge-neutral';
  hideResultCard();
}

function hideResultCard() {
  recognitionResultCard.classList.add('hidden');
  alertBox.classList.add('hidden');
  latestRecognitionResult = null;
}

async function runProbeVerification() {
  if (!currentProbeFile) return;

  verifyBtn.disabled = true;
  probeStatusBadge.textContent = 'Processing...';
  probeStatusBadge.className = 'badge badge-warning';

  const formData = new FormData();
  formData.append('file', currentProbeFile);

  try {
    const res = await fetch(`${API_BASE}/recognition/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Recognition API error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    latestRecognitionResult = data;
    displayRecognitionResult(data);
  } catch (err) {
    alert(`Recognition Failed: ${err.message}`);
    probeStatusBadge.textContent = 'Error';
    probeStatusBadge.className = 'badge badge-danger';
  } finally {
    verifyBtn.disabled = false;
  }
}

function displayRecognitionResult(data) {
  recognitionResultCard.classList.remove('hidden');

  if (!data.faces_detected || data.faces_detected === 0) {
    matchStatusPill.textContent = 'No Face Found';
    matchStatusPill.className = 'badge badge-danger';
    resStudentId.textContent = 'None';
    resStudentName.textContent = 'No face detected in image';
    resSimilarity.textContent = '0.0000';
    resLiveness.textContent = 'N/A';
    resSimilarityPercent.textContent = '0%';
    resSimilarityFill.style.width = '0%';
    showAlert('danger', 'Detection Failed', 'No valid human face was detected in the probe image.');
    markFromProbeBtn.disabled = true;
    probeStatusBadge.textContent = 'No Face';
    probeStatusBadge.className = 'badge badge-danger';
    return;
  }

  const match = data.matches[0];
  const sim = match.similarity;
  const simPercent = Math.max(0, Math.min(100, (sim * 100))).toFixed(1);

  resSimilarity.textContent = sim.toFixed(4);
  resSimilarityPercent.textContent = `${simPercent}%`;
  resSimilarityFill.style.width = `${simPercent}%`;

  // Check Liveness
  const isSpoof = match.liveness_status === 'SPOOF';
  const isLive = !isSpoof;
  const livenessScore = match.liveness_score != null ? (match.liveness_score * 100).toFixed(0) + '%' : 'Verified';
  resLiveness.textContent = isLive ? `Live (${livenessScore})` : `Spoof/Photo (${livenessScore})`;

  if (isSpoof) {
    matchStatusPill.textContent = 'Spoof Detected';
    matchStatusPill.className = 'badge badge-danger';
    resStudentId.textContent = match.student_id || 'Unknown';
    resStudentName.textContent = match.name || 'Unknown';
    showAlert('danger', 'Anti-Spoofing Alert', 'Presentation attack or 2D photo detected. Attendance rejected.');
    markFromProbeBtn.disabled = true;
    probeStatusBadge.textContent = 'Spoof';
    probeStatusBadge.className = 'badge badge-danger';
    return;
  }

  if (match.is_recognized) {
    matchStatusPill.textContent = 'Matched';
    matchStatusPill.className = 'badge badge-emerald';
    resStudentId.textContent = match.student_id;
    resStudentName.textContent = match.name;
    showAlert('success', 'Identity Verified', `Confidence ${sim.toFixed(4)} exceeds threshold &tau;=0.60.`);
    markFromProbeBtn.disabled = !activeSessionId;
    probeStatusBadge.textContent = 'Identified';
    probeStatusBadge.className = 'badge badge-emerald';
  } else {
    matchStatusPill.textContent = 'Unknown Face';
    matchStatusPill.className = 'badge badge-warning';
    resStudentId.textContent = 'UNKNOWN';
    resStudentName.textContent = 'Unregistered / Below Threshold';
    showAlert('warning', 'Unknown Person', `Face detected with cosine similarity ${sim.toFixed(4)} < 0.60.`);
    markFromProbeBtn.disabled = true;
    probeStatusBadge.textContent = 'Unknown';
    probeStatusBadge.className = 'badge badge-warning';
  }
}

function showAlert(type, title, message) {
  alertBox.className = `alert-box alert-${type}`;
  alertIcon.textContent = type === 'danger' ? '⛔' : type === 'warning' ? '⚠️' : '✅';
  alertText.innerHTML = `<strong>${title}:</strong> ${escapeHtml(message)}`;
  alertBox.classList.remove('hidden');
}

function formatApiError(errData, fallback = 'Operation failed') {
  if (!errData) return fallback;
  if (typeof errData === 'string') return errData;
  if (typeof errData.detail === 'string') return errData.detail;
  if (Array.isArray(errData.detail)) {
    return errData.detail.map(d => {
      const field = d.loc ? d.loc.filter(x => x !== 'body').join('.') : '';
      return `${field ? field + ': ' : ''}${d.msg || JSON.stringify(d)}`;
    }).join('\n');
  }
  if (typeof errData.detail === 'object') return JSON.stringify(errData.detail);
  if (errData.message) return errData.message;
  return JSON.stringify(errData);
}

async function markAttendanceFromProbe() {
  if (!activeSessionId || !latestRecognitionResult) return;

  const match = latestRecognitionResult.matches[0];
  if (!match || !match.student_id || !match.is_recognized) return;

  markFromProbeBtn.disabled = true;

  try {
    const payload = {
      session_id: activeSessionId,
      student_id: match.student_id,
      similarity_score: match.similarity,
      confidence: match.similarity,
      bounding_box: match.bounding_box || [0, 0, 100, 100],
      liveness_score: match.liveness_score ?? 1.0,
      is_live: match.is_live ?? true,
      observation_count: 5,
      is_confirmed: true,
      marked_by: 'web_dashboard',
    };

    const res = await fetch(`${API_BASE}/attendance/mark`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => null);
      throw new Error(formatApiError(errData, `HTTP ${res.status}`));
    }

    const data = await res.json();
    if (data.status === 'PRESENT') {
      showAlert('success', 'Attendance Recorded', `${match.name} marked PRESENT successfully!`);
    } else if (data.status === 'ALREADY_MARKED') {
      showAlert('warning', 'Already Logged', `${match.name} was already marked for this session.`);
    } else {
      showAlert('warning', data.status, data.message || 'Status processed.');
    }

    // Refresh roster
    await loadSessionDetails(activeSessionId);
    await loadOverview();
  } catch (err) {
    showAlert('danger', 'Mark Error', err.message);
  } finally {
    markFromProbeBtn.disabled = false;
  }
}

// ==============================================================================
// Create New Session
// ==============================================================================
async function handleCreateSession(e) {
  e.preventDefault();

  const sessionId = document.getElementById('newSessionId').value.trim();
  const courseName = document.getElementById('newCourseName').value.trim();
  const department = document.getElementById('newDepartment').value.trim() || null;
  const room = document.getElementById('newRoom').value.trim() || null;
  const threshold = parseFloat(document.getElementById('newThreshold').value) || 0.60;

  try {
    const payload = {
      session_id: sessionId,
      course_name: courseName,
      department: department,
      room_number: room,
      min_similarity_threshold: threshold,
      similarity_threshold: threshold,
    };

    const res = await fetch(`${API_BASE}/attendance/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => null);
      throw new Error(formatApiError(errData, `HTTP ${res.status}`));
    }

    // Reset form and close modal
    createSessionForm.reset();
    newSessionModal.classList.add('hidden');

    // Reload sessions and select new one
    await loadSessions();
    sessionSelect.value = sessionId;
    activeSessionId = sessionId;
    await loadSessionDetails(sessionId);
  } catch (err) {
    alert(`Failed to create session:\n${err.message}`);
  }
}

// ==============================================================================
// Utility Functions
// ==============================================================================
function escapeHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ==============================================================================
// Enrollment Modal Helpers
// ==============================================================================

function resetEnrollmentModal() {
  // Reset step to 1
  enrollStep1.classList.remove('hidden');
  enrollStep2.classList.add('hidden');
  stepIndicator1.classList.add('active');
  stepIndicator1.classList.remove('done');
  stepIndicator2.classList.remove('active', 'done');
  if (stepConnector) stepConnector.classList.remove('done');

  // Reset form
  registerStudentForm.reset();

  // Reset photo state
  enrollSelectedFiles = [];
  pendingStudentId = null;
  enrollPreviewGrid.innerHTML = '';
  enrollDropZone.classList.remove('has-files');
  enrollProgress.classList.add('hidden');
  enrollProgressFill.style.width = '0%';
  hideEnrollStatus();
  updateEnrollPhotoUI();
}

function goToEnrollStep2() {
  const sid = document.getElementById('enrollStudentId').value.trim();
  const sname = document.getElementById('enrollName').value.trim();
  const display = document.getElementById('enrollTargetStudentDisplay');
  if (display) {
    display.textContent = `${sname || 'Student'} (${sid || '--'})`;
  }

  // Transition step indicator
  stepIndicator1.classList.remove('active');
  stepIndicator1.classList.add('done');
  stepIndicator2.classList.add('active');
  if (stepConnector) stepConnector.classList.add('done');

  enrollStep1.classList.add('hidden');
  enrollStep2.classList.remove('hidden');
  hideEnrollStatus();
  updateEnrollPhotoUI();
}

function addEnrollFiles(newFiles) {
  const imageFiles = newFiles.filter(f => f.type.startsWith('image/'));
  const remaining = 10 - enrollSelectedFiles.length;
  const toAdd = imageFiles.slice(0, remaining);

  toAdd.forEach(file => {
    enrollSelectedFiles.push(file);
    renderEnrollThumb(file, enrollSelectedFiles.length - 1);
  });

  updateEnrollPhotoUI();

  if (imageFiles.length > remaining) {
    showEnrollStatus('warning', `Maximum 10 photos allowed. ${imageFiles.length - remaining} file(s) were skipped.`);
  }
}

function renderEnrollThumb(file, index) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const wrap = document.createElement('div');
    wrap.className = 'enroll-thumb-wrap';
    wrap.dataset.index = index;

    const img = document.createElement('img');
    img.src = e.target.result;
    img.alt = `Photo ${index + 1}`;

    const numLabel = document.createElement('span');
    numLabel.className = 'enroll-thumb-num';
    numLabel.textContent = `#${index + 1}`;

    const removeBtn = document.createElement('button');
    removeBtn.className = 'enroll-thumb-remove';
    removeBtn.innerHTML = '&times;';
    removeBtn.title = 'Remove photo';
    removeBtn.addEventListener('click', (ev) => {
      ev.stopPropagation();
      removeEnrollFile(parseInt(wrap.dataset.index));
    });

    wrap.appendChild(img);
    wrap.appendChild(numLabel);
    wrap.appendChild(removeBtn);
    enrollPreviewGrid.appendChild(wrap);
  };
  reader.readAsDataURL(file);
}

function removeEnrollFile(index) {
  enrollSelectedFiles.splice(index, 1);
  // Re-render all thumbnails
  enrollPreviewGrid.innerHTML = '';
  enrollSelectedFiles.forEach((file, i) => renderEnrollThumb(file, i));
  updateEnrollPhotoUI();
}

function updateEnrollPhotoUI() {
  const count = enrollSelectedFiles.length;
  const isValid = count >= 5 && count <= 10;

  enrollPhotoCountText.textContent = `${count} photo${count !== 1 ? 's' : ''} selected`;

  if (count === 0) {
    enrollPhotoCountBadge.textContent = 'Need 5–10';
    enrollPhotoCountBadge.className = 'badge badge-neutral';
    enrollDropZone.classList.remove('has-files');
  } else if (!isValid) {
    enrollPhotoCountBadge.textContent = `Need ${count < 5 ? 5 - count + ' more' : 'max 10'}`;
    enrollPhotoCountBadge.className = 'badge badge-warning';
    enrollDropZone.classList.add('has-files');
  } else {
    enrollPhotoCountBadge.textContent = `${count} ✓ Ready`;
    enrollPhotoCountBadge.className = 'badge badge-emerald';
    enrollDropZone.classList.add('has-files');
  }

  enrollSubmitBtn.disabled = !isValid;
  const submitText = document.getElementById('enrollSubmitBtnText');
  if (submitText) {
    if (isValid) {
      submitText.textContent = `Save & Add to Database (${count} Ready)`;
    } else {
      submitText.textContent = `Save & Add to Database (${count}/5 min)`;
    }
  }
}

function showEnrollStatus(type, message) {
  enrollStatusMsg.className = `enroll-status ${type}`;
  enrollStatusMsg.innerHTML = message;
  enrollStatusMsg.classList.remove('hidden');
}

function hideEnrollStatus() {
  enrollStatusMsg.classList.add('hidden');
  enrollStatusMsg.className = 'enroll-status hidden';
  enrollStatusMsg.innerHTML = '';
}

function setEnrollProgress(pct, label) {
  enrollProgressFill.style.width = `${pct}%`;
  enrollProgressText.textContent = label;
  enrollProgressPct.textContent = `${Math.round(pct)}%`;
}

async function handleEnrollSubmit() {
  const studentId = document.getElementById('enrollStudentId').value.trim();
  const name = document.getElementById('enrollName').value.trim();
  const department = document.getElementById('enrollDept').value.trim() || 'Computer Science';
  const semester = document.getElementById('enrollSemester').value.trim() || '1st';
  const section = document.getElementById('enrollSection').value.trim() || 'A';

  if (!studentId || !name) {
    showEnrollStatus('error', 'Student ID and Name are required. Please go back to Step 1 and fill them in.');
    return;
  }

  if (studentId.length < 2) {
    showEnrollStatus('error', 'Student ID must be at least 2 characters long (e.g. STU-001). Please go back to Step 1 and update it.');
    return;
  }

  if (enrollSelectedFiles.length < 5 || enrollSelectedFiles.length > 10) {
    showEnrollStatus('warning', `Please provide between 5 and 10 face photos (currently ${enrollSelectedFiles.length}).`);
    return;
  }

  enrollSubmitBtn.disabled = true;
  enrollBackBtn.disabled = true;
  hideEnrollStatus();
  enrollProgress.classList.remove('hidden');

  const step2Body = document.getElementById('modalStep2Body');
  if (step2Body) {
    step2Body.scrollTo({ top: step2Body.scrollHeight, behavior: 'smooth' });
  }

  try {
    // ── Phase 1: Create Student (30%) ──────────────────────────────────────────
    setEnrollProgress(10, 'Creating student record...');

    const studentPayload = { student_id: studentId, name, department, semester, section };
    const createRes = await fetch(`${API_BASE}/students`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(studentPayload),
    });

    if (!createRes.ok) {
      const errData = await createRes.json().catch(() => null);
      // 409 = already exists — still proceed to enrollment
      if (createRes.status !== 409) {
        throw new Error(formatApiError(errData, `Failed to create student (HTTP ${createRes.status})`));
      }
      showEnrollStatus('warning', `⚠️ Student <strong>${escapeHtml(studentId)}</strong> already exists — enrolling new face photos onto existing record.`);
    } else {
      setEnrollProgress(30, 'Student created ✓ — uploading face photos...');
    }

    pendingStudentId = studentId;

    // ── Phase 2: Enroll Face Photos (30%→95%) ─────────────────────────────────
    setEnrollProgress(40, `Uploading ${enrollSelectedFiles.length} photos to biometric engine...`);

    const formData = new FormData();
    enrollSelectedFiles.forEach((file, i) => {
      formData.append('files', file, file.name || `face_${i + 1}.jpg`);
    });

    // Animate progress from 40 → 85 while waiting
    let fakePct = 40;
    const fakeInterval = setInterval(() => {
      fakePct = Math.min(fakePct + 3, 85);
      setEnrollProgress(fakePct, 'Extracting ArcFace embeddings...');
    }, 300);

    const enrollRes = await fetch(`${API_BASE}/students/${encodeURIComponent(studentId)}/enroll?replace_existing=false`, {
      method: 'POST',
      body: formData,
    });

    clearInterval(fakeInterval);
    setEnrollProgress(95, 'Finalizing enrollment...');

    if (!enrollRes.ok) {
      const errData = await enrollRes.json().catch(() => null);
      const msg = errData?.detail?.message || formatApiError(errData, `Enrollment failed (HTTP ${enrollRes.status})`);
      throw new Error(msg);
    }

    const enrollData = await enrollRes.json();
    setEnrollProgress(100, 'Enrollment complete!');

    // ── Success ────────────────────────────────────────────────────────────────
    showEnrollStatus('success',
      `✅ <strong>${escapeHtml(name)}</strong> enrolled successfully!<br>` +
      `📸 ${enrollData.total_enrolled} of ${enrollData.total_submitted} photos accepted.`
    );

    // Mark step 2 as done
    stepIndicator2.classList.remove('active');
    stepIndicator2.classList.add('done');

    // Refresh student table and overview metrics
    await loadStudents();
    await loadOverview();
    if (activeSessionId) {
      await loadSessionDetails(activeSessionId);
    }

    // Auto-close after 2.5 seconds
    setTimeout(() => {
      registerStudentModal.classList.add('hidden');
      resetEnrollmentModal();
    }, 2500);

  } catch (err) {
    setEnrollProgress(0, 'Enrollment failed');
    enrollProgress.classList.add('hidden');
    showEnrollStatus('error', `❌ <strong>Error:</strong> ${escapeHtml(err.message)}`);
  } finally {
    enrollSubmitBtn.disabled = false;
    enrollBackBtn.disabled = false;
  }
}

// ==============================================================================
// Camera Mode – Probe Panel (Live Attendance)
// ==============================================================================
let liveAttendStream = null;
let liveAttendInterval = null;
let liveAttendMarkedCount = 0;
const liveAttendCooldowns = new Map();
const ATTEND_COOLDOWN_MS = 10000;

const probeTabUpload     = document.getElementById('probeTabUpload');
const probeTabLive       = document.getElementById('probeTabLive');
const probeModeUpload    = document.getElementById('probeModeUpload');
const probeModeLive      = document.getElementById('probeModeLive');
const liveAttendVideo    = document.getElementById('liveAttendVideo');
const liveAttendCanvas   = document.getElementById('liveAttendCanvas');
const liveAttendOverlay  = document.getElementById('liveAttendOverlay');
const liveAttendDot      = document.getElementById('liveAttendDot');
const liveAttendStatusText = document.getElementById('liveAttendStatusText');
const liveAttendResult   = document.getElementById('liveAttendResult');
const liveResultIcon     = document.getElementById('liveResultIcon');
const liveResultName     = document.getElementById('liveResultName');
const liveResultMeta     = document.getElementById('liveResultMeta');
const liveResultBadge    = document.getElementById('liveResultBadge');
const liveMarkedBadge    = document.getElementById('liveMarkedBadge');
const autoMarkToggle     = document.getElementById('autoMarkToggle');
const livenessCheckToggle = document.getElementById('livenessCheckToggle');
const startLiveAttendBtn = document.getElementById('startLiveAttendBtn');
const stopLiveAttendBtn  = document.getElementById('stopLiveAttendBtn');
// Name overlay refs (on video)
const liveNameOverlay = document.getElementById('liveNameOverlay');
const liveNameBadge   = document.getElementById('liveNameBadge');
const liveNameIcon    = document.getElementById('liveNameIcon');
const liveNameText    = document.getElementById('liveNameText');
const liveNameSub     = document.getElementById('liveNameSub');
const liveNameStatus  = document.getElementById('liveNameStatus');

probeTabUpload.addEventListener('click', () => {
  probeTabUpload.classList.add('active');
  probeTabLive.classList.remove('active');
  probeModeUpload.classList.remove('hidden');
  probeModeLive.classList.add('hidden');
  stopLiveAttendance();
});

probeTabLive.addEventListener('click', () => {
  probeTabLive.classList.add('active');
  probeTabUpload.classList.remove('active');
  probeModeLive.classList.remove('hidden');
  probeModeUpload.classList.add('hidden');
});

startLiveAttendBtn.addEventListener('click', startLiveAttendance);
stopLiveAttendBtn.addEventListener('click', stopLiveAttendance);

async function startLiveAttendance() {
  if (!activeSessionId) {
    alert('Please select an active session first from the navbar.');
    return;
  }
  try {
    liveAttendStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    liveAttendVideo.srcObject = liveAttendStream;
    startLiveAttendBtn.disabled = true;
    stopLiveAttendBtn.disabled = false;
    liveAttendOverlay.classList.add('active');
    liveAttendDot.className = 'cam-status-dot scanning';
    liveAttendStatusText.textContent = 'Scanning for faces...';
    liveAttendResult.classList.remove('hidden');
    liveResultName.textContent = 'Scanning...';
    liveResultMeta.textContent = '';
    liveResultBadge.textContent = '';
    liveResultIcon.textContent = '🔍';
    liveAttendInterval = setInterval(scanFrameForAttendance, 1500);
  } catch (err) {
    alert(`Camera access denied: ${err.message}`);
  }
}

function stopLiveAttendance() {
  clearInterval(liveAttendInterval);
  liveAttendInterval = null;
  if (liveAttendStream) {
    liveAttendStream.getTracks().forEach(t => t.stop());
    liveAttendStream = null;
  }
  liveAttendVideo.srcObject = null;
  startLiveAttendBtn.disabled = false;
  stopLiveAttendBtn.disabled = true;
  liveAttendOverlay.classList.remove('active');
  liveAttendDot.className = 'cam-status-dot';
  liveAttendStatusText.textContent = 'Camera off';
  liveAttendResult.classList.add('hidden');
}

async function scanFrameForAttendance() {
  if (!liveAttendStream) return;
  if (!liveAttendVideo.readyState || liveAttendVideo.readyState < 2) return;
  const vw = liveAttendVideo.videoWidth || 640;
  const vh = liveAttendVideo.videoHeight || 480;
  liveAttendCanvas.width = vw;
  liveAttendCanvas.height = vh;
  const ctx = liveAttendCanvas.getContext('2d');
  ctx.drawImage(liveAttendVideo, 0, 0, vw, vh);
  const base64 = liveAttendCanvas.toDataURL('image/jpeg', 0.8);
  try {
    const checkLiveness = livenessCheckToggle ? livenessCheckToggle.checked : true;
    const res = await fetch(`${API_BASE}/recognition/predict-base64`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: base64, check_liveness: checkLiveness }),
    });
    if (!res.ok) return;
    const data = await res.json();

    if (!data.faces || data.faces.length === 0) {
      liveAttendDot.className = 'cam-status-dot scanning';
      liveAttendStatusText.textContent = 'No face detected';
      liveResultIcon.textContent = '👤';
      liveResultName.textContent = 'No face in frame';
      liveResultMeta.textContent = '';
      liveResultBadge.textContent = '';
      liveAttendResult.className = 'cam-live-result';
      // Hide overlay on video
      liveNameOverlay.classList.add('hidden');
      return;
    }

    const face = data.faces[0];
    const isSpoof   = checkLiveness && (face.liveness_status === 'SPOOF');
    const isMatched = face.matched;
    const sim       = face.similarity_score ?? 0;
    const name      = face.student_name || face.student_id || 'Unknown';
    const simPct    = `${(sim * 100).toFixed(1)}%`;

    if (isSpoof) {
      liveAttendDot.className = 'cam-status-dot error';
      liveAttendStatusText.textContent = 'Spoof detected!';
      liveResultIcon.textContent = '⛔';
      liveResultName.textContent = 'Presentation Attack';
      liveResultMeta.textContent = face.liveness_reason || 'Anti-spoofing check failed';
      liveResultBadge.textContent = 'SPOOF';
      liveResultBadge.className = 'badge badge-danger';
      liveAttendResult.className = 'cam-live-result spoof';
      // Show overlay on video – spoof
      setLiveNameOverlay('⛔', 'Spoof Detected', 'Anti-spoofing failed', 'SPOOF', 'spoof');
      return;
    }

    if (!isMatched) {
      liveAttendDot.className = 'cam-status-dot error';
      liveAttendStatusText.textContent = 'Unknown face';
      liveResultIcon.textContent = '❓';
      liveResultName.textContent = 'Unknown Person';
      liveResultMeta.textContent = `Similarity: ${simPct} (below threshold)`;
      liveResultBadge.textContent = 'UNKNOWN';
      liveResultBadge.className = 'badge badge-warning';
      liveAttendResult.className = 'cam-live-result unknown';
      // Show overlay on video – unknown
      setLiveNameOverlay('❓', 'Unknown Person', `Sim: ${simPct} — below threshold`, 'UNKNOWN', 'unknown');
      return;
    }

    // ── Matched + Live ──
    liveAttendDot.className = 'cam-status-dot matched';
    liveAttendStatusText.textContent = `Identified: ${name}`;
    liveResultIcon.textContent = '✅';
    liveResultName.textContent = name;
    liveResultMeta.textContent = `${face.student_id} • Similarity: ${simPct}`;
    liveResultBadge.textContent = 'MATCHED';
    liveResultBadge.className = 'badge badge-emerald';
    liveAttendResult.className = 'cam-live-result matched';
    // Show overlay on video – matched
    setLiveNameOverlay('✅', name, `${face.student_id} • ${simPct}`, 'MATCHED', 'matched');

    if (autoMarkToggle.checked && face.student_id) {
      const lastMark = liveAttendCooldowns.get(face.student_id) || 0;
      if (Date.now() - lastMark > ATTEND_COOLDOWN_MS) {
        liveAttendCooldowns.set(face.student_id, Date.now());
        await autoMarkLiveAttendance(face);
      }
    }
  } catch (_) {}
}

function setLiveNameOverlay(icon, name, sub, statusLabel, stateClass) {
  liveNameIcon.textContent = icon;
  liveNameText.textContent = name;
  liveNameSub.textContent = sub;
  liveNameStatus.textContent = statusLabel;
  liveNameBadge.className = `cam-name-badge ${stateClass}`;
  liveNameOverlay.classList.remove('hidden');
  // Auto-hide after 4 seconds if nothing new comes in
  clearTimeout(liveNameOverlay._hideTimer);
  liveNameOverlay._hideTimer = setTimeout(() => {
    liveNameOverlay.classList.add('hidden');
  }, 4000);
}

async function autoMarkLiveAttendance(face) {
  try {
    const payload = {
      session_id: activeSessionId,
      student_id: face.student_id,
      similarity_score: face.similarity_score,
      confidence: face.similarity_score,
      bounding_box: face.bbox || [0, 0, 100, 100],
      liveness_score: face.liveness_score ?? 1.0,
      is_live: face.is_live ?? true,
      observation_count: 5,
      is_confirmed: true,
      marked_by: 'live_camera',
    };
    const res = await fetch(`${API_BASE}/attendance/mark`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      const d = await res.json();
      if (d.status === 'PRESENT') {
        liveAttendMarkedCount++;
        liveMarkedBadge.textContent = `${liveAttendMarkedCount} Marked`;
        liveMarkedBadge.className = 'badge badge-emerald';
        liveAttendStatusText.textContent = `✓ ${face.student_name || face.student_id} marked present!`;
        setLiveNameOverlay('✅', face.student_name || face.student_id, `${face.student_id} • Marked Present!`, 'PRESENT', 'matched');
        await loadSessionDetails(activeSessionId);
        await loadOverview();
      } else if (d.status === 'ALREADY_MARKED') {
        liveAttendStatusText.textContent = `ℹ ${face.student_name || face.student_id} already marked!`;
        setLiveNameOverlay('ℹ️', face.student_name || face.student_id, `${face.student_id} • Already Marked`, 'LOGGED', 'matched');
        await loadSessionDetails(activeSessionId);
      } else {
        console.warn('Attendance mark response:', d);
        liveAttendStatusText.textContent = `${d.status}: ${d.message || ''}`;
        liveAttendCooldowns.delete(face.student_id);
      }
    } else {
      const errData = await res.json().catch(() => null);
      console.error('Attendance mark HTTP error:', errData);
      liveAttendStatusText.textContent = `Error: ${errData?.detail || res.statusText}`;
      liveAttendCooldowns.delete(face.student_id);
    }
  } catch (err) {
    console.error('autoMarkLiveAttendance error:', err);
    liveAttendCooldowns.delete(face.student_id);
  }
}

// ==============================================================================
// Camera Mode – Enrollment Live Capture
// ==============================================================================
let enrollCamStream = null;
let enrollCamAutoInterval = null;
const ENROLL_TARGET = 10;

const enrollTabUpload        = document.getElementById('enrollTabUpload');
const enrollTabCamera        = document.getElementById('enrollTabCamera');
const enrollModeUpload       = document.getElementById('enrollModeUpload');
const enrollModeCamera       = document.getElementById('enrollModeCamera');
const enrollCamVideo         = document.getElementById('enrollCamVideo');
const enrollCamCanvas        = document.getElementById('enrollCamCanvas');
const enrollCamFlash         = document.getElementById('enrollCamFlash');
const enrollCamCount         = document.getElementById('enrollCamCount');
const enrollCamDots          = document.getElementById('enrollCamDots');
const enrollStartCamBtn      = document.getElementById('enrollStartCamBtn');
const enrollCaptureBtnManual = document.getElementById('enrollCaptureBtnManual');
const enrollStopCamBtn       = document.getElementById('enrollStopCamBtn');

enrollTabUpload.addEventListener('click', () => {
  enrollTabUpload.classList.add('active');
  enrollTabCamera.classList.remove('active');
  enrollModeUpload.classList.remove('hidden');
  enrollModeCamera.classList.add('hidden');
  stopEnrollCamera();
});

enrollTabCamera.addEventListener('click', () => {
  enrollTabCamera.classList.add('active');
  enrollTabUpload.classList.remove('active');
  enrollModeCamera.classList.remove('hidden');
  enrollModeUpload.classList.add('hidden');
  initEnrollDots();
});

enrollStartCamBtn.addEventListener('click', startEnrollCamera);
enrollStopCamBtn.addEventListener('click', stopEnrollCamera);
enrollCaptureBtnManual.addEventListener('click', () => captureEnrollFrame());

function initEnrollDots() {
  enrollCamDots.innerHTML = '';
  for (let i = 0; i < ENROLL_TARGET; i++) {
    const dot = document.createElement('div');
    dot.className = 'cam-dot';
    dot.id = `enrollDot${i}`;
    enrollCamDots.appendChild(dot);
  }
}

async function startEnrollCamera() {
  try {
    enrollCamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    enrollCamVideo.srcObject = enrollCamStream;
    enrollStartCamBtn.disabled = true;
    enrollCaptureBtnManual.disabled = false;
    enrollStopCamBtn.disabled = false;

    // Remove any previously camera-captured files
    enrollSelectedFiles = enrollSelectedFiles.filter(f => !f._fromCamera);
    initEnrollDots();
    updateEnrollCamCount();

    // Auto-capture every 1.2 seconds
    enrollCamAutoInterval = setInterval(() => {
      if (enrollSelectedFiles.length < ENROLL_TARGET) {
        captureEnrollFrame();
      } else {
        clearInterval(enrollCamAutoInterval);
        enrollCamAutoInterval = null;
        enrollCaptureBtnManual.disabled = true;
      }
    }, 1200);
  } catch (err) {
    alert(`Camera access denied: ${err.message}`);
  }
}

function stopEnrollCamera() {
  clearInterval(enrollCamAutoInterval);
  enrollCamAutoInterval = null;
  if (enrollCamStream) {
    enrollCamStream.getTracks().forEach(t => t.stop());
    enrollCamStream = null;
  }
  enrollCamVideo.srcObject = null;
  enrollStartCamBtn.disabled = false;
  enrollCaptureBtnManual.disabled = true;
  enrollStopCamBtn.disabled = true;
}

function captureEnrollFrame() {
  if (!enrollCamStream || enrollSelectedFiles.length >= ENROLL_TARGET) return;
  if (!enrollCamVideo.readyState || enrollCamVideo.readyState < 2) return;
  const vw = enrollCamVideo.videoWidth || 640;
  const vh = enrollCamVideo.videoHeight || 480;
  enrollCamCanvas.width = vw;
  enrollCamCanvas.height = vh;
  const ctx = enrollCamCanvas.getContext('2d');
  ctx.translate(vw, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(enrollCamVideo, 0, 0, vw, vh);

  // Flash
  enrollCamFlash.classList.add('flash-active');
  setTimeout(() => enrollCamFlash.classList.remove('flash-active'), 120);

  enrollCamCanvas.toBlob((blob) => {
    if (!blob) return;
    const idx = enrollSelectedFiles.length;
    const file = new File([blob], `cam_face_${idx + 1}.jpg`, { type: 'image/jpeg' });
    file._fromCamera = true;
    enrollSelectedFiles.push(file);
    renderEnrollThumb(file, idx);
    updateEnrollCamCount();
    updateEnrollPhotoUI();

    const dot = document.getElementById(`enrollDot${idx}`);
    if (dot) { dot.classList.remove('capturing'); dot.classList.add('filled'); }
    const nextDot = document.getElementById(`enrollDot${idx + 1}`);
    if (nextDot) nextDot.classList.add('capturing');

    if (enrollSelectedFiles.length >= ENROLL_TARGET) {
      stopEnrollCamera();
      showEnrollStatus('success', `✅ <strong>All ${ENROLL_TARGET} photos captured!</strong> Saving to database in 2 seconds (or click button below)...`);
      const step2Body = document.getElementById('modalStep2Body');
      if (step2Body) {
        step2Body.scrollTo({ top: step2Body.scrollHeight, behavior: 'smooth' });
      }
      enrollSubmitBtn.disabled = false;
      enrollSubmitBtn.classList.add('btn-pulse');
      setTimeout(() => {
        enrollSubmitBtn.classList.remove('btn-pulse');
        if (!enrollSubmitBtn.disabled && enrollSelectedFiles.length >= 5) {
          handleEnrollSubmit();
        }
      }, 2000);
    }
  }, 'image/jpeg', 0.85);
}

function updateEnrollCamCount() {
  if (enrollCamCount) enrollCamCount.textContent = `${enrollSelectedFiles.length} / ${ENROLL_TARGET}`;
}

// Override resetEnrollmentModal to also stop cameras and reset tabs
function resetEnrollmentModal() {
  stopEnrollCamera();
  stopLiveAttendance();
  enrollStep1.classList.remove('hidden');
  enrollStep2.classList.add('hidden');
  stepIndicator1.classList.add('active');
  stepIndicator1.classList.remove('done');
  stepIndicator2.classList.remove('active', 'done');
  if (stepConnector) stepConnector.classList.remove('done');
  registerStudentForm.reset();
  enrollSelectedFiles = [];
  pendingStudentId = null;
  enrollPreviewGrid.innerHTML = '';
  enrollDropZone.classList.remove('has-files');
  enrollProgress.classList.add('hidden');
  enrollProgressFill.style.width = '0%';
  enrollSubmitBtn.classList.remove('btn-pulse');
  const submitText = document.getElementById('enrollSubmitBtnText');
  if (submitText) submitText.textContent = 'Save & Add to Database';
  hideEnrollStatus();
  updateEnrollPhotoUI();
  // Reset mode tabs
  if (enrollTabUpload) enrollTabUpload.classList.add('active');
  if (enrollTabCamera) enrollTabCamera.classList.remove('active');
  if (enrollModeUpload) enrollModeUpload.classList.remove('hidden');
  if (enrollModeCamera) enrollModeCamera.classList.add('hidden');
}
