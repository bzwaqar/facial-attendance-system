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

    kpiSessionTitle.textContent = data.course_name;
    kpiSessionMeta.textContent = `ID: ${data.session_id} • Room: ${data.room_number || 'N/A'}`;
    kpiPresentStudents.textContent = data.present_count;
    kpiAttendanceRate.textContent = `${data.attendance_rate.toFixed(1)}% of registered students`;

    renderAttendanceTable(data.attendees);
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
    const dateStr = new Date(att.marked_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const simScore = att.confidence != null ? att.confidence.toFixed(4) : 'N/A';
    const livenessBadge = att.liveness_score != null && att.liveness_score >= 0.5
      ? `<span class="badge badge-emerald">Live (${(att.liveness_score * 100).toFixed(0)}%)</span>`
      : `<span class="badge badge-warning">Unchecked</span>`;

    return `
      <tr>
        <td class="mono"><strong>${escapeHtml(att.student_id)}</strong></td>
        <td>${escapeHtml(att.student_name)}</td>
        <td class="mono">${dateStr}</td>
        <td class="mono">${simScore}</td>
        <td>${livenessBadge}</td>
        <td><span class="badge badge-emerald">Present</span></td>
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
  const isLive = match.is_live !== false;
  const livenessScore = match.liveness_score != null ? (match.liveness_score * 100).toFixed(0) + '%' : 'Verified';
  resLiveness.textContent = isLive ? `Live (${livenessScore})` : `Spoof/Photo (${livenessScore})`;

  if (!isLive) {
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

async function markAttendanceFromProbe() {
  if (!activeSessionId || !latestRecognitionResult) return;

  const match = latestRecognitionResult.matches[0];
  if (!match || !match.student_id || !match.is_recognized) return;

  markFromProbeBtn.disabled = true;

  try {
    const payload = {
      session_id: activeSessionId,
      student_id: match.student_id,
      confidence: match.similarity,
      bounding_box: match.bounding_box || [0, 0, 100, 100],
      liveness_score: match.liveness_score ?? 1.0,
      marked_by: 'web_dashboard',
    };

    const res = await fetch(`${API_BASE}/attendance/mark`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to mark attendance' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
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
      similarity_threshold: threshold,
    };

    const res = await fetch(`${API_BASE}/attendance/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to create session' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
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
    alert(`Failed to create session: ${err.message}`);
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
