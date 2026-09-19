from fastapi import APIRouter
from app.api.routes.attendance import router as attendance_router
from app.api.routes.health import router as health_router
from app.api.routes.recognition import router as recognition_router
from app.api.routes.reports import router as reports_router
from app.api.routes.students import router as students_router

api_v1_router = APIRouter()

# Register health check under /api/v1 as well for API consumers
api_v1_router.include_router(health_router, prefix="", tags=["Health"])

# Student registration and biometric facial enrollment routes
api_v1_router.include_router(students_router, prefix="/students", tags=["Students & Enrollment"])

# On-demand face recognition and biometric verification routes
api_v1_router.include_router(recognition_router, prefix="/recognition", tags=["Face Recognition"])

# Attendance engine and session management routes
api_v1_router.include_router(attendance_router, prefix="/attendance", tags=["Attendance Engine"])

# Dashboard overview and attendance reporting routes
api_v1_router.include_router(reports_router, prefix="/reports", tags=["Reports & Analytics"])


