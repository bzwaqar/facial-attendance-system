"""Tests for health check endpoints."""

from fastapi import status
from fastapi.testclient import TestClient

from app.core.config import settings


def test_root_endpoint(client: TestClient) -> None:
    """Test the root informational endpoint."""
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == settings.PROJECT_NAME
    assert data["status"] == "online"
    assert data["health_check"] == "/health"


def test_health_check_endpoint(client: TestClient) -> None:
    """Test the GET /health endpoint returns 200 OK and expected structure."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == settings.PROJECT_NAME
    assert data["version"] == settings.VERSION
    assert data["environment"] == settings.ENVIRONMENT
    assert "timestamp" in data


def test_api_v1_health_check_endpoint(client: TestClient) -> None:
    """Test that the health endpoint is also accessible via the /api/v1 prefix."""
    response = client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == settings.VERSION


def test_dashboard_and_static_files(client: TestClient) -> None:
    """Test that /dashboard serves the HTML page and /static serves CSS assets."""
    dash_resp = client.get("/dashboard")
    assert dash_resp.status_code == status.HTTP_200_OK
    assert "<!DOCTYPE html>" in dash_resp.text
    assert "FaceID Biometrics" in dash_resp.text

    css_resp = client.get("/static/style.css")
    assert css_resp.status_code == status.HTTP_200_OK
    assert "--bg-dark" in css_resp.text

    # Root redirect when browser sends Accept: text/html
    root_browser_resp = client.get("/", headers={"Accept": "text/html"}, follow_redirects=False)
    assert root_browser_resp.status_code == status.HTTP_307_TEMPORARY_REDIRECT
    assert root_browser_resp.headers["location"] == "/dashboard"

