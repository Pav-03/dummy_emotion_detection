
class TestAuth:
    """  Auth tests — login and JWT token validation for protected endpoints """

    # Login Tests

    def test_login_success(self, client):
        """correct credentials should return 200"""
        response = client.post("/auth/login", json={
            "username": "pavan",
            "password": "secure123"
        })
        assert response.status_code == 200

    def test_login_returns_token(self, client):
        """login response must have access_token field"""
        response = client.post("/auth/login", json={
            "username": "pavan",
            "password": "secure123"
        })
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data

    def test_login_token_type_is_bearer(self, client):
        """token type must be bearer"""
        response = client.post("/auth/login", json={
            "username": "pavan",
            "password": "secure123"
        })
        assert response.json()["token_type"] == "bearer"

    def test_login_token_is_non_empty_string(self, client):
        """token must be a non empty string"""
        response = client.post("/auth/login", json={
            "username": "pavan",
            "password": "secure123"
        })
        token = response.json()["access_token"]
        assert isinstance(token, str)
        assert len(token) > 0

    def test_login_wrong_password(self, client):
        """wrong password should return 401"""
        response = client.post("/auth/login", json={
            "username": "pavan",
            "password": "wrongpassword"
        })
        assert response.status_code == 401

    def test_login_wrong_username(self, client):
        """non existent user should return 401"""
        response = client.post("/auth/login", json={
            "username": "hacker",
            "password": "secure123"
        })
        assert response.status_code == 401

    def test_login_empty_password(self, client):
        """empty password should return 401"""
        response = client.post("/auth/login", json={
            "username": "pavan",
            "password": ""
        })
        assert response.status_code == 401

    # JWT Protection Tests for /predict endpoint

    def test_protected_endpoint_without_token(self, client):
        """no token should return 401"""
        response = client.post("/predict",
            json={"text": "I am happy"})
        assert response.status_code == 401

    def test_protected_endpoint_with_fake_token(self, client):
        """made up token should return 401"""
        response = client.post("/predict",
            headers={"Authorization": "Bearer faketoken123"},
            json={"text": "I am happy"})
        assert response.status_code == 401

    def test_protected_endpoint_wrong_format(self, client):
        """
        token without Bearer prefix should return 401
        must be: "Bearer <token>"
        not just: "<token>"
        """
        response = client.post("/predict",
            headers={"Authorization": "invalidformat"},
            json={"text": "I am happy"})
        assert response.status_code == 401

    def test_protected_endpoint_with_valid_token(self, client, auth_headers):
        """valid token should NOT return 401"""
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "I am happy"})
        assert response.status_code != 401