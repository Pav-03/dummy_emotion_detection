# tests/test_security.py

class TestSecurity:
    """
    Security tests — can someone hack API?
    Tests malicious inputs and attack vectors

    200 = handled it fine
    400 = rejected bad input
    413 = payload too large
    422 = validation error (pydantic caught it)
    401 = unauthorized (for JWT attacks)
    """

    # Injection Attack Tests 

    def test_sql_injection_attempt(self, client, auth_headers):
        """
        SQL injection — classic hacker attack
        attacker tries to break database with:
        '; DROP TABLE users; --
        should be handled gracefully, not crash
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "'; DROP TABLE users; --"})
        assert response.status_code in [200, 400]

    def test_script_injection_attempt(self, client, auth_headers):
        """
        XSS attack — attacker tries to inject javascript
        should be handled gracefully, not execute
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "<script>alert('xss')</script>"})
        assert response.status_code in [200, 400]

    def test_command_injection_attempt(self, client, auth_headers):
        """
        Command injection , attacker tries to run system commands
        should be handled gracefully
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "happy; rm -rf /"})
        assert response.status_code in [200, 400]

    # Input Size Tests

    def test_oversized_input(self, client, auth_headers):
        """
        Extremely large input — attacker tries to crash server
        with massive text payload (DOS attack)
        should be rejected, not crash
        """
        huge_text = "a" * 100_000
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": huge_text})
        assert response.status_code in [200, 400, 413]
        

    def test_empty_input(self, client, auth_headers):
        """
        Empty string should be rejected
        not crash the model
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": ""})
        assert response.status_code == 400

    def test_whitespace_only_input(self, client, auth_headers):
        """
        Whitespace only string is effectively empty
        should be rejected
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "     "})
        assert response.status_code == 400

    def test_special_characters_input(self, client, auth_headers):
        """
        Special characters should not crash the API
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "!@#$%^&*()_+-=[]{}|;':\",./<>?"})
        assert response.status_code in [200, 400]

    def test_unicode_input(self, client, auth_headers):
        """
        Unicode characters should not crash the API
        real users might send emojis or non-english text
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "I am happy 😊 जिंदगी خوشی"})
        assert response.status_code in [200, 400]

    def test_null_input(self, client, auth_headers):
        """
        Null value should return 422 (validation error)
        pydantic should catch this before it hits your code
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": None})
        assert response.status_code == 422

    def test_missing_text_field(self, client, auth_headers):
        """
        Missing required field entirely
        pydantic should catch this and return 422
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={})
        assert response.status_code == 422

    def test_wrong_data_type(self, client, auth_headers):
        """
        Sending integer instead of string
        pydantic should catch this and return 422
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": 12345})
        assert response.status_code == 422

    # JWT Attack Tests 

    def test_jwt_algorithm_none_attack(self, client):
        """
        Classic JWT attack:
        hacker sets algorithm to 'none'
        to bypass signature verification entirely
        must be rejected with 401
        """
        fake_token = "eyJhbGciOiJub25lIn0.eyJ1c2VybmFtZSI6InBhdmFuIn0."
        response = client.post("/predict",
            headers={"Authorization": f"Bearer {fake_token}"},
            json={"text": "I am happy"})
        assert response.status_code == 401

    def test_jwt_tampered_payload(self, client):
        """
        Hacker gets a valid token then manually edits the payload
        tampered token signature won't match → must reject
        """
        tampered_token = "eyJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6ImhhY2tlciJ9.fakesignature"
        response = client.post("/predict",
            headers={"Authorization": f"Bearer {tampered_token}"},
            json={"text": "I am happy"})
        assert response.status_code == 401

    def test_jwt_empty_token(self, client):
        """
        Empty Bearer token should return 401
        """
        response = client.post("/predict",
            headers={"Authorization": "Bearer "},
            json={"text": "I am happy"})
        assert response.status_code == 401

    # Header Tests

    def test_no_content_type_header(self, client, auth_headers):
        """
        Sending raw text without content-type header
        should return 422, not crash
        """
        headers = {**auth_headers, "Content-Type": "text/plain"}
        response = client.post("/predict",
            headers=headers,
            content="I am happy")
        assert response.status_code == 422