from tests.conftest import auth_headers


class TestErrorHandling:
    """
    Error handling tests — does your API fail gracefully?
    Bad input, wrong methods, wrong endpoints
    A good API never crashes — always returns proper error response
    """

    # HTTP Method Tests
    
    def test_wrong_method_on_predict(self, client, auth_headers):
        """
        GET /predict should return 405 Method Not Allowed
        predict only accepts POST
        not GET, not PUT, not DELETE
        """
        response = client.get("/predict",
            headers=auth_headers)
        assert response.status_code == 405

    def test_wrong_method_on_login(self, client, auth_headers):
        """
        GET /auth/login should return 405
        login only accepts POST
        """
        response = client.get("/auth/login")
        assert response.status_code == 405

    def test_wrong_method_on_batch_predict(self, client, auth_headers):
        """
        GET /predict/batch should return 405
        batch predict only accepts POST
        """
        response = client.get("/predict/batch",
            headers=auth_headers)
        assert response.status_code == 405

    # 404 Not Found Tests

    def test_unknown_endpoint_returns_404(self, client, auth_headers):
        """
        Hitting a route that doesn't exist
        should return 404 Not Found
        not 500 Internal Server Error
        """
        response = client.get("/this-does-not-exist",
                              headers=auth_headers)
        assert response.status_code == 404

    def test_unknown_nested_endpoint_returns_404(self, client, auth_headers):
        """
        Deep nested unknown route should also return 404
        """
        response = client.get("/api/v1/something/random",
                              headers=auth_headers)
        assert response.status_code == 404

    def test_typo_in_endpoint_returns_404(self, client, auth_headers):
        """
        Common typos should return 404
        not 500
        """
        response = client.get("/healt", headers=auth_headers)     # Typo mistake intenstionally
        assert response.status_code == 404

    # Request Body Tests

    def test_empty_body_returns_422(self, client, auth_headers):
        """
        Sending completely empty body to /predict
        pydantic should catch this and return 422
        422 = Unprocessable Entity
           = request was received but data is wrong
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={})
        assert response.status_code == 422
        assert response.headers["content-type"] == "application/json"

    def test_invalid_json_returns_422(self, client, auth_headers):
        """
        Sending wrong field names
        pydantic should catch this
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"wrong_field": "I am happy"})
        assert response.status_code == 422

    def test_extra_fields_ignored(self, client, auth_headers):
        """
        Extra fields in request body should be ignored
        not cause a crash or error
        pydantic ignores unknown fields by default
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={
                "text": "I am happy",
                "extra_field": "should be ignored",
                "another_field": 12345
            })
        assert response.status_code == 200

    def test_batch_predict_empty_list_returns_400(self, client, auth_headers):
        """
        Empty list for batch predict should return 400
        not 200 with empty response
        not 500 crash
        """
        response = client.post("/predict/batch",
            headers=auth_headers,
            json={"texts": []})
        assert response.status_code == 400

    def test_batch_predict_wrong_field_returns_422(self, client, auth_headers):
        """
        Wrong field name for batch predict
        pydantic should catch this
        """
        response = client.post("/predict/batch",
            headers=auth_headers,
            json={"wrong_field": ["I am happy"]})
        assert response.status_code == 422

    # Error Response Format Tests

    def test_404_returns_json_not_html(self, client, auth_headers):
        """
        Error responses should be JSON
        not raw HTML error pages
        APIs should always return JSON even for errors
        """
        response = client.get("/does-not-exist",
                              headers=auth_headers)
        assert response.status_code == 404
        assert response.headers["content-type"] == "application/json"

    def test_422_returns_detail_field(self, client, auth_headers):
        """
        Validation errors should have detail field
        explaining what went wrong
        so caller knows how to fix their request
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_401_returns_json(self, client, auth_headers):
        """
        Auth errors should return JSON
        not HTML, not plain text
        """
        response = client.post("/predict",
            json={"text": "I am happy"})
        assert response.status_code == 401
        assert "application/json" in response.headers["content-type"]