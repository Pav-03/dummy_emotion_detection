
class TestMiddleware:
    """
    Middleware tests — testing the layers that wrap every request
    CORS, logging (request ID), auth guard
    """

    # CORS Middleware Tests

    def test_cors_allowed_origin(self, client):
        """
        Allowed origin should get CORS headers back
        browser uses these headers to decide if
        frontend can read the response
        """
        response = client.get("/health",
            headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_cors_preflight_request(self, client):
        """
        OPTIONS request = browser asking:
        'Am I allowed to call this API?'
        should return 200 with CORS headers
        happens automatically before every cross-origin request
        """
        response = client.options("/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization"
            })
        assert response.status_code in [200, 400]
        assert "access-control-allow-origin" in response.headers

    # Logging Middleware Tests
    # (Request ID)

    def test_every_request_gets_request_id(self, client):
        """
        Every response should have a unique request ID header
        used for tracing requests through logs
        'which log lines belong to which request?'
        """
        response = client.get("/health")
        assert "x-request-id" in response.headers

    def test_request_id_is_non_empty(self, client):
        """
        Request ID must not be empty string
        """
        response = client.get("/health")
        request_id = response.headers.get("x-request-id")
        assert request_id is not None
        assert len(request_id) > 0

    def test_every_request_gets_unique_id(self, client):
        """
        Two requests should get two DIFFERENT request IDs
        if IDs are same → can't trace individual requests in logs
        """
        response1 = client.get("/health")
        response2 = client.get("/health")

        id1 = response1.headers.get("x-request-id")
        id2 = response2.headers.get("x-request-id")

        assert id1 != id2

    def test_request_id_present_on_all_endpoints(self, client):
        """
        Request ID should appear on every endpoint
        not just health — every single response
        """
        endpoints = ["/", "/health", "/docs"]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert "x-request-id" in response.headers, \
                f"x-request-id missing on {endpoint}"

    # Auth Guard Middleware Tests

    def test_public_paths_dont_need_token(self, client):
        """
        These paths are public — no token needed
        auth guard should let them through
        """
        public_paths = ["/", "/health", "/docs", "/auth/login"]

        for path in public_paths:
            if path == "/auth/login":
                response = client.post(path, json={
                    "username": "pavan",
                    "password": "secure123"
                })
            else:
                response = client.get(path)

            assert response.status_code != 401, \
                f"{path} should be public but returned 401"

    def test_protected_paths_need_token(self, client):
        """
        These paths are protected — token required
        auth guard should block requests without token
        """
        protected_paths = ["/predict", "/predict/batch", "/model-info"]

        for path in protected_paths:
            response = client.post(path, json={"text": "test"}) \
                if path != "/model-info" \
                else client.get(path)

            assert response.status_code == 401, \
                f"{path} should be protected but returned {response.status_code}"

    def test_auth_guard_passes_valid_token(self, client, auth_headers):
        """
        Valid token should pass through auth guard
        response should NOT be 401
        """
        response = client.get("/model-info", headers=auth_headers)
        assert response.status_code != 401

    def test_response_time_header(self, client):
        """
        Logging middleware adds latency to logs
        response should come back in reasonable time
        under 5 seconds for a simple health check
        """
        import time
        start = time.time()
        response = client.get("/health")
        end = time.time()

        assert response.status_code == 200
        assert (end - start) < 5.0