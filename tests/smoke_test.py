class TestSmoke:
    
    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
    
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        # check fields exist
        assert "status" in data
        assert "model_loaded" in data
        assert "vectorizer_loaded" in data
    
        # check types are correct
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["vectorizer_loaded"], bool)
    
    def test_docs_endpoiont(self, client):
        response = client.get("/docs")
        assert response.status_code == 200