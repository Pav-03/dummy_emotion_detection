
class TestPredict:
    """
    Predict tests — does the ML part actually work?
    Single prediction, batch prediction, response format
    """

    # Single Prediction Tests

    def test_predict_returns_200(self, client, auth_headers):
        """
        Basic happy path — valid text should return 200
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "I am feeling very happy today"})
        assert response.status_code == 200

    def test_predict_returns_emotion_field(self, client, auth_headers):
        """
        Response must have emotion field
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "I am feeling very happy today"})
        data = response.json()
        assert "emotion" in data

    def test_predict_returns_confidence_field(self, client, auth_headers):
        """
        Response must have confidence field
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "I am feeling very happy today"})
        data = response.json()
        assert "confidence" in data

    def test_predict_returns_model_version_field(self, client, auth_headers):
        """
        Response must have model_version field
        so caller knows which model made the prediction
        important for debugging and auditing
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "I am feeling very happy today"})
        data = response.json()
        assert "model_version" in data

    def test_predict_emotion_is_valid_value(self, client, auth_headers):
        """
        emotion must be a known valid value
        not some random string
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "I am feeling very happy today"})
        data = response.json()
        assert data["emotion"] in ["positive", "negative"]

    def test_predict_confidence_is_float(self, client, auth_headers):
        """
        confidence must be a float number
        not string, not int
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "I am feeling very happy today"})
        data = response.json()
        assert isinstance(data["confidence"], float)

    def test_predict_confidence_between_0_and_1(self, client, auth_headers):
        """
        confidence score must be between 0.0 and 1.0
        0.0 = model has no idea
        1.0 = model is 100% sure
        anything outside this range = bug in your model
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "I am feeling very happy today"})
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_model_version_is_string(self, client, auth_headers):
        """
        model_version must be a non empty string
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "I am feeling very happy today"})
        data = response.json()
        assert isinstance(data["model_version"], str)
        assert len(data["model_version"]) > 0

    def test_predict_happy_text(self, client, auth_headers):
        """
        Clearly happy text should return positive
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "I am so happy and excited today!"})
        assert response.status_code == 200
        assert response.json()["emotion"] == "positive"

    def test_predict_sad_text(self, client, auth_headers):
        """
        Clearly sad text should return negative
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "I am so sad and depressed today"})
        assert response.status_code == 200
        assert response.json()["emotion"] == "negative"

    def test_predict_empty_text_returns_400(self, client, auth_headers):
        """
        Empty text should be rejected with 400
        not crash the model
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": ""})
        assert response.status_code == 400

    def test_predict_whitespace_only_returns_400(self, client, auth_headers):
        """
        Whitespace only is effectively empty
        should be rejected with 400
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "     "})
        assert response.status_code == 400

    def test_predict_response_has_no_extra_fields(self, client, auth_headers):
        """
        Response should only have expected fields
        no leaking internal data
        no unexpected fields
        """
        response = client.post("/predict",
            headers=auth_headers,
            json={"text": "I am happy"})
        data = response.json()
        allowed_fields = {"emotion", "confidence", "model_version"}
        assert set(data.keys()) == allowed_fields

    # Batch Prediction Tests

    def test_batch_predict_returns_200(self, client, auth_headers):
        """
        Valid batch request should return 200
        """
        response = client.post("/predict/batch",
            headers=auth_headers,
            json={"texts": ["I am happy", "I am sad"]})
        assert response.status_code == 200

    def test_batch_predict_returns_predictions_field(self, client, auth_headers):
        """
        Response must have predictions field
        """
        response = client.post("/predict/batch",
            headers=auth_headers,
            json={"texts": ["I am happy", "I am sad"]})
        data = response.json()
        assert "predictions" in data

    def test_batch_predict_returns_total_field(self, client, auth_headers):
        """
        Response must have total field
        showing how many predictions were made
        """
        response = client.post("/predict/batch",
            headers=auth_headers,
            json={"texts": ["I am happy", "I am sad"]})
        data = response.json()
        assert "total" in data

    def test_batch_predict_count_matches_input(self, client, auth_headers):
        """
        If you send 3 texts → get 3 predictions back
        total must match number of texts sent
        """
        texts = ["I am happy", "I am sad", "life is beautiful"]
        response = client.post("/predict/batch",
            headers=auth_headers,
            json={"texts": texts})
        data = response.json()
        assert data["total"] == 3
        assert len(data["predictions"]) == 3

    def test_batch_predict_each_prediction_has_correct_fields(self, client, auth_headers):
        """
        Each prediction in batch must have
        same fields as single prediction
        """
        response = client.post("/predict/batch",
            headers=auth_headers,
            json={"texts": ["I am happy", "I am sad"]})
        predictions = response.json()["predictions"]

        for prediction in predictions:
            assert "emotion" in prediction
            assert "confidence" in prediction
            assert prediction["emotion"] in ["positive", "negative"]
            assert 0.0 <= prediction["confidence"] <= 1.0

    def test_batch_predict_empty_list_returns_400(self, client, auth_headers):
        """
        Empty list should return 400
        not 200 with empty predictions
        not 500 crash
        """
        response = client.post("/predict/batch",
            headers=auth_headers,
            json={"texts": []})
        assert response.status_code == 400

    def test_batch_predict_single_text_works(self, client, auth_headers):
        """
        Batch with just one text should work fine
        """
        response = client.post("/predict/batch",
            headers=auth_headers,
            json={"texts": ["I am happy"]})
        assert response.status_code == 200
        assert response.json()["total"] == 1

    # Model Info Tests

    def test_model_info_returns_200(self, client, auth_headers):
        """
        GET /model-info should return 200
        """
        response = client.get("/model-info",
            headers=auth_headers)
        assert response.status_code == 200

    def test_model_info_returns_version(self, client, auth_headers):
        """
        model info should include version
        """
        response = client.get("/model-info",
            headers=auth_headers)
        data = response.json()
        assert "version" in data