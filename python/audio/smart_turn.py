"""Smart Turn v3 ONNX 모델을 사용하여 발화 종료를 예측합니다."""

import logging

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

logger = logging.getLogger(__name__)


class SmartTurnProcessor:
    """Smart Turn v3 ONNX 모델을 사용하여 발화 종료를 예측하는 클래스."""

    def __init__(self, onnx_path):
        try:
            so = ort.SessionOptions()
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=['CPUExecutionProvider'])
            self.feature_extractor = WhisperFeatureExtractor(chunk_length=8)
            logger.info(f"Smart Turn 모델 로드 완료: {onnx_path}")
        except Exception as e:
            logger.error(f"Smart Turn 모델 로드 실패: {e}", exc_info=True)
            self.session = None

    def _truncate_or_pad_audio(self, audio_array, n_seconds=8, sample_rate=16000):
        max_samples = n_seconds * sample_rate
        if len(audio_array) > max_samples:
            return audio_array[-max_samples:]
        elif len(audio_array) < max_samples:
            padding = max_samples - len(audio_array)
            return np.pad(audio_array, (padding, 0), mode='constant', constant_values=0)
        return audio_array

    def predict(self, audio_array_f32: np.ndarray) -> dict:
        """
        오디오 세그먼트의 발화 종료 여부를 예측합니다.
        Returns: {"prediction": 0 or 1, "probability": float}
        """
        if not self.session:
            logger.warning("Smart Turn 모델이 로드되지 않았습니다. 항상 '진행 중'으로 반환합니다.")
            return {"prediction": 0, "probability": 0.0}

        audio_array = self._truncate_or_pad_audio(audio_array_f32, n_seconds=8)
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="np",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True,
        )
        input_features = np.expand_dims(inputs.input_features.squeeze(0), axis=0).astype(np.float32)
        outputs = self.session.run(None, {"input_features": input_features})
        probability = outputs[0][0].item()
        prediction = 1 if probability > 0.5 else 0
        return {"prediction": prediction, "probability": probability}
