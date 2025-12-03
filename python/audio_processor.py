import time
import math
import json
import queue
import asyncio
import logging
import threading
from collections import deque

import sounddevice as sd
import numpy as np
import torch # VAD에 필요
from google.cloud import speech # STT에 필요
from google.api_core import exceptions

# SmartTurn 모델에 필요한 라이브러리
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

from config import (
    SMART_TURN_MODEL_PATH, TURN_END_SILENCE_CHUNKS, MAX_TURN_CHUNKS, SMART_TURN_GRACE_PERIOD_S
)


# --- 로깅 설정 (단독 실행 시 필요) ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s',
    datefmt='%H:%M:%S'
)

# ==================================================================================
# SmartTurn
# ==================================================================================
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
            logging.info(f"✅ Smart Turn 모델 로드 완료: {onnx_path}")
        except Exception as e:
            logging.error(f"❌ Smart Turn 모델 로드 실패: {e}", exc_info=True)
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
            # 모델 로드 실패 시, 항상 '진행 중'으로 판단하여 대화가 끊기지 않도록 함
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


# ==================================================================================
# 유틸리티 함수
# ==================================================================================

def find_input_device(device_name_substring: str = 'pipewire') -> int | None:
    """
    주어진 문자열이 포함된 오디오 입력 장치를 검색합니다.
    """
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device_name_substring.lower() in device['name'].lower() and device['max_input_channels'] > 0:
            logging.info(f"🔍 발견된 입력 장치: [{idx}] {device['name']}")
            return idx
    logging.warning(f"⚠️ '{device_name_substring}'가 포함된 입력 장치를 찾지 못했습니다.")
    return None

# ==================================================================================
# Component 1: 마이크 스트림 담당 (Producer)
# ==================================================================================

class MicrophoneStream:
    """마이크로부터 오디오 데이터를 읽어 큐에 넣는 클래스."""

    def __init__(self, mic_audio_queue: queue.Queue, sample_rate: int, chunk_size: int, channels: int, dtype: str, device_idx: int | None = None):
        self.mic_audio_queue = mic_audio_queue
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.dtype = dtype
        self.device_idx = device_idx
        self.stream: sd.InputStream | None = None
    
    def start(self):
        if self.stream is not None and self.stream.active:
            logging.warning("MicrophoneStream이 이미 활성화되어 있습니다.")
            return

        logging.info("🎙️ MicrophoneStream 시작.")
        self.stream = sd.InputStream(samplerate=self.sample_rate,
                                     blocksize=self.chunk_size,
                                     channels=self.channels,
                                     dtype=self.dtype,
                                     device=self.device_idx,
                                     callback=self._callback)
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            logging.info("🎙️ MicrophoneStream 중지.")
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logging.warning(f"[오디오 상태] {status}")
        self.mic_audio_queue.put(indata.copy())


# ==================================================================================
# Component 2: VAD (Voice Activity Detection) 담당
# ==================================================================================

class VADProcessor:
    """Silero VAD 모델을 사용하여 음성 활동을 감지하는 클래스."""

    def __init__(self, sample_rate: int, chunk_size: int, threshold: float = 0.5, consecutive_chunks: int = 3, reset_interval: float = 20.0):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.consecutive_chunks_required = consecutive_chunks
        self.reset_interval = reset_interval

        # VAD 모델 로드
        try:
            model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=True)
            self.vad_model = model
            logging.info("✅ Silero VAD 초기화 완료")
        except Exception as e:
            logging.error(f"❌ Silero VAD 로드 실패: {e}")
            self.vad_model = None
        
        self.vad_buffer = torch.tensor([])
        self.consecutive_speech_chunks = 0
        self.vad_detection_start_time = time.time()
    
    def process_chunk(self, audio_chunk_int16: np.ndarray) -> bool:
        """
        오디오 청크를 처리하고 음성 감지 여부를 반환합니다.
        
        Returns:
            bool: 음성 시작 조건(연속 청크 수)이 충족되었는지 여부.
        """
        if self.vad_model is None:
            return False
        
        # float32로 변환 (Silero VAD 요구사항)
        audio_chunk_float32 = audio_chunk_int16.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_chunk_float32.flatten())
        
        self.vad_buffer = torch.cat([self.vad_buffer, audio_tensor])
        
        speech_detected = False

        while len(self.vad_buffer) >= self.chunk_size:
            vad_chunk = self.vad_buffer[:self.chunk_size]
            self.vad_buffer = self.vad_buffer[self.chunk_size:]
            
            speech_prob = self.vad_model(vad_chunk, self.sample_rate).item()
            
            if speech_prob > self.threshold:
                self.consecutive_speech_chunks += 1
                self.vad_detection_start_time = time.time() # 감지 시점 갱신
            else:
                self.consecutive_speech_chunks = 0

            if self.consecutive_speech_chunks >= self.consecutive_chunks_required:
                speech_detected = True
                break # 음성 감지 조건 충족 시 루프 종료

        return speech_detected
    
    def reset_if_inactive(self):
        """
        일정 시간 동안 음성 감지가 없으면 VAD 상태를 초기화합니다.
        """
        if time.time() - self.vad_detection_start_time > self.reset_interval:
            logging.info(f"{self.reset_interval}초 동안 음성 감지가 없어 VAD 상태를 초기화합니다.")
            self.reset()
            self.vad_detection_start_time = time.time()
    
    def reset(self):
        """
        VAD 상태를 초기화합니다.
        """
        if self.vad_model:
            self.vad_model.reset_states()
        self.vad_buffer = torch.tensor([])
        self.consecutive_speech_chunks = 0
        logging.info("VAD 상태 초기화 완료.")

# ==================================================================================
# Component 3: Google STT 스트리머 담당
# ==================================================================================

class GoogleSTTStreamer:
    """Google Speech-to-Text API로 오디오 스트림을 처리하는 클래스."""
    
    def __init__(self, stt_result_queue: asyncio.Queue, main_loop: asyncio.AbstractEventLoop, websocket, sample_rate: int, stt_audio_queue: queue.Queue, stt_stop_event: threading.Event):
        self.stt_result_queue = stt_result_queue
        self.main_loop = main_loop
        self.websocket = websocket
        self.stt_audio_queue = stt_audio_queue
        self.stt_stop_event = stt_stop_event
        
        self.stt_client = speech.SpeechClient()
        self.stt_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="ko-KR",
            enable_automatic_punctuation=True,
        )
        self.stt_streaming_config = speech.StreamingRecognitionConfig(
            config=self.stt_config,
            interim_results=True,
            single_utterance=False,
        )
        logging.info("✅ Google STT 클라이언트 초기화 완료")

    def _stt_audio_generator(self):
        """STT API에 오디오를 공급하는 제너레이터."""
        while not self.stt_stop_event.is_set():
            try:
                chunk = self.stt_audio_queue.get(timeout=0.1)
                yield speech.StreamingRecognizeRequest(audio_content=chunk.tobytes())
            except queue.Empty:
                continue
        logging.info("STT 오디오 공급 중단됨.")

    def run_stt_session(self):
        """단일 STT 세션을 실행하고 최종 결과를 큐에 넣음."""
        logging.info("🚀 STT 세션 스레드 시작.")
        first_response_received = False

        accumulated_transcripts = []
        current_interim_transcript = ""

        try:
            audio_gen = self._stt_audio_generator()
            responses = self.stt_client.streaming_recognize(self.stt_streaming_config, audio_gen)
            
            for response in responses:
                if not first_response_received:
                    first_response_received = True
                    # C++ 클라이언트에게 인터럽션 신호 전송
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send(json.dumps({"type": "user_interruption"})),
                        self.main_loop
                    )

                if not response.results or not response.results[0].alternatives:
                    continue

                result = response.results[0]
                transcript = result.alternatives[0].transcript.strip()

                if result.is_final:
                    accumulated_transcripts.append(transcript)
                    current_interim_transcript = ""
                    logging.info(f"✅ STT 최종 결과 조각: '{transcript}'")
                else:
                    current_interim_transcript = transcript
                    logging.info(f"🟩 STT 중간 결과: '{transcript}'")
                
                if self.stt_stop_event.is_set():
                    break

                # if result.is_final and self.stt_stop_event.is_set():
                #     final_text = transcript.strip()
                #     logging.info(f"✅ STT 최종 결과: '{final_text}'")
                #     if final_text:
                #         # 메인 스레드로 결과 전송
                #         self.main_loop.call_soon_threadsafe(self.stt_result_queue.put_nowait, final_text)
                    
                #     # C++ 클라이언트에 STT 완료 신호 전송
                #     stt_completion_time = int(time.time() * 1000)
                #     asyncio.run_coroutine_threadsafe(
                #         self.websocket.send(json.dumps({"type": "stt_done", "stt_done_time": stt_completion_time})),
                #         self.main_loop
                #     )
                #     break # 최종 결과를 받으면 루프 종료
                # else:
                #     logging.info(f"✅ STT 중간 결과: '{transcript}'")
                    
        except exceptions.DeadlineExceeded as e:
            logging.warning(f"STT 세션 타임아웃(DeadlineExceeded): {e}")
        except Exception as e:
            logging.error(f"STT 세션 중 오류 발생: {e}", exc_info=True)
        finally:
            # 최종 결과 반환
            final_text_parts = accumulated_transcripts.copy()
            if current_interim_transcript:
                final_text_parts.append(current_interim_transcript)
            final_text = " ".join(final_text_parts).strip()

            if final_text:
                logging.info(f"✅ STT 최종 결과: '{final_text}'")

                # 메인 스레드로 결과 전송
                self.main_loop.call_soon_threadsafe(self.stt_result_queue.put_nowait, final_text)

                # C++ 클라이언트에 STT 완료 신호 전송
                stt_completion_time = int(time.time() * 1000)
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps({"type": "stt_done", "stt_done_time": stt_completion_time})),
                    self.main_loop
                )
            else:
                logging.info("❎ STT 인식 결과가 없습니다.")
            logging.info("🚀 STT 세션 스레드 종료.")

# ==================================================================================
# Orchestrator: 오디오 처리 파이프라인 관리
# ==================================================================================

class AudioProcessor:
    """마이크 스트림, VAD, SmartTurn, STT를 총괄하여 오디오 처리 파이프라인을 관리."""
    
    def __init__(self, stt_result_queue: asyncio.Queue, main_loop: asyncio.AbstractEventLoop, websocket, config: dict):
               
        # 설정 값
        self.sample_rate = config['SAMPLE_RATE']
        self.channels = config['CHANNELS']
        self.audio_dtype = config['AUDIO_DTYPE']
        self.vad_chunk_size = config['VAD_CHUNK_SIZE']
        self.pre_buffer_duration = config['PRE_BUFFER_DURATION']
        
        # 통신 채널
        self.stt_result_queue = stt_result_queue
        self.main_loop = main_loop
        self.websocket = websocket
        
        # 오디오 큐
        self.mic_audio_queue = queue.Queue()
        self.stt_audio_queue = queue.Queue()
        pre_buffer_max_chunks = math.ceil(self.sample_rate * self.pre_buffer_duration / self.vad_chunk_size)
        self.stt_pre_buffer = deque(maxlen=pre_buffer_max_chunks)
        self.current_turn_audio = []

        # 컴포넌트 초기화
        device_idx = find_input_device()
        self.mic_stream = MicrophoneStream(
            mic_audio_queue=self.mic_audio_queue,
            sample_rate=self.sample_rate,
            chunk_size=self.vad_chunk_size,
            channels=self.channels,
            dtype=self.audio_dtype,
            device_idx=device_idx
        )
        self.vad_processor = VADProcessor(
            sample_rate=self.sample_rate,
            chunk_size=self.vad_chunk_size,
            threshold=config['VAD_THRESHOLD'],
            consecutive_chunks=config['VAD_CONSECUTIVE_CHUNKS'],
            reset_interval=config['VAD_RESET_INTERVAL']
        )
        self.smart_turn_processor = SmartTurnProcessor(SMART_TURN_MODEL_PATH)
        self.stt_stop_event = threading.Event()
        self.stt_streamer = GoogleSTTStreamer(
            stt_result_queue=stt_result_queue,
            main_loop=main_loop,
            websocket=websocket,
            sample_rate=self.sample_rate,
            stt_audio_queue=self.stt_audio_queue,
            stt_stop_event=self.stt_stop_event
        )

        # 상태 관리
        self._is_running = threading.Event()
        self.user_is_speaking = False
        self.silent_chunks_count = 0
        self.turn_chunks_count = 0
        self.in_grace_period = False
        self.grace_period_start_time = 0.0

        self._thread: threading.Thread | None = None

    def _processing_loop(self):
        """
        오디오 큐에서 데이터를 소비하여 처리하는 메인 루프.
        """
        logging.info("🎧 오디오 처리 루프 시작.")
        
        # MicrophoneStream 시작 (sounddevice 내부 스레드 시작)
        self.mic_stream.start()

        while self._is_running.is_set():
            try:
                chunk = self.mic_audio_queue.get(timeout=0.1)
            except queue.Empty:
                if self.user_is_speaking:
                    # 사용자 발화 중 타임아웃 발생 시, 강제로 턴 종료 (예외 처리)
                    logging.warning("사용자 발화 중 오디오 입력 타임아웃. 강제로 턴을 종료합니다.")
                    self._end_turn()
                continue
            
            if not self.user_is_speaking:
                self._handle_silence_state(chunk)
            else:
                self._handle_speaking_state(chunk)

        logging.info("🎧 오디오 처리 루프 종료.")

    def _handle_silence_state(self, chunk: np.ndarray):
        """사용자가 말하고 있지 않을 때의 로직 (발화 시작 감지)"""
        self.stt_pre_buffer.append(chunk)
        self.vad_processor.reset_if_inactive()
        
        if self.vad_processor.process_chunk(chunk):
            logging.info("🗣️ 사용자 발화 시작 감지!")
            self.user_is_speaking = True
            self.silent_chunks_count = 0
            self.turn_chunks_count = 0
            self.in_grace_period = False
            self.current_turn_audio.clear()
            self.stt_stop_event.clear()
            
            # STT 세션 시작
            threading.Thread(
                target=self.stt_streamer.run_stt_session,
                name="STTSessionThread"
            ).start()
            
            # 사전 버퍼를 STT 큐로 전송
            for pre_chunk in self.stt_pre_buffer:
                self.stt_audio_queue.put(pre_chunk)
                self.current_turn_audio.append(pre_chunk)
            
            self.stt_audio_queue.put(chunk)
            self.current_turn_audio.append(chunk)
            self.vad_processor.reset()

    def _handle_speaking_state(self, chunk: np.ndarray):
        """사용자가 말하고 있을 때의 로직 (발화 종료 감지)"""
        # STT 및 내부 버퍼로 오디오 전달
        self.stt_audio_queue.put(chunk)
        self.current_turn_audio.append(chunk)
        self.turn_chunks_count += 1

        # VAD로 무음 감지
        is_speech_in_chunk = self.vad_processor.process_chunk(chunk)
        if is_speech_in_chunk:
            self.silent_chunks_count = 0
            if self.in_grace_period:
                logging.info("⏳ 유예 기간 중 추가 발화 감지. 유예 기간을 취소합니다.")
                self.in_grace_period = False
        else:
            self.silent_chunks_count += 1
        
        # 종료 조건 확인
        turn_ended = False

        if self.in_grace_period and (time.time() - self.grace_period_start_time) > SMART_TURN_GRACE_PERIOD_S:
            logging.info("⏳ 유예 기간 종료. 턴을 종료합니다.")
            turn_ended = True

        elif not self.in_grace_period and self.silent_chunks_count > TURN_END_SILENCE_CHUNKS:
            concatenated_audio_int16 = np.concatenate([c.flatten() for c in self.current_turn_audio])
            full_audio_float32 = concatenated_audio_int16.astype(np.float32) / 32768.0
            
            start_time = time.time()
            result = self.smart_turn_processor.predict(full_audio_float32)
            duration_ms = (time.time() - start_time) * 1000
            
            logging.info(f"🤖 SmartTurn 예측: {'종료' if result['prediction'] == 1 else '진행중'} (확률: {result['probability']:.2f}, 소요시간: {duration_ms:.1f}ms)")
            
            if result['prediction'] == 1:
                turn_ended = True
            else:
                logging.info(f"⏳ SmartTurn이 '진행중'으로 판단. {SMART_TURN_GRACE_PERIOD_S}초의 유예 시간을 시작합니다.")
                self.in_grace_period = True
                self.grace_period_start_time = time.time()
        
        # elif self.turn_chunks_count > MAX_TURN_CHUNKS:
        #     logging.warning(f"최대 발화 길이({MAX_TURN_CHUNKS * 0.032:.1f}초) 초과. 턴을 종료합니다.")
        #     turn_ended = True
        
        if turn_ended:
            self._end_turn()

    def _end_turn(self):
        """현재 발화 턴을 종료하는 헬퍼 함수"""
        if not self.user_is_speaking: return
        
        logging.info("🤫 인식 종료. STT 오디오 공급을 중단합니다.")
        self.stt_stop_event.set()
        self.user_is_speaking = False
        self.in_grace_period = False
        
        # 남아있을 수 있는 큐를 비워 다음 턴에 영향이 없도록 함
        with self.stt_audio_queue.mutex:
            self.stt_audio_queue.queue.clear()

    def __enter__(self):
        """AudioProcessor의 생명주기 시작."""
        logging.info("AudioProcessor 컨텍스트 시작...")
        self._is_running.set()
        
        # 오디오 처리 루프 스레드 시작
        self._thread = threading.Thread(target=self._processing_loop, name="AudioProcessingThread")
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """AudioProcessor의 생명주기 종료."""
        logging.info("AudioProcessor 컨텍스트 종료...")
        
        # 1. 처리 루프 스레드에 종료 신호 전송
        self._is_running.clear()
        self.stt_stop_event.set() # 대기 중인 스레드를 깨움
        
        # 2. 처리 루프 스레드 종료 대기
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        # 3. MicrophoneStream 중지 (sounddevice 내부 스레드 종료)
        self.mic_stream.stop()
        
        logging.info("AudioProcessor가 성공적으로 종료되었습니다.")
