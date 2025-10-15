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


# --- 로깅 설정 (단독 실행 시 필요) ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s',
    datefmt='%H:%M:%S'
)

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

    def __init__(self, audio_queue: queue.Queue, sample_rate: int, chunk_size: int, channels: int, dtype: str, device_idx: int | None = None):
        self.audio_queue = audio_queue
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
        self.audio_queue.put(indata.copy())

# ==================================================================================
# Component 2: VAD (Voice Activity Detection) 담당
# ==================================================================================

class VADProcessor:
    """Silero VAD 모델을 사용하여 음성 활동을 감지하는 클래스."""

    def __init__(self, sample_rate: int, chunk_size: int, threshold: float = 0.5, consecutive_chunks: int = 3, reset_interval: float = 10.0):
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
    
    def __init__(self, stt_result_queue: asyncio.Queue, main_loop: asyncio.AbstractEventLoop, websocket, sample_rate: int, adaptation_config=None):
        self.stt_result_queue = stt_result_queue
        self.main_loop = main_loop
        self.websocket = websocket
        
        self.stt_client = speech.SpeechClient()
        self.stt_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="ko-KR",
            enable_automatic_punctuation=True,
            adaptation=adaptation_config
        )
        self.stt_streaming_config = speech.StreamingRecognitionConfig(
            config=self.stt_config,
            interim_results=True,
            single_utterance=True,
        )
        logging.info("✅ Google STT 클라이언트 초기화 완료")
    
    def _stt_audio_generator(self, pre_buffer: deque, audio_queue: queue.Queue, stt_stop_flag: threading.Event, inactivity_stop_flag: threading.Event):
        """
        STT API에 오디오를 공급하는 제너레이터.
        """
        # 1. 사전 버퍼(pre-buffer) 전송
        if pre_buffer:
            combined_audio = np.concatenate(list(pre_buffer))
            duration_sec = len(combined_audio) / self.stt_config.sample_rate_hertz
            yield speech.StreamingRecognizeRequest(audio_content=combined_audio.tobytes())
            logging.info(f"STT 사전 버퍼 ({duration_sec:.2f}초) 전송 완료")
            pre_buffer.clear()

        # 2. 실시간 오디오 전송
        while not stt_stop_flag.is_set() and not inactivity_stop_flag.is_set():
            try:
                # MicrophoneStream이 채워주는 큐에서 데이터를 가져옴
                chunk = audio_queue.get(timeout=0.1)
                yield speech.StreamingRecognizeRequest(audio_content=chunk.tobytes())
            except queue.Empty:
                continue
            except Exception as e:
                logging.debug(f"STT 오디오 생성기 오류: {e}")
                break

    def run_stt_session(self, pre_buffer: deque, audio_queue: queue.Queue, vad_active_flag: threading.Event):
        """
        단일 STT 세션을 실행하고 결과를 반환. 이 함수는 동기적으로 실행됨 (별도 스레드에서 호출됨).
        """
        
        FIRST_RESPONSE_TIMEOUT = 3.0
        END_OF_SPEECH_TIMEOUT = 3.0
        
        stt_stop_flag = threading.Event()
        end_of_speech_flag = threading.Event()
        first_response_event = threading.Event()
        last_response_time = time.time()

        def false_start_checker():
            """첫 응답 타임아웃 체크. VAD 오감지 등으로 STT가 시작됐지만 실제 음성이 없을 때를 대비."""
            if not first_response_event.wait(timeout=FIRST_RESPONSE_TIMEOUT):
                logging.warning(f"STT 첫 응답 타임아웃 - 세션 종료")
                stt_stop_flag.set()

        def speech_end_checker():
            """사용자의 발화가 끝났는지(일정 시간 동안 응답이 없는지) 체크."""
            while not stt_stop_flag.is_set() and not end_of_speech_flag.is_set():
                if time.time() - last_response_time > END_OF_SPEECH_TIMEOUT:
                    logging.info(f"{END_OF_SPEECH_TIMEOUT}초 동안 STT 응답이 없어 오디오 전송을 중단합니다.")
                    end_of_speech_flag.set()
                    break
                time.sleep(0.1)

        false_start_check_thread = threading.Thread(target=false_start_checker, daemon=True, name="STTFalseStartChecker")
        false_start_check_thread.start()

        speech_end_check_thread = None

        try:
            audio_gen = self._stt_audio_generator(pre_buffer, audio_queue, stt_stop_flag, end_of_speech_flag)
            responses = self.stt_client.streaming_recognize(self.stt_streaming_config, audio_gen)
            
            for response in responses:
                if stt_stop_flag.is_set(): return
                
                last_response_time = time.time()

                if not first_response_event.is_set():
                    first_response_event.set()
                    speech_end_check_thread = threading.Thread(target=speech_end_checker, daemon=True, name="STTSpeechEndChecker")
                    speech_end_check_thread.start()

                    # C++ 클라이언트에게 인터럽션 신호 전송 (asyncio 루프에 스케줄링)
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send(json.dumps({"type": "user_interruption"})),
                        self.main_loop
                    )

                if not response.results or not response.results[0].alternatives:
                    continue

                result = response.results[0]
                transcript = result.alternatives[0].transcript

                if result.is_final:
                    final_text = transcript.strip()
                    logging.info(f"✅ STT 최종 결과: '{final_text}'")
                    if final_text:
                        self.main_loop.call_soon_threadsafe(self.stt_result_queue.put_nowait, final_text)
                    
                    stt_completion_time = int(time.time() * 1000)
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send(json.dumps({"type": "stt_done", "stt_done_time": stt_completion_time})),
                        self.main_loop
                    )
                    return
                else:
                    logging.debug(f"✅ STT 중간 결과: '{transcript}'")
                    
        except exceptions.DeadlineExceeded as e:
            logging.error(f"STT 세션 타임아웃(DeadlineExceeded): {e}")
        except Exception as e:
            logging.error(f"STT 세션 중 오류: {e}", exc_info=True)
        finally:
            stt_stop_flag.set()
            end_of_speech_flag.set()
            vad_active_flag.set() # VAD 감지 재개 신호
            logging.info("STT 세션 종료.")
    
# ==================================================================================
# Orchestrator: 오디오 처리 파이프라인 관리
# ==================================================================================

class AudioProcessor:
    """마이크 스트림, VAD, STT를 총괄하여 오디오 처리 파이프라인을 관리하는 클래스."""
    
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
        
        # 오디오 버퍼
        self.audio_queue = queue.Queue() # MicrophoneStream이 채우는 큐
        pre_buffer_max_chunks = math.ceil(self.sample_rate * self.pre_buffer_duration / self.vad_chunk_size)
        self.stt_pre_buffer = deque(maxlen=pre_buffer_max_chunks)

        # 컴포넌트 초기화
        device_idx = find_input_device()
        self.mic_stream = MicrophoneStream(
            audio_queue=self.audio_queue,
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
        self.stt_streamer = GoogleSTTStreamer(
            stt_result_queue=stt_result_queue,
            main_loop=main_loop,
            websocket=websocket,
            sample_rate=self.sample_rate
        )

        # 상태 관리
        self._is_running = threading.Event()
        self.vad_active_flag = threading.Event()
        self.vad_active_flag.set() # 초기에는 VAD 감지 활성화 상태
        self._thread: threading.Thread | None = None

    def _processing_loop(self):
        """
        오디오 큐에서 데이터를 소비하여 처리하는 메인 루프.
        """
        logging.info("🎧 오디오 처리 루프 시작.")
        
        # MicrophoneStream 시작 (sounddevice 내부 스레드 시작)
        self.mic_stream.start()

        while self._is_running.is_set():
            # STT 실행 중일 경우 대기
            self.vad_active_flag.wait()
            if not self._is_running.is_set(): break
            
            self.vad_processor.reset_if_inactive()

            try:
                audio_chunk_int16 = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # 사전 버퍼 저장
            self.stt_pre_buffer.append(audio_chunk_int16)

            # VAD 처리 및 음성 감지 확인
            if self.vad_processor.process_chunk(audio_chunk_int16):
                
                # 음성 감지 시 VAD 루프를 대기 상태로 전환
                self.vad_active_flag.clear() 
                logging.info(f"🗣️ 음성 시작 감지! STT 시작.")
                
                # STT 세션은 별도의 스레드에서 동기적으로 실행
                threading.Thread(
                    target=self.stt_streamer.run_stt_session, 
                    args=(self.stt_pre_buffer, self.audio_queue, self.vad_active_flag),
                    name="STTSessionThread"
                ).start()
                
                # STT 시작과 함께 VAD 상태 초기화
                self.vad_processor.reset()
        
        logging.info("🎧 오디오 처리 루프 종료.")

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
        self.vad_active_flag.set() # 대기 중인 스레드를 깨움
        
        # 2. 처리 루프 스레드 종료 대기
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        # 3. MicrophoneStream 중지 (sounddevice 내부 스레드 종료)
        self.mic_stream.stop()
        
        logging.info("AudioProcessor가 성공적으로 종료되었습니다.")