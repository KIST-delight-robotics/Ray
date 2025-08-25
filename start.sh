# 스크립트 종료시 cleanup 함수 실행
trap cleanup EXIT

cleanup() {
    echo -e "\nExiting... Shutting down background processes."
    # PYTHON_PID 변수가 설정되어 있을 때만 kill 명령 실행
    if [ ! -z "$PYTHON_PID" ]; then
        kill $PYTHON_PID 2>/dev/null
        sleep 1
        # 프로세스가 아직 살아있으면 강제 종료
        if ps -p $PYTHON_PID > /dev/null 2>&1; then
            kill -9 $PYTHON_PID 2>/dev/null
        fi
        echo "Python server stopped."
    fi
    
    # 혹시 남아있는 포트 5000 점유 프로세스 정리
    pkill -f "python.*gpt_server.py" 2>/dev/null
    exit
} 

# Conda 환경 활성화
eval "$(conda shell.bash hook)"
conda activate ray

# 1. 파이썬 서버를 백그라운드에서 실행
echo "Starting Python server..."
python python/gpt_server.py &

# 2. 파이썬 서버의 프로세스 ID를 저장 (종료 시 사용)
PYTHON_PID=$!

sleep 3  # 서버가 시작될 때까지 잠시 대기

# 3. C++ 클라이언트 실행
echo "Starting C++ client..."
./bin/Ray

echo "All processes finished."