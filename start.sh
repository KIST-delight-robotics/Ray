# 스크립트 종료시 cleanup 함수 실행
trap cleanup EXIT

cleanup() {
    echo -e "\nExiting... Shutting down background processes."
    # PYTHON_PID 변수가 설정되어 있을 때만 kill 명령 실행
    if [ ! -z "$PYTHON_PID" ]; then
        kill $PYTHON_PID
        echo "Python server stopped."
    fi
    exit
} 

# 1. 파이썬 서버를 백그라운드에서 실행
echo "Starting Python server..."
python3 python/gpt_server.py &

# 2. 파이썬 서버의 프로세스 ID를 저장 (종료 시 사용)
PYTHON_PID=$!

sleep 3  # 서버가 시작될 때까지 잠시 대기

# 3. C++ 클라이언트 실행
echo "Starting C++ client..."
./bin/Ray

echo "All processes finished."