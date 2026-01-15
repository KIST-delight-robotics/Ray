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
    pkill -f "python.*main.py" 2>/dev/null
    exit
} 

# 만일을 대비한 오디오 소켓 경로 지정
# 사용자 ID(uid)를 가져와서 런타임 디렉토리 지정
if [ -z "$XDG_RUNTIME_DIR" ]; then
    export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi

# 만약 DBus 주소가 없다면 설정
if [ -z "$DBUS_SESSION_BUS_ADDRESS" ]; then
    export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"
fi

# 시작 전에 혹시 남아있는 파이썬 서버 정리
pkill -f "python.*main.py" 2>/dev/null

# 0. 오디오 서비스 재시작
echo "Restarting audio services..."
systemctl --user restart pipewire wireplumber pipewire-pulse
sleep 2 
echo "Audio services restarted."


# 1. Python venv 환경 활성화
VENV_DIR="ray" # 가상환경 디렉토리 이름 설정

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "Error: Cannot find virtual environment at '$VENV_DIR'."
    echo "Please check the directory name."
    exit 1
fi


# 2. 파이썬 서버를 백그라운드에서 실행
echo "Starting Python server..."
python python/main.py &


# 3. 파이썬 서버의 프로세스 ID를 저장 (종료 시 사용)
PYTHON_PID=$!

sleep 1  # 서버가 시작될 때까지 잠시 대기


# 4. C++ 클라이언트 실행
echo "Starting C++ client..."
./build/Ray

echo "All processes finished."