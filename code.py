import board
import displayio
import fourwire
import terminalio
import digitalio
from adafruit_display_text import label
from adafruit_display_shapes.circle import Circle # 원형 그리기를 위해 추가
import adafruit_gc9a01a

# 1. 초기화 (백라이트 & 디스플레이)
led_bl = digitalio.DigitalInOut(board.D6)
led_bl.direction = digitalio.Direction.OUTPUT
led_bl.value = True

displayio.release_displays()
spi = board.SPI()
display_bus = fourwire.FourWire(spi, command=board.D3, chip_select=board.D1, reset=board.D2)
display = adafruit_gc9a01a.GC9A01A(display_bus, width=240, height=240)

# 2. 메인 화면 그룹 생성
main_screen = displayio.Group()

# 3. 배경에 원 그리기 (청진기 접촉부 느낌)
inner_circle = Circle(120, 120, 100, outline=0x00FFFF, stroke=3)
main_screen.append(inner_circle)

# 4. 상단 제목 레이블
title_label = label.Label(terminalio.FONT, text="AI STETHOSCOPE", color=0xFFFFFF, x=75, y=40)
main_screen.append(title_label)

# 5. 중앙 상태 메시지 (강조)
status_label = label.Label(terminalio.FONT, text="PRESS TO\nSTART", color=0x00FF00, x=90, y=115)
main_screen.append(status_label)

# 6. 하단 배터리 정보 (핀 맵 P0.14 데이터용)
bat_label = label.Label(terminalio.FONT, text="BAT: 100%", color=0xAAAAAA, x=95, y=200)
main_screen.append(bat_label)

# 화면에 적용
display.root_group = main_screen

print("새로운 UI 화면 적용 완료!")

while True:
    pass
