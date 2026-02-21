import time
time.sleep(3)

import board
import busio
import displayio
import fourwire
import digitalio
import adafruit_gc9a01a
import terminalio
from adafruit_display_text import label
from adafruit_display_shapes.circle import Circle
from adafruit_display_shapes.rect import Rect

displayio.release_displays()

spi = busio.SPI(clock=board.D8, MOSI=board.D9)
cs  = digitalio.DigitalInOut(board.D1)
dc  = digitalio.DigitalInOut(board.D0)
rst = digitalio.DigitalInOut(board.D7)

display_bus = fourwire.FourWire(spi, command=dc, chip_select=cs, reset=rst, baudrate=1000000)
display = adafruit_gc9a01a.GC9A01A(display_bus, width=240, height=240)

splash = displayio.Group()
display.root_group = splash

# 배경
bg_bitmap = displayio.Bitmap(240, 240, 1)
bg_palette = displayio.Palette(1)
bg_palette[0] = 0x000000
bg = displayio.TileGrid(bg_bitmap, pixel_shader=bg_palette)
splash.append(bg)

# 외곽 원 (파랑)
splash.append(Circle(120, 120, 115, outline=0x0088FF))
splash.append(Circle(120, 120, 112, outline=0x004488))

# 내부 원
splash.append(Circle(120, 120, 85, outline=0x003366))

# 심장 (빨간 원)
splash.append(Circle(120, 108, 20, fill=0xFF0033, outline=0xFF6666))

# 제목 (중앙 정렬)
title = label.Label(terminalio.FONT, text="AI STETHOSCOPE", color=0x00AAFF)
title.x = 120 - title.bounding_box[2] // 2
title.y = 32
splash.append(title)

# 구분선
splash.append(Rect(60, 140, 120, 2, fill=0x003366))

# READY 텍스트 (중앙)
status = label.Label(terminalio.FONT, text="READY", color=0x00FF88)
status.x = 120 - status.bounding_box[2] // 2
status.y = 158
splash.append(status)

# 레벨미터 배경
splash.append(Rect(55, 172, 130, 10, fill=0x001133, outline=0x003366))

# 레벨미터 바 (4칸으로 나눔)
for i in range(4):
    splash.append(Rect(58 + i * 32, 174, 28, 6, fill=0x0055FF))

# BLE 상태
ble = label.Label(terminalio.FONT, text="BLE: OFF", color=0x334455)
ble.x = 120 - ble.bounding_box[2] // 2
ble.y = 200
splash.append(ble)

print("UI OK")

while True:
    pass
