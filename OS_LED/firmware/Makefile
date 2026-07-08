# OS_LED ATtiny85 firmware build & flash
#
# Targets:
#   make            -> build firmware.hex
#   make fuses      -> program lfuse=0xE1 (16 MHz PLL, no CKDIV8)
#   make flash      -> upload firmware.hex via Pi 5 linuxspi
#   make clean

MCU      := attiny85
F_CPU    := 16000000UL

CC       := avr-gcc
OBJCOPY  := avr-objcopy
SIZE     := avr-size

CFLAGS   := -mmcu=$(MCU) -DF_CPU=$(F_CPU) -Os -Wall -Wextra \
            -std=gnu11 -ffunction-sections -fdata-sections
LDFLAGS  := -mmcu=$(MCU) -Wl,--gc-sections

SRC      := main/main.c
TARGET   := firmware

AVRDUDE      := sudo /usr/bin/avrdude
AVRDUDE_PROG := -c linuxspi -P /dev/spidev0.0:/dev/gpiochip4:22 -B 100 -i 100 -x disable_no_cs
AVRDUDE_PART := -p t85

all: $(TARGET).hex

$(TARGET).elf: $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^
	$(SIZE) --mcu=$(MCU) --format=avr $@

$(TARGET).hex: $(TARGET).elf
	$(OBJCOPY) -O ihex -R .eeprom $< $@

fuses:
	$(AVRDUDE) $(AVRDUDE_PROG) $(AVRDUDE_PART) \
	    -U lfuse:w:0xE1:m -U hfuse:w:0xDF:m -U efuse:w:0xFF:m

flash: $(TARGET).hex
	$(AVRDUDE) $(AVRDUDE_PROG) $(AVRDUDE_PART) -e
	$(AVRDUDE) $(AVRDUDE_PROG) $(AVRDUDE_PART) -D -U flash:w:$<:i

clean:
	rm -f $(TARGET).elf $(TARGET).hex

.PHONY: all fuses flash clean
