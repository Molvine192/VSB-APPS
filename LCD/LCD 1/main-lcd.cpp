#include "mbed.h"
#include "lcd_lib.h"
#include "fonts/font10x16_lsb.h"

extern uint8_t g_font8x8[256][8];

DigitalOut g_led_PTA1(PTA1, 0);
DigitalOut g_led_PTA2(PTA2, 0);

uint16_t l_color_red = 0xF800;
uint16_t l_color_green = 0x07E0;
uint16_t l_color_blue = 0x001F;
uint16_t l_color_white = 0xFFFF;

class Button {
private:
	DigitalIn m_button;
	bool previousState = 0;
	bool isActive = 0;

public:
	Button(PinName button) : m_button(button) {};

	bool getIsActive() { return this->isActive; }
	bool getPressStatus() { return this->m_button; }
	void usedActiveState() { this->isActive = 0; }

	void updateState()
	{
		if (!this->m_button && this->previousState)
		{
			this->previousState = 0;
			this->isActive = !this->isActive;

		}
		if (this->m_button && !this->previousState)
		{
			this->previousState = 1;
		}
	}
	;

};

Button button = PTC9;

void drawChar(int char_height, int char_width, int x, int y, const uint16_t character[], uint16_t Color)
{
	for (int i = 0; i < char_height; ++i)
	{
		for (int j = 0; j < char_width; ++j)
		{
			if ((character[i] >> (char_width - 1 - j)) & 0x01)
			{
				lcd_put_pixel(x + j, y + i, Color);
			}
		}
	}
}

void drawFont(int char_height, int char_width, uint16_t Color)
{
	for (int i = 0; i < 256; i++)
	{
		drawChar(char_height, char_width, (char_width + 1) * i % (320 - char_width), char_height * ((char_width + 1) * i / (320 - char_width)), font[i], Color);
	}
}

void drawRectangle(int x, int y, int height, int widht, uint16_t Color)
{
	for (int i = 0; i < height; i++) {
		lcd_put_pixel(i + x, y, Color);
		lcd_put_pixel(i + x, y + widht, Color);
	}
	for (int j = 0; j < widht ; j++) {
		lcd_put_pixel(x, y + j, Color);
		lcd_put_pixel(x + height, y + j, Color);
	}
}

void drawCircle(int x0, int y0, int radius, uint16_t Color)
{
    int x = radius;
    int y = 0;
    int err = 0;

    while (x >= y) {
        lcd_put_pixel(x0 + x, y0 + y, Color);
        lcd_put_pixel(x0 + y, y0 + x, Color);
        lcd_put_pixel(x0 - y, y0 + x, Color);
        lcd_put_pixel(x0 - x, y0 + y, Color);
        lcd_put_pixel(x0 - x, y0 - y, Color);
        lcd_put_pixel(x0 - y, y0 - x, Color);
        lcd_put_pixel(x0 + y, y0 - x, Color);
        lcd_put_pixel(x0 + x, y0 - y, Color);

        if (err <= 0) {
            y += 1;
            err += 2 * y + 1;
        }

        if (err > 0) {
            x -= 1;
            err -= 2 * x + 1;
        }
    }
}

void drawLine(int x0, int y0, int x1, int y1, uint16_t Color)
{
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx, sy;

    if (x0 < x1) sx = 1;
    else sx = -1;

    if (y0 < y1) sy = 1;
    else sy = -1;

    int err = dx - dy;
    int e2;

    while (true)
    {
        lcd_put_pixel(x0, y0, Color);

        if (x0 == x1 && y0 == y1) break;

        e2 = 2 * err;

        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }

        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

void drawFigureByID(int id)
{
	if (id == 0)
		drawFont(16,10, l_color_white);
	else if (id == 1)
		drawCircle(160, 120, 64, l_color_green);
	else if (id == 2)
	{
		drawRectangle(50, 50, 128, 128, l_color_white);
		drawLine(178, 50, 50, 178, l_color_red);
		drawLine(50, 50, 178, 178, l_color_blue);
	}
}

int main() {
	lcd_init();

	int state = 0;
	drawFigureByID(state);

	while (1)
	{
		button.updateState();
		if (button.getIsActive())
		{
			button.usedActiveState();
			lcd_init();
			state++;
			if (state >= 3) { state = 0; }
			drawFigureByID(state);
		}
	}
	return 0;
}
