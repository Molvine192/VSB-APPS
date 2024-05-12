#include "mbed.h"
#include "lcd_lib.h"
#include "fonts/font10x16_msb.h"

extern uint8_t g_font8x8[256][8];

DigitalOut g_led_PTA1(PTA1, 0);
DigitalOut g_led_PTA2(PTA2, 0);

uint16_t l_color_red = 0xF800;
uint16_t l_color_green = 0x07E0;
uint16_t l_color_blue = 0x001F;
uint16_t l_color_white = 0xFFFF;
uint16_t l_color_black = 0x0000;

Ticker ticker;
//this->m_ticker.attach(callback(this, &PWMLED::pwm), 1ms);

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

void drawRectangle(int x, int y, int height, int widht, uint16_t Color)
{
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < widht ; j++) {
			lcd_put_pixel(i + x, j + y, Color);
		}
	}
}

void cleanScore(int score)
{
	if (score/10 > 0)
	{
		drawChar(16, 16, 260+10, 10+16, font[66+(score/10)-18], l_color_black);
		drawChar(16, 16, 260+20, 10+16, font[66+(score%10)-18], l_color_black);
	}
	else
	{
		drawChar(16, 16, 260+10, 10+16, font[66+score-18], l_color_black);
	}
}

void drawScore(int score)
{
	if (score/10 > 0)
	{
		drawChar(16, 16, 260+10, 10+16, font[66+(score/10)-18], l_color_white);
		drawChar(16, 16, 260+20, 10+16, font[66+(score%10)-18], l_color_white);
	}
	else
	{
		drawChar(16, 16, 260+10, 10+16, font[66+score-18], l_color_white);
	}
}

int platform[] = {1, 30, 20, 0};
int circle[] = {5, 5, 160, 120};
int sx = 1, sy = 1;
int score = 10;

DigitalIn buttonPTC9 = PTC9;
DigitalIn buttonPTC10 = PTC10;
DigitalIn buttonPTC11 = PTC11;
DigitalIn buttonPTC12 = PTC12;

int main() {
	lcd_init();

	drawRectangle(240, 0, 5, 240, l_color_white);
	drawRectangle(platform[2], platform[3], platform[0], platform[1], l_color_white);
	drawRectangle(circle[2], circle[3], circle[0], circle[1], l_color_white);

	drawChar(16, 16, 260, 10, font[101-18], l_color_red);
	drawChar(16, 16, 260+10, 10, font[85-18], l_color_red);
	drawChar(16, 16, 260+20, 10, font[97-18], l_color_red);
	drawChar(16, 16, 260+30, 10, font[100-18], l_color_red);
	drawChar(16, 16, 260+40, 10, font[87-18], l_color_red);
	drawScore(score);

	while (1)
	{
		if (!buttonPTC9)
		{
			drawRectangle(platform[2], platform[3], platform[0], platform[1], l_color_black);
			platform[3]-=5;
			drawRectangle(platform[2], platform[3], platform[0], platform[1], l_color_white);
		}

		if (!buttonPTC10)
		{
			drawRectangle(platform[2], platform[3], platform[0], platform[1], l_color_black);
			platform[3]+=5;
			drawRectangle(platform[2], platform[3], platform[0], platform[1], l_color_white);
		}

		if (circle[2]+sx+circle[0] >= 240)
		{
			sx = -sx;
		}
		else if (circle[2]+sy < 0)
		{
			sx = -sx;
			cleanScore(score);
			score = 0;
			drawScore(score);
			drawRectangle(circle[2], circle[3], circle[0], circle[1], l_color_black);
			circle[2] = 160;
			circle[3] = 120;
		}
		else if (sx < 0 && circle[2]+sx-platform[2]-platform[0] <= 0 && (circle[3] + circle[1] >= platform[3] && circle[3] < platform[3] + platform[1]))
		{
			sx = -sx;
			cleanScore(score);
			score++;
			drawScore(score);
		}
		else if (circle[3]+sy <= 0 || circle[3]+sy+circle[1] >= 240)
			sy = -sy;

		drawRectangle(circle[2], circle[3], circle[0], circle[1], l_color_black);
		circle[2]+=sx;
		circle[3]+=sy;
		drawRectangle(circle[2], circle[3], circle[0], circle[1], l_color_white);
	}
	return 0;
}
