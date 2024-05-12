#include "mbed.h"

class PWMLED
{
public:
    Ticker m_ticker;
    Ticker b_ticker;
    DigitalOut m_led;
    bool zvyseni = true;
    float max_period = 15;
    float current_delay = 0;
    int m_ticks = 0;
    bool can_blink = false;
    PWMLED(PinName led) : m_led(led)
    {
        this->m_ticker.attach(callback(this, &PWMLED::pwm), 1ms);
        this->b_ticker.attach(callback(this, &PWMLED::ticker_jas), 50ms);
    }

    void ticker_jas()
    {
    	if (!this->can_blink) {return;}
        if (this->zvyseni)
        {
            this->current_delay += this->max_period * 0.05;
            if (this->current_delay >= this->max_period)
            {
                this->zvyseni = false;
            }
        }

        else
        {
            this->current_delay -= this->max_period * 0.05;
            if (this->current_delay <= 0)
            {
                this->zvyseni = true;
            }
        }
    }

    void nastav_jas(float jas)
    {
        if (jas > 1)
        {
            jas = 1;
        }
        if (jas < 0)
        {
            jas = 0;
        }
        this->current_delay = (max_period * (1 - jas));
    }

    void pwm()
    {
        if (this->m_ticks < this->current_delay)
        {
            this->m_led.write(0);
        }
        else
        {
            this->m_led.write(1);
        }
        this->m_ticks++;
        if (this->m_ticks >= this->max_period)
        {

            this->m_ticks = 0;
        }
    }

    void setBlinkStatus(bool b)
    {
    	this->can_blink = b;
    }
};

class Button
{
private:
    DigitalIn m_button;
    bool previousState = 0;
    bool isActive = 0;

public:
    int clickCount = 0;

    Button(PinName button) : m_button(button){};

    bool getIsActive() { return this->isActive; }
    bool getPressStatus() { return this->m_button; }
    void usedActiveState() { this->isActive = 0; clickCount = 0; }

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
            clickCount++;
        }
    };
};

class ledcontrols
{
private:
    DigitalOut m_led;
    int current_tick = 0;
    float max_delay_period = 26;
    float current_delay = 10;
    float current_brightness = 1;

public:
    ledcontrols(PinName led) : m_led(led)
    {
    }

    void setBrightness(float brightness)
    {
        if (brightness > 1)
        {
            brightness = 1;
        }
        if (brightness < 0)
        {
            brightness = 0;
        }
        this->current_brightness = brightness;
        this->current_delay = (this->max_delay_period * (1 - brightness));
    }

    void turnOff()
    {
        this->setBrightness(0);
    }

    void pwm()
    {
        if (this->current_tick <= this->current_delay)
        {
            this->m_led.write(0);
        }
        else
        {
            this->m_led.write(1);
        }
        this->current_tick++;
        if (this->current_tick > this->max_delay_period)
        {
            this->current_tick = 0;
        }
    }

    float getBrightness()
    {
        return this->current_brightness;
    }
};

ledcontrols rgb_led[] =
    {
        {PTB2},
        {PTB3},
        {PTB9},
        {PTB10},
        {PTB11},
        {PTB18},
        {PTB19},
        {PTB20},
        {PTB23}};

PWMLED red_led_left[] ={{ PTC0 },{ PTC1 },{ PTC2 },{ PTC3 }};
PWMLED red_led_right[] ={{ PTC4 },{ PTC5 },{ PTC7 },{ PTC8 }};

Button buttonPTC9 = PTC9;
Button buttonPTC10 = PTC10;
Button buttonPTC11 = PTC11;
Button buttonPTC12 = PTC12;

void allRgbLightsControl()
{
    for (int i = 0; i < 9; i++)
    {
        rgb_led[i].pwm();
    }
}

void turnOffAllRgbLights()
{
    for (int i = 0; i < 9; i++)
    {
        rgb_led[i].setBrightness(0);
    }
}

void turnOffLeftLedLights()
{
    for (int i = 0; i < 4; i++)
    {
    	red_led_left[i].nastav_jas(0);
    }
}

void turnOffRightLedLights()
{
    for (int i = 0; i < 4; i++)
    {
    	red_led_right[i].nastav_jas(0);
    }
}

int main()
{
	turnOffAllRgbLights();
	turnOffLeftLedLights();
	turnOffRightLedLights();

    Ticker ticker;
    ticker.attach(callback(allRgbLightsControl), 1ms);

    while (1)
    {
    	buttonPTC9.updateState();
    	buttonPTC10.updateState();
    	buttonPTC11.updateState();
    	buttonPTC12.updateState();

    	if (!buttonPTC9.getPressStatus() & !buttonPTC10.getPressStatus() & !buttonPTC11.getPressStatus() & !buttonPTC12.getPressStatus())
    	{
    		for (int i = 0; i < 4; i++)
    		{
    			red_led_right[i].nastav_jas(0);
    			red_led_right[i].setBlinkStatus(true);
    		}
    		for (int i = 0; i < 4; i++)
    		{
    			red_led_left[i].nastav_jas(0);
    			red_led_left[i].setBlinkStatus(true);
    		}
    		buttonPTC9.usedActiveState();
    		buttonPTC12.usedActiveState();
    	}
    	else
    	{
    		if (!buttonPTC11.getPressStatus())
    		{
    			turnOffAllRgbLights();
    			rgb_led[0].setBrightness(1);
    			rgb_led[6].setBrightness(1);
    			buttonPTC11.usedActiveState();
    		}
    		else if (!buttonPTC10.getPressStatus())
    		{
    			turnOffAllRgbLights();
    			rgb_led[0].setBrightness(0.05);
    			rgb_led[6].setBrightness(0.05);
    			rgb_led[3].setBrightness(0.25);
    			rgb_led[4].setBrightness(0.25);
    			rgb_led[5].setBrightness(0.25);
    			buttonPTC10.usedActiveState();
    		}
    		else if (buttonPTC9.clickCount >= 3)
    		{
    			for (int i = 0; i < 4; i++)
    			{
    				red_led_right[i].nastav_jas(0);
    				red_led_right[i].setBlinkStatus(false);
    			}
    			for (int i = 0; i < 4; i++)
    			{
    				red_led_left[i].nastav_jas(0);
    				red_led_left[i].setBlinkStatus(true);
    			}
    			buttonPTC9.usedActiveState();
    		}
    		else if (buttonPTC12.clickCount >= 3)
    		{
    			for (int i = 0; i < 4; i++)
    			{
    				red_led_left[i].nastav_jas(0);
    				red_led_left[i].setBlinkStatus(false);
    			}
    			for (int i = 0; i < 4; i++)
    			{
    				red_led_right[i].nastav_jas(0);
    				red_led_right[i].setBlinkStatus(true);
    			}
    			buttonPTC12.usedActiveState();
    		}
    	}
    }
}
