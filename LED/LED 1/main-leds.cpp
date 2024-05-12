#include "mbed.h"


class PWMLED  {
public:
	Ticker m_ticker;
    Ticker b_ticker;
    DigitalOut m_led;
    bool zvyseni = true;
    float max_period = 15;
    float current_delay = 0;
    int m_ticks = 0;

    PWMLED(PinName led) : m_led(led)
    {
    	this->m_ticker.attach(callback(this, &PWMLED::pwm), 1ms);
        this->b_ticker.attach(callback(this, &PWMLED::ticker_jas), 100ms);
    }

    void ticker_jas()
    {
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
};

PWMLED g_red_led[] = {{PTC0}, {PTC1}, {PTC2}, {PTC3}, {PTC4}, {PTC5}, {PTC7}, {PTC8}};

int main()
{
    g_red_led[0].nastav_jas(0.05);
    g_red_led[1].nastav_jas(0.95);
    g_red_led[2].nastav_jas(0.05);
    g_red_led[3].nastav_jas(0.95);
    g_red_led[4].nastav_jas(0.05);
    g_red_led[5].nastav_jas(0.95);
    g_red_led[6].nastav_jas(0.05);
    g_red_led[7].nastav_jas(0.95);

    while (1)
    {

    }
}
