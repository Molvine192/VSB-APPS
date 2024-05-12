#include <mbed.h>

#include "i2c-lib.h"
#include "si4735-lib.h"

#define R 0b00000001
#define W 0b00000000

int zmena = 0;

#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

DigitalOut led1(PTA1, 0);
DigitalOut led2(PTA2, 0);

DigitalIn buttonPTC9(PTC9);
DigitalIn buttonPTC10(PTC10);
DigitalIn buttonPTC11(PTC11);
DigitalIn buttonPTC12(PTC12);

uint8_t i2c_out_in(uint8_t t_adr, uint8_t *t_out_data, uint32_t t_out_len,
		uint8_t *t_in_data, uint32_t t_in_len)
{
	i2c_start();

	uint8_t l_ack = i2c_output(t_adr | W);

	if (l_ack == 0)
	{
		for (int i = 0; i < t_out_len; i++)
		{
			l_ack |= i2c_output(t_out_data[i]); // send all t_out_data
		}
	}

	if (l_ack != 0) // error?
	{
		i2c_stop();
		return l_ack;
	}

	if (t_in_data != nullptr)
	{
		i2c_start(); // repeated start

		l_ack |= i2c_output(t_adr | R);

		for (int i = 0; i < t_in_len; i++)
		{
			i2c_ack();
			t_in_data[i] = i2c_input(); // receive all t_data_in
		}

		i2c_nack();
	}

	i2c_stop();

	return l_ack;
}

class Expander
{
public:
	void LED(int login)
	{
		int l_ack = 0;
		uint8_t led = 0b00000000;
		for (uint8_t i = 0; i < login; i++)
		{
			i2c_start();
			l_ack = i2c_output(0b01000000);
			l_ack = i2c_output(led);
			i2c_stop();
			led = led << 1;
			led += 1;
			wait_us(250000);
		}
		for (uint8_t i = login; i > 0; i--)
		{
			i2c_start();
			l_ack = i2c_output(0b01000000);
			l_ack = i2c_output(led);
			i2c_stop();
			led = led >> 1;
			wait_us(250000);
		}
		i2c_start();
		l_ack = i2c_output(0b01000000);
		l_ack = i2c_output(0b00000000);
		i2c_stop();
	}
};

class Radio
{
public:
	uint8_t set_volume(uint16_t sila)
	{
		uint8_t l_data_out[6] =
		{ 0x12, 0x00, 0x40, 0x00, 0x00, sila };
		return i2c_out_in( SI4735_ADDRESS, l_data_out, 6, nullptr, 0);
	}
	uint8_t search_freq(int s)
	{
		uint8_t l_data_out[2] =
		{ 0x21, s };
		return i2c_out_in(SI4735_ADDRESS, l_data_out, 2, nullptr, 0);
	}
	uint8_t set_freq(uint16_t t_freq)
	{
		uint8_t l_data_out[5] =
		{ 0x20, 0x00, t_freq >> 8, t_freq & 0xFF, 0 };
		return i2c_out_in( SI4735_ADDRESS, l_data_out, 5, nullptr, 0);
	}
	uint8_t get_tune_status(uint8_t *t_data_status, uint32_t t_data_len)
	{
		uint8_t l_data_out[2] =
		{ 0x22, 0 };
		return i2c_out_in( SI4735_ADDRESS, l_data_out, 2, t_data_status,
				t_data_len);
	}
};

int main()
{
	Expander led;
	int login = 8;

	int volume = 30;
	int freq = 8750;
	Radio radio;
	int l_ack;

	i2c_init();
	if ((l_ack = si4735_init() != 0))
	{
		printf("Error (%d)\r\n", l_ack);
		return 0;
	}
	else
		printf("Initialized.\n"
				"Current state:\t volume: %d\t frequency: %d.%dMHz\r\n\n",
				volume, freq / 100, freq % 100);

	l_ack = radio.set_freq(freq);
	l_ack = radio.set_volume(volume);

	bool is_search = true;

	while (1)
	{
		if (!buttonPTC10)
		{
			volume += 1;
			if (volume >= 63)
				volume = 20;
			l_ack = radio.set_volume(volume);
		}

		if (!buttonPTC11)
		{
			if (is_search)
			{
				l_ack = radio.search_freq(0b00000100);
			}
			else
			{
				freq -= 75;
				if (freq < 8300)
					freq = 10800;

				l_ack = radio.set_freq(freq);
			}
		}

		if (!buttonPTC12)
		{
			if (is_search)
			{
				l_ack = radio.search_freq(0b00001100);
			}
			else {
				freq += 75;
				if (freq >= 10800)
					freq = 8300;

				l_ack = radio.set_freq(freq);
			}

		}

		if (!buttonPTC9)
		{
			led.LED(login);
		}

		if (!buttonPTC10 || !buttonPTC11)
		{
			if (l_ack != 0)
				printf("Error!\r\n");
			else
				printf("Current state:\t volume: %d\t frequency: %d.%dMHz\r\n\n", volume, freq / 100, freq % 100);
		}

	}

	return 0;
}
