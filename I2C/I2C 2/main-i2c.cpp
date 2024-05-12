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
		}
	}
};

class Radio
{
public:
	int lastQuality = 0;
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

	void rssi_snr()
	{
		uint8_t l_ack = 0;
		uint8_t l_S1, l_S2, l_RSSI, l_SNR, l_MULT, l_CAP;
		int l_freq;
		i2c_start();
		l_ack |= i2c_output( SI4735_ADDRESS | W);
		l_ack |= i2c_output(0x22);
		l_ack |= i2c_output(0x00);
		i2c_start();
		l_ack |= i2c_output( SI4735_ADDRESS | R);
		l_S1 = i2c_input();
		i2c_ack();
		l_S2 = i2c_input();
		i2c_ack();
		l_freq = (uint32_t) i2c_input() << 8;
		i2c_ack();
		l_freq |= i2c_input();
		i2c_ack();
		l_RSSI = i2c_input();
		i2c_ack();
		l_SNR = i2c_input();
		i2c_ack();
		l_MULT = i2c_input();
		i2c_ack();
		l_CAP = i2c_input();
		i2c_nack();
		i2c_stop();
		printf("RSSI:%d, SNR:%d\r\n\n", l_RSSI, l_SNR);
		lastQuality = (((float)l_SNR/(float)l_RSSI)*8);
		printf("Quality:%d\n", lastQuality);
	}

	int getQuality()
	{
		return lastQuality;
	}
};

int main()
{
	Expander led;

	int volume = 30;
	int freq = 9100;
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

	while (1)
	{
		if (!buttonPTC9 && !buttonPTC10)
		{
			if (l_ack != 0)
				printf("Error!\r\n");
			else
				printf(
						"Current state:\t volume: %d\t frequency: %d.%dMHz\r\n\n",
						volume, freq / 100, freq % 100);
			while (!buttonPTC9 && !buttonPTC10)
				;
		}

		if (!buttonPTC9)
		{
			radio.rssi_snr();
			while (!buttonPTC9 && buttonPTC10)
				;
		}

		if (!buttonPTC10)
		{
			volume += 1;
			if (volume >= 63)
				volume = 20;
			l_ack = radio.set_volume(volume);
			while (!buttonPTC10 && buttonPTC9)
				;
		}

		if (!buttonPTC11)
		{
			freq -= 75;
			if (freq < 8300)
				freq = 10800;

			l_ack = radio.set_freq(freq);
			led.LED(radio.getQuality());
			while (!buttonPTC11 && buttonPTC12)
				;
		}

		if (!buttonPTC12)
		{
			freq += 75;
			if (freq >= 10800)
				freq = 8300;

			l_ack = radio.set_freq(freq);
			while (!buttonPTC12 && buttonPTC11)
				;
		}

		if (!buttonPTC11 && !buttonPTC12)
		{
			radio.search_freq(0b00001100);
			while (!buttonPTC11 && !buttonPTC12)
				;
		}

	}

	return 0;
}
