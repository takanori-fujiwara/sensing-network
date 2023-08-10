/*
CapacitiveSensorR4.cpp and CapacitiveSensorR4.h

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae
All rights reserved.

The library is built by referring to CapacitiveSensor Libary (Copyright (c) 2009 Paul Bagder)
https://github.com/PaulStoffregen/CapacitiveSensor
(CapacitiveSensor Libary does not support Arduino Uno R4)

CapacitiveSensorR4 uses Arduino's default digitalWrite, pinMode, digitalRead functions
as Uno R4 has a much faster microcontroller than Uno R3.
*/

#include "Arduino.h"
#include "CapacitiveSensorR4.h"

CapacitiveSensor::CapacitiveSensor(uint8_t sendPin, uint8_t receivePin)
{
	sPin = sendPin;
	rPin = receivePin;

	set_CS_Timeout_Millis(2000); // => R4: 1860000, R3: 620000
	set_CS_AutocaL_Millis(0xFFFFFFFF);

	// get pin mapping and port for send Pin - from PinMode function in core
	if (sendPin >= NUM_DIGITAL_PINS)
		error = -1;
	if (receivePin >= NUM_DIGITAL_PINS)
		error = -1;

	pinMode(sendPin, OUTPUT);	// sendpin to OUTPUT
	pinMode(receivePin, INPUT); // receivePin to INPUT
	digitalWrite(sendPin, LOW);

	// get pin mapping and port for receive Pin - from digital pin functions in Wiring.c
	leastTotal = 0x0FFFFFFFL; // input large value for autocalibrate begin
	lastCal = millis();		  // set millis for start
}

long CapacitiveSensor::capacitiveSensor(uint8_t samples)
{
	total = 0;
	if (samples == 0)
		return 0;
	if (error < 0)
		return -1; // bad pin

	for (uint8_t i = 0; i < samples; i++)
	{ // loop for samples parameter - simple lowpass filter
		if (SenseOneCycle() < 0)
			return -2; // variable over timeout
	}

	// only calibrate if time is greater than CS_AutocaL_Millis and total is less than 10% of baseline
	// this is an attempt to keep from calibrating when the sensor is seeing a "touched" signal
	unsigned long diff = (total > leastTotal) ? total - leastTotal : leastTotal - total;
	if ((millis() - lastCal > CS_AutocaL_Millis) && diff < (int)(.10 * (float)leastTotal))
	{
		leastTotal = 0x0FFFFFFFL; // reset for "autocalibrate"
		lastCal = millis();
	}

	// routine to subtract baseline (non-sensed capacitance) from sensor return
	if (total < leastTotal)
		leastTotal = total; // set floor value to subtract from sensed value

	return (total - leastTotal);
}

long CapacitiveSensor::capacitiveSensorRaw(uint8_t samples)
{
	total = 0;
	if (samples == 0)
		return 0;
	if (error < 0)
		return -1; // bad pin - this appears not to work

	for (uint8_t i = 0; i < samples; i++)
	{ // loop for samples parameter - simple lowpass filter
		if (SenseOneCycle() < 0)
			return -2; // variable over timeout
	}

	return total;
}

int CapacitiveSensor::SenseOneCycle(void)
{
	// For Uno R4, using nomal digitalWrite, pinMode, digitalRead
	// These are expected to be slower than DIGITAL_WRITE_LOW, etc in CapacitiveSensor Library.
	// But, as Uno R4 uses a fast microprocessor, this should be fine
	// TODO: If possible, prepare and use DIGITAL_WRITE_LOW, etc.
	noInterrupts();
	digitalWrite(sPin, LOW);
	pinMode(rPin, INPUT);
	pinMode(rPin, OUTPUT);
	digitalWrite(rPin, LOW);
	delayMicroseconds(100);
	pinMode(rPin, INPUT);
	digitalWrite(sPin, HIGH);
	interrupts();

	// while receive pin is LOW AND total is positive value
	while (!digitalRead(rPin) && (total < CS_Timeout_Millis))
		total++;

	if (total > CS_Timeout_Millis)
		return -2; //  total variable over timeout

	// set receive pin HIGH briefly to charge up fully - because the while loop above will exit when pin is ~ 2.5V
	noInterrupts();
	digitalWrite(rPin, HIGH);
	pinMode(rPin, OUTPUT);
	digitalWrite(rPin, HIGH);
	pinMode(rPin, INPUT);
	digitalWrite(sPin, LOW);
	interrupts();

	// while receive pin is HIGH  AND total is less than timeout
	while (digitalRead(rPin) && (total < CS_Timeout_Millis))
		total++;

	if (total >= CS_Timeout_Millis)
		return -2; // total variable over timeout
	else
		return 1;
}
