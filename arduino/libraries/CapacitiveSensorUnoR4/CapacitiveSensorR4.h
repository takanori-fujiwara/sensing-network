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

#ifndef CapacitiveSensorR4_h
#define CapacitiveSensorR4_h

class CapacitiveSensor
{
public:
    CapacitiveSensor(uint8_t sendPin, uint8_t receivePin);
    long capacitiveSensorRaw(uint8_t samples);
    long capacitiveSensor(uint8_t samples);
    void set_CS_Timeout_Millis(unsigned long timeout_millis)
    {
        CS_Timeout_Millis = (timeout_millis * (float)loopTimingFactor * (float)F_CPU) / 16000000; // floats to deal with large numbers
    };
    void reset_CS_AutoCal() { leastTotal = 0x0FFFFFFFL; };
    void set_CS_AutocaL_Millis(unsigned long autoCal_millis) { CS_AutocaL_Millis = autoCal_millis; };

private:
    const unsigned int loopTimingFactor = 310; // determined empirically (from Capancitive Sensor Library);
    int error = 1;
    unsigned long leastTotal;
    unsigned long CS_Timeout_Millis;
    unsigned long CS_AutocaL_Millis;
    unsigned long lastCal;
    unsigned long total;
    uint8_t sPin;
    uint8_t rPin;
    int SenseOneCycle(void);
};

#endif
