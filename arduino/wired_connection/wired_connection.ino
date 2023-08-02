#include <CapacitiveSensorR4.h> // to use Arduino Uno R3, change this to Paul Bagder's Capacitive Sensing Library

CapacitiveSensor cs = CapacitiveSensor(5, 2);
void setup() {
  Serial.begin(9600); // opens serial port, sets data rate to 9600 bps
}

void loop() {
   //resistor should be pin 5 (start => in node)
  long sensorValue = cs.capacitiveSensor(30);
  Serial.println(sensorValue);
} 
