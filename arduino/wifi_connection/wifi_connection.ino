/*
Copyright (c) 2023 Takanori Fujiwara and S. Sandra Bae
This source code is created by referring to the examples available here:
https://docs.arduino.cc/tutorials/uno-r4-wifi/wifi-examples
*/

#include <CapacitiveSensorR4.h> // to use Arduino Uno R3, change this to Paul Bagder's Capacitive Sensing Library
#include <WiFiS3.h>

// WiFi and sensor settings
bool directConnection = true; // if true, arduino will be a WiFi access point (IP: 192.168.4.1)
char ssid[] = "networkphys"; // must 8 or more chars for direct connection mode
char pass[] = "sensing!"; // must 8 or more chars for direct connection mode
int port = 1021;
CapacitiveSensor cs = CapacitiveSensor(5, 2); // NOTE: resistor should be pin 5 (start => in node)

int status = WL_IDLE_STATUS;
WiFiServer server(port);

void printWiFiStatus() {
  // print the SSID of the network you're attached to:
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  // print your WiFi shield's IP address:
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  // print where to go in a browser:
  Serial.print("To see the response, from Terminal, run: nc ");
  Serial.print(ip);
  Serial.print(" ");
  Serial.println(port);

  if (directConnection) {
    Serial.print("Don't forget connecting Arduino WiFi");
  }
}

void setup() {
  Serial.begin(9600); // opens serial port, sets data rate to 9600 bps
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }

  // check for the WiFi module:
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    // don't continue
    while (true);
  }

  String fv = WiFi.firmwareVersion();
  if (fv < WIFI_FIRMWARE_LATEST_VERSION) {
    Serial.println("Please upgrade the firmware");
  }

  if (directConnection) {
    // Create open network.
    status = WiFi.beginAP(ssid, pass);
    if (status != WL_AP_LISTENING) {
      Serial.println("Creating access point failed");
      // don't continue
      while (true);
    }

    // wait 10 seconds for connection:
    delay(10000);
  } else {
    // attempt to connect to WiFi network:
    while (status != WL_CONNECTED) {
      Serial.print("Attempting to connect to SSID: ");
      Serial.println(ssid);
      status = WiFi.begin(ssid, pass);
      delay(10000);
    }
  }

  server.begin();
  printWiFiStatus();
}

void loop() {
  // listen for incoming clients
  WiFiClient client = server.available();
  long sensorValue = cs.capacitiveSensor(30);

  Serial.println(sensorValue);
  if (client) {
    client.println(sensorValue);
  }
}